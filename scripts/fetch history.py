# -*- coding: utf-8 -*-
"""
历史日线数据抓取与清洗脚本
==================================================
给 scripts/backtest.py 准备"全历史股票池(含退市股)+ 后复权日线"数据。

不接入 15 分钟的实时定时任务 —— 这个脚本要跑很久(几千只股票逐个请求),
建议手动跑,或者单独挂一个"每周一次"的 workflow。

设计上的几个关键点(和 backtest.py 头部注释呼应):
  - 后复权(adjust="hfq"):避免除权缺口让均线/涨跌幅失真。
  - pre_close 自己算:用同一只股票的上一交易日后复权收盘价(shift(1)),
    不用 akshare 另外给的涨跌额反推,避免复权口径不一致。
  - 停牌:akshare 的历史行情接口本身就不会返回停牌当天的行数,所以停牌日在
    数据里表现为"这只股票这天直接没有记录"——backtest.py 按 date 分组构建
    当天数据时,缺记录的股票自然就不会被处理,等价于停牌跳过,不需要额外
    构造一个 suspended=True 的占位行。
  - 行业:用当前行业分类整体套用到这只股票的全部历史行,不是"当年那个时点"
    真实的行业归属(那需要一套点时行业变更历史,工作量大很多,这里没做,
    是一个已知的简化,不是遗漏)。
  - 断点续传:每只股票存一个独立的 parquet 文件,已经抓过的直接跳过,
    脚本中途失败重跑不会从头再来。

!! 最不确定的一块:获取"已退市股票列表"用的是 ak.stock_info_sh_delist() /
   ak.stock_info_sz_delist(),这是我印象里对应的 akshare 接口,但我这边的
   沙盒连不上 akshare 真实数据源,没法在线验证函数名和返回的列名是否还准确。
   如果跑起来这两个函数报"不存在"或者列名对不上,把报错发我,我再改——
   即使这部分彻底失败,脚本也不会崩,只是会退化成"只用当前存活股票"
   (此时回测会有幸存者偏差,收益会偏乐观),并且会显式打印警告,不是静默出错。 !!
"""

import argparse
import logging
import re
import time
from pathlib import Path
from typing import Optional

import akshare as ak
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch_history")

OUTPUT_DIR = Path("data/history")
DEFAULT_START = "20200101"
SLEEP_SECONDS = 0.3   # 每只股票请求间隔,避免请求过于频繁被数据源限流/封IP
MAX_RETRIES = 3


def get_universe() -> pd.DataFrame:
    """全历史股票池 = 当前存活 + 已退市。返回列: code, name, delisted(bool)"""
    frames = []

    # 当前存活股票:这个接口很基础常用,把握比较大
    try:
        alive = ak.stock_info_a_code_name()
        alive = alive.rename(columns={c: c.lower() for c in alive.columns})
        alive = alive.rename(columns={"code": "code", "name": "name"})
        alive["delisted"] = False
        frames.append(alive[["code", "name", "delisted"]])
        logger.info(f"当前存活股票: {len(alive)} 只")
    except Exception as e:
        logger.error(f"获取存活股票列表失败: {e} —— 这是最基础的数据源,无法继续")
        raise

    # 已退市股票:这部分接口名我没法离线验证,失败会显式警告并降级,不会静默出错
    try:
        delist_sh = ak.stock_info_sh_delist()
        delist_sz = ak.stock_info_sz_delist()
        delisted = pd.concat([delist_sh, delist_sz], ignore_index=True)

        code_col = next((c for c in delisted.columns if "代码" in str(c) or str(c).lower() == "code"), None)
        name_col = next((c for c in delisted.columns if "名称" in str(c) or str(c).lower() == "name"), None)
        if code_col is None:
            raise ValueError(f"无法识别退市列表的代码列,实际列名: {list(delisted.columns)}")

        rename_map = {code_col: "code"}
        if name_col:
            rename_map[name_col] = "name"
        delisted = delisted.rename(columns=rename_map)
        if "name" not in delisted.columns:
            delisted["name"] = delisted["code"]
        delisted["delisted"] = True
        frames.append(delisted[["code", "name", "delisted"]])
        logger.info(f"已退市股票: {len(delisted)} 只")
    except Exception as e:
        logger.warning(
            f"获取退市股票列表失败({e})—— 回测股票池将只包含当前存活股票,"
            f"这会带来幸存者偏差,回测收益会偏乐观,请留意。"
            f"如果这个报错反复出现,大概率是 akshare 的接口名或返回字段变了,发我看看。"
        )

    universe = pd.concat(frames, ignore_index=True).drop_duplicates(subset="code").reset_index(drop=True)
    return universe


def fetch_one_stock(code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """抓取单只股票的后复权(hfq)日线数据,失败自动重试。"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date, end_date=end_date, adjust="hfq",
            )
            if df is None or df.empty:
                return None

            df = df.rename(columns={
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
            })
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # pre_close 用同一只股票自己的前一日后复权收盘价算,不用 akshare 另给的
            # 涨跌额反推,避免复权口径不一致
            df["pre_close"] = df["close"].shift(1)
            df = df.dropna(subset=["pre_close"]).reset_index(drop=True)
            df["code"] = code

            return df[["date", "code", "open", "high", "low", "close", "pre_close", "volume"]]
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{code} 第{attempt}次抓取失败: {e},重试中...")
                time.sleep(1.5 * attempt)
            else:
                logger.error(f"{code} 抓取失败,已放弃(重试{MAX_RETRIES}次): {e}")
                return None


def fetch_industry(code: str) -> str:
    try:
        info = ak.stock_individual_info_em(symbol=code)
        row = info.loc[info["item"] == "行业", "value"]
        if not row.empty:
            return str(row.values[0])
    except Exception:
        pass
    return "未分类"


def main(start_date: str = DEFAULT_START, end_date: Optional[str] = None, sample: Optional[int] = None):
    end_date = end_date or pd.Timestamp.today().strftime("%Y%m%d")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    universe = get_universe()
    if sample:
        universe = universe.head(sample)
        logger.info(f"测试模式:只抓取前 {sample} 只股票")

    total = len(universe)
    ok, skipped, failed = 0, 0, 0

    for i, row in enumerate(universe.itertuples(), start=1):
        code = re.sub(r"[^0-9]", "", str(row.code))[-6:]
        out_path = OUTPUT_DIR / f"{code}.parquet"
        if out_path.exists():
            skipped += 1
            continue  # 断点续传:已经抓过的直接跳过

        hist = fetch_one_stock(code, start_date, end_date)
        if hist is None or hist.empty:
            logger.warning(f"[{i}/{total}] {code} 无数据,跳过(可能是退市股在这个区间内没有交易记录)")
            failed += 1
            time.sleep(SLEEP_SECONDS)
            continue

        hist["industry"] = "已退市" if getattr(row, "delisted", False) else fetch_industry(code)
        hist.to_parquet(out_path, index=False)
        ok += 1

        if i % 50 == 0:
            logger.info(f"进度: {i}/{total}(成功 {ok},跳过 {skipped},失败 {failed})")
        time.sleep(SLEEP_SECONDS)

    logger.info(f"全部完成。成功 {ok},跳过(已存在) {skipped},失败 {failed},总计 {total}")


def load_all_history() -> pd.DataFrame:
    """给 backtest.py 用:把 data/history/ 下所有股票的 parquet 拼成一个大 DataFrame。"""
    files = list(OUTPUT_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{OUTPUT_DIR} 下没有找到任何历史数据,先跑一遍: python scripts/fetch_history.py")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="抓取全历史A股(含退市股)后复权日线数据")
    parser.add_argument("--start", default=DEFAULT_START, help="起始日期,格式 YYYYMMDD")
    parser.add_argument("--end", default=None, help="结束日期,默认今天")
    parser.add_argument("--sample", type=int, default=None, help="只抓前 N 只股票,用于先小规模测试")
    args = parser.parse_args()
    main(start_date=args.start, end_date=args.end, sample=args.sample)
