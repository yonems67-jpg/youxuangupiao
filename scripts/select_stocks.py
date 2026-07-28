import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import re
import time
from typing import Optional
from datetime import datetime, timedelta, timezone, time as dtime


def to_json_safe(obj):
    """
    递归把 numpy/pandas 的标量类型转成原生 Python 类型。
    根因:compute_risk() 里 top_share > 0.4 这类比较,算出来的是 numpy.bool_
    而不是 Python 原生 bool,json 标准库不认这个类型,哪怕它看起来就是 True/False,
    会报 "Object of type bool is not JSON serializable"。这里做成通用的递归转换,
    顺便把 numpy.int64/float64、NaN 也一起处理掉,不只是头痛医头。
    """
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and pd.isna(obj):
        return None
    return obj


def select_stocks():
    # 1. 获取股票池 / 行情数据(腾讯接口)
    stock_list = ak.stock_zh_a_spot_tx()

    # 2. 市场概览(基于全市场快照的客观宽度指标,不是主观判断)
    market = compute_market_overview(stock_list)

    # 3. 第一轮筛选:涨跌幅/换手率/量比/振幅/流动性/相对强度,产出候选池(比最终结果宽)
    candidates = robust_strategy_logic(stock_list)

    # 4. 第二轮:中期趋势确认(20日均线上方)+ 仓位建议,产出最终 15 只
    selected_df = apply_trend_filter_and_finalize(candidates, market)

    # 5. 行业 + 操作建议标签
    selected_df = enrich_with_industry(selected_df)
    selected_df = add_action_tag(selected_df)

    # 6. 风险提示(基于市场概览 + 行业集中度推导,规则透明)
    risk = compute_risk(selected_df, market)

    # 7. 整理输出
    result = {
        "update_date": get_beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": market,
        "risk": risk,
        "stocks": selected_df.to_dict(orient="records"),
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(to_json_safe(result), f, ensure_ascii=False, indent=2)

    print(f"候选 {len(candidates)} 只,趋势确认后最终选出 {len(selected_df)} 只,"
          f"市场评分 {market['market_score']},已写入 site/data/latest.json")


def compute_market_overview(raw_df: pd.DataFrame) -> dict:
    """
    市场概览:完全基于全市场涨跌家数计算,不是主观判断。
    market_score = (上涨家数 - 下跌家数) / 总家数,映射到 0~100。
    50 分代表涨跌家数持平,越接近 100 说明普涨,越接近 0 说明普跌。

    !! 注意:scripts/backtest.py 里的 assess_market_regime() 用的是完全相同的
       0~100 打分公式和 60/40 分档阈值,只是数据来源从"实时快照"换成"历史日线涨跌幅"。
       两边的阈值必须保持一致,否则回测出来的"市场环境判断"和实盘运行的不是同一套规则,
       回测结果就没法代表实盘表现了。 !!
    """
    df = raw_df.copy()
    df["zdf"] = pd.to_numeric(df["zdf"], errors="coerce")

    total = int(df["zdf"].notna().sum())
    up = int((df["zdf"] > 0).sum())
    down = int((df["zdf"] < 0).sum())
    limit_up = int((df["zdf"] >= 9.5).sum())
    limit_down = int((df["zdf"] <= -9.5).sum())

    breadth = (up - down) / total if total else 0  # -1 ~ 1
    market_score = round((breadth + 1) / 2 * 100, 1)  # 0 ~ 100

    if market_score >= 60:
        status = "强势"
        suggested_position = "6~8成"
    elif market_score >= 40:
        status = "震荡"
        suggested_position = "3~5成"
    else:
        status = "弱势"
        suggested_position = "1~2成或观望"

    return {
        "market_score": market_score,
        "market_status": status,
        "suggested_position": suggested_position,
        "up_count": up,
        "down_count": down,
        "limit_up_count": limit_up,
        "limit_down_count": limit_down,
        "total_count": total,
    }


def get_beijing_now() -> datetime:
    # GitHub Actions 跑在 UTC,这里统一转换成北京时间
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)


def trading_minutes_elapsed(now: datetime) -> float:
    """
    返回"用于折算换手率的已交易分钟数",不是字面意义上"现在几点"。
    交易时段:09:30-11:30(120分钟)+ 13:00-15:00(120分钟),全天共240分钟。

    开盘前(< 9:30)单独返回 240,而不是 0——因为这个时间点快照里的换手率其实是
    "昨天收盘的最终值",不是"今天才过了0分钟的部分数据",如果按0分钟折算会把
    一个已经是完整全天的数字硬乘几十倍,数值会离谱地虚高。
    """
    t = now.time()
    open_am, close_am = dtime(9, 30), dtime(11, 30)
    open_pm, close_pm = dtime(13, 0), dtime(15, 0)

    if t < open_am:
        return 240.0
    if t <= close_am:
        return (datetime.combine(now.date(), t) - datetime.combine(now.date(), open_am)).total_seconds() / 60
    if t < open_pm:
        return 120.0
    if t <= close_pm:
        return 120.0 + (datetime.combine(now.date(), t) - datetime.combine(now.date(), open_pm)).total_seconds() / 60
    return 240.0


def robust_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    """
    第一轮筛选,产出候选池(比最终 15 只宽,留给趋势确认再筛一遍)。

    v4 改动(针对策略复盘指出的短板):
      6. 振幅过滤:振幅(zf)超过 8% 的剔除,避免日内波动过于剧烈、风险不可控的票。
      7. 流动性过滤:流通市值(ltsz)低于阈值的剔除,避免小盘股容易被资金快速拉升
         又快速回落。!! ltsz 的具体单位(万元/元/亿元)我没法离线验证,第一次跑
         会打印几个原始样例数值,如果这条过滤器直接把候选股清零了会自动跳过
         (不会让当天结果直接归零),但阈值需要你看了真实数值之后告诉我调整 !!
      8. 相对强度过滤:剔除涨幅没有跑赢当天全市场平均涨幅的股票——不能只看自己
         涨了多少,还要看是不是真的比大盘强。

    v3 及更早的改动(不变):剔除滞后的主力净流入因子、涨跌幅收窄到2%~7%、
    量比>1.8、换手率按已交易时长折算成全天预估值(见 trading_minutes_elapsed)。

    每一条新增过滤都带了"安全阀":如果套上这条规则后候选股直接清零,会打印警告
    并跳过这条规则,不会让参数猜错的某一条过滤器把当天结果整批清空。
    """
    df = stock_list.copy()

    numeric_cols = ["zdf", "hsl", "lb", "zxj", "zd", "zf", "zsz", "ltsz"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    market_avg_zdf = df["zdf"].mean()

    df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]
    df = df[(df["zdf"] >= 2.0) & (df["zdf"] <= 7.0)]

    elapsed = max(trading_minutes_elapsed(get_beijing_now()), 10.0)
    projection_factor = 240.0 / elapsed
    df["hsl_projected"] = df["hsl"] * projection_factor
    df = df[(df["hsl"] >= 0.5) & (df["hsl_projected"] >= 5.0) & (df["hsl_projected"] <= 15.0)]

    df = df[df["lb"] > 1.8]

    # 振幅过滤(带安全阀)
    zf_filtered = df[df["zf"] <= 8.0]
    if not zf_filtered.empty:
        df = zf_filtered
    else:
        print("提示: 振幅过滤后候选股清零,本次跳过振幅过滤")

    # 流通市值过滤(带安全阀;阈值单位未离线验证,见函数说明)
    LTSZ_THRESHOLD = 200000  # 假设单位是"万元",约等于20亿流通市值,需要用下面打印的样例核对
    if "ltsz" in df.columns and df["ltsz"].notna().any():
        print(f"流通市值原始样例(用来核对单位是否猜对): {df['ltsz'].dropna().head(5).tolist()}")
        ltsz_filtered = df[df["ltsz"] >= LTSZ_THRESHOLD]
        if not ltsz_filtered.empty:
            df = ltsz_filtered
        else:
            print("提示: 流通市值过滤后候选股清零(阈值单位可能猜错了),本次跳过这条过滤")

    # 相对强度:今日涨幅要跑赢全市场平均涨幅(带安全阀)
    df["relative_strength"] = df["zdf"] - market_avg_zdf
    rs_filtered = df[df["relative_strength"] > 0]
    if not rs_filtered.empty:
        df = rs_filtered
    else:
        print("提示: 相对强度过滤后候选股清零,本次跳过这条过滤")

    df["score"] = df["hsl_projected"] * 0.6 + df["zdf"] * 0.4
    candidates = df.sort_values("score", ascending=False).head(40)

    return candidates[["code", "name", "zxj", "zdf", "hsl", "hsl_projected", "lb",
                        "zf", "ltsz", "relative_strength", "score"]].rename(
        columns={
            "code": "代码", "name": "名称", "zxj": "最新价", "zdf": "涨跌幅",
            "hsl": "换手率", "hsl_projected": "预估全天换手率", "lb": "量比",
            "zf": "振幅", "ltsz": "流通市值", "relative_strength": "相对强度",
        }
    )


def fetch_trend_confirmed(code_clean: str) -> Optional[bool]:
    """
    用腾讯的历史行情接口(不是被墙的东方财富)确认股票是否处于20日均线上方的
    中期上升趋势,用来过滤"下跌趋势里的反弹股"这个策略复盘指出的第一优先级短板。

    返回 True(确认在均线上方)/ False(在均线下方)/ None(数据拿不到,无法判断——
    调用方会把 None 当"未确认通过"处理)。

    !! 用腾讯而不是东方财富的理由:腾讯的实时快照接口在 GitHub Actions 里一直
       稳定工作,历史接口大概率是同一套基础设施,值得一试,但我这边没法离线验证
       这个接口现在是否可用、字段名是否准确,失败了会打印出来,不影响整体流程
       (见 apply_trend_filter_and_finalize 的安全阀说明)。 !!
    """
    try:
        end = get_beijing_now()
        start = end - timedelta(days=45)
        hist = ak.stock_zh_a_hist_tx(
            symbol=code_clean,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if hist is None or len(hist) < 20:
            return None

        close_col = "close" if "close" in hist.columns else ("收盘" if "收盘" in hist.columns else None)
        if close_col is None:
            print(f"趋势确认失败 code={code_clean}: 无法识别收盘价列,实际列名 {list(hist.columns)}")
            return None

        closes = pd.to_numeric(hist[close_col], errors="coerce").dropna()
        if len(closes) < 20:
            return None

        ma20 = closes.tail(20).mean()
        return bool(closes.iloc[-1] > ma20)
    except Exception as e:
        print(f"趋势确认失败 code={code_clean}: {e}")
        return None


def apply_trend_filter_and_finalize(candidates: pd.DataFrame, market: dict) -> pd.DataFrame:
    """
    对第一轮候选股(通常几十只,远小于全市场)逐个做中期趋势确认,过滤掉"下跌趋势
    里的反弹股"。同时算出每只股票的建议仓位(三、仓位管理模块:市场越强、排名
    越靠前,建议仓位越高,单票仓位有硬上限)。

    !! 安全阀:如果趋势确认这一步全部未通过(数量从几十只变成0),大概率是接口
       异常而不是"今天真的没有符合趋势的股票",这种情况下会自动跳过趋势过滤、
       退回用第一轮候选名单,不会让当天选股结果直接清零。 !!
    """
    if candidates.empty:
        return candidates

    trend_ok = []
    for _, row in candidates.iterrows():
        code_clean = re.sub(r"[^0-9]", "", str(row["代码"]))[-6:]
        trend_ok.append(fetch_trend_confirmed(code_clean) is True)
        time.sleep(0.5)

    candidates = candidates.copy()
    candidates["_trend_confirmed"] = trend_ok
    confirmed_df = candidates[candidates["_trend_confirmed"]]

    if confirmed_df.empty:
        print("警告: 趋势确认这一步全部未通过,大概率是接口异常而不是真的没有符合条件的股票,"
              "本次跳过趋势过滤,直接用第一轮候选名单")
        final = candidates
    else:
        final = confirmed_df

    final = final.drop(columns=["_trend_confirmed"]).sort_values("score", ascending=False).head(15).reset_index(drop=True)

    # 三、仓位管理:市场越强、排名越靠前,建议仓位越高;单票仓位设硬上限
    n = len(final)
    market_multiplier = {"强势": 1.0, "震荡": 0.5, "弱势": 0.0}.get(market["market_status"], 0.5)
    base_pct, max_pct = 20.0, 25.0

    if n > 0:
        ranks = final["score"].rank(ascending=False, method="first")
        signal_factor = 0.5 + 0.5 * (n - ranks) / max(n - 1, 1)
        final["建议仓位"] = (base_pct * market_multiplier * signal_factor).clip(upper=max_pct).round(1)

    return final


def guess_exchange_prefix(code_clean: str) -> str:
    """从6位代码猜交易所前缀,给雪球接口用(它要 SH600000 这种带前缀的格式)。"""
    if code_clean.startswith(("60", "68", "90")):
        return "SH"
    if code_clean.startswith(("8", "43", "92")):
        return "BJ"
    return "SZ"


def fetch_industry_live(code_clean: str) -> str:
    """
    尝试雪球的接口
    """
    try:
        code_prefixed = guess_exchange_prefix(code_clean) + code_clean
        info = ak.stock_individual_basic_info_xq(symbol=code_prefixed)
        row = info.loc[info["item"].astype(str).str.contains("行业|industry", case=False, na=False), "value"]
        if not row.empty and str(row.values[0]).strip():
            return str(row.values[0])
    except Exception:
        pass
    return ""


def enrich_with_industry(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    给入选股票查行业分类,三层兜底:
      1. 先试雪球接口(fetch_industry_live)—— 不同数据源,值得试,但没把握一定能用
      2. 雪球失败就查本地静态映射表 site/data/industry_map.json(用
         scripts/build_industry_map.py 在国内网络生成、手动提交更新)
      3. 都没有就标"未分类"(不会导致脚本崩溃)
    """
    map_path = "site/data/industry_map.json"
    industry_map = {}
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            industry_map = json.load(f)
    else:
        print(f"提示: {map_path} 不存在,雪球接口如果也失败,行业会显示为未分类。"
              f"可以在本地跑一遍 python scripts/build_industry_map.py 再提交进仓库作为兜底。")

    industries = []
    for code in selected_df["代码"]:
        code_clean = re.sub(r"[^0-9]", "", str(code))[-6:]
        industry = fetch_industry_live(code_clean)
        if not industry:
            industry = industry_map.get(code_clean, "未分类")
        industries.append(industry)

    selected_df = selected_df.copy()
    selected_df["行业"] = industries
    return selected_df


def add_action_tag(selected_df: pd.DataFrame) -> pd.DataFrame:
    """操作建议:按入选列表内部的相对排名分三档,是信号强弱标签,不是买卖指令。"""
    df = selected_df.copy()
    n = len(df)
    if n == 0:
        df["操作建议"] = []
        return df

    ranks = df["score"].rank(ascending=False, method="first")

    def tag(rank):
        if rank <= max(1, n // 3):
            return "重点关注"
        elif rank <= max(1, 2 * n // 3):
            return "观察"
        else:
            return "谨慎"

    df["操作建议"] = ranks.apply(tag)
    return df


def compute_risk(selected_df: pd.DataFrame, market: dict) -> dict:
    """风险提示:行业集中度 + 市场风险,全部从已经算好的数据推导,规则透明可追溯。"""
    notes = []

    if len(selected_df) > 0 and "行业" in selected_df.columns:
        counts = selected_df["行业"].value_counts()
        top_industry = counts.index[0]
        top_share = round(counts.iloc[0] / len(selected_df), 2)
    else:
        top_industry = None
        top_share = 0

    industry_warning = top_share > 0.4
    if industry_warning:
        notes.append(f"入选股票中「{top_industry}」占比达 {int(top_share*100)}%,行业集中度偏高")

    market_warning = market["market_status"] == "弱势"
    if market_warning:
        notes.append("当前市场宽度偏弱(下跌家数明显多于上涨家数)")

    suggest_reduce = industry_warning or market_warning

    return {
        "top_industry": top_industry,
        "top_industry_share": top_share,
        "industry_concentration_warning": industry_warning,
        "market_risk_warning": market_warning,
        "suggest_reduce_position": suggest_reduce,
        "notes": notes,
    }


if __name__ == "__main__":
    select_stocks()
