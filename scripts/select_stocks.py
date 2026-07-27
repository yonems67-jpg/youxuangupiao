import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import re
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

    # 2. 市场概览(基于全市场快照的客观宽度指标,不是主观打分)
    market = compute_market_overview(stock_list)

    # 3. 选股逻辑(v2:换手率主导 + 收窄涨跌幅区间,剔除滞后的主力净流入因子——
    #    理由见 robust_strategy_logic 的说明)
    selected_df = robust_strategy_logic(stock_list)

    # 4. 行业 + 操作建议标签
    selected_df = enrich_with_industry(selected_df)
    selected_df = add_action_tag(selected_df)

    # 5. 风险提示(基于市场概览 + 行业集中度推导,规则透明)
    risk = compute_risk(selected_df, market)

    # 6. 整理输出
    result = {
        "update_date": get_beijing_now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": market,
        "risk": risk,
        "stocks": selected_df.to_dict(orient="records"),
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(to_json_safe(result), f, ensure_ascii=False, indent=2)

    print(f"选出 {len(selected_df)} 只股票,市场评分 {market['market_score']},已写入 site/data/latest.json")


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

    breadth = (up - down) / total if total else 0  # -1 ~ 1
    market_score = round((breadth + 1) / 2 * 100, 1)  # 0 ~ 100

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
        return 240.0  # 开盘前:数据是昨天收盘的最终值,不折算
    if t <= close_am:
        elapsed = (datetime.combine(now.date(), t) - datetime.combine(now.date(), open_am)).total_seconds() / 60
        return elapsed
    if t < open_pm:
        return 120.0  # 午休:按上午收盘时的累计分钟数算,是今天的部分数据
    if t <= close_pm:
        return 120.0 + (datetime.combine(now.date(), t) - datetime.combine(now.date(), open_pm)).total_seconds() / 60
    return 240.0  # 收盘后:今天已经是完整一天的数据,不折算


def robust_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    """
    选股逻辑(相对上一版的改动说明):
      1. 不再使用"主力净流入(zljlr)"——这是收盘后才公布的滞后数据,盘中/实时选股用它
         会造成时间错配,本质上是一种"伪因子"(和 scripts/backtest.py 头部注释的
         维度七呼应)。
      2. 涨跌幅收窄到 2%~7%:回避一字涨停(想买也买不进)和刚起步、趋势尚不明朗的
         弱势股,聚焦"温和放量上涨"这种相对健康的短线形态。
      3. 量比下限 1.8:提高"资金确实在活跃换手"的门槛,过滤掉无量空涨的伪强势股。
      4. 换手率改用"按已交易时长折算的全天预估值"而不是原始累计值(v3 新增)——
         换手率是全天累计指标,开盘不久天然就低,如果直接卡"换手率≥5%"这种绝对
         阈值,早盘几乎选不出票,得等到真实累计值爬到5%以上才会出结果,这会导致
         选股结果集中出现在下午,错过早盘的进场时机。折算公式:
         预估全天换手率 = 当前换手率 ×(240 / 已交易分钟数)。已交易分钟数低于10分钟
         时按10分钟算,避免开盘头几分钟样本太小,折算出离谱的数字;同时保留原始
         换手率 ≥0.5% 的地板,过滤纯噪音。
      5. 打分权重"换手率0.6 + 涨跌幅0.4"不变,但换手率部分换成折算后的预估值,
         和筛选口径保持一致,避免"筛选用折算值、排序用原始值"这种自相矛盾。
    """
    df = stock_list.copy()

    # 腾讯接口返回的数值字段经常是字符串类型,直接比较大小会报
    # TypeError: '>' not supported between instances of 'str' and 'int'
    numeric_cols = ["zdf", "hsl", "lb", "zxj", "zd", "zf", "zsz", "ltsz"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. 排除 ST / 退市风险股(name = 名称)
    df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]

    # 2. 涨跌幅区间:2%~7%,剔除一字板(买不进)和涨幅不足的弱势股(zdf = 涨跌幅,单位 %)
    df = df[(df["zdf"] >= 2.0) & (df["zdf"] <= 7.0)]

    # 3. 换手率按已交易时长折算成全天预估值,不用原始累计值(见函数说明第4点)
    elapsed = max(trading_minutes_elapsed(get_beijing_now()), 10.0)
    projection_factor = 240.0 / elapsed
    df["hsl_projected"] = df["hsl"] * projection_factor

    df = df[(df["hsl"] >= 0.5) & (df["hsl_projected"] >= 5.0) & (df["hsl_projected"] <= 15.0)]

    # 4. 量比 > 1.8:量比的官方定义本身已经按同一时段历史均值折算过,不需要再处理(lb = 量比)
    df = df[df["lb"] > 1.8]

    # 5. 综合打分:换手率(折算后)为主、涨跌幅为辅,取分数最高的 15 只
    df["score"] = df["hsl_projected"] * 0.6 + df["zdf"] * 0.4
    selected = df.sort_values("score", ascending=False).head(15)

    return selected[["code", "name", "zxj", "zdf", "hsl", "hsl_projected", "lb", "score"]].rename(
        columns={
            "code": "代码",
            "name": "名称",
            "zxj": "最新价",
            "zdf": "涨跌幅",
            "hsl": "换手率",
            "hsl_projected": "预估全天换手率",
            "lb": "量比",
        }
    )


def guess_exchange_prefix(code_clean: str) -> str:
    """从6位代码猜交易所前缀,给雪球接口用(它要 SH600000 这种带前缀的格式)。"""
    if code_clean.startswith(("60", "68", "90")):
        return "SH"
    if code_clean.startswith(("8", "43", "92")):
        return "BJ"
    return "SZ"


def fetch_industry_live(code_clean: str) -> str:
    """
    尝试雪球的接口——和东方财富是不同公司的数据源,东方财富被墙不代表这个也被墙,
    但我这边同样没法离线验证这个接口现在是否可用、字段名是否还准确,失败很正常,
    失败了会自动落到本地静态映射表兜底,不影响整体流程。
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

    背景:实测 ak.stock_individual_info_em()(东方财富)在 GitHub Actions 环境里
    15/15 全部查询失败,加了重试、请求间隔也没救回来任何一个,更像是东方财富把
    GitHub Actions 用的海外 IP 段整体限制/屏蔽了,所以不再单独依赖它。
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
