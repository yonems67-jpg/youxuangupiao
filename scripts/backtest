import akshare as ak
import pandas as pd
import json
import os
import re
from datetime import datetime


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
        "update_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market": market,
        "risk": risk,
        "stocks": selected_df.to_dict(orient="records"),
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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


def robust_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    """
    选股逻辑(相对上一版的改动说明):
      1. 不再使用"主力净流入(zljlr)"——这是收盘后才公布的滞后数据,盘中/实时选股用它
         会造成时间错配,本质上是一种"伪因子"(和 scripts/backtest.py 头部注释的
         维度七呼应)。
      2. 涨跌幅收窄到 2%~7%:回避一字涨停(想买也买不进)和刚起步、趋势尚不明朗的
         弱势股,聚焦"温和放量上涨"这种相对健康的短线形态。
      3. 换手率下限提到 5%(原 3%),量比下限提到 1.8(原 1.5):提高"资金确实在
         活跃换手"的门槛,过滤掉无量空涨的伪强势股。
      4. 打分权重从"涨跌幅/量比各半"改为"换手率0.6 + 涨跌幅0.4":短线里资金活跃度
         (换手率)比单纯涨幅更能反映当下的关注度和参与意愿。
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

    # 3. 换手率区间:5%~15%,说明有持续增量资金参与换手,而非无量空涨(hsl = 换手率,单位 %)
    df = df[(df["hsl"] >= 5.0) & (df["hsl"] <= 15.0)]

    # 4. 量比 > 1.8:今日成交量相对近期明显放大,说明确有资金在动(lb = 量比)
    df = df[df["lb"] > 1.8]

    # 5. 综合打分:换手率为主、涨跌幅为辅,取分数最高的 15 只
    df["score"] = df["hsl"] * 0.6 + df["zdf"] * 0.4
    selected = df.sort_values("score", ascending=False).head(15)

    return selected[["code", "name", "zxj", "zdf", "hsl", "lb", "score"]].rename(
        columns={
            "code": "代码",
            "name": "名称",
            "zxj": "最新价",
            "zdf": "涨跌幅",
            "hsl": "换手率",
            "lb": "量比",
        }
    )


def enrich_with_industry(selected_df: pd.DataFrame) -> pd.DataFrame:
    """
    给入选股票逐个查行业分类。
    注意:ak.stock_individual_info_em() 我这边没法离线验证实际返回字段
    (沙盒里连不了 akshare 真实数据源),如果跑起来行业全变成"未分类",
    大概率是这个接口的字段名或代码格式对不上,把 except 里的 print 输出发我看一下就行。
    """
    industries = []
    for code in selected_df["代码"]:
        industry = "未分类"
        try:
            code_clean = re.sub(r"[^0-9]", "", str(code))[-6:]
            info = ak.stock_individual_info_em(symbol=code_clean)
            row = info.loc[info["item"] == "行业", "value"]
            if not row.empty:
                industry = str(row.values[0])
        except Exception as e:
            print(f"查询行业失败 code={code}: {e}")
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
