import akshare as ak
import pandas as pd
import json
import os
from datetime import datetime


def select_stocks():
    # 1. 获取股票池 / 行情数据(腾讯接口)
    stock_list = ak.stock_zh_a_spot_tx()

    # 2. 接入短线选股逻辑
    selected_df = your_strategy_logic(stock_list)

    # 3. 整理输出为 JSON
    result = {
        "update_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": selected_df.to_dict(orient="records"),
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"选出 {len(selected_df)} 只股票,已写入 site/data/latest.json")


def your_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    df = stock_list.copy()

    # 腾讯接口返回的数值字段经常是字符串类型,直接比较大小会报
    # TypeError: '>' not supported between instances of 'str' and 'int'
    # 这里把用到的数值列统一转成数字,转换失败的值会变成 NaN(筛选时自动被排除,不会报错)
    numeric_cols = ["zdf", "hsl", "lb", "zljlr", "zxj", "zd", "zf", "zsz", "ltsz"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 排除 ST / 退市风险股(name = 名称)
    df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]

    # 排除涨停(追高性价比低)和跌停(zdf = 涨跌幅,单位 %)
    df = df[(df["zdf"] > 0) & (df["zdf"] < 9.5)]

    # 换手率控制在合理区间:太低=关注度不够,太高=可能已炒过热(hsl = 换手率,单位 %)
    df = df[(df["hsl"] > 3) & (df["hsl"] < 15)]

    # 量比 > 1.5:今日成交量相对近期明显放大,说明有资金在动(lb = 量比)
    df = df[df["lb"] > 1.5]

    # 主力资金净流入为正(zljlr = 主力净流入,单位元)
    # 如果这个字段跑起来不符合预期,把下面两行删掉即可
    if "zljlr" in df.columns:
        df = df[df["zljlr"] > 0]

    # 综合打分(涨跌幅 + 量比各占一半权重),取分数最高的 20 只
    df["score"] = df["zdf"] * 0.5 + df["lb"] * 0.5
    selected = df.sort_values("score", ascending=False).head(20)

    # 输出时换成中文列名,方便展示页面直接渲染
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


if __name__ == "__main__":
    select_stocks()
