import akshare as ak
import pandas as pd
import json
import os
from datetime import datetime

def select_stocks():
    # 1. 获取股票池 / 行情数据(按你策略需要的接口替换)
    stock_list = ak.stock_zh_a_spot_tx()
    print("=== 当前数据包含的列名有 ===")
    print(stock_list.columns.tolist())
    print("==========================")

    # ... 后续的 your_strategy_logic ...


    # 2. TODO: 接入你现有的短线选股逻辑
    #    对 stock_list 做筛选/打分,返回一个 DataFrame
    selected_df = your_strategy_logic(stock_list)

    # 3. 整理输出为 JSON
    result = {
        "update_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": selected_df.to_dict(orient="records")
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"选出 {len(selected_df)} 只股票,已写入 site/data/latest.json")

def your_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    df = stock_list.copy()

    # 排除 ST / 退市风险股
    df = df[~df['名称'].str.contains('ST|退', na=False)]

    # 排除涨停(追高性价比低)和跌停
    df = df[(df['涨跌幅'] > 0) & (df['涨跌幅'] < 9.5)]

    # 换手率控制在合理区间:太低=关注度不够,太高=可能已炒过热
    df = df[(df['换手率'] > 3) & (df['换手率'] < 15)]

    # 量比 > 1.5:今日成交量相对近期明显放大,说明有资金在动
    df = df[df['量比'] > 1.5]

    # 综合打分(涨跌幅 + 量比各占一半权重),取分数最高的 20 只
    df['score'] = df['涨跌幅'] * 0.5 + df['量比'] * 0.5
    selected = df.sort_values('score', ascending=False).head(20)

    return selected[['代码', '名称', '最新价', '涨跌幅', '换手率', '量比', 'score']]

if __name__ == "__main__":
    select_stocks()
