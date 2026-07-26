import akshare as ak
import pandas as pd
import json
import os
import time
import random
from datetime import datetime
from zoneinfo import ZoneInfo

# 全局设置超时时间，防止卡死
ak.set_timeout((12, 18))

def select_stocks():
    max_retry = 3
    df = None
    
    # 重试机制：应对 GitHub 海外节点访问国内数据源的网络抖动
    for attempt in range(max_retry):
        try:
            print(f"第{attempt+1}次尝试拉取全市场行情...")
            stock_list = ak.stock_zh_a_spot_em()
            df = stock_list.copy()
            break
        except Exception as e:
            print(f"拉取失败：{str(e)}")
            time.sleep(random.uniform(2, 5))
    
    if df is None:
        print("多次重试获取行情失败，程序退出。")
        return

    # 运行选股策略
    selected_df = your_strategy_logic(df)
    
    # 填充 NaN 值，防止前端 JS 解析 JSON 失败
    selected_df = selected_df.fillna(0)

    # 整理输出（强制使用北京时间）
    result = {
        "update_date": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": selected_df.to_dict(orient="records")
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"选出 {len(selected_df)} 只股票,已写入 site/data/latest.json")

def your_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    df = stock_list.copy()
    
    # 过滤ST、退市
    df = df[~df['名称'].str.contains('ST|退', na=False)]
    
    # 涨跌幅区间 (剔除跌停和涨停，保留活口)
    df = df[(df['涨跌幅'] > 0) & (df['涨跌幅'] < 9.5)]
    
    # 换手率、量比条件
    df = df[(df['换手率'] > 3) & (df['换手率'] < 15)]
    df = df[df['量比'] > 1.5]
    
    # 打分排序
    df['score'] = df['涨跌幅'] * 0.5 + df['量比'] * 0.5
    selected = df.sort_values('score', ascending=False).head(20)
    
    return selected[['代码', '名称', '最新价', '涨跌幅', '换手率', '量比', 'score']]

if __name__ == "__main__":
    select_stocks()
