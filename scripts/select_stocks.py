#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import akshare as ak
import pandas as pd
import json
import os
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


# ============ 配置 ============
CACHE_FILE = Path("site/data/cache.json")      # 本地缓存路径
OUTPUT_FILE = Path("site/data/latest.json")    # 输出路径
CACHE_MAX_AGE_HOURS = 24                       # 缓存有效期（小时）
USE_DEMO_FALLBACK = True                       # 是否允许回退到测试数据


def get_demo_data():
    """测试数据回退：结构与真实输出保持一致"""
    print("[INFO] 使用测试数据生成 latest.json")
    
    demo_stocks = [
        {"代码": "000001", "名称": "平安银行", "最新价": 10.25, "涨跌幅": 2.15, "换手率": 5.32, "量比": 2.1, "score": 2.125},
        {"代码": "000002", "名称": "万科A", "最新价": 15.80, "涨跌幅": 1.85, "换手率": 4.15, "量比": 1.8, "score": 1.825},
        {"代码": "000858", "名称": "五粮液", "最新价": 145.60, "涨跌幅": 3.21, "换手率": 6.20, "量比": 2.5, "score": 2.855},
        {"代码": "002415", "名称": "海康威视", "最新价": 32.40, "涨跌幅": 2.56, "换手率": 3.80, "量比": 2.2, "score": 2.380},
        {"代码": "002594", "名称": "比亚迪", "最新价": 268.50, "涨跌幅": 4.12, "换手率": 7.50, "量比": 3.1, "score": 3.560},
        {"代码": "300750", "名称": "宁德时代", "最新价": 198.30, "涨跌幅": 1.92, "换手率": 4.60, "量比": 1.9, "score": 1.910},
        {"代码": "600000", "名称": "浦发银行", "最新价": 7.85, "涨跌幅": 1.15, "换手率": 3.20, "量比": 1.6, "score": 1.375},
        {"代码": "600519", "名称": "贵州茅台", "最新价": 1750.00, "涨跌幅": 2.08, "换手率": 1.50, "量比": 1.8, "score": 1.940},
        {"代码": "600036", "名称": "招商银行", "最新价": 34.20, "涨跌幅": 2.35, "换手率": 3.90, "量比": 2.0, "score": 2.175},
        {"代码": "601318", "名称": "中国平安", "最新价": 48.60, "涨跌幅": 1.68, "换手率": 4.10, "量比": 1.7, "score": 1.690},
    ]
    
    return {
        "update_date": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": "DEMO_DATA",
        "note": "当前使用测试数据（akshare 接口不可用）",
        "stocks": demo_stocks
    }


def load_cache():
    """读取本地缓存"""
    if not CACHE_FILE.exists():
        return None
    
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        
        cached_time = datetime.fromisoformat(cache.get("update_date", "2000-01-01 00:00:00").replace(" ", "T"))
        age = datetime.now(ZoneInfo("Asia/Shanghai")) - cached_time
        
        if age > timedelta(hours=CACHE_MAX_AGE_HOURS):
            print(f"[INFO] 缓存已过期（{age.total_seconds()/3600:.1f} 小时），将尝试重新获取")
            return None
        
        print(f"[INFO] 使用本地缓存（{age.total_seconds()/3600:.1f} 小时前生成）")
        cache["data_source"] = cache.get("data_source", "CACHE") + "_FALLBACK"
        cache["note"] = f"使用缓存数据（{age.total_seconds()/3600:.1f} 小时前）"
        return cache
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"[WARN] 缓存文件损坏: {e}")
        return None


def save_cache(data):
    """保存数据到本地缓存"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[INFO] 数据已缓存到 {CACHE_FILE}")
    except Exception as e:
        print(f"[WARN] 缓存保存失败: {e}")


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


def select_stocks():
    data = None
    
    # ========== 1. 尝试从 akshare 获取真实数据 ==========
    max_retry = 3
    df = None
    
    for attempt in range(max_retry):
        try:
            print(f"第{attempt+1}次尝试拉取全市场行情...")
            stock_list = ak.stock_zh_a_spot_em()
            df = stock_list.copy()
            break
        except Exception as e:
            print(f"拉取失败：{str(e)}")
            time.sleep(random.uniform(2, 5))
    
    if df is not None:
        # 运行选股策略
        selected_df = your_strategy_logic(df)
        selected_df = selected_df.fillna(0)
        
        data = {
            "update_date": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
            "data_source": "AKSHARE_REAL",
            "note": "实时数据",
            "stocks": selected_df.to_dict(orient="records")
        }
        print(f"[SUCCESS] 从 akshare 获取实时数据，选出 {len(selected_df)} 只股票")
    
    # ========== 2. 真实数据失败 → 尝试本地缓存 ==========
    if data is None:
        data = load_cache()
    
    # ========== 3. 缓存也没有 → 使用测试数据 ==========
    if data is None and USE_DEMO_FALLBACK:
        data = get_demo_data()
    
    # ========== 4. 全部失败 ==========
    if data is None:
        print("[FATAL] 无法获取任何数据（远程、缓存、测试数据均不可用）")
        return
    
    # ========== 写入输出文件 ==========
    os.makedirs("site/data", exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"已写入 {OUTPUT_FILE} | 数据源: {data['data_source']} | 股票数: {len(data['stocks'])}")
    
    # 如果是真实数据，同时更新缓存
    if data.get("data_source") == "AKSHARE_REAL":
        save_cache(data)


if __name__ == "__main__":
    select_stocks()
