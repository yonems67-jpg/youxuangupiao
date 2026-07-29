import akshare as ak
import pandas as pd
import numpy as np
import json
import os
import re
import time
from typing import Optional
from datetime import datetime, timedelta, timezone, time as dtime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 以下参数未经历史回测验证,是起始值,建议跑一段时间/跑一次回测后再调整
AMOUNT_TIERS = [
    {"tier": 0, "name": "标准模式",       "amount_threshold": 5.0e8, "ma_required": True},
    {"tier": 1, "name": "存量/轻缩量模式", "amount_threshold": 3.5e8, "ma_required": True},
    {"tier": 2, "name": "极度缩量防御模式", "amount_threshold": 2.5e8, "ma_required": False},
]
TARGET_COUNT_BY_STATUS = {"强势": 15, "震荡": 9, "弱势": 4}

VOLUME_SPIKE_MULTIPLIER = 1.8   # 今日成交量 >= 前5日均量(不含今天) * 这个倍数
BREAKOUT_RATIO = 0.97           # 收盘价 >= 近20日最高价 * 这个比例

SCORE_WEIGHTS = {
    "turnover": 0.25,
    "volume_ratio": 0.15,
    "industry_strength": 0.20,
    "high20_ratio": 0.15,
    "tech_bonus": 0.25,
}  # 五项权重和为1.0

FEW_STOCKS_THRESHOLD = 3
SINGLE_STOCK_CAP_WHEN_FEW = 5.0

CANDIDATE_POOL_SIZE_FOR_TECH_FETCH = 50  # 只对初筛后排名靠前的这么多只做技术因子拉取
TECH_FETCH_MAX_WORKERS = 8


def to_json_safe(obj):
    """递归把 numpy/pandas 标量转成原生 Python 类型,否则 json.dump 会报错。"""
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


def get_beijing_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=8)


def trading_minutes_elapsed(now: datetime) -> float:
    """已交易分钟数,用于换手率折算全天预估值。开盘前按240(用昨收数据)。"""
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


def load_industry_map() -> dict:
    """读取本地行业映射表(build_industry_map.py 生成)。用绝对路径防止 FC 环境找不到文件。"""
    # 获取 select_stocks.py 所在的绝对目录
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 拼接绝对路径：/code/site/data/industry_map.json
    map_path = os.path.join(base_dir, "site", "data", "industry_map.json")
    
    if os.path.exists(map_path):
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"✅ [FC] 成功加载行业字典: {map_path} (共 {len(data)} 条记录)")
                return data
        except Exception as e:
            print(f"❌ [FC] 读取行业字典失败: {e}")
            return {}
            
    print(f"⚠️ [FC] 提示: {map_path} 文件不存在，行业将退化为未分类")
    return {}


def min_max_normalize(s: pd.Series) -> pd.Series:
    """Min-Max 归一化到0~100;全部相等/取不到值/单个NaN 都填中性分50,不报错不拖垮整体加权。"""
    s = pd.to_numeric(s, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(50.0, index=s.index)
    return ((s - lo) / (hi - lo) * 100).fillna(50.0)


def compute_market_overview(raw_df: pd.DataFrame) -> dict:
    """
    market_score = (上涨家数-下跌家数)/总家数,映射到0~100,50为持平。
    注意: scripts/backtest.py 的 assess_market_regime() 用同一套公式和60/40分档阈值,
    两边必须保持一致,否则回测结果不能代表实盘。
    """
    df = raw_df.copy()
    df["zdf"] = pd.to_numeric(df["zdf"], errors="coerce")

    total = int(df["zdf"].notna().sum())
    up = int((df["zdf"] > 0).sum())
    down = int((df["zdf"] < 0).sum())
    limit_up = int((df["zdf"] >= 9.5).sum())
    limit_down = int((df["zdf"] <= -9.5).sum())

    breadth = (up - down) / total if total else 0
    market_score = round((breadth + 1) / 2 * 100, 1)

    if market_score >= 60:
        status, suggested_position = "强势", "6~8成"
    elif market_score >= 40:
        status, suggested_position = "震荡", "3~5成"
    else:
        status, suggested_position = "弱势", "1~2成或观望"

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
    第一轮筛选,产出技术因子拉取候选池(不超过 CANDIDATE_POOL_SIZE_FOR_TECH_FETCH 只)。
    每条新增过滤都带安全阀:套上后候选股清零就跳过这条规则并打印提示,不让当天结果整批清空。
    """
    df = stock_list.copy()
    print(f"stock_zh_a_spot_tx() 原始字段列表: {df.columns.tolist()}")

    numeric_cols = ["zdf", "hsl", "lb", "zxj", "zd", "zf", "zsz", "ltsz"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 行业相对强度(全市场层面,纯本地计算,不额外调接口)
    industry_map = load_industry_map()
    df["_code6"] = df["code"].astype(str).str.replace(r"\D", "", regex=True).str[-6:]
    df["行业"] = df["_code6"].map(industry_map).fillna("未分类")

    market_avg_zdf = df["zdf"].mean()
    industry_avg = df.loc[df["行业"] != "未分类"].groupby("行业")["zdf"].mean()
    df["_industry_avg_zdf"] = df["行业"].map(industry_avg)
    df["行业相对强度"] = np.where(
        df["_industry_avg_zdf"].notna(),
        df["zdf"] - df["_industry_avg_zdf"],
        df["zdf"] - market_avg_zdf,
    )

    df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]
    # 2%~7%区间本身已排除各板块的涨停/一字板情况,不需要额外按板块判断涨停线
    df = df[(df["zdf"] >= 2.0) & (df["zdf"] <= 7.0)]

    elapsed = max(trading_minutes_elapsed(get_beijing_now()), 10.0)
    projection_factor = 240.0 / elapsed
    df["hsl_projected"] = df["hsl"] * projection_factor
    df = df[(df["hsl"] >= 0.5) & (df["hsl_projected"] >= 5.0) & (df["hsl_projected"] <= 15.0)]

    df = df[df["lb"] > 1.8]

    zf_filtered = df[df["zf"] <= 8.0]
    if not zf_filtered.empty:
        df = zf_filtered
    else:
        print("提示: 振幅过滤后候选股清零,本次跳过")

    # 流动性:优先用成交额,找不到该字段退回流通市值。字段名/单位未离线核实,打印样例供核对。
    AMOUNT_COL_CANDIDATES = ["cje", "amount", "成交额"]
    amount_col = next((c for c in AMOUNT_COL_CANDIDATES if c in df.columns), None)
    loosest_amount_threshold = AMOUNT_TIERS[-1]["amount_threshold"]

    if amount_col:
        df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce")
        print(f"成交额字段(推断为'{amount_col}')样例: {df[amount_col].dropna().head(5).tolist()}")
        amt_filtered = df[df[amount_col] >= loosest_amount_threshold]
        if not amt_filtered.empty:
            df = amt_filtered.rename(columns={amount_col: "成交额"})
        else:
            print(f"提示: 成交额过滤后候选股清零(阈值{loosest_amount_threshold:.0f}),本次跳过,请核对单位")
    else:
        print("提示: 未找到成交额字段,回退到流通市值过滤")
        LTSZ_THRESHOLD = 200000  # 假设单位"万元",约20亿流通市值
        if "ltsz" in df.columns and df["ltsz"].notna().any():
            print(f"流通市值样例: {df['ltsz'].dropna().head(5).tolist()}")
            ltsz_filtered = df[df["ltsz"] >= LTSZ_THRESHOLD]
            if not ltsz_filtered.empty:
                df = ltsz_filtered
            else:
                print("提示: 流通市值过滤后候选股清零,本次跳过")

    rs_filtered = df[df["行业相对强度"] > 0]
    if not rs_filtered.empty:
        df = rs_filtered
    else:
        print("提示: 行业相对强度过滤后候选股清零,本次跳过")

    df["_pre_score"] = df["hsl_projected"].fillna(0) * 0.5 + df["行业相对强度"].fillna(0) * 0.5
    candidates = df.sort_values("_pre_score", ascending=False).head(CANDIDATE_POOL_SIZE_FOR_TECH_FETCH)

    keep_cols = ["_code6", "code", "name", "zxj", "zdf", "hsl", "hsl_projected", "lb",
                 "zf", "行业", "行业相对强度"]
    if "成交额" in candidates.columns:
        keep_cols.append("成交额")
    elif "ltsz" in candidates.columns:
        keep_cols.append("ltsz")

    result = candidates[keep_cols].rename(columns={
        "code": "代码", "name": "名称", "zxj": "最新价", "zdf": "涨跌幅",
        "hsl": "换手率", "hsl_projected": "预估全天换手率", "lb": "量比",
        "zf": "振幅", "ltsz": "流通市值",
    })
    return result.reset_index(drop=True)


def fetch_advanced_tech_factors(code_clean: str) -> dict:
    """
    多线程调用。用同一次历史K线算出三个因子:
      ma_aligned: MA5>MA10>MA20(均线多头排列)
      volume_spike: 今日成交量 >= 前5日均量(不含今天) * VOLUME_SPIKE_MULTIPLIER
      breakout / high20_ratio: 收盘 vs 近20日最高价
    任何一项拿不到都返回 None,调用方按"未确认"处理,不当 False 扣分/剔除。
    close/volume 列名已通过 akshare 源码核实,可靠。
    """
    result = {"ma_aligned": None, "volume_spike": None, "breakout": None, "high20_ratio": None}
    try:
        end = get_beijing_now()
        start = end - timedelta(days=60)
        hist = ak.stock_zh_a_hist_tx(
            symbol=code_clean,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if hist is None or len(hist) < 20:
            return result

        close_col = "close" if "close" in hist.columns else ("收盘" if "收盘" in hist.columns else None)
        if close_col is None:
            print(f"技术因子计算失败 code={code_clean}: 无法识别收盘价列 {list(hist.columns)}")
            return result

        closes = pd.to_numeric(hist[close_col], errors="coerce").dropna()
        if len(closes) < 20:
            return result

        ma5, ma10, ma20 = closes.tail(5).mean(), closes.tail(10).mean(), closes.tail(20).mean()
        result["ma_aligned"] = bool(ma5 > ma10 > ma20)

        high20 = closes.tail(20).max()
        last_close = closes.iloc[-1]
        result["high20_ratio"] = float(last_close / high20) if high20 else None
        result["breakout"] = bool(last_close >= high20 * BREAKOUT_RATIO)

        vol_col = next((c for c in ["volume", "vol", "成交量"] if c in hist.columns), None)
        if vol_col:
            vols = pd.to_numeric(hist[vol_col], errors="coerce").dropna()
            if len(vols) >= 6:
                avg5 = vols.iloc[-6:-1].mean()  # 前5日,不含今天
                today_vol = vols.iloc[-1]
                if avg5 and not np.isnan(avg5):
                    result["volume_spike"] = bool(today_vol >= avg5 * VOLUME_SPIKE_MULTIPLIER)

        return result
    except Exception as e:
        print(f"技术因子计算失败 code={code_clean}: {e}")
        return result


def fetch_tech_factors_parallel(codes: list) -> dict:
    """并发拉取技术因子。如果日志里失败率明显升高(被限流),调小 TECH_FETCH_MAX_WORKERS。"""
    results = {}
    with ThreadPoolExecutor(max_workers=TECH_FETCH_MAX_WORKERS) as executor:
        future_map = {executor.submit(fetch_advanced_tech_factors, code): code for code in codes}
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                results[code] = future.result()
            except Exception as e:
                print(f"技术因子线程异常 code={code}: {e}")
                results[code] = {"ma_aligned": None, "volume_spike": None, "breakout": None, "high20_ratio": None}
    return results


def compute_multi_factor_score(pool: pd.DataFrame) -> pd.DataFrame:
    """多因子 Min-Max 归一化加权评分,权重见 SCORE_WEIGHTS(未回测验证的起始值)。"""
    pool = pool.copy()
    pool["_norm_turnover"] = min_max_normalize(pool["预估全天换手率"])
    pool["_norm_volume_ratio"] = min_max_normalize(pool["量比"])
    pool["_norm_industry_strength"] = min_max_normalize(pool["行业相对强度"])
    pool["_norm_high20_ratio"] = min_max_normalize(pool["距20日新高比例"])
    pool["_tech_bonus"] = np.where(
        (pool["ma_aligned"] == True) & ((pool["volume_spike"] == True) | (pool["breakout"] == True)),
        100.0, 0.0
    )
    pool["score"] = (
        SCORE_WEIGHTS["turnover"] * pool["_norm_turnover"]
        + SCORE_WEIGHTS["volume_ratio"] * pool["_norm_volume_ratio"]
        + SCORE_WEIGHTS["industry_strength"] * pool["_norm_industry_strength"]
        + SCORE_WEIGHTS["high20_ratio"] * pool["_norm_high20_ratio"]
        + SCORE_WEIGHTS["tech_bonus"] * pool["_tech_bonus"]
    ).round(1)
    return pool


def select_with_tiered_fallback(candidates: pd.DataFrame, tech_factors: dict, market: dict):
    """
    三级降级:Tier0(标准)->Tier1(存量)->Tier2(极度缩量,兜底不再降级)。
    技术因子只拉一次,这里只是按不同门槛在内存里反复筛选,不会多发网络请求。
    """
    target = TARGET_COUNT_BY_STATUS.get(market["market_status"], 9)

    pool = candidates.copy()
    pool["ma_aligned"] = pool["_code6"].map(lambda c: tech_factors.get(c, {}).get("ma_aligned"))
    pool["volume_spike"] = pool["_code6"].map(lambda c: tech_factors.get(c, {}).get("volume_spike"))
    pool["breakout"] = pool["_code6"].map(lambda c: tech_factors.get(c, {}).get("breakout"))
    pool["距20日新高比例"] = pool["_code6"].map(lambda c: tech_factors.get(c, {}).get("high20_ratio"))

    has_amount_col = "成交额" in pool.columns
    chosen = None
    final_df = pd.DataFrame()

    for tier_cfg in AMOUNT_TIERS:
        subset = pool
        if has_amount_col:
            subset = subset[subset["成交额"] >= tier_cfg["amount_threshold"]]
        if tier_cfg["ma_required"]:
            subset = subset[subset["ma_aligned"] == True]

        if subset.empty and tier_cfg["tier"] < 2:
            continue

        scored_sorted = compute_multi_factor_score(subset).sort_values("score", ascending=False) \
            if not subset.empty else subset

        if len(scored_sorted) >= target or tier_cfg["tier"] == 2:
            final_df = scored_sorted.head(target)
            chosen = tier_cfg
            break

    if chosen is None:
        chosen = AMOUNT_TIERS[-1]
        final_df = pd.DataFrame()

    actual = len(final_df)
    fallback_status = {
        "used_tier": chosen["tier"],
        "tier_name": chosen["name"],
        "target_count": target,
        "actual_count": actual,
        "fallback_note": (
            f"市场状态「{market['market_status']}」目标选出 {target} 只,"
            f"经 Tier{chosen['tier']}({chosen['name']}) 筛选后实际选出 {actual} 只。"
            + ("" if chosen["tier"] == 0 else " 已触发降级,当前盘面偏冷/缩量。")
        ),
    }

    n = actual
    market_multiplier = {"强势": 1.0, "震荡": 0.5, "弱势": 0.0}.get(market["market_status"], 0.5)
    base_pct, max_pct = 20.0, 25.0
    if n > 0:
        final_df = final_df.copy()
        ranks = final_df["score"].rank(ascending=False, method="first")
        signal_factor = 0.5 + 0.5 * (n - ranks) / max(n - 1, 1)
        final_df["建议仓位"] = (base_pct * market_multiplier * signal_factor).clip(upper=max_pct).round(1)
        if n <= FEW_STOCKS_THRESHOLD:
            final_df["建议仓位"] = final_df["建议仓位"].clip(upper=SINGLE_STOCK_CAP_WHEN_FEW)
            fallback_status["fallback_note"] += f" 入选≤{FEW_STOCKS_THRESHOLD}只,单票仓位已压缩至{SINGLE_STOCK_CAP_WHEN_FEW}%上限。"

    return final_df.reset_index(drop=True), fallback_status


def add_action_tag(selected_df: pd.DataFrame) -> pd.DataFrame:
    """操作建议:按入选列表内部相对排名分三档,是信号强弱标签,不是买卖指令。"""
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
        return "谨慎"

    df["操作建议"] = ranks.apply(tag)
    return df


def compute_risk(selected_df: pd.DataFrame, market: dict) -> dict:
    """
    风险提示:行业集中度 + 市场风险。
    "未分类"不算真实行业集中度信号(可能只是没有映射表),否则每天误报。
    """
    notes = []

    if len(selected_df) > 0 and "行业" in selected_df.columns:
        known = selected_df[selected_df["行业"] != "未分类"]
        unclassified_share = round(1 - len(known) / len(selected_df), 2)
        if len(known) > 0:
            counts = known["行业"].value_counts()
            top_industry = counts.index[0]
            top_share = round(counts.iloc[0] / len(known), 2)
        else:
            top_industry, top_share = None, 0
    else:
        top_industry, top_share, unclassified_share = None, 0, 0

    industry_warning = top_industry is not None and top_share > 0.4
    if industry_warning:
        notes.append(f"入选股票中「{top_industry}」占比达 {int(top_share*100)}%,行业集中度偏高")
    if unclassified_share >= 0.5:
        notes.append(f"约 {int(unclassified_share*100)}% 的入选股票没有行业分类数据,行业集中度判断仅供参考")

    market_warning = market["market_status"] == "弱势"
    if market_warning:
        notes.append("当前市场宽度偏弱(下跌家数明显多于上涨家数)")

    return {
        "top_industry": top_industry,
        "top_industry_share": top_share,
        "industry_concentration_warning": industry_warning,
        "market_risk_warning": market_warning,
        "suggest_reduce_position": industry_warning or market_warning,
        "notes": notes,
    }


def build_result():
    # ... 前面原有的选股逻辑 ...
    
    # 假设你筛选出的最终股票 DataFrame 变量叫 res_df / df / candidate_df
    # 这里做个兼容，自动寻找已存在的 DataFrame 变量：
    target_df = None
    for var_name in ['df', 'res_df', 'candidate_df', 'selected_df']:
        if var_name in locals() and isinstance(locals()[var_name], pd.DataFrame):
            target_df = locals()[var_name]
            break

    stocks_list = []
    if target_df is not None and not target_df.empty:
        # 1. 自动找到代码列
        code_col = 'code' if 'code' in target_df.columns else ('代码' if '代码' in target_df.columns else target_df.columns[0])
        target_df['_code6'] = target_df[code_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6).str[-6:]

        # 2. 读取并清洗行业字典进行映射
        industry_map = load_industry_map()
        clean_map = {str(k).zfill(6)[-6:]: v for k, v in industry_map.items()}
        
        target_df['industry_clean'] = target_df['_code6'].map(clean_map)
        
        # 优先用映射出来的行业，如果没有映射上且原 DataFrame 有行业/industry 列则保留，否则填未分类
        if 'industry' in target_df.columns:
            target_df['industry'] = target_df['industry_clean'].fillna(target_df['industry']).fillna('未分类')
        elif '行业' in target_df.columns:
            target_df['industry'] = target_df['industry_clean'].fillna(target_df['行业']).fillna('未分类')
        else:
            target_df['industry'] = target_df['industry_clean'].fillna('未分类')

        # 3. 构造输出列表
        for _, row in target_df.iterrows():
            c_code = str(row.get('_code6', ''))
            c_name = str(row.get('name', row.get('名称', '')))
            c_ind = str(row.get('industry', '未分类'))
            
            stocks_list.append({
                "code": c_code,
                "name": c_name,
                "industry": c_ind,
                "代码": c_code,
                "名称": c_name,
                "行业": c_ind,
                "zdf": float(row.get('zdf', 0.0)) if pd.notna(row.get('zdf')) else 0.0,
            })

    result = {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "market_score": market_score if 'market_score' in locals() else 50,
        "stocks": stocks_list
            # === 假设原本构造的字典叫 result / ret / res_dict ===
    # 只要在最终 return 之前，加下面这几行补全行业字段：
    
    industry_map = load_industry_map()
    clean_map = {str(k).zfill(6)[-6:]: v for k, v in industry_map.items()}

    # 遍历已生成的 stocks 列表，修补补全 code、name 和 industry
    if "stocks" in result and isinstance(result["stocks"], list):
        for s in result["stocks"]:
            if isinstance(s, dict):
                # 1. 提取 6 位代码
                raw_code = str(s.get("code") or s.get("代码") or "")
                c_code = "".join(filter(str.isdigit, raw_code)).zfill(6)[-6:]
                
                # 2. 如果已有代码，补全字段
                if c_code:
                    s["code"] = c_code
                    s["代码"] = c_code
                    
                    # 3. 匹配行业（优先用字典映射，找不到再用原有的）
                    mapped_ind = clean_map.get(c_code)
                    existing_ind = s.get("industry") or s.get("行业")
                    
                    final_ind = mapped_ind if mapped_ind else (existing_ind if existing_ind and existing_ind != "None" else "未分类")
                    
                    s["industry"] = final_ind
                    s["行业"] = final_ind

    return result

    }
    
    



def select_stocks():
    """本地/GitHub Actions 用的入口:算完直接写本地文件。阿里云 FC 用 fc_handler.py 调 build_result()。"""
    result = build_result()
    
    # 确保目标目录存在
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "site", "data")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "latest.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 成功写入 {out_file} (共 {len(result.get('stocks', []))} 只股票)")



if __name__ == "__main__":
    select_stocks()
