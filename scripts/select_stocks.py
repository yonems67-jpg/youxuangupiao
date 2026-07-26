"""
A股量化选股框架 - 合二为一版本

【核心流程】
  1. 获取腾讯API实时行情 → 基础筛选（ST、涨跌停、换手率、量比、资金流）
  2. 综合评分 → 选出today's推荐20只
  3. 生成推荐理由 + 风险提示 → "值得入手"的股票清单
  4. JSON + 漂亮表格双格式输出

【数据源】
  - 实时行情: 腾讯 API (ak.stock_zh_a_spot_tx)
  - 输出格式: JSON + 详细解读
"""

import time
import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple

import akshare as ak
import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════
# 日志配置
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("a_share_selector")


# ═══════════════════════════════════════════════════════════════════════════
# 配置类
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SelectionConfig:
    """选股策略配置"""
    # 【快速筛选规则】
    exclude_st: bool = True              # 排除ST股
    zdf_min: float = 0.0                 # 涨跌幅最小值 (%)
    zdf_max: float = 9.5                 # 涨跌���最大值 (%)，避免涨停
    hsl_min: float = 3.0                 # 换手率最小值 (%)，需要流动性
    hsl_max: float = 15.0                # 换手率最大值 (%)，避免过热
    lb_threshold: float = 1.5            # 量比阈值，需放量
    zljlr_positive: bool = True          # 主力净流入需为正
    
    # 【评分权重】
    zdf_weight: float = 0.5              # 涨跌幅权重
    lb_weight: float = 0.5               # 量比权重
    top_n: int = 20                      # 最终选出的股票数量


@dataclass
class StockRecommendation:
    """最终推荐的股票"""
    code: str
    name: str
    price: float
    zdf: float
    hsl: float
    lb: float
    score: float
    reason: str  # 推荐理由
    risk: str    # 风险提示


# ═══════════════════════════════════════════════════════════════════════════
# 核心选股类
# ═══════════════════════════════════════════════════════════════════════════

class AShareStockSelector:
    """A股智能选股系统 - 实时选股 + 推荐理由生成"""
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self.logger = logger
        self.realtime_data: Optional[pd.DataFrame] = None
        self.filtered_stocks: Optional[pd.DataFrame] = None
        self.recommendations: List[StockRecommendation] = []
    
    # ─────────────────────────────────────────────────────────────────────
    # 【第1步】数据准备：转换数值类型
    # ─────────────────────────────────────────────────────────────────────
    
    def _prepare_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        腾讯API返回的数值字段是字符串，需转换为浮点数
        失败的转换会变成NaN（自动被筛选排除）
        """
        numeric_cols = ["zdf", "hsl", "lb", "zljlr", "zxj", "zd", "zf", "zsz", "ltsz"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    
    # ─────────────────────────────────────────────────────────────────────
    # 【第2步】基础筛选：排除不符合条件的股票
    # ─────────────────────────────────────────────────────────────────────
    
    def apply_base_filters(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        应用5层筛选规则
        
        Returns:
          (filtered_df, filter_stats) - 过滤后的数据 + 每层的筛选统计
        """
        stats = {"初始": len(df)}
        df = df.copy()
        df = self._prepare_numeric_columns(df)
        
        # 【规则1】排除ST/退市风险股
        if self.config.exclude_st and "name" in df.columns:
            before = len(df)
            df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]
            stats["排除ST后"] = len(df)
            self.logger.info(f"  ✓ 排除ST/退市股: {before} → {len(df)} (去除{before-len(df)}只)")
        
        # 【规则2】涨跌幅筛选 (避免追高、止损)
        before = len(df)
        df = df[(df["zdf"] > self.config.zdf_min) & (df["zdf"] < self.config.zdf_max)]
        stats["涨跌幅筛选后"] = len(df)
        self.logger.info(f"  ✓ 涨跌幅 {self.config.zdf_min}%~{self.config.zdf_max}%: {before} → {len(df)} (去除{before-len(df)}只)")
        
        # 【规则3】换手率筛选 (平衡流动性和过热度)
        before = len(df)
        df = df[(df["hsl"] > self.config.hsl_min) & (df["hsl"] < self.config.hsl_max)]
        stats["换手率筛选后"] = len(df)
        self.logger.info(f"  ✓ 换手率 {self.config.hsl_min}%~{self.config.hsl_max}%: {before} → {len(df)} (去除{before-len(df)}只)")
        
        # 【规则4】量比筛选 (需要明显放量)
        before = len(df)
        df = df[df["lb"] > self.config.lb_threshold]
        stats["量比筛选后"] = len(df)
        self.logger.info(f"  ✓ 量比 > {self.config.lb_threshold}: {before} → {len(df)} (去除{before-len(df)}只)")
        
        # 【规则5】主力资金净流入 (可选)
        if self.config.zljlr_positive and "zljlr" in df.columns:
            before = len(df)
            df = df[df["zljlr"] > 0]
            stats["资金流筛选后"] = len(df)
            self.logger.info(f"  ✓ 主力净流入>0: {before} → {len(df)} (去除{before-len(df)}只)")
        
        return df, stats
    
    # ─────────────────────────────────────────────────────────────────────
    # 【第3步】评分计算：加权综合评分
    # ─────────────────────────────────────────────────────────────────────
    
    def calculate_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        评分公式：score = zdf * weight_zdf + lb * weight_lb
        
        逻辑：
          - zdf (涨跌幅): 反映市场情绪强度
          - lb (量比): 反映资金进场意愿
          两者都强 → 值得关注的机会
        """
        df = df.copy()
        df["score"] = (
            df["zdf"] * self.config.zdf_weight +
            df["lb"] * self.config.lb_weight
        )
        return df
    
    # ─────────────────────────────────────────────────────────────────────
    # 【第4步】排序选出Top N
    # ─────────────────────────────────────────────────────────────────────
    
    def select_top_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """按评分从高到低排序，取前N只"""
        df = df.sort_values("score", ascending=False).head(self.config.top_n)
        return df.reset_index(drop=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # 【第5步】生成推荐理由
    # ─────────────────────────────────────────────────────────────────────
    
    def generate_recommendation_reason(self, row: pd.Series) -> Tuple[str, str]:
        """
        生成推荐理由和风险提示
        
        Returns:
          (reason, risk) - 推荐理由 + 风险提示
        """
        reasons = []
        
        # 理由1: 涨跌幅
        if row["zdf"] > 5:
            reasons.append(f"涨幅{row['zdf']:.2f}%，上升趋势强")
        elif row["zdf"] > 2:
            reasons.append(f"温和上涨{row['zdf']:.2f}%，节奏适中")
        else:
            reasons.append(f"小幅上涨{row['zdf']:.2f}%，低吸机会")
        
        # 理由2: 量比
        if row["lb"] > 3:
            reasons.append(f"量比{row['lb']:.2f}，大幅放量，资金进场积极")
        elif row["lb"] > 2:
            reasons.append(f"量比{row['lb']:.2f}，明显放量，关注度提升")
        else:
            reasons.append(f"量比{row['lb']:.2f}，温和放量")
        
        # 理由3: 换手率
        if row["hsl"] > 10:
            reasons.append(f"换手率{row['hsl']:.2f}%，高活跃度")
        elif row["hsl"] > 5:
            reasons.append(f"换手率{row['hsl']:.2f}%，适度活跃")
        
        reason_text = " | ".join(reasons)
        
        # 风险提示
        risks = []
        if row["zdf"] > 7:
            risks.append("⚠️ 涨幅过大，需防回调")
        if row["hsl"] > 12:
            risks.append("⚠️ 换手过快，可能存在出货压力")
        if row.get("zljlr", 0) is not None and row["zljlr"] > 0 and row["zljlr"] < 1_000_000:
            risks.append("⚠️ 主力进场资金量有限")
        
        risk_text = " ".join(risks) if risks else "✅ 风险可控"
        
        return reason_text, risk_text
    
    # ─────────────────────────────────────────────────────────────────────
    # 【第6步】实时选股主流程
    # ─────────────────────────────────────────────────────────────────────
    
    def run_realtime_selection(self) -> List[StockRecommendation]:
        """
        实时选股完整流程：
        获取数据 → 筛选 → 评分 → 生成推荐
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("【实时选股模式】")
        self.logger.info("="*80)
        
        try:
            # Step 1: 获取腾讯API实时行情
            self.logger.info("\n📡 Step 1: 获取腾讯API实时行情...")
            self.realtime_data = ak.stock_zh_a_spot_tx()
            self.logger.info(f"✓ 获取 {len(self.realtime_data)} 只股票")
            
            # Step 2: 应用基础筛选
            self.logger.info("\n🔍 Step 2: 应用基础筛选规则...")
            self.filtered_stocks, stats = self.apply_base_filters(self.realtime_data)
            self.logger.info(f"✓ 筛选结果: {len(self.filtered_stocks)} 只股票通过")
            
            # Step 3: 计算评分
            self.logger.info("\n⭐ Step 3: 计算综合评分...")
            scored = self.calculate_score(self.filtered_stocks)
            self.logger.info("✓ 评分完成")
            
            # Step 4: 选出Top N
            self.logger.info(f"\n🎯 Step 4: 选出Top {self.config.top_n}...")
            top_stocks = self.select_top_stocks(scored)
            self.logger.info(f"✓ 最终选出 {len(top_stocks)} 只")
            
            # Step 5: 生成推荐
            self.logger.info("\n💡 Step 5: 生成推荐理由...")
            self.recommendations = []
            for _, row in top_stocks.iterrows():
                reason, risk = self.generate_recommendation_reason(row)
                rec = StockRecommendation(
                    code=row["code"],
                    name=row["name"],
                    price=row["zxj"],
                    zdf=row["zdf"],
                    hsl=row["hsl"],
                    lb=row["lb"],
                    score=row["score"],
                    reason=reason,
                    risk=risk,
                )
                self.recommendations.append(rec)
            
            self.logger.info(f"✓ 生成了 {len(self.recommendations)} 条推荐")
            
            return self.recommendations
        
        except Exception as e:
            self.logger.error(f"❌ 实时选股失败: {e}")
            raise
    
    # ─────────────────────────────────────────────────────────────────────
    # 【输出】保存结果到JSON
    # ─────────────────────────────────────────────────────────────────────
    
    def save_results(self, output_path: str = "site/data/latest.json"):
        """
        保存推荐结果为JSON
        
        输出格式：
        {
          "update_time": "2024-01-15 14:30:00",
          "total_count": 20,
          "recommendations": [
            {
              "code": "002001",
              "name": "新和成",
              "price": 45.67,
              "zdf": 3.45,
              "hsl": 8.92,
              "lb": 2.34,
              "score": 2.90,
              "reason": "...",
              "risk": "..."
            },
            ...
          ]
        }
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        result = {
            "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_count": len(self.recommendations),
            "recommendations": [
                asdict(rec) for rec in self.recommendations
            ]
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"✅ 已保存到 {output_path}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 【打印】控制台输出 - 漂亮的表格
    # ─────────────────────────────────────────────────────────────────────
    
    def print_recommendations(self):
        """打印推荐清单到控制台"""
        if not self.recommendations:
            self.logger.warning("没有推荐结果")
            return
        
        self.logger.info("\n" + "="*140)
        self.logger.info(f"【值得入手的股票】 共 {len(self.recommendations)} 只  更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*140)
        
        # 表头
        header = f"{'排名':<4} {'代码':<8} {'名称':<8} {'价格':<8} {'涨幅%':<8} {'换手率%':<10} {'量比':<6} {'评分':<8} {'推荐理由':<50} {'风险提示':<45}"
        self.logger.info(header)
        self.logger.info("-"*140)
        
        # 数据行
        for idx, rec in enumerate(self.recommendations, 1):
            row = (
                f"{idx:<4} "
                f"{rec.code:<8} "
                f"{rec.name:<8} "
                f"{rec.price:<8.2f} "
                f"{rec.zdf:<8.2f} "
                f"{rec.hsl:<10.2f} "
                f"{rec.lb:<6.2f} "
                f"{rec.score:<8.2f} "
                f"{rec.reason[:48]:<50} "
                f"{rec.risk[:43]:<45}"
            )
            self.logger.info(row)
        
        self.logger.info("-"*140)
        self.logger.info(f"【温馨提示】")
        self.logger.info(f"  1. 这只是基于技术面的初步筛选，需结合基本面分析")
        self.logger.info(f"  2. 建议在小时线上再观察一次，确认买点")
        self.logger.info(f"  3. 严格设置止损，建议止损位置在2-3%")
        self.logger.info(f"  4. 持仓周期：1~3天的短线持仓")
        self.logger.info("="*140 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# 主程序：完整流程
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """主入口：执行完整的选股流程"""
    
    # 【配置】
    config = SelectionConfig(
        exclude_st=True,
        zdf_min=0.0,
        zdf_max=9.5,
        hsl_min=3.0,
        hsl_max=15.0,
        lb_threshold=1.5,
        zljlr_positive=True,
        top_n=20,
    )
    
    # 【运行选股】
    selector = AShareStockSelector(config)
    recommendations = selector.run_realtime_selection()
    
    # 【打印结果】
    selector.print_recommendations()
    
    # 【保存结果】
    selector.save_results("site/data/latest.json")
    
    print("\n✨ 选股完成！")
    print(f"📊 共推荐 {len(recommendations)} 只股票")
    print(f"💾 结果已保存到 site/data/latest.json")


if __name__ == "__main__":
    main()
