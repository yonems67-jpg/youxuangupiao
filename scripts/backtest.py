# -*- coding: utf-8 -*-
"""
A股短线量化策略 —— 最终整合版(回测 / 模拟盘引擎)
==================================================

在原有回测框架基础上,新增三个独立模块,把"策略"从单纯的选股信号
升级成一套完整的交易系统:

一、市场环境模块 assess_market_regime()  —— 决定"今天能不能开新仓"
    强势 -> 正常仓位(系数 1.0)
    震荡 -> 仓位减半(系数 0.5)
    弱势 -> 停止开新仓(系数 0,但已有持仓的止盈止损判断照常执行)

二、卖出规则模块 generate_sell_signal()  —— 决定"什么时候卖、卖多少"
    按优先级从高到低依次判断,命中即执行,不再往下判断:
      1. 止损:跌破成本价一定比例,无条件全部卖出
      2. 趋势走弱:跌破10日均线
      3. 超过最长持有天数仍未有明显收益,离场
      4. 止盈:分两档分批卖出(先落袋一部分,让剩余仓位继续跑)

三、仓位管理模块 PositionSizing            —— 决定"买多少"
    单票仓位 = 基准仓位 × 市场环境系数 × 信号强度系数,
    并且不管信号多强、市场多好,单票和单一行业都有硬性仓位上限。

!! 市场环境判断和 scripts/select_stocks.py 里的 compute_market_overview() 用的是
   完全相同的打分公式(涨跌家数比映射到 0~100,60/40 分档),这是有意为之的一致性
   设计——如果回测用一套市场环境规则、实盘用另一套,回测结果就没法代表实盘表现。 !!

沿用自上一版、未改动的部分:
  - T+1 结算(当日买入份额当日不可卖)
  - 涨跌停部分成交建模(区分"一字板锁死"和"触及但有成交")
  - 滑点 / 双边佣金 / 印花税(仅卖出)
  - 单日最大回撤熔断
  - 行业集中度上限(在买入时检查)
  - 日历/政策风险日仓位打折
  - submit_order_with_retry:实盘下单容错重试

使用前必读(和上一版相同,再次强调):
  - 本文件不含真实行情数据,需自行接入券商/数据商的"后复权"日线或分钟线数据。
  - 股票池必须包含历史上已退市/被ST摘牌的个股,且要覆盖足够多的股票用于市场环境的
    涨跌家数统计,否则回测收益会被"幸存者偏差"虚高,市场环境判断也会失真。
  - 这里的胜率/平均持仓天数/最大回撤,全部来自 compute_trade_stats() 对
    trade_log 的真实统计,不是编出来的数字——但也仅仅是"这份历史数据 + 这套规则"
    跑出来的结果,不代表未来一定重复。
  - 这是教学框架,实盘接入前务必用小资金、模拟盘充分验证。
"""

import time
import logging
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import akshare as ak

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("a_share_quant")


# ===========================================================================
# 第一部分:实时选股(和 scripts/select_stocks.py 保持同一套核心规则)
# ===========================================================================
def select_stocks():
    """从全市场实时行情中筛选短线候选股,并写出 site/data/latest.json"""
    logger.info("正在获取全市场实时行情...")
    stock_list = ak.stock_zh_a_spot_tx()
    selected_df = robust_strategy_logic(stock_list)

    result = {
        "update_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stocks": selected_df.to_dict(orient="records"),
    }

    os.makedirs("site/data", exist_ok=True)
    with open("site/data/latest.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"筛选完成,共选出 {len(selected_df)} 只股票,已写入 site/data/latest.json")


def robust_strategy_logic(stock_list: pd.DataFrame) -> pd.DataFrame:
    """选股逻辑说明见 scripts/select_stocks.py 里的同名函数,两边保持一致。"""
    df = stock_list.copy()
    numeric_cols = ["zdf", "hsl", "lb", "zxj", "zd", "zf", "zsz", "ltsz"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[~df["name"].astype(str).str.contains("ST|退", na=False)]
    df = df[(df["zdf"] >= 2.0) & (df["zdf"] <= 7.0)]
    df = df[(df["hsl"] >= 5.0) & (df["hsl"] <= 15.0)]
    df = df[df["lb"] > 1.8]

    df["score"] = df["hsl"] * 0.6 + df["zdf"] * 0.4
    selected = df.sort_values("score", ascending=False).head(15)

    return selected[["code", "name", "zxj", "zdf", "hsl", "lb", "score"]].rename(
        columns={
            "code": "代码", "name": "名称", "zxj": "最新价",
            "zdf": "涨跌幅", "hsl": "换手率", "lb": "量比",
        }
    )


# ===========================================================================
# 第二部分:一、市场环境模块
# ===========================================================================
def assess_market_regime(day_data: pd.DataFrame) -> tuple[str, float]:
    """
    市场环境评估:用当日全市场(day_data 里的全部股票)涨跌家数比打分,
    映射到 0~100,60/40 分档 —— 和 select_stocks.py 里 compute_market_overview()
    的公式完全一致,只是数据来源从"实时快照"换成了"历史日线的 close/pre_close"。

    返回 (状态文字, 仓位系数):
      强势 -> ("强势", 1.0)   正常仓位
      震荡 -> ("震荡", 0.5)   仓位减半
      弱势 -> ("弱势", 0.0)   当日停止开新仓(但已有持仓的卖出规则照常执行)
    """
    valid = day_data.dropna(subset=["close", "pre_close"])
    valid = valid[valid["pre_close"] > 0]
    if valid.empty:
        return "震荡", 0.5

    pct_change = (valid["close"] - valid["pre_close"]) / valid["pre_close"]
    total = len(pct_change)
    up = int((pct_change > 0).sum())
    down = int((pct_change < 0).sum())
    breadth = (up - down) / total if total else 0
    market_score = (breadth + 1) / 2 * 100

    if market_score >= 60:
        return "强势", 1.0
    elif market_score >= 40:
        return "震荡", 0.5
    else:
        return "弱势", 0.0


# ===========================================================================
# 第三部分:交易成本 / 风控 / 卖出规则 / 仓位管理 —— 参数集中管理
# ===========================================================================
@dataclass
class TradingCosts:
    commission_rate: float = 0.00025
    stamp_duty_rate: float = 0.0005
    slippage_rate: float = 0.003
    min_commission: float = 5.0


@dataclass
class RiskControl:
    max_daily_drawdown: float = -0.03
    max_industry_exposure_pct: float = 0.35
    triggered_today: bool = False


@dataclass
class SellRules:
    """二、卖出规则模块的参数。优先级见类注释:止损 > 趋势走弱 > 超期未涨 > 止盈。"""
    stop_loss_pct: float = -0.05                # 止损线:相对成本价跌幅
    max_holding_days: int = 10                  # 最长持有交易日数
    max_holding_no_gain_threshold: float = 0.0  # 超过最长持有天数时,收益率不超过这个值就离场
    take_profit_tier1_pct: float = 0.08         # 止盈第一档:盈利达到这个比例
    take_profit_tier1_fraction: float = 0.5     # 第一档卖出的仓位比例
    take_profit_tier2_pct: float = 0.15         # 止盈第二档
    take_profit_tier2_fraction: float = 1.0     # 第二档卖出剩余全部


@dataclass
class PositionSizing:
    """三、仓位管理模块的参数。最终单票仓位 = base × 市场系数 × 信号强度系数,且不超过上限。"""
    base_position_size_pct: float = 0.20   # 强势市场、满强度信号下的基准单票仓位(占总权益比例)
    max_single_stock_pct: float = 0.25     # 单票仓位硬上限,不管信号多强都不能突破
    signal_strength_floor: float = 0.5     # 信号强度的下限系数(0.5~1.0 之间线性映射,不会因为信号弱就完全不买)


# ===========================================================================
# 第四部分:核心回测/模拟盘引擎
# ===========================================================================
class AShareBacktester:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        costs: Optional[TradingCosts] = None,
        risk: Optional[RiskControl] = None,
        sell_rules: Optional[SellRules] = None,
        position_sizing: Optional[PositionSizing] = None,
        risk_dates: Optional[set] = None,
        risk_date_position_discount: float = 0.5,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            必须包含列 ['date', 'code', 'open', 'high', 'low', 'close', 'pre_close']。
            可选列:'volume'(缩量判断用) / 'suspended' / 'industry'
            !! 价格必须是"后复权"或"前复权"价格 !!
            !! 股票池必须是全历史(含退市股),且要覆盖足够多股票用于市场环境的
               涨跌家数统计,不能只用"当前存活"的少数股票 !!
        """
        required_cols = {"date", "code", "open", "high", "low", "close", "pre_close"}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"输入数据缺少必要列: {missing}")

        self.data = data.sort_values(["code", "date"]).reset_index(drop=True)
        if "suspended" not in self.data.columns:
            self.data["suspended"] = False
        if "industry" not in self.data.columns:
            self.data["industry"] = "未分类"
        if "volume" not in self.data.columns:
            self.data["volume"] = np.nan

        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: dict[str, list[dict]] = {}
        self.industry_of: dict[str, str] = {}

        self.costs = costs or TradingCosts()
        self.risk = risk or RiskControl()
        self.sell_rules = sell_rules or SellRules()
        self.position_sizing = position_sizing or PositionSizing()
        self.risk_dates = risk_dates or set()
        self.risk_date_position_discount = risk_date_position_discount

        self.trade_log: list[dict] = []
        self.daily_equity: list[dict] = []
        self._day_start_equity = initial_cash
        self._date_index: dict = {}
        self.last_price: dict[str, float] = {}  # 停牌等原因当天没有行情时,用于估值兜底(见 _current_equity)

    # ------------------------------------------------------------------
    # 涨跌停判定
    # ------------------------------------------------------------------
    @staticmethod
    def _limit_pct(code: str, is_st: bool = False) -> float:
        if is_st:
            return 0.05
        if code.startswith(("300", "301", "688")):
            return 0.20
        if code.startswith(("8", "43", "92")):
            return 0.30
        return 0.10

    def _fill_ratio_at_limit(self, row: pd.Series, direction: str) -> float:
        limit_pct = self._limit_pct(row["code"])
        if direction == "buy":
            limit_price = round(row["pre_close"] * (1 + limit_pct), 2)
            one_word = (row["open"] == row["high"] == row["low"]) and (row["close"] >= limit_price - 0.01)
        else:
            limit_price = round(row["pre_close"] * (1 - limit_pct), 2)
            one_word = (row["open"] == row["high"] == row["low"]) and (row["close"] <= limit_price + 0.01)

        if one_word:
            return 0.0

        day_range = row["high"] - row["low"]
        if day_range <= 0:
            return 0.0

        touched_but_open = 0.5 * min(1.0, day_range / (row["pre_close"] * limit_pct))
        return float(np.clip(touched_but_open, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 一、买入信号(只负责"要不要买"和"信号有多强",不管仓位大小)
    # ------------------------------------------------------------------
    def generate_buy_signal(self, hist_slice: pd.DataFrame) -> tuple[bool, float]:
        """
        多头趋势下的短期缩量回调(低吸):
          - 多头:最新收盘价在10日均线上方
          - 缩量:最新一日成交量相较前一日明显萎缩(视为洗盘而非出货)
        返回 (是否触发买入, 信号强度 0~1)。信号强度只影响仓位大小(见三、仓位管理模块),
        不影响"要不要买"这个二元判断。

        !! hist_slice 必须严格早于当前交易日 T(date < T),绝不能引用 T 日数据,
           否则就是未来函数。!!
        """
        if len(hist_slice) < 15:
            return False, 0.0

        closes = hist_slice["close"].values
        volumes = hist_slice["volume"].values

        ma10 = closes[-10:].mean()
        current_close = closes[-1]
        if current_close <= ma10:
            return False, 0.0

        prev_vol, last_vol = volumes[-2], volumes[-1]
        if np.isnan(prev_vol) or np.isnan(last_vol) or prev_vol == 0:
            return False, 0.0  # 量能数据缺失,保持观望,不伪造信号

        shrink_ratio = last_vol / prev_vol
        if shrink_ratio >= 0.8:
            return False, 0.0

        # 信号强度:缩量越明显 + 离10日均线越远,强度越高(简单线性映射,便于理解和调参)
        shrink_strength = float(np.clip((0.8 - shrink_ratio) / 0.8, 0.0, 1.0))
        trend_strength = float(np.clip((current_close - ma10) / ma10 / 0.05, 0.0, 1.0))
        strength = 0.5 * shrink_strength + 0.5 * trend_strength
        return True, strength

    # ------------------------------------------------------------------
    # 二、卖出规则模块
    # ------------------------------------------------------------------
    def generate_sell_signal(
        self, code: str, hist_slice: pd.DataFrame, current_row: pd.Series
    ) -> Optional[tuple[float, str]]:
        """
        返回 (卖出比例 0~1, 触发原因) 或 None(不卖)。
        优先级从高到低,命中即返回,不再继续判断:
          1. 止损 —— 保护本金优先于一切
          2. 趋势走弱(跌破10日均线)
          3. 超过最长持有天数仍未有明显收益
          4. 止盈(分批)
        """
        avg_cost = self._avg_cost(code)
        if avg_cost <= 0:
            return None

        current_price = current_row["close"]
        pnl_pct = (current_price - avg_cost) / avg_cost

        # 1. 止损:无条件全部卖出
        if pnl_pct <= self.sell_rules.stop_loss_pct:
            return 1.0, f"止损(浮亏{pnl_pct:.1%})"

        # 2. 趋势走弱:跌破10日均线
        if len(hist_slice) >= 10:
            ma10 = hist_slice["close"].values[-10:].mean()
            if current_price < ma10:
                return 1.0, "趋势走弱(跌破10日均线)"

        # 3. 超过最长持有天数仍未有明显收益
        lots = self.positions.get(code, [])
        if lots:
            oldest_buy_date = min(lot["buy_date"] for lot in lots)
            current_idx = self._date_index.get(current_row["date"])
            buy_idx = self._date_index.get(oldest_buy_date)
            if current_idx is not None and buy_idx is not None:
                holding_days = current_idx - buy_idx
                if holding_days >= self.sell_rules.max_holding_days and \
                        pnl_pct <= self.sell_rules.max_holding_no_gain_threshold:
                    return 1.0, f"持有{holding_days}个交易日未达预期,离场"

        # 4. 止盈:分两档
        if pnl_pct >= self.sell_rules.take_profit_tier2_pct:
            return self.sell_rules.take_profit_tier2_fraction, f"止盈第二档(盈利{pnl_pct:.1%})"
        if pnl_pct >= self.sell_rules.take_profit_tier1_pct:
            return self.sell_rules.take_profit_tier1_fraction, f"止盈第一档(盈利{pnl_pct:.1%})"

        return None

    # ------------------------------------------------------------------
    # T+1 / 均价 辅助函数
    # ------------------------------------------------------------------
    def _sellable_shares(self, code: str, current_date) -> int:
        if code not in self.positions:
            return 0
        return sum(lot["shares"] for lot in self.positions[code] if lot["buy_date"] < current_date)

    def _total_shares(self, code: str) -> int:
        if code not in self.positions:
            return 0
        return sum(lot["shares"] for lot in self.positions[code])

    def _avg_cost(self, code: str) -> float:
        if code not in self.positions or not self.positions[code]:
            return 0.0
        total_shares = sum(lot["shares"] for lot in self.positions[code])
        total_cost = sum(lot["shares"] * lot["cost_price"] for lot in self.positions[code])
        return total_cost / total_shares if total_shares else 0.0

    def _current_industry_exposure(self, industry: str, price_by_code: dict) -> float:
        equity = self._current_equity(price_by_code)
        if equity <= 0:
            return 0.0
        industry_value = 0.0
        for code, lots in self.positions.items():
            if self.industry_of.get(code) == industry:
                price = price_by_code.get(code, 0.0)
                industry_value += sum(lot["shares"] for lot in lots) * price
        return industry_value / equity

    def _current_equity(self, price_by_code: dict) -> float:
        """
        !! 这里有一处修复:如果某只股票当天停牌(不在 price_by_code 里),之前的写法会把
           它的持仓市值直接算作 0,导致停牌期间权益曲线出现一个"凭空消失又凭空出现"的
           假跳空,还可能误触发单日回撤熔断。现在改成用 self.last_price 里记录的
           最后一次已知收盘价估值 —— 停牌期间市值按最后已知价格"冻结",而不是清零。 !!
        """
        equity = self.cash
        for code, lots in self.positions.items():
            price = price_by_code.get(code)
            if price is None:
                price = self.last_price.get(code)
            if price is not None:
                equity += sum(lot["shares"] for lot in lots) * price
        return equity

    # ------------------------------------------------------------------
    # 三、仓位管理:计算本次买入的目标金额
    # ------------------------------------------------------------------
    def _compute_position_value(self, signal_strength: float, market_multiplier: float, equity: float) -> float:
        signal_factor = self.position_sizing.signal_strength_floor + \
            (1 - self.position_sizing.signal_strength_floor) * signal_strength
        position_pct = self.position_sizing.base_position_size_pct * market_multiplier * signal_factor
        position_pct = min(position_pct, self.position_sizing.max_single_stock_pct)
        return equity * position_pct

    # ------------------------------------------------------------------
    # 交易执行:滑点 + 税费 + 涨跌停部分成交 + 停牌 + T+1 + 行业集中度
    # ------------------------------------------------------------------
    def _execute_buy(self, row: pd.Series, target_value: float, is_risk_date: bool = False) -> bool:
        code = row["code"]
        try:
            if bool(row.get("suspended", False)):
                logger.info(f"{row['date']} {code} 停牌中,买入跳过")
                return False

            fill_ratio = self._fill_ratio_at_limit(row, direction="buy")
            if fill_ratio <= 0:
                logger.warning(f"{row['date']} {code} 涨停一字板锁死,买入失败(无法成交)")
                return False

            industry = row.get("industry", "未分类")
            price_by_code_now = {code: row["close"]}
            projected_exposure = self._current_industry_exposure(industry, price_by_code_now) + (
                target_value / max(self._current_equity(price_by_code_now), 1e-9)
            )
            if projected_exposure > self.risk.max_industry_exposure_pct:
                logger.info(
                    f"{row['date']} {code} 行业[{industry}]敞口将超过上限"
                    f"({projected_exposure:.1%} > {self.risk.max_industry_exposure_pct:.1%}),买入跳过"
                )
                return False

            effective_target_value = target_value
            if is_risk_date:
                effective_target_value *= self.risk_date_position_discount
                logger.info(f"{row['date']} 处于日历风险日,{code} 新开仓仓位按{self.risk_date_position_discount:.0%}打折")

            exec_price = row["open"] * (1 + self.costs.slippage_rate)
            vol = row.get("volume", np.nan)
            max_shares_by_fill = int(vol * fill_ratio) if not np.isnan(vol) else np.inf
            shares = int(effective_target_value / exec_price // 100 * 100)
            shares = int(min(shares, max_shares_by_fill)) if max_shares_by_fill != np.inf else shares
            if shares <= 0:
                return False

            gross_amount = shares * exec_price
            commission = max(gross_amount * self.costs.commission_rate, self.costs.min_commission)
            total_cost = gross_amount + commission

            if total_cost > self.cash:
                logger.info(f"{row['date']} {code} 资金不足,买入失败(需{total_cost:.2f},现金{self.cash:.2f})")
                return False

            self.cash -= total_cost
            self.positions.setdefault(code, []).append({
                "shares": shares, "cost_price": exec_price, "buy_date": row["date"],
            })
            self.industry_of[code] = industry

            self.trade_log.append({
                "date": row["date"], "code": code, "action": "BUY",
                "price": exec_price, "shares": shares, "commission": commission,
                "fill_ratio": fill_ratio, "reason": "买入信号触发",
            })
            return True
        except Exception as e:
            logger.error(f"买入执行异常 {code} {row.get('date')}: {e}")
            return False

    def _execute_sell(self, row: pd.Series, shares_to_sell: Optional[int] = None, reason: str = "") -> bool:
        code = row["code"]
        current_date = row["date"]
        try:
            if bool(row.get("suspended", False)):
                logger.info(f"{current_date} {code} 停牌中,卖出跳过(仓位被锁死,无法操作)")
                return False

            sellable = self._sellable_shares(code, current_date)
            if sellable <= 0:
                if self._total_shares(code) > 0:
                    logger.info(f"{current_date} {code} 当日买入份额受 T+1 限制,今日不可卖出")
                return False

            fill_ratio = self._fill_ratio_at_limit(row, direction="sell")
            if fill_ratio <= 0:
                logger.warning(f"{current_date} {code} 跌停一字板锁死,卖出失败(无法成交)")
                return False

            target_shares = shares_to_sell or sellable
            target_shares = min(target_shares, sellable)
            vol = row.get("volume", np.nan)
            max_shares_by_fill = int(vol * fill_ratio) if not np.isnan(vol) else np.inf
            shares = int(min(target_shares, max_shares_by_fill)) if max_shares_by_fill != np.inf else target_shares
            if shares <= 0:
                return False

            exec_price = row["open"] * (1 - self.costs.slippage_rate)
            gross_amount = shares * exec_price
            commission = max(gross_amount * self.costs.commission_rate, self.costs.min_commission)
            stamp_duty = gross_amount * self.costs.stamp_duty_rate
            net_amount = gross_amount - commission - stamp_duty

            self.cash += net_amount

            remaining = shares
            lots = self.positions.get(code, [])
            sellable_lots = sorted(
                (lot for lot in lots if lot["buy_date"] < current_date),
                key=lambda l: l["buy_date"],
            )
            for lot in sellable_lots:
                if remaining <= 0:
                    break
                deduct = min(lot["shares"], remaining)
                lot["shares"] -= deduct
                remaining -= deduct
            self.positions[code] = [lot for lot in lots if lot["shares"] > 0]
            if not self.positions[code]:
                del self.positions[code]

            self.trade_log.append({
                "date": current_date, "code": code, "action": "SELL",
                "price": exec_price, "shares": shares,
                "commission": commission, "stamp_duty": stamp_duty,
                "fill_ratio": fill_ratio, "reason": reason,
            })
            return True
        except Exception as e:
            logger.error(f"卖出执行异常 {code} {row.get('date')}: {e}")
            return False

    # ------------------------------------------------------------------
    # 风控:单日最大回撤熔断
    # ------------------------------------------------------------------
    def _check_daily_drawdown_breaker(self, current_equity: float) -> bool:
        if self._day_start_equity <= 0:
            return False
        drawdown = (current_equity - self._day_start_equity) / self._day_start_equity
        if drawdown <= self.risk.max_daily_drawdown:
            if not self.risk.triggered_today:
                logger.warning(f"触发单日最大回撤熔断({drawdown:.2%}),强制平仓可卖份额并停止今日新开仓")
                self.risk.triggered_today = True
            return True
        return False

    def _force_liquidate_all(self, row_by_code: dict):
        for code in list(self.positions.keys()):
            if code in row_by_code:
                self._execute_sell(row_by_code[code], reason="单日回撤熔断强平")

    # ------------------------------------------------------------------
    # 主回测循环
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        dates = sorted(self.data["date"].unique())
        self._date_index = {d: i for i, d in enumerate(dates)}

        for date in dates:
            day_data = self.data[self.data["date"] == date]
            row_by_code = {r["code"]: r for _, r in day_data.iterrows()}
            is_risk_date = date in self.risk_dates

            # 一、市场环境模块:今天算强势/震荡/弱势,仓位系数是多少
            market_status, market_multiplier = assess_market_regime(day_data)

            self._day_start_equity = self._current_equity(
                {c: r["open"] for c, r in row_by_code.items()}
            ) or self.cash
            self.risk.triggered_today = False

            for code, row in row_by_code.items():
                if bool(row.get("suspended", False)):
                    continue

                hist_slice = self.data[
                    (self.data["code"] == code) & (self.data["date"] < date)
                ].tail(60)

                current_equity = self._current_equity({c: r["close"] for c, r in row_by_code.items()})
                if self._check_daily_drawdown_breaker(current_equity):
                    continue

                # 二、卖出规则模块:不管市场环境如何,已有持仓的止盈止损始终优先判断
                if code in self.positions:
                    sell_decision = self.generate_sell_signal(code, hist_slice, row)
                    if sell_decision:
                        fraction, reason = sell_decision
                        sellable = self._sellable_shares(code, date)
                        shares_to_sell = int(sellable * fraction // 100 * 100)
                        if shares_to_sell > 0:
                            self._execute_sell(row, shares_to_sell=shares_to_sell, reason=reason)
                    continue  # 当日同一只股票不会既卖又买

                # 弱势市场:停止开新仓(一、市场环境模块的核心作用)
                if market_multiplier <= 0:
                    continue

                should_buy, strength = self.generate_buy_signal(hist_slice)
                if should_buy:
                    # 三、仓位管理模块:市场越强、信号越强,仓位越高,但不超过单票硬上限
                    equity_now = self._current_equity({c: r["close"] for c, r in row_by_code.items()})
                    target_value = self._compute_position_value(strength, market_multiplier, equity_now)
                    self._execute_buy(row, target_value, is_risk_date=is_risk_date)

            if self.risk.triggered_today:
                self._force_liquidate_all(row_by_code)

            equity_eod = self._current_equity({c: r["close"] for c, r in row_by_code.items()})
            self.last_price.update({c: r["close"] for c, r in row_by_code.items()})
            self.daily_equity.append({"date": date, "equity": equity_eod, "market_status": market_status})

        return pd.DataFrame(self.daily_equity)

    # ------------------------------------------------------------------
    # 结果统计
    # ------------------------------------------------------------------
    def summary(self) -> dict:
        eq_df = pd.DataFrame(self.daily_equity)
        if eq_df.empty:
            return {}
        eq_df["returns"] = eq_df["equity"].pct_change()

        total_return = eq_df["equity"].iloc[-1] / self.initial_cash - 1
        max_drawdown = ((eq_df["equity"] / eq_df["equity"].cummax()) - 1).min()
        annual_vol = eq_df["returns"].std() * np.sqrt(252)
        sharpe = (eq_df["returns"].mean() * 252) / annual_vol if annual_vol > 0 else 0.0

        base = {
            "总收益率": f"{total_return:.2%}",
            "最大回撤": f"{max_drawdown:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率(简化,未扣无风险利率)": f"{sharpe:.2f}",
            "交易次数": len(self.trade_log),
            "剩余现金": f"{self.cash:.2f}",
            "剩余持仓数": len(self.positions),
        }
        base.update(self.compute_trade_stats())
        return base

    def compute_trade_stats(self) -> dict:
        """
        从 trade_log 里按 FIFO 配对 BUY/SELL,计算每笔完整交易的收益率和持有天数,
        得到胜率、平均持仓天数 —— 这是唯一"真实计算"而非编造的回测指标来源。
        """
        trades_by_code: dict[str, list[dict]] = {}
        for t in self.trade_log:
            trades_by_code.setdefault(t["code"], []).append(t)

        completed = []
        for code, log in trades_by_code.items():
            buy_queue = []
            for t in log:
                if t["action"] == "BUY":
                    buy_queue.append({"date": t["date"], "price": t["price"], "shares": t["shares"]})
                elif t["action"] == "SELL":
                    remaining = t["shares"]
                    while remaining > 0 and buy_queue:
                        lot = buy_queue[0]
                        matched = min(lot["shares"], remaining)
                        holding_days = self._date_index.get(t["date"], 0) - self._date_index.get(lot["date"], 0)
                        ret = (t["price"] - lot["price"]) / lot["price"]
                        completed.append({"code": code, "return": ret, "holding_days": holding_days})
                        lot["shares"] -= matched
                        remaining -= matched
                        if lot["shares"] <= 0:
                            buy_queue.pop(0)

        if not completed:
            return {"完整交易笔数": 0}

        returns = [c["return"] for c in completed]
        holding_days = [c["holding_days"] for c in completed]
        win_rate = sum(1 for r in returns if r > 0) / len(returns)

        return {
            "完整交易笔数": len(completed),
            "胜率": f"{win_rate:.1%}",
            "平均持仓天数": round(sum(holding_days) / len(holding_days), 1),
            "平均单笔收益率": f"{sum(returns) / len(returns):.2%}",
        }


# ===========================================================================
# 第五部分:实盘下单容错封装(未改动)
# ===========================================================================
def submit_order_with_retry(
    order_func,
    *args,
    max_retries: int = 3,
    retry_interval: float = 2.0,
    **kwargs,
):
    for attempt in range(1, max_retries + 1):
        try:
            result = order_func(*args, **kwargs)
            if result and result.get("status") == "accepted":
                return result
            logger.warning(f"第{attempt}次报单被拒: {result}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"第{attempt}次下单网络异常: {e},{retry_interval}s 后重试")
            time.sleep(retry_interval)
        except Exception as e:
            logger.error(f"下单发生未预期异常: {e}")
            break
    logger.error("多次重试后下单仍失败,已触发报警(可在此接入短信/邮件/钉钉/企业微信通知)")
    return None


if __name__ == "__main__":
    select_stocks()
