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

  覆盖十个核心维度：
  一、策略逻辑：严格避免未来函数（look-ahead bias），策略参数保持宽容，避免过拟合
  二、交易摩擦：涨跌停部分成交建模、滑点、印花税（仅卖出）、双边佣金
  三、数据品质：提示使用全历史股票池（含退市股）+ 后复权/前复权价格，避免幸存者偏差
  四、实盘风控：单日最大回撤熔断 + 下单容错重试（断网/超时/废单不崩程序）
  五、T+1 结算：当日买入的份额，当日不可卖出（A股与美股/港股最根本的结构差异）
  六、涨跌停部分成交：区分"一字板完全锁死"与"触及但有成交"两种情况
  七、信息滞后：龙虎榜/主力资金净流入等数据是收盘后才公布的，只能作为 T+1 的参考信号
  八、停牌风险：停牌期间无法买卖，需要显式标记并跳过
  九、行业集中度：单一行业敞口设上限，避免"分散持仓"实际是同一题材的重复暴露
  十、日历/政策事件：巨型IPO、两会窗口等日历效应，作为独立风险过滤条件

使用前必读：
  - 本文件不含真实行情数据，需自行接入券商/数据商的"后复权"日线或分钟线数据。
  - 股票池必须包含历史上已退市/被ST摘牌的个股，否则回测收益会被"幸存者偏差"严重虚高。
  - 这是一个教学框架，实盘接入前请务必用小资金、模拟盘充分验证。
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 日志配置：实盘容错的基础——任何异常都要留痕，而不是让程序静默崩溃
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("a_share_quant")


# ---------------------------------------------------------------------------
# 配置：交易成本与风控参数（集中管理，方便调整，也方便做敏感性分析）
# ---------------------------------------------------------------------------
@dataclass
class TradingCosts:
    commission_rate: float = 0.00025   # 双边佣金，示例万2.5（各券商不同，自行核对）
    stamp_duty_rate: float = 0.0005    # 印花税，仅卖出收取，万5
    slippage_rate: float = 0.003       # 滑点，建议 0.2%~0.5% 区间，短线换手越高越要放大
    min_commission: float = 5.0        # 部分券商单笔最低佣金


@dataclass
class RiskControl:
    max_daily_drawdown: float = -0.03      # 单日最大回撤熔断阈值，达到即强平+停止当日新开仓
    max_industry_exposure_pct: float = 0.35  # 单一行业最大敞口占总权益比例（维度九）
    triggered_today: bool = False


# ---------------------------------------------------------------------------
# 核心回测引擎
# ---------------------------------------------------------------------------
class AShareBacktester:
    def __init__(
        self,
        data: pd.DataFrame,
        initial_cash: float = 100_000.0,
        costs: Optional[TradingCosts] = None,
        risk: Optional[RiskControl] = None,
        position_size_pct: float = 0.2,
        risk_dates: Optional[set] = None,
        risk_date_position_discount: float = 0.5,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            必须包含列 ['date', 'code', 'open', 'high', 'low', 'close', 'pre_close']。
            可选列：
              'volume'    : 成交量，用于估算涨跌停部分成交比例（维度六）
              'suspended' : bool，当日是否停牌（维度八）
              'industry'  : 所属申万一级行业，用于行业集中度控制（维度九）
            !! 价格必须是"后复权"或"前复权"价格，否则除权缺口会让均线等指标严重失真 !!
            !! 股票池必须是全历史（含退市股），不能只用"当前存活"的股票列表回测历史 !!
        position_size_pct : float
            单只股票买入时，最多使用当前现金的比例（简单仓位管理，避免单票集中度过高）
        risk_dates : set[date]
            日历/政策风险日集合（维度十），例如巨型IPO申购日、上市日、两会窗口等。
            这些日期内，新开仓仓位会被 risk_date_position_discount 打折。
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
            self.data["volume"] = np.nan  # 缺失时退化为"完全锁死/未锁死"二元判定

        self.cash = initial_cash
        self.initial_cash = initial_cash
        # positions[code] = list of lots: {'shares', 'cost_price', 'buy_date'}
        # 用 lot（批次）而不是单一汇总，是为了精确支持 T+1（维度五）的分批可卖判定
        self.positions: dict[str, list[dict]] = {}
        self.industry_of: dict[str, str] = {}  # code -> industry，买入时记录

        self.costs = costs or TradingCosts()
        self.risk = risk or RiskControl()
        self.position_size_pct = position_size_pct
        self.risk_dates = risk_dates or set()
        self.risk_date_position_discount = risk_date_position_discount

        self.trade_log: list[dict] = []
        self.daily_equity: list[dict] = []
        self._day_start_equity = initial_cash

    # ------------------------------------------------------------------
    # 涨跌停判定（维度六：区分"完全锁死"与"触及但有成交"）
    # ------------------------------------------------------------------
    @staticmethod
    def _limit_pct(code: str, is_new_stock_5d: bool = False, is_st: bool = False) -> float:
        """
        按代码/状态判断涨跌停幅度：
          - 科创板/创业板新股上市前5个交易日：无涨跌幅限制（is_new_stock_5d=True 时按 None 处理）
          - ST股：±5%
          - 创业板(300/301)/科创板(688)：±20%
          - 北交所(8/4/9开头，简化示例，实际需按具体规则细分)：±30%
          - 其余主板：±10%
        """
        if is_st:
            return 0.05
        if code.startswith(("300", "301", "688")):
            return 0.20
        if code.startswith(("8", "43", "92")):
            return 0.30
        return 0.10

    def _fill_ratio_at_limit(self, row: pd.Series, direction: str) -> float:
        """
        估算涨跌停价位上的成交比例（0~1），而不是简单的 0/1 二元判定。

        规则（简化启发式，若能接入真实盘口委托队列数据，应替换为更精确的估算）：
          - 全天开=高=低=收 且达到涨跌停价 -> 判定为"一字板完全锁死"，fill_ratio = 0
          - 触及涨跌停价，但当日价格区间(high-low)有实质波动
            -> 说明涨跌停板曾被打开过，允许部分/全部成交，
               用 "非涨跌停价的价格区间占比" 作为成交概率的粗略代理
          - 缺失 volume 数据时，退化为上述二元判定
        """
        limit_pct = self._limit_pct(row["code"])
        if direction == "buy":
            limit_price = round(row["pre_close"] * (1 + limit_pct), 2)
            one_word = (row["open"] == row["high"] == row["low"]) and (row["close"] >= limit_price - 0.01)
        else:
            limit_price = round(row["pre_close"] * (1 - limit_pct), 2)
            one_word = (row["open"] == row["high"] == row["low"]) and (row["close"] <= limit_price + 0.01)

        if one_word:
            return 0.0  # 完全锁死，实盘完全买/卖不进去

        day_range = row["high"] - row["low"]
        if day_range <= 0:
            return 0.0

        # 曾经打开过涨跌停：用价格区间的相对宽度粗略代理"有多少成交机会"，
        # 并做保守压缩（乘 0.5），避免过度乐观估计实盘可成交量
        touched_but_open = 0.5 * min(1.0, day_range / (row["pre_close"] * limit_pct))
        return float(np.clip(touched_but_open, 0.0, 1.0))

    # ------------------------------------------------------------------
    # 信号生成：这是"未来函数"最容易混进来的地方，务必只用 T 日之前的数据
    # ------------------------------------------------------------------
    def generate_signal(self, hist_slice: pd.DataFrame) -> str:
        """
        hist_slice: 必须是严格早于当前交易日 T 的历史切片（date < T）。
        !! 这里绝对不能出现对 T 日 close/high/low 的引用 !!
        !! 同理，龙虎榜/主力净流入净流出等数据是"收盘后才公布"的（维度七），
           如果要用这类因子，也只能取 T-1 日及更早的已公布数据，
           绝不能假装能在 T 日盘中拿到当天的龙虎榜结果 !!

        示例策略刻意保持简单、参数宽容（5日均线上/下穿20日均线），
        目的是演示"避免过拟合"——真实研究中可以替换为你自己的信号，
        但要抵制住"加更多条件让历史曲线更好看"的诱惑。
        """
        if len(hist_slice) < 21:
            return "HOLD"

        closes = hist_slice["close"].values
        ma5, ma20 = closes[-5:].mean(), closes[-20:].mean()
        ma5_prev, ma20_prev = closes[-6:-1].mean(), closes[-21:-1].mean()

        if ma5_prev <= ma20_prev and ma5 > ma20:
            return "BUY"
        if ma5_prev >= ma20_prev and ma5 < ma20:
            return "SELL"
        return "HOLD"

    # ------------------------------------------------------------------
    # T+1 辅助函数（维度五）
    # ------------------------------------------------------------------
    def _sellable_shares(self, code: str, current_date) -> int:
        """只统计 buy_date < current_date 的批次，当日新买入的份额不可卖"""
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

    # ------------------------------------------------------------------
    # 行业集中度辅助函数（维度九）
    # ------------------------------------------------------------------
    def _current_industry_exposure(self, industry: str, price_by_code: dict) -> float:
        """返回某行业当前持仓市值占总权益的比例"""
        equity = self._current_equity(price_by_code)
        if equity <= 0:
            return 0.0
        industry_value = 0.0
        for code, lots in self.positions.items():
            if self.industry_of.get(code) == industry:
                price = price_by_code.get(code, 0.0)
                industry_value += sum(lot["shares"] for lot in lots) * price
        return industry_value / equity

    # ------------------------------------------------------------------
    # 交易执行：滑点 + 税费 + 涨跌停部分成交 + 停牌 + T+1 + 行业集中度
    # ------------------------------------------------------------------
    def _execute_buy(self, row: pd.Series, target_value: float, is_risk_date: bool = False) -> bool:
        code = row["code"]
        try:
            if bool(row.get("suspended", False)):
                logger.info(f"{row['date']} {code} 停牌中，买入跳过")
                return False

            fill_ratio = self._fill_ratio_at_limit(row, direction="buy")
            if fill_ratio <= 0:
                logger.warning(f"{row['date']} {code} 涨停一字板锁死，买入失败（无法成交）")
                return False

            # 行业集中度检查（维度九）
            industry = row.get("industry", "未分类")
            price_by_code_now = {code: row["close"]}
            projected_exposure = self._current_industry_exposure(industry, price_by_code_now) + (
                target_value / max(self._current_equity(price_by_code_now), 1e-9)
            )
            if projected_exposure > self.risk.max_industry_exposure_pct:
                logger.info(
                    f"{row['date']} {code} 行业[{industry}]敞口将超过上限"
                    f"({projected_exposure:.1%} > {self.risk.max_industry_exposure_pct:.1%})，买入跳过"
                )
                return False

            # 日历/政策风险日（维度十）：新开仓仓位打折，而不是一刀切禁止交易
            effective_target_value = target_value
            if is_risk_date:
                effective_target_value *= self.risk_date_position_discount
                logger.info(f"{row['date']} 处于日历风险日，{code} 新开仓仓位按{self.risk_date_position_discount:.0%}打折")

            exec_price = row["open"] * (1 + self.costs.slippage_rate)
            max_shares_by_fill = int(row.get("volume", np.inf) * fill_ratio) if not np.isnan(row.get("volume", np.nan)) else np.inf
            shares = int(effective_target_value / exec_price // 100 * 100)
            shares = int(min(shares, max_shares_by_fill)) if max_shares_by_fill != np.inf else shares
            if shares <= 0:
                return False

            gross_amount = shares * exec_price
            commission = max(gross_amount * self.costs.commission_rate, self.costs.min_commission)
            total_cost = gross_amount + commission

            if total_cost > self.cash:
                logger.info(f"{row['date']} {code} 资金不足，买入失败（需{total_cost:.2f}，现金{self.cash:.2f}）")
                return False

            self.cash -= total_cost
            self.positions.setdefault(code, []).append({
                "shares": shares, "cost_price": exec_price, "buy_date": row["date"],
            })
            self.industry_of[code] = industry

            self.trade_log.append({
                "date": row["date"], "code": code, "action": "BUY",
                "price": exec_price, "shares": shares, "commission": commission,
                "fill_ratio": fill_ratio,
            })
            return True
        except Exception as e:  # 单笔交易异常不能让整个回测/实盘崩溃
            logger.error(f"买入执行异常 {code} {row.get('date')}: {e}")
            return False

    def _execute_sell(self, row: pd.Series, shares_to_sell: Optional[int] = None) -> bool:
        code = row["code"]
        current_date = row["date"]
        try:
            if bool(row.get("suspended", False)):
                logger.info(f"{current_date} {code} 停牌中，卖出跳过（仓位被锁死，无法操作）")
                return False

            sellable = self._sellable_shares(code, current_date)
            if sellable <= 0:
                if self._total_shares(code) > 0:
                    logger.info(f"{current_date} {code} 当日买入份额受 T+1 限制，今日不可卖出")
                return False

            fill_ratio = self._fill_ratio_at_limit(row, direction="sell")
            if fill_ratio <= 0:
                logger.warning(f"{current_date} {code} 跌停一字板锁死，卖出失败（无法成交）")
                return False

            target_shares = shares_to_sell or sellable
            target_shares = min(target_shares, sellable)
            max_shares_by_fill = int(row.get("volume", np.inf) * fill_ratio) if not np.isnan(row.get("volume", np.nan)) else np.inf
            shares = int(min(target_shares, max_shares_by_fill)) if max_shares_by_fill != np.inf else target_shares
            if shares <= 0:
                return False

            exec_price = row["open"] * (1 - self.costs.slippage_rate)
            gross_amount = shares * exec_price
            commission = max(gross_amount * self.costs.commission_rate, self.costs.min_commission)
            stamp_duty = gross_amount * self.costs.stamp_duty_rate  # 印花税只在卖出时收
            net_amount = gross_amount - commission - stamp_duty

            self.cash += net_amount

            # FIFO 从最早的可卖批次里扣减
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
                "fill_ratio": fill_ratio,
            })
            return True
        except Exception as e:
            logger.error(f"卖出执行异常 {code} {row.get('date')}: {e}")
            return False

    # ------------------------------------------------------------------
    # 风控：单日最大回撤熔断
    # ------------------------------------------------------------------
    def _check_daily_drawdown_breaker(self, current_equity: float) -> bool:
        if self._day_start_equity <= 0:
            return False
        drawdown = (current_equity - self._day_start_equity) / self._day_start_equity
        if drawdown <= self.risk.max_daily_drawdown:
            if not self.risk.triggered_today:
                logger.warning(f"触发单日最大回撤熔断（{drawdown:.2%}），强制平仓可卖份额并停止今日新开仓")
                self.risk.triggered_today = True
            return True
        return False

    def _force_liquidate_all(self, row_by_code: dict):
        """强平所有可卖份额；注意受 T+1 限制的当日新买入份额依然无法平掉，
        这正是熔断机制在 A 股实盘中必须承认的局限——不是万能的"""
        for code in list(self.positions.keys()):
            if code in row_by_code:
                self._execute_sell(row_by_code[code])

    def _current_equity(self, price_by_code: dict) -> float:
        equity = self.cash
        for code, lots in self.positions.items():
            price = price_by_code.get(code)
            if price is not None:
                equity += sum(lot["shares"] for lot in lots) * price
        return equity

    # ------------------------------------------------------------------
    # 主回测循环
    # ------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        dates = sorted(self.data["date"].unique())

        for date in dates:
            day_data = self.data[self.data["date"] == date]
            row_by_code = {r["code"]: r for _, r in day_data.iterrows()}
            is_risk_date = date in self.risk_dates

            self._day_start_equity = self._current_equity(
                {c: r["open"] for c, r in row_by_code.items()}
            ) or self.cash
            self.risk.triggered_today = False

            for code, row in row_by_code.items():
                if bool(row.get("suspended", False)):
                    continue  # 停牌股当日完全不参与任何交易判断（维度八）

                # 关键：严格只取 date < 当前交易日 的历史数据，杜绝未来函数
                hist_slice = self.data[
                    (self.data["code"] == code) & (self.data["date"] < date)
                ].tail(60)

                signal = self.generate_signal(hist_slice)

                current_equity = self._current_equity({c: r["close"] for c, r in row_by_code.items()})
                if self._check_daily_drawdown_breaker(current_equity):
                    continue  # 熔断后禁止新开仓，但已有可卖持仓会在下面统一强平

                if signal == "BUY" and code not in self.positions:
                    target_value = self.cash * self.position_size_pct
                    self._execute_buy(row, target_value, is_risk_date=is_risk_date)
                elif signal == "SELL" and code in self.positions:
                    self._execute_sell(row)

            if self.risk.triggered_today:
                self._force_liquidate_all(row_by_code)

            equity_eod = self._current_equity({c: r["close"] for c, r in row_by_code.items()})
            self.daily_equity.append({"date": date, "equity": equity_eod})

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

        return {
            "总收益率": f"{total_return:.2%}",
            "最大回撤": f"{max_drawdown:.2%}",
            "年化波动率": f"{annual_vol:.2%}",
            "夏普比率（简化，未扣无风险利率）": f"{sharpe:.2f}",
            "交易次数": len(self.trade_log),
            "剩余现金": f"{self.cash:.2f}",
            "剩余持仓数": len(self.positions),
        }


# ---------------------------------------------------------------------------
# 实盘下单容错封装（QMT / PTrade 等接口风格示意）
# 断网、券商服务器超时、报单被拒（资金不足/废单）时，
# 程序必须能自动重试或报警，绝不能直接崩溃挂掉
# ---------------------------------------------------------------------------
def submit_order_with_retry(
    order_func,
    *args,
    max_retries: int = 3,
    retry_interval: float = 2.0,
    **kwargs,
):
    """
    order_func: 实际的下单函数（对接 QMT/PTrade/券商 API），
                约定返回形如 {'status': 'accepted'|'rejected', ...} 的字典。
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = order_func(*args, **kwargs)
            if result and result.get("status") == "accepted":
                return result
            logger.warning(f"第{attempt}次报单被拒: {result}")
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"第{attempt}次下单网络异常: {e}，{retry_interval}s 后重试")
            time.sleep(retry_interval)
        except Exception as e:
            logger.error(f"下单发生未预期异常: {e}")
            break
    logger.error("多次重试后下单仍失败，已触发报警（可在此接入短信/邮件/钉钉/企业微信通知）")
    return None



if __name__ == "__main__":
    select_stocks()
