# -*- coding: utf-8 -*-
"""
行业分类映射构建脚本 —— 请在你自己电脑上手动运行,不要放进 GitHub Actions
==================================================
背景:实测 ak.stock_individual_info_em() 在 GitHub Actions 的运行环境里
15 只股票全部查询失败(加了重试、请求间隔也没用),报错都是同一种"响应体为空"
(Expecting value: line 1 column 1 (char 0))。全军覆没不像是偶发限流,
更像是东方财富把 GitHub Actions(Azure 海外IP段)整体限制/屏蔽了。

解决思路:与其每次运行都指望这个不稳定的实时查询,不如在国内网络环境下
跑一次这个脚本,把"代码 -> 行业"的映射整个生成好、提交进仓库,
select_stocks.py 之后只需要读这份本地映射表做字典查找,不再发起任何网络请求。
行业分类变化很慢,这份映射表不需要每天更新——发现行业信息过时了、
或者新上市股票缺失映射了,再重新跑一次这个脚本更新即可(支持断点续传)。

用法(在你自己电脑的终端里,不是 GitHub Actions):
  pip install akshare pandas
  python scripts/build_industry_map.py
  git add site/data/industry_map.json
  git commit -m "更新行业分类映射"
  git push

跑全市场 5000+ 只股票会比较慢(每只间隔 0.3 秒,预计 25~30 分钟),
中途每 100 只会自动保存一次,断网/中断了直接重新跑这个脚本即可继续,
已经抓到的不会重复请求。
"""

import json
import re
import time
import logging
from pathlib import Path

import akshare as ak
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_industry_map")

OUTPUT_PATH = Path("site/data/industry_map.json")
SLEEP_SECONDS = 0.3
MAX_RETRIES = 2


def fetch_industry(code: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            info = ak.stock_individual_info_em(symbol=code)
            row = info.loc[info["item"] == "行业", "value"]
            return str(row.values[0]) if not row.empty else "未分类"
        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(1.0)
            else:
                logger.warning(f"{code} 查询行业失败: {e}")
                return "未分类"


def main():
    stock_list = ak.stock_info_a_code_name()
    code_col = "code" if "code" in stock_list.columns else stock_list.columns[0]

    # 断点续传:已有映射的直接跳过,不用每次从头跑一遍全市场
    existing = {}
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        logger.info(f"已有映射 {len(existing)} 条,增量更新")

    mapping = dict(existing)
    codes = stock_list[code_col].tolist()
    total = len(codes)

    for i, code in enumerate(codes, start=1):
        code_clean = re.sub(r"[^0-9]", "", str(code))[-6:]
        if code_clean in mapping:
            continue

        mapping[code_clean] = fetch_industry(code_clean)

        if i % 100 == 0:
            logger.info(f"进度 {i}/{total},已写入 {len(mapping)} 条,中途保存一次")
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(SLEEP_SECONDS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"完成,共 {len(mapping)} 条行业映射,已写入 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
