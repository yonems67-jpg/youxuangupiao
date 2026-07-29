## -*- coding: utf-8 -*-
"""
阿里云函数计算 FC 入口文件 (fc_handler.py)
==================================================
"""

import base64
import json
import logging
import os
import time
import requests

from select_stocks import build_result
from fortune import compute_fortune

# 强制将当前 Python 进程设置为北京时间 (东八区)
os.environ['TZ'] = os.getenv('TZ', 'Asia/Shanghai')
if hasattr(time, 'tzset'):
    time.tzset()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

GITHUB_API_BASE = "https://api.github.com"


def _github_config():
    """获取环境变量配置"""
    return {
        "owner": os.environ["GITHUB_OWNER"],
        "repo": os.environ["GITHUB_REPO"],
        "token": os.environ["GITHUB_TOKEN"],
        "branch": os.environ.get("GITHUB_BRANCH", "main"),
    }


def _get_headers(token: str):
    """构造请求头（含 User-Agent 防拦截）"""
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "Aliyun-FC-StockPicker-Script",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _get_file_sha(cfg, path):
    """获取 GitHub 仓库中原文件的 sha（更新需要，新建返回 None）"""
    url = f"{GITHUB_API_BASE}/repos/{cfg['owner']}/{cfg['repo']}/contents/{path}"
    headers = _get_headers(cfg["token"])
    params = {"ref": cfg["branch"]}

    resp = requests.get(url, headers=headers, params=params)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def upload_to_github(path: str, data: dict, message: str = "Update stock picks"):
    """把字典转为 JSON 并推送到 GitHub"""
    cfg = _github_config()
    headers = _get_headers(cfg["token"])

    content_str = json.dumps(data, ensure_ascii=False, indent=2)
    content_b64 = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")

    url = f"{GITHUB_API_BASE}/repos/{cfg['owner']}/{cfg['repo']}/contents/{path}"
    sha = _get_file_sha(cfg, path)

    payload = {
        "message": message,
        "content": content_b64,
        "branch": cfg["branch"],
    }

    if sha:
        payload["sha"] = sha

    resp = requests.put(url, headers=headers, json=payload)

    if resp.status_code not in (200, 201):
        logger.error(f"GitHub 上传失败 [HTTP {resp.status_code}]: {resp.text[:300]}")
        resp.raise_for_status()

    logger.info(f"成功更新 {path} 到 GitHub ({cfg['owner']}/{cfg['repo']}@{cfg['branch']})")
    return resp.json()



def handler(event, context):
    """阿里云 FC 默认入口函数"""

    logger.info("阿里云 FC 选股任务开始运行...")

    try:
        result = build_result()

        upload_to_github(
            "site/data/latest.json",
            result
        )

        logger.info("选股任务执行完毕并已推送到 GitHub！")


        logger.info("阿里云 FC 八字任务开始运行...")

        fortune_result = compute_fortune()

        upload_to_github(
            "site/data/fortune.json",
            fortune_result,
            "Update fortune"
        )

        logger.info("八字任务执行完毕并已推送到 GitHub！")


        return {
            "status": "ok",
            "market_score": (
                result.get("market", {}).get("market_score")
                if isinstance(result, dict)
                else None
            ),
            "fortune_updated": True,
            "stock_updated": True
        }


    except Exception as e:
        logger.error(
            f"执行失败: {e}",
            exc_info=True
        )
        raise 

