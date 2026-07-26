# youxuangupiao

## 项目说明

这是一个每日动量选股项目，通过 `akshare` 拉取 A 股实时行情，并生成 `site/data/latest.json` 供前端页面展示。

## 目录结构

- `requirements.txt`：Python 依赖
- `scripts/select_stocks.py`：核心选股脚本
- `.github/workflows/daily.yml`：GitHub Actions 定时工作流
- `site/index.html`：前端展示页面
- `site/data/.gitkeep`：数据目录占位文件

## 使用方法

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 本地运行：

```bash
python scripts/select_stocks.py
```

3. 打开 `site/index.html` 查看结果。

## Cloudflare Pages 托管

1. 在 Cloudflare Pages 创建一个新项目，连接 GitHub 仓库 `youxuangupiao`。
2. 设置构建命令：无（留空）或 `echo 'no build'`
3. 输出目录：`site`
4. 分支：`main`

> 注意：由于 `site/data/latest.json` 由 GitHub Actions 生成并提交到仓库，Cloudflare Pages 会自动部署最新结果。

## GitHub Actions 说明

工作流 `.github/workflows/daily.yml` 会在每个工作日 UTC 10:00（北京时间 18:00）执行一次，运行选股脚本并将生成结果提交回仓库。

## 额外说明

- `site/data/latest.json` 由脚本生成，不应直接提交。
- 若希望手动触发，可以在 Actions 页面执行 `workflow_dispatch`。
