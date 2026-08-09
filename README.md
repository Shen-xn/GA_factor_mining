# GA Factor Mining — Stock Development

当前分支为 `stock_dev` 的个股研究线，只维护个股遗传因子、滚动模型和 Top-50 选股。板块业务代码位于独立的 `sector_dev` 分支。

## 目录

```text
configs/stock/                     个股配置
data/stock/                        个股原始数据，不提交 Git
outputs/stock/{v1,top50}/          缓存、模型、预测和机器指标
reports/stock/{v1,top50}/          Markdown/PDF 人工报告
src/ga_factor_mining/common/       两分支共用基础组件
src/ga_factor_mining/stock/        个股业务代码
tests/{common,stock}/              公共与个股测试
```

## 运行

```powershell
python -m pip install -e .

# 第一代因子与模型对照
python -m ga_factor_mining.stock.v1.run_pipeline all

# Top-50 V2
python -m ga_factor_mining.stock.top50.run_experiments --stage all-validation
python -m ga_factor_mining.stock.top50.run_experiments --stage final-test
```

默认配置为 `configs/stock/v1.json` 和 `configs/stock/top50.json`。

数据、Parquet 缓存和模型文件均被 `.gitignore` 排除。机器产物只写入 `outputs/stock/`，人工报告只写入 `reports/stock/`。
