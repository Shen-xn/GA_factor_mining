# GA Factor Mining — Sector Development

当前分支为 `sector_dev`，只维护板块因子挖掘与行业/概念板块轮动。个股研究代码位于独立的 `stock_dev` 分支，在本分支工作区中不可见。

## 目录

```text
configs/sector/                         板块配置
data/sector/                            板块原始数据，不提交 Git
outputs/sector/{rotation,factor_mining}/
                                        缓存、持仓和机器指标
reports/sector/{rotation,factor_mining}/
                                        Markdown/PDF/HTML 和图表
src/ga_factor_mining/common/            两分支共用基础组件
src/ga_factor_mining/sector/            板块业务代码
tests/{common,sector}/                   公共与板块测试
```

## 运行

```powershell
python -m pip install -e .

# 板块轮动与模型实验
python -m ga_factor_mining.sector.rotation.run_experiments --with-lgbm
python -m ga_factor_mining.sector.rotation.summarize_results
python -m ga_factor_mining.sector.rotation.rolling_validation
python -m ga_factor_mining.sector.rotation.build_paper_report

# 板块遗传因子挖掘
python -m ga_factor_mining.sector.factor_mining.run_factor_mining
```

板块因子默认配置为 `configs/sector/factor_mining.json`。

机器中间产物只写入 `outputs/sector/`，人工报告只写入 `reports/sector/`。数据、Parquet 缓存和大型模型均被 `.gitignore` 排除。

## 与 stock_dev 同步公共代码

公共组件的修改应单独提交，然后在另一开发分支执行：

```powershell
git cherry-pick <common-commit>
```

不要直接合并整个开发分支，否则会把另一条业务线的目录带入当前分支。
