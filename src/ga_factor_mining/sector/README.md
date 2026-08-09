# Sector Research

- `rotation/`：板块特征构造、公式与 LightGBM 实验、滚动验证及报告生成。
- `factor_mining/`：以未来 10 日 Top-K 超额收益为目标的遗传因子挖掘。

两条子项目共享 `outputs/sector/rotation/sector_feature_panel.parquet`，但分别写入自己的输出和报告目录。
