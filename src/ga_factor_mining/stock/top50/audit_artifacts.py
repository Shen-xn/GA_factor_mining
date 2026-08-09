"""审计正式产物的Top-50完整性、标签隔离和冻结状态。"""
from pathlib import Path
import json
import pandas as pd
from .data import load_config

def main():
 c=load_config("configs/stock/top50.json");root=Path(c["paths"]["artifacts"]);errors=[];h=int(c["target"]["horizon"])
 dates=pd.read_parquet(c["paths"]["prepared_data"],columns=["trade_date"]).trade_date.astype(str).drop_duplicates().sort_values().tolist();position={d:i for i,d in enumerate(dates)}
 for path in root.rglob("models.csv"):
  table=pd.read_csv(path,dtype={"date":str,"train_end":str})
  for row in table.itertuples():
   if position.get(row.train_end,10**9)>position.get(row.date,-1)-h-1:errors.append(f"标签隔离失败: {path} {row.date} train_end={row.train_end}")
 top_path=root/"top50_predictions.parquet"
 if top_path.exists():
  top=pd.read_parquet(top_path,columns=["trade_date","ts_code"]);counts=top.groupby("trade_date").size()
  if counts.min()!=50 or counts.max()!=50 or top.duplicated(["trade_date","ts_code"]).any():errors.append("最终Top50不是每天50只不同股票")
 else:errors.append("缺少top50_predictions.parquet")
 frozen_path=root/"FINAL_STRATEGY_CONFIG.json"
 if not frozen_path.exists():errors.append("缺少冻结配置")
 else:
  frozen=json.loads(frozen_path.read_text(encoding="utf-8"))
  if not frozen.get("frozen") or frozen.get("selection_data_end")!="20251231":errors.append("冻结配置状态或选择截止日错误")
  library_path=Path(frozen["library"]);library_path=library_path if library_path.is_absolute() else root/library_path
  library=json.loads(library_path.read_text(encoding="utf-8"))
  if library.get("options",{}).get("quick"):errors.append("冻结配置错误地引用了quick因子库")
 if errors:raise RuntimeError("\n".join(errors))
 print(f"[audit] PASS models={len(list(root.rglob('models.csv')))} final_days={len(counts)} top50_per_day=50")

if __name__=="__main__":main()
