"""分阶段实验、验证选择、配置冻结与2026确认测试。"""
from __future__ import annotations
import argparse,csv,gc,hashlib,json,shutil,sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import lightgbm as lgb
from .data import category_of,feature_names,load_config,load_frame
from .ga import GAOptions,run_ga
from .models import build_factor_frame,rolling_backtest

FACTOR_EXPERIMENTS={
 "F00":dict(objective="top3_ic",months="48"),"F01":dict(objective="top3_ic",months="all"),"F02":dict(objective="top3_ndcg",months="all"),"F03":dict(objective="contiguous_ndcg",months="all")}
OPERATOR_EXPERIMENTS={"O00":(),"O01":("delta_5","slope_5"),"O02":("mean_5","std_20","zscore_20","ts_rank_20"),"O03":("delta_5","slope_5","mean_5","std_20","zscore_20","ts_rank_20")}
GA_EXPERIMENTS={"G00":dict(selection_mode="elite_uniform",crossover_mode="insert_parent",mutation_mode="subtree",initialization="free"),"G01":dict(selection_mode="tournament",crossover_mode="symmetric_subtree",mutation_mode="mixed",initialization="free"),"G02":dict(selection_mode="tournament",crossover_mode="symmetric_subtree",mutation_mode="mixed",initialization="category_balanced")}
MODEL_EXPERIMENTS={"M00":("ic_vote",10),"M01-10":("binning",10),"M01-20":("binning",20),"M02":("regression",10),"M03":("classifier",10),"M04":("lambdarank",10)}
WINDOW_EXPERIMENTS={f"W{l}-{u}":(l,u,{30:5,60:10,120:20}[l]) for l in (30,60,120) for u in (5,1)}
DECAY_EXPERIMENTS={"D00":0.0,"D10":10.0,"D20":20.0}
ABLATION_EXPERIMENTS={"A00":None,"A01":"price_tech","A02":"moneyflow","A03":"valuation"}

def save_manifest(path,c,stage,experiment_id,summary):
 out=Path(path);out.mkdir(parents=True,exist_ok=True);raw=json.dumps(c,sort_keys=True,ensure_ascii=False).encode("utf-8");manifest={"stage":stage,"experiment_id":experiment_id,"seed":c["seed"],"config_sha256":hashlib.sha256(raw).hexdigest(),"summary":summary};(out/"config_snapshot.json").write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding="utf-8");(out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")

def write_matrix(root):
 rows=[]
 for x,v in FACTOR_EXPERIMENTS.items():rows.append(["factor",x,str(v),"因子目标/月覆盖"])
 for x,v in OPERATOR_EXPERIMENTS.items():rows.append(["operators",x,str(v),"时间算子"])
 for x,v in GA_EXPERIMENTS.items():rows.append(["ga",x,str(v),"遗传方式"])
 for x,v in MODEL_EXPERIMENTS.items():rows.append(["models",x,str(v),"滚动模型"])
 for x,v in WINDOW_EXPERIMENTS.items():rows.append(["windows",x,str(v),"训练/更新窗口"])
 for x,v in DECAY_EXPERIMENTS.items():rows.append(["decay",x,str(v),"近期衰减"])
 for x,v in ABLATION_EXPERIMENTS.items():rows.append(["ablation",x,str(v),"因子类别消融"])
 pd.DataFrame(rows,columns=["stage","experiment_id","setting","purpose"]).to_csv(root/"EXPERIMENT_MATRIX.csv",index=False,encoding="utf-8-sig")

def choose(results,c,baseline_ndcg=None):
 ok=[r for r in results if r.get("top50_excess_5d_2024",-1)>0 and r.get("top50_excess_5d_2025",-1)>0 and r.get("monthly_win_rate",0)>=c["selection"]["minimum_monthly_win_rate"] and r.get("ndcg_at_50",-1)>r.get("random_ndcg_at_50",0) and (baseline_ndcg is None or r.get("ndcg_at_50",-1)>baseline_ndcg)]
 diagnostic=[r for r in results if r.get("ndcg_at_50",-1)>r.get("random_ndcg_at_50",0) and (baseline_ndcg is None or r.get("ndcg_at_50",-1)>baseline_ndcg)];pool=ok or diagnostic or results;return sorted(pool,key=lambda r:(r.get("monthly_median_excess",-9),r.get("ndcg_at_50",-9),-r.get("turnover",9)),reverse=True)[0],bool(ok)

def winner_file(root):return root/"stage_winners.json"
def winners(root):return json.loads(winner_file(root).read_text(encoding="utf-8")) if winner_file(root).exists() else {}
def save_winner(root,stage,winner,eligible):
 w=winners(root);w[stage]={"experiment_id":winner["experiment_id"],"eligible_pool":eligible,"summary":winner};winner_file(root).write_text(json.dumps(w,ensure_ascii=False,indent=2),encoding="utf-8")

def inherited_search_options(root):
 w=winners(root);factor_id=w.get("factor",{}).get("experiment_id","F03");operator_id=w.get("operators",{}).get("experiment_id","O03")
 return FACTOR_EXPERIMENTS[factor_id],OPERATOR_EXPERIMENTS[operator_id]

def library_path(root,allow_quick=False):
 w=winners(root)
 for stage in ("ga","operators","factor"):
  if stage in w:
   path=root/stage/w[stage]["experiment_id"]/"factor_library.json";payload=json.loads(path.read_text(encoding="utf-8"))
   if payload.get("options",{}).get("quick") and not allow_quick:raise RuntimeError("当前赢家来自--quick冒烟搜索，不能进入正式模型或冻结流程")
   return path
 raise RuntimeError("尚无因子库，请先运行 factor 阶段")

def evaluate_library(lib_path,c,exp_dir,model="lambdarank",lookback=60,update=5,valdays=10,half=10,bins=10,quick=False):
 data=load_frame(c,"20240601" if quick else "20230601",c["split"]["validation_end"]);lib=json.loads(Path(lib_path).read_text(encoding="utf-8"));frame,factors=build_factor_frame(data,lib)
 return rolling_backtest(frame,factors,c,model,"20250101" if quick else c["split"]["validation_start"],c["split"]["validation_end"],lookback,valdays,20 if quick else update,half,bins,exp_dir/"validation")

def run_ga_stage(stage,defs,c,root,quick,base=None):
 discovery=load_frame(c,c["split"]["discovery_start"],c["split"]["discovery_end"]);terms=feature_names(c);generated=[]
 inherited_factor,inherited_temporal=inherited_search_options(root)
 for eid,setting in defs.items():
  kwargs=dict(experiment_id=eid,objective=inherited_factor["objective"],months=inherited_factor["months"],quick=quick)
  if stage=="factor":kwargs.update(setting)
  elif stage=="operators":kwargs["temporal"]=setting
  else:kwargs.update(setting);kwargs["temporal"]=OPERATOR_EXPERIMENTS[base] if base else inherited_temporal
  opt=GAOptions(**kwargs);edir=root/stage/eid;reuse=None
  if stage=="operators" and eid=="O00":
   fw=winners(root).get("factor",{}).get("experiment_id");reuse=root/"factor"/fw/"factor_library.json" if fw else None
  elif stage=="ga" and eid=="G00":
   ow=winners(root).get("operators",{}).get("experiment_id");reuse=root/"operators"/ow/"factor_library.json" if ow else None
  if reuse and reuse.exists():
   source=json.loads(reuse.read_text(encoding="utf-8"))
   if source.get("options",{}).get("quick")==quick:
    edir.mkdir(parents=True,exist_ok=True);payload=dict(source);payload["options"]=asdict(opt);payload["reused_from"]=str(reuse.resolve());(edir/"factor_library.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");generated.append((eid,edir,payload));continue
  payload,ctx=run_ga(discovery,terms,c,opt,edir);del ctx;generated.append((eid,edir,payload))
 del discovery;gc.collect();results=[]
 for eid,edir,payload in generated:
  summary=evaluate_library(edir/"factor_library.json",c,edir,quick=quick);summary.update({"experiment_id":eid,"quick":quick,"search_seconds":payload["seconds"],"unique_expressions":payload["unique_expressions"]});results.append(summary)
  save_manifest(edir,c,stage,eid,summary)
 pd.DataFrame(results).to_csv(root/stage/"results.csv",index=False,encoding="utf-8-sig");win,eligible=choose(results,c);save_winner(root,stage,win,eligible);return win

def run_model_stage(c,root,quick=False):
 lib=library_path(root,allow_quick=quick);data=load_frame(c,"20240601" if quick else "20230601",c["split"]["validation_end"]);library=json.loads(lib.read_text(encoding="utf-8"));frame,factors=build_factor_frame(data,library);results=[]
 for eid,(model,bins) in MODEL_EXPERIMENTS.items():
  s=rolling_backtest(frame,factors,c,model,"20250101" if quick else c["split"]["validation_start"],c["split"]["validation_end"],60,10,20 if quick else 5,10,bins,root/"models"/eid);s.update({"experiment_id":eid,"bins":bins});results.append(s);save_manifest(root/"models"/eid,c,"models",eid,s)
 pd.DataFrame(results).to_csv(root/"models"/"results.csv",index=False,encoding="utf-8-sig");baseline=next((r["ndcg_at_50"] for r in results if r["experiment_id"]=="M00"),None);win,eligible=choose(results,c,baseline);save_winner(root,"models",win,eligible);return win

def run_windows(c,root):
 lib=library_path(root);data=load_frame(c,"20230601",c["split"]["validation_end"]);frame,factors=build_factor_frame(data,json.loads(lib.read_text(encoding="utf-8")));model_winner=winners(root).get("models",{});mw=model_winner.get("summary",{});model=mw.get("model","lambdarank");bins=mw.get("bins",20 if model_winner.get("experiment_id")=="M01-20" else 10);results=[]
 for eid,(look,update,val) in WINDOW_EXPERIMENTS.items():
  s=rolling_backtest(frame,factors,c,model,c["split"]["validation_start"],c["split"]["validation_end"],look,val,update,10,bins,root/"windows"/eid);s.update({"experiment_id":eid,"bins":bins});results.append(s);save_manifest(root/"windows"/eid,c,"windows",eid,s)
 win,_=choose(results,c);decays=[]
 for eid,half in DECAY_EXPERIMENTS.items():
  s=rolling_backtest(frame,factors,c,model,c["split"]["validation_start"],c["split"]["validation_end"],win["lookback_days"],{30:5,60:10,120:20}[win["lookback_days"]],win["update_days"],half,bins,root/"decay"/eid);s.update({"experiment_id":eid,"bins":bins});decays.append(s);save_manifest(root/"decay"/eid,c,"decay",eid,s)
 final,eligible=choose(decays,c);pd.DataFrame(results).to_csv(root/"windows"/"results.csv",index=False,encoding="utf-8-sig");pd.DataFrame(decays).to_csv(root/"decay"/"results.csv",index=False,encoding="utf-8-sig");save_winner(root,"windows",final,eligible)
 ablations=[]
 for eid,removed in ABLATION_EXPERIMENTS.items():
  subset=factors if removed is None else [item["name"] for item in json.loads(lib.read_text(encoding="utf-8"))["factors"] if item.get("category")!=removed]
  if eid=="A00":s=dict(final)
  else:s=rolling_backtest(frame,subset,c,model,c["split"]["validation_start"],c["split"]["validation_end"],final["lookback_days"],{30:5,60:10,120:20}[final["lookback_days"]],final["update_days"],final["half_life"],bins,root/"ablations"/eid)
  s.update({"experiment_id":eid,"removed_category":removed,"factor_count":len(subset)});ablations.append(s);save_manifest(root/"ablations"/eid,c,"ablations",eid,s)
 pd.DataFrame(ablations).to_csv(root/"ablations"/"results.csv",index=False,encoding="utf-8-sig")
 if not eligible:
  write_validation_report(Path(c["paths"]["reports"]),final);raise RuntimeError("没有方案同时通过2024、2025、月度胜率和NDCG门槛，拒绝冻结配置")
 frozen={"frozen":True,"selection_data_end":c["split"]["validation_end"],"library":str(lib.relative_to(root)).replace("\\","/"),"model":model,"bins":bins,"lookback_days":final["lookback_days"],"validation_days":{30:5,60:10,120:20}[final["lookback_days"]],"update_days":final["update_days"],"half_life":final["half_life"],"validation_summary":final};(root/"FINAL_STRATEGY_CONFIG.json").write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding="utf-8");write_validation_report(Path(c["paths"]["reports"]),final);return final

def write_validation_report(root,summary):
 root.mkdir(parents=True,exist_ok=True)
 lines=["# Top-50 V2 验证报告","","> 参数选择仅使用2024-2025数据；2026未参与本报告中的选择。","","## 冻结方案","",f"- 模型：`{summary.get('model')}`",f"- 训练窗口：{summary.get('lookback_days')}个交易日",f"- 更新间隔：{summary.get('update_days')}个交易日",f"- 时间衰减半衰期：{summary.get('half_life')}个交易日","","## 核心指标","", "| 指标 | 数值 |","|---|---:|"]
 for key in ("top50_return_5d","top50_excess_5d","top50_mean_percentile","precision_at_50","ndcg_at_50","monthly_win_rate","monthly_median_excess","worst_month_excess","turnover","rank_ic"):
  if key in summary:lines.append(f"| `{key}` | {summary[key]:.6f} |")
 lines.extend(["","## 分年表现","","| 年份 | Top50收益 | Top50超额 | NDCG@50 | Precision@50 | Rank IC | 换手率 |","|---|---:|---:|---:|---:|---:|---:|"])
 for year in ("2024","2025"):lines.append(f"| {year} | {summary.get('top50_return_5d_'+year,float('nan')):.6f} | {summary.get('top50_excess_5d_'+year,float('nan')):.6f} | {summary.get('ndcg_at_50_'+year,float('nan')):.6f} | {summary.get('precision_at_50_'+year,float('nan')):.6f} | {summary.get('rank_ic_'+year,float('nan')):.6f} | {summary.get('turnover_'+year,float('nan')):.6f} |")
 (root/"VALIDATION_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def final_test(c,root):
 p=root/"FINAL_STRATEGY_CONFIG.json"
 if not p.exists():raise RuntimeError("最终配置尚未冻结，拒绝运行2026测试")
 f=json.loads(p.read_text(encoding="utf-8"))
 if not f.get("frozen") or f.get("selection_data_end")!=c["split"]["validation_end"]:raise RuntimeError("冻结文件无效，拒绝运行2026测试")
 # 冻结配置使用相对 artifacts/ 的路径，整个项目移动后仍可直接运行。
 lib_path=Path(f["library"]);lib_path=lib_path if lib_path.is_absolute() else root/lib_path
 frozen_before=p.read_bytes();data=load_frame(c,"20250601",c["split"]["test_end"]);lib=json.loads(lib_path.read_text(encoding="utf-8"));frame,factors=build_factor_frame(data,lib);test=rolling_backtest(frame,factors,c,f["model"],c["split"]["test_start"],c["split"]["test_end"],f["lookback_days"],f["validation_days"],f["update_days"],f["half_life"],f.get("bins",10),root/"final_test");val=f["validation_summary"];numeric=(int,float);gap={k:{"validation":val.get(k),"test":test.get(k),"absolute_gap":test.get(k)-val.get(k) if isinstance(test.get(k),numeric) and isinstance(val.get(k),numeric) else None,"retention":test.get(k)/val.get(k) if isinstance(test.get(k),numeric) and isinstance(val.get(k),numeric) and val.get(k) else None} for k in ("top50_excess_5d","ndcg_at_50","precision_at_50","monthly_win_rate","turnover")}
 if p.read_bytes()!=frozen_before:raise RuntimeError("确认测试期间冻结配置发生变化")
 (root/"TEST_2026_REPORT.json").write_text(json.dumps(test,ensure_ascii=False,indent=2),encoding="utf-8");(root/"VALIDATION_TEST_GAP.json").write_text(json.dumps(gap,ensure_ascii=False,indent=2),encoding="utf-8");shutil.copy2(root/"final_test"/"top50_predictions.parquet",root/"top50_predictions.parquet");write_test_reports(root,Path(c["paths"]["reports"]),test,gap);return test

def write_test_reports(root,report_root,test,gap):
 report_root.mkdir(parents=True,exist_ok=True)
 test_lines=["# Top-50 V2 2026确认测试","","> 此结果来自冻结配置，未用于调参。","","| 指标 | 2026 |","|---|---:|"]
 for key,value in test.items():
  if isinstance(value,(int,float)):test_lines.append(f"| `{key}` | {value:.6f} |")
 monthly=pd.read_csv(root/"final_test"/"monthly.csv");test_lines.extend(["","## 逐月表现","","| 月份 | Top50收益 | Top50超额 | NDCG@50 | Precision@50 | Rank IC | 换手率 |","|---|---:|---:|---:|---:|---:|---:|"])
 for _,x in monthly.iterrows():test_lines.append(f"| {int(x['month'])} | {x['top50_return_5d']:.6f} | {x['top50_excess_5d']:.6f} | {x['ndcg_at_50']:.6f} | {x['precision_at_50']:.6f} | {x['rank_ic']:.6f} | {x['turnover']:.6f} |")
 frozen=json.loads((root/"FINAL_STRATEGY_CONFIG.json").read_text(encoding="utf-8"));library_path=Path(frozen["library"]);library_path=library_path if library_path.is_absolute() else root/library_path;library=json.loads(library_path.read_text(encoding="utf-8"));counts=pd.Series([x.get("category","other") for x in library["factors"]]).value_counts();test_lines.extend(["","## 因子库类别","",*([f"- `{name}`：{count}个" for name,count in counts.items()])])
 validation_model=root/"decay"/frozen["validation_summary"]["experiment_id"]/"latest_model.txt";test_model=root/"final_test"/"latest_model.txt"
 if validation_model.exists() and test_model.exists():
  vb=lgb.Booster(model_file=str(validation_model));tb=lgb.Booster(model_file=str(test_model));vi=pd.Series(vb.feature_importance("gain"),index=vb.feature_name());ti=pd.Series(tb.feature_importance("gain"),index=tb.feature_name());vi=vi/(vi.sum() or 1);ti=ti/(ti.sum() or 1);imp=pd.DataFrame({"validation":vi,"test":ti}).fillna(0);imp["change"]=imp.test-imp.validation;imp=imp.reindex(imp.test.sort_values(ascending=False).head(10).index);test_lines.extend(["","## 最近模型特征重要性变化","","| 因子 | 验证末期 | 测试末期 | 变化 |","|---|---:|---:|---:|"])
  for name,x in imp.iterrows():test_lines.append(f"| `{name}` | {x.validation:.6f} | {x.test:.6f} | {x.change:+.6f} |")
 (report_root/"TEST_2026_REPORT.md").write_text("\n".join(test_lines)+"\n",encoding="utf-8")
 gap_lines=["# 验证与测试差异","","| 指标 | 验证 | 测试 | 绝对差 | 保持率 |","|---|---:|---:|---:|---:|"]
 for key,x in gap.items():gap_lines.append(f"| `{key}` | {x['validation']:.6f} | {x['test']:.6f} | {x['absolute_gap']:.6f} | {x['retention']:.6f} |")
 (report_root/"VALIDATION_TEST_GAP.md").write_text("\n".join(gap_lines)+"\n",encoding="utf-8")

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/stock/top50.json");ap.add_argument("--stage",choices=("factor","operators","ga","models","windows","final-test","all-validation"),required=True);ap.add_argument("--quick",action="store_true");a=ap.parse_args();c=load_config(a.config);root=Path(c["paths"]["artifacts"]);write_matrix(root)
 if a.quick and a.stage in ("windows","all-validation","final-test"):raise RuntimeError("快速冒烟模式不得冻结配置或运行2026确认测试")
 if a.stage in ("factor","all-validation"):run_ga_stage("factor",FACTOR_EXPERIMENTS,c,root,a.quick)
 if a.stage in ("operators","all-validation"):run_ga_stage("operators",OPERATOR_EXPERIMENTS,c,root,a.quick)
 if a.stage in ("ga","all-validation"):run_ga_stage("ga",GA_EXPERIMENTS,c,root,a.quick)
 if a.stage in ("models","all-validation"):run_model_stage(c,root,a.quick)
 if a.stage in ("windows","all-validation"):run_windows(c,root)
 if a.stage=="final-test":final_test(c,root)
if __name__=="__main__":main()
