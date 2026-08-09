"""五类滚动模型与无泄露Top-50回测。"""
from __future__ import annotations
import json,time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from .ga import Evaluator
from .metrics import evaluate_predictions

def build_factor_frame(data,library):
 d=data.sort_values(["ts_code","trade_date"]).reset_index(drop=True);ev=Evaluator(d);out=d[["ts_code","trade_date","future_ret_5d","target_rank","relevance","is_target_topk"]].copy()
 for item in library["factors"]:out[item["name"]]=item.get("direction",1)*ev.eval(item["expression"])
 factors=[x["name"] for x in library["factors"]];out=out.sort_values(["trade_date","ts_code"]).reset_index(drop=True);out[factors]=out[factors].groupby(out.trade_date).rank(pct=True).fillna(.5).astype("float32");return out,factors

def _weights(frame,dates,half_life):
 if not half_life:return None
 pos={d:i for i,d in enumerate(dates)};age=frame.trade_date.map(lambda d:len(dates)-1-pos[d]).to_numpy();return np.power(.5,age/half_life)

def _lgb(c,mode,iterations=None):
 p=c["lightgbm"];common=dict(n_estimators=iterations or p["n_estimators"],learning_rate=p["learning_rate"],max_depth=p["max_depth"],num_leaves=p["num_leaves"],min_child_samples=p["min_child_samples"],max_bin=p["max_bin"],reg_alpha=p["reg_alpha"],reg_lambda=p["reg_lambda"],feature_fraction=p["feature_fraction"],bagging_fraction=p["bagging_fraction"],bagging_freq=p["bagging_freq"],random_state=c["seed"],n_jobs=-1,verbosity=-1,deterministic=True,force_col_wise=True)
 if mode=="lambdarank":return lgb.LGBMRanker(objective="lambdarank",label_gain=[0,1,2,4,8],lambdarank_truncation_level=50,**common)
 if mode=="classifier":return lgb.LGBMClassifier(objective="binary",scale_pos_weight=50,**common)
 return lgb.LGBMRegressor(objective="regression_l2",**common)

def _bin_predict(train,test,factors,bins=10,shrink=500):
 pred=np.zeros(len(test))
 for f in factors:
  b=np.minimum((train[f]*bins).astype(int),bins-1);stats=train.assign(_b=b).groupby("_b").target_rank.agg(["mean","count"]);mapping=((stats["mean"]*stats["count"]+.5*shrink)/(stats["count"]+shrink)).to_dict();tb=np.minimum((test[f]*bins).astype(int),bins-1);pred+=tb.map(mapping).fillna(.5).to_numpy()-.5
 return pred/len(factors)+.5

def rolling_backtest(frame,factors,c,model,start,end,lookback=60,validation_days=10,update_days=5,half_life=10,bins=10,out_dir=None):
 t0=time.time();dates=np.array(sorted(frame.trade_date.unique()));eval_dates=dates[(dates>=start)&(dates<=end)];h=c["target"]["horizon"];preds=[];records=[];last_fitted=None
 for n in range(0,len(eval_dates),update_days):
  block=eval_dates[n:n+update_days];position=int(np.searchsorted(dates,block[0]));hist_end=position-h;hist_dates=dates[max(0,hist_end-lookback):hist_end]
  if len(hist_dates)<lookback:continue
  train=frame[frame.trade_date.isin(hist_dates)&frame.target_rank.notna()];test=frame[frame.trade_date.isin(block)].copy();sw=_weights(train,hist_dates,half_life)
  if model=="ic_vote":
   raw=[]
   for f in factors:
    by=train.groupby("trade_date").apply(lambda g:g[f].corr(g.target_rank),include_groups=False);age=np.arange(len(by)-1,-1,-1);w=np.ones(len(by)) if not half_life else np.power(.5,age/half_life);raw.append(np.nansum(by*w)/np.nansum(w))
   weights=np.array(raw);weights/=np.sum(np.abs(weights)) or 1;score=(test[factors].to_numpy()-.5)@weights;iteration=0
  elif model=="binning":score=_bin_predict(train,test,factors,bins);iteration=0
  else:
   fit_dates=hist_dates[:-validation_days];val_dates=hist_dates[-validation_days:];fit=train[train.trade_date.isin(fit_dates)];val=train[train.trade_date.isin(val_dates)];fitw=_weights(fit,fit_dates,half_life);m=_lgb(c,model);cb=[lgb.early_stopping(c["lightgbm"]["early_stopping_rounds"],verbose=False),lgb.log_evaluation(0)]
   if model=="lambdarank":m.fit(fit[factors],fit.relevance.astype(int),group=fit.groupby("trade_date").size().tolist(),sample_weight=fitw,eval_set=[(val[factors],val.relevance.astype(int))],eval_group=[val.groupby("trade_date").size().tolist()],eval_at=[50],callbacks=cb)
   elif model=="classifier":m.fit(fit[factors],fit.is_target_topk,sample_weight=fitw,eval_set=[(val[factors],val.is_target_topk)],eval_metric="binary_logloss",callbacks=cb)
   else:m.fit(fit[factors],fit.target_rank,sample_weight=fitw,eval_set=[(val[factors],val.target_rank)],eval_metric="l1",callbacks=cb)
   iteration=max(1,m.best_iteration_);final=_lgb(c,model,iteration)
   if model=="lambdarank":final.fit(train[factors],train.relevance.astype(int),group=train.groupby("trade_date").size().tolist(),sample_weight=sw,callbacks=[lgb.log_evaluation(0)])
   else:final.fit(train[factors],train.is_target_topk if model=="classifier" else train.target_rank,sample_weight=sw,callbacks=[lgb.log_evaluation(0)])
   score=final.predict_proba(test[factors])[:,1] if model=="classifier" else final.predict(test[factors]);last_fitted=final
  test["score"]=score;test["rebalance_date"]=block[0];preds.append(test[["ts_code","trade_date","rebalance_date","future_ret_5d","target_rank","relevance","is_target_topk","score"]]);records.append({"date":block[0],"train_start":hist_dates[0],"train_end":hist_dates[-1],"rows":len(train),"iteration":iteration})
  if n%max(update_days*20,1)==0:print(f"[model-v2] {model} date={block[0]} rows={len(train):,}")
 pred=pd.concat(preds,ignore_index=True);ranked,daily,monthly,summary=evaluate_predictions(pred,int(c["target"]["top_k"]));summary.update({"model":model,"lookback_days":lookback,"validation_days":validation_days,"update_days":update_days,"half_life":half_life,"seconds":time.time()-t0})
 if out_dir:
  out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);ranked.to_parquet(out/"predictions.parquet",index=False);ranked[ranked.is_pred_topk].to_parquet(out/"top50_predictions.parquet",index=False);daily.to_csv(out/"daily.csv",index=False,encoding="utf-8-sig");monthly.to_csv(out/"monthly.csv",encoding="utf-8-sig");(out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8");pd.DataFrame(records).to_csv(out/"models.csv",index=False)
  if last_fitted is not None:last_fitted.booster_.save_model(str(out/"latest_model.txt"))
 return summary
