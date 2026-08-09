"""严格Top-K与横截面评估指标。"""
from __future__ import annotations
import numpy as np
import pandas as pd

def _ndcg(group,k):
 g=group.sort_values(["score","ts_code"],ascending=[False,True]).head(k); ideal=group.nlargest(k,"relevance")
 w=1/np.log2(np.arange(2,len(g)+2)); dcg=np.sum((np.power(2,g.relevance.to_numpy())-1)*w); iw=1/np.log2(np.arange(2,len(ideal)+2)); ideal_dcg=np.sum((np.power(2,ideal.relevance.to_numpy())-1)*iw)
 return float(dcg/ideal_dcg) if ideal_dcg>0 else np.nan

def _rank_corr(a,b):
 x=a.rank().to_numpy();y=b.rank().to_numpy()
 return float(np.corrcoef(x,y)[0,1]) if np.nanstd(x)>1e-12 and np.nanstd(y)>1e-12 else np.nan

def evaluate_predictions(frame,k=50):
 x=frame.dropna(subset=["score","future_ret_5d"]).sort_values(["trade_date","ts_code"]).copy(); x["pred_position"]=x.groupby("trade_date").score.rank(method="first",ascending=False); x["is_pred_topk"]=x.pred_position<=k
 rows=[]; previous=None
 for date,g in x.groupby("trade_date",sort=False):
  top=g[g.is_pred_topk]; universe=g.future_ret_5d.mean(); names=set(top.ts_code); turnover=np.nan if previous is None else 1-len(names&previous)/k; previous=names
  gains=np.power(2,g.relevance.to_numpy())-1;ideal=np.sort(gains)[::-1][:k];w=1/np.log2(np.arange(2,min(k,len(g))+2));random_dcg=gains.mean()*w.sum();ideal_dcg=np.sum(ideal*w)
  rows.append({"trade_date":date,"top50_return_5d":top.future_ret_5d.mean(),"universe_return_5d":universe,"top50_excess_5d":top.future_ret_5d.mean()-universe,"top50_mean_percentile":top.target_rank.mean(),"precision_at_50":top.is_target_topk.mean(),"random_precision_at_50":min(k,len(g))/len(g),"ndcg_at_50":_ndcg(g,k),"random_ndcg_at_50":random_dcg/ideal_dcg if ideal_dcg>0 else np.nan,"turnover":turnover,"rank_ic":_rank_corr(g.score,g.future_ret_5d)})
 daily=pd.DataFrame(rows); daily["month"]=daily.trade_date.str[:6]; daily["year"]=daily.trade_date.str[:4]
 monthly=daily.groupby("month").agg(top50_excess_5d=("top50_excess_5d","mean"),top50_return_5d=("top50_return_5d","mean"),ndcg_at_50=("ndcg_at_50","mean"),precision_at_50=("precision_at_50","mean"),rank_ic=("rank_ic","mean"),turnover=("turnover","mean"))
 summary={"days":len(daily),"top50_return_5d":daily.top50_return_5d.mean(),"top50_excess_5d":daily.top50_excess_5d.mean(),"top50_mean_percentile":daily.top50_mean_percentile.mean(),"precision_at_50":daily.precision_at_50.mean(),"random_precision_at_50":daily.random_precision_at_50.mean(),"ndcg_at_50":daily.ndcg_at_50.mean(),"random_ndcg_at_50":daily.random_ndcg_at_50.mean(),"monthly_win_rate":float((monthly.top50_excess_5d>0).mean()),"monthly_median_excess":monthly.top50_excess_5d.median(),"worst_month_excess":monthly.top50_excess_5d.min(),"turnover":daily.turnover.mean(),"rank_ic":daily.rank_ic.mean()}
 for year,g in daily.groupby("year"):
  for key in ("top50_return_5d","top50_excess_5d","top50_mean_percentile","precision_at_50","ndcg_at_50","turnover","rank_ic"):summary[f"{key}_{year}"]=g[key].mean()
 return x,daily,monthly,summary
