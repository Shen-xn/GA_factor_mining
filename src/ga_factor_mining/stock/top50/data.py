"""V2 数据协议、严格时间切分和Top-50标签。"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

EXCLUDED = {"is_basic_missing","is_moneyflow_missing","is_tech_warmup","is_recent_listing","valid_feature_ratio","valuation_missing_count","is_loss_or_pe_missing"}
CATEGORIES = {
 "price_tech": {"ret_1d","open_gap","intraday_ret","high_low_range","close_position","upper_shadow","lower_shadow","ret_5d","ret_20d","ret_1d_rank_cs","ret_5d_rank_cs","ret_20d_rank_cs","range_rank_cs","close_position_rank_cs","macd_dif","macd_dea","macd","kdj_k","kdj_j","rsi_6","rsi_24","cci","boll_width","boll_position","macd_rank_cs","rsi_6_rank_cs"},
 "liquidity": {"turnover_rate","turnover_rate_f","volume_ratio","amount_z_20d","amount_z_60d","turnover_z_20d","vol_z_20d","amount_rank_cs","vol_rank_cs","turnover_rank_cs","volume_ratio_rank_cs","liquidity_rank_cs"},
 "valuation": {"total_mv_rank_cs","circ_mv_rank_cs","pe_ttm_rank_cs","pb_rank_cs","ps_ttm_rank_cs","dv_ttm_rank_cs"},
 "moneyflow": {"net_mf_amount_ratio","net_mf_vol_ratio","sm_net_amount_ratio","md_net_amount_ratio","lg_net_amount_ratio","elg_net_amount_ratio","lg_buy_amount_ratio","elg_buy_amount_ratio","lg_sell_amount_ratio","elg_sell_amount_ratio","net_mf_amount_ratio_5d","elg_net_amount_ratio_5d","lg_net_amount_ratio_5d","net_mf_rank_cs","elg_net_rank_cs","main_force_rank_cs"}
}

def load_config(path="config.json"):
 p=Path(path).resolve(); c=json.loads(p.read_text(encoding="utf-8")); base=p.parent
 for k in ("prepared_data","feature_meta","artifacts"): c["paths"][k]=str((base/c["paths"][k]).resolve())
 Path(c["paths"]["artifacts"]).mkdir(parents=True,exist_ok=True); return c

def feature_names(c):
 m=json.loads(Path(c["paths"]["feature_meta"]).read_text(encoding="utf-8")); return [x for x in m["feature_columns"] if x not in EXCLUDED]

def load_frame(c, start=None, end=None, columns=None):
 cols=["ts_code","trade_date",c["target"]["column"],*(columns or feature_names(c))]
 filters=[]
 if start:filters.append(("trade_date",">=",start))
 if end:filters.append(("trade_date","<=",end))
 df=pd.read_parquet(c["paths"]["prepared_data"],columns=list(dict.fromkeys(cols)),filters=filters or None); df["trade_date"]=df.trade_date.astype(str)
 return add_labels(df,c)

def add_labels(df,c):
 y=c["target"]["column"]; df=df.copy(); df["target_rank"]=df.groupby("trade_date",sort=False)[y].rank(pct=True)
 edges=c["target"]["relevance_edges"]; r=np.zeros(len(df),dtype=np.int8)
 for edge in edges: r+=(df.target_rank.to_numpy()>=edge)
 r[df.target_rank.isna().to_numpy()]=0; df["relevance"]=r
 df=df.sort_values(["trade_date","ts_code"]); k=int(c["target"]["top_k"])
 df["target_position"]=df.groupby("trade_date",sort=False)[y].rank(method="first",ascending=False); df["is_target_topk"]=(df.target_position<=k).astype(np.int8)
 return df.reset_index(drop=True)

def stable_hash(code,seed=42): return int.from_bytes(hashlib.blake2b(f"{seed}:{code}".encode(),digest_size=8).digest(),"little")

def category_of(feature):
 for name,items in CATEGORIES.items():
  if feature in items:return name
 return "other"
