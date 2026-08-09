"""Top-50目标、时间终端和可配置遗传编程。"""
from __future__ import annotations
import copy,json,random,time
from collections import OrderedDict
from dataclasses import dataclass,asdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from data import CATEGORIES,category_of,stable_hash

UNARY=("neg","abs","signed_log","signed_sqrt")
BINARY=("add","sub","mul","div","min","max")
TEMPORAL=("delta_5","slope_5","mean_5","std_20","zscore_20","ts_rank_20")

def canonical(x):return json.dumps(x,separators=(",",":"),ensure_ascii=False)
def depth(x):return 0 if isinstance(x,str) else 1+max(depth(v) for v in x[1:])
def nodes(x):return 1 if isinstance(x,str) else 1+sum(nodes(v) for v in x[1:])
def text(x):
 if isinstance(x,str):return x
 return f"{x[0]}({','.join(text(v) for v in x[1:])})"
def expression_category(e):
 terms=[]
 def visit(node):
  if isinstance(node,str):terms.append(category_of(node))
  else:
   for child in node[1:]:visit(child)
 visit(e);counts={x:terms.count(x) for x in set(terms)}
 return max(counts,key=counts.get) if counts else "other"

class Evaluator:
 def __init__(self,frame,max_abs=1e6,temporal_cache_size=16):self.df=frame;self.max_abs=max_abs;self.temp=OrderedDict();self.temporal_cache_size=temporal_cache_size
 def temporal(self,op,col):
  key=(op,col)
  if key in self.temp:self.temp.move_to_end(key);return self.temp[key]
  g=self.df.groupby("ts_code",sort=False)[col]; x=self.df[col].astype(float)
  if op=="delta_5": out=x-g.shift(5)
  elif op=="slope_5": out=(-2*g.shift(4)-g.shift(3)+g.shift(1)+2*x)/10
  elif op=="mean_5": out=g.transform(lambda s:s.rolling(5,min_periods=3).mean())
  elif op=="std_20": out=g.transform(lambda s:s.rolling(20,min_periods=5).std())
  elif op=="zscore_20":
   m=g.transform(lambda s:s.rolling(20,min_periods=5).mean()); sd=g.transform(lambda s:s.rolling(20,min_periods=5).std()); out=(x-m)/sd.replace(0,np.nan)
  elif op=="ts_rank_20": out=g.transform(lambda s:s.rolling(20,min_periods=5).rank(pct=True))
  self.temp[key]=out.replace([np.inf,-np.inf],np.nan).astype("float32");self.temp.move_to_end(key)
  while len(self.temp)>self.temporal_cache_size:self.temp.popitem(last=False)
  return self.temp[key]
 def eval(self,e):
  if isinstance(e,str):return self.df[e].astype(float)
  op=e[0]
  if op in TEMPORAL:return self.temporal(op,e[1])
  a=self.eval(e[1])
  if op=="neg":o=-a
  elif op=="abs":o=a.abs()
  elif op=="signed_log":o=np.sign(a)*np.log1p(a.abs())
  elif op=="signed_sqrt":o=np.sign(a)*np.sqrt(a.abs())
  else:
   b=self.eval(e[2]);o={"add":lambda:a+b,"sub":lambda:a-b,"mul":lambda:a*b,"div":lambda:a/b.where(b.abs()>1e-8),"min":lambda:np.minimum(a,b),"max":lambda:np.maximum(a,b)}[op]()
  return pd.Series(np.clip(o,-self.max_abs,self.max_abs),index=self.df.index).replace([np.inf,-np.inf],np.nan)

def prepare_search_context(discovery,c,months="all"):
 d=discovery.copy(deep=False);hash_map={x:stable_hash(x,c["seed"]) for x in d.ts_code.unique()};hashes=d.ts_code.map(hash_map);limit=int(c["ga"]["sample_stocks_per_day"]);search=hashes.groupby(d.trade_date).rank(method="first")<=limit
 if months!="all":
  rng=np.random.default_rng(c["seed"]);available=sorted(d.trade_date.str[:6].unique());chosen=set(rng.choice(available,min(int(months),len(available)),replace=False));search&=d.trade_date.str[:6].isin(chosen)
 d["_search"]=search.to_numpy()
 return d

class Scorer:
 def __init__(self,frame,mask,k=50,min_days=12):
  self.f=frame.loc[mask,["ts_code","trade_date","relevance","target_rank"]].sort_values(["trade_date","ts_code"]);self.k=k;self.min_days=min_days;self.groups=[g.index.to_numpy() for _,g in self.f.groupby("trade_date",sort=False)];self.dates=[d for d,_ in self.f.groupby("trade_date",sort=False)]
  self.rel=self.f.relevance.to_numpy(); self.target=self.f.target_rank.to_numpy(); self.local=[np.arange(len(g)) for g in self.groups]
 def _daily(self,v,objective):
  vals=pd.Series(v,index=self.f.index).to_numpy(); pos=[];neg=[]
  offset=0
  for idx in self.groups:
   n=len(idx); a=vals[offset:offset+n]; rel=self.rel[offset:offset+n]; y=self.target[offset:offset+n]; offset+=n; valid=np.isfinite(a)&np.isfinite(y)
   if valid.sum()<max(100,self.k):pos.append(np.nan);neg.append(np.nan);continue
   av=a[valid];rv=rel[valid];yv=y[valid]; kk=min(self.k,len(av)); hi=np.argpartition(av,-kk)[-kk:];lo=np.argpartition(av,kk-1)[:kk]
   if np.nanstd(av)<1e-12:
    pos.append(np.nan);neg.append(np.nan);continue
   if objective=="ic":
    rr=rankdata(av);yy=rankdata(yv)
    if np.nanstd(rr)<1e-12 or np.nanstd(yy)<1e-12:pos.append(np.nan);neg.append(np.nan);continue
    corr=np.corrcoef(rr,yy)[0,1];pos.append(corr);neg.append(-corr)
   else:
    ideal=np.sort(rv)[::-1][:kk];w=1/np.log2(np.arange(2,kk+2));den=np.sum((2**ideal-1)*w)
    def nd(sel,ascending=False):
     order=sel[np.argsort(av[sel]) if ascending else np.argsort(av[sel])[::-1]]
     return np.sum((2**rv[order]-1)*w)/den if den>0 else np.nan
    pos.append(nd(hi)); neg.append(nd(lo,ascending=True))
  return pd.DataFrame({"date":self.dates,"pos":pos,"neg":neg})
 def score(self,v,objective="top3_ndcg"):
  base="ic" if objective.endswith("ic") else "ndcg"; daily=self._daily(v,base);daily["month"]=daily.date.str[:6]
  monthly=daily.groupby("month").agg(pos=("pos","mean"),neg=("neg","mean"),count=("pos","count"));monthly=monthly[monthly["count"]>=self.min_days];monthly["score"]=monthly[["pos","neg"]].max(axis=1);monthly["direction"]=np.where(monthly.pos>=monthly.neg,1,-1)
  if monthly.empty:return -1e9,"",0,1,monthly
  if objective.startswith("contiguous"):
   rolling=monthly.score.rolling(2).mean();raw=float(rolling.max());peak=str(rolling.idxmax());end=monthly.index.get_loc(peak);selected_months=monthly.iloc[max(0,end-1):end+1]
  else:
   top=monthly.score.nlargest(3);raw=float(top.mean());peak=str(top.index[0]);selected_months=monthly.loc[top.index]
  direction=1 if selected_months.pos.mean()>=selected_months.neg.mean() else -1
  chosen=np.where(daily.pos>=daily.neg,daily.pos,daily.neg);return raw,peak,float(np.nanstd(chosen)),direction,monthly

def random_expr(terminals,max_depth,rng,temporal=(),d=0):
 if d>=max_depth or (d>0 and rng.random()<.30):return rng.choice(terminals)
 if temporal and rng.random()<.18:return [rng.choice(temporal),rng.choice(terminals)]
 if rng.random()<.35:return [rng.choice(UNARY),random_expr(terminals,max_depth,rng,temporal,d+1)]
 return [rng.choice(BINARY),random_expr(terminals,max_depth,rng,temporal,d+1),random_expr(terminals,max_depth,rng,temporal,d+1)]

def paths(e,p=()):
 out=[p]
 if not isinstance(e,str):
  for i in range(1,len(e)):out+=paths(e[i],p+(i,))
 return out
def valid_expr(e):
 if isinstance(e,str):return True
 if e[0] in TEMPORAL:return len(e)==2 and isinstance(e[1],str)
 return all(valid_expr(x) for x in e[1:])
def get(e,p):
 for i in p:e=e[i]
 return e
def put(e,p,v):
 if not p:return copy.deepcopy(v)
 o=copy.deepcopy(e);cur=o
 for i in p[:-1]:cur=cur[i]
 cur[p[-1]]=copy.deepcopy(v);return o
def crossover(a,b,rng,mode):
 if mode=="symmetric_subtree":return put(a,rng.choice(paths(a)),get(b,rng.choice(paths(b))))
 if isinstance(a,str) or rng.random()<.30:return copy.deepcopy(b)
 o=copy.deepcopy(a);i=rng.randrange(1,len(o));o[i]=crossover(o[i],b,rng,mode);return o
def mutate(e,terminals,max_depth,rng,mode,temporal):
 if mode=="mixed":
  kind=rng.choice(("terminal","operator","subtree"));p=rng.choice(paths(e));node=get(e,p)
  if kind=="terminal":return put(e,p,rng.choice(terminals))
  if kind=="operator" and not isinstance(node,str):
   choices=UNARY if len(node)==2 else BINARY; n=copy.deepcopy(node);n[0]=rng.choice(choices);return put(e,p,n)
  return put(e,p,random_expr(terminals,max_depth,rng,temporal))
 if isinstance(e,str) or rng.random()<.25:return random_expr(terminals,max_depth,rng,temporal)
 o=copy.deepcopy(e);i=rng.randrange(1,len(o));o[i]=mutate(o[i],terminals,max_depth-1 if max_depth>1 else 1,rng,mode,temporal);return o

@dataclass
class GAOptions:
 experiment_id:str; objective:str="top3_ndcg"; months:str="all"; temporal:tuple=(); selection_mode:str="elite_uniform"; crossover_mode:str="insert_parent"; mutation_mode:str="subtree"; initialization:str="free"; quick:bool=False

def run_ga(discovery,terminals,c,opt,out_dir):
 start=time.time();ctx=prepare_search_context(discovery,c,opt.months);ev=Evaluator(ctx);sc=Scorer(ctx,ctx._search,int(c["target"]["top_k"]),int(c["ga"]["min_month_days"]));rng=random.Random(c["seed"]);g=c["ga"];pop=18 if opt.quick else g["population_size"];gens=3 if opt.quick else g["generations"];elite=min(6 if opt.quick else g["elite_size"],pop)
 if opt.initialization=="category_balanced":
  population=[]
  for cat in CATEGORIES:
   ts=[x for x in terminals if category_of(x)==cat]
   population += [random_expr(ts,g["max_depth"],rng,opt.temporal) for _ in range(pop//4)]
  while len(population)<pop:population.append(random_expr(terminals,g["max_depth"],rng,opt.temporal))
 else:population=[random_expr(terminals,g["max_depth"],rng,opt.temporal) for _ in range(pop)]
 cache={};hall={}
 for gen in range(gens):
  scored=[]
  for e in population:
   key=canonical(e)
   if key not in cache:
    val=ev.eval(e);v=val.loc[sc.f.index].to_numpy();raw,peak,vol,direction,_=sc.score(v,opt.objective);score=raw-g["daily_volatility_penalty"]*vol-g["complexity_penalty"]*nodes(e);cache[key]=(score,peak,raw,vol,direction)
   fit=cache[key];scored.append((fit[0],e,fit));hall[key]=(fit[0],e,fit)
  scored.sort(key=lambda x:x[0],reverse=True);elites=scored[:elite];nextp=[x[1] for x in elites];imm=max(1,int(pop*g["random_immigrant_rate"]));limit=pop-imm
  def parent():
   if opt.selection_mode=="tournament":return max(rng.sample(scored,min(g["tournament_size"],len(scored))),key=lambda x:x[0])[1]
   return rng.choice(elites)[1]
  while len(nextp)<limit:
   a=parent();child=crossover(a,parent(),rng,opt.crossover_mode) if rng.random()<g["crossover_rate"] else a
   if rng.random()<g["mutation_rate"]:child=mutate(child,terminals,g["max_depth"],rng,opt.mutation_mode,opt.temporal)
   if depth(child)>g["max_depth"] or not valid_expr(child):child=random_expr(terminals,g["max_depth"],rng,opt.temporal)
   nextp.append(child)
  while len(nextp)<pop:nextp.append(random_expr(terminals,g["max_depth"],rng,opt.temporal))
  population=nextp;print(f"[ga-v2] {opt.experiment_id} gen={gen+1:02d} best={scored[0][0]:.5f} unique={len(cache)}")
 candidates=sorted(hall.values(),key=lambda x:x[0],reverse=True)[:g["candidate_keep"]];reviewed_count=0
 if not opt.quick and opt.experiment_id!="F00":
  full_ev=Evaluator(discovery);full_sc=Scorer(discovery,discovery.target_rank.notna(),int(c["target"]["top_k"]),int(c["ga"]["min_month_days"]));reviewed=[]
  for sampled_score,e,fit in candidates:
   value=full_ev.eval(e).loc[full_sc.f.index].to_numpy();raw,peak,vol,direction,_=full_sc.score(value,opt.objective);full_score=raw-g["daily_volatility_penalty"]*vol-g["complexity_penalty"]*nodes(e);reviewed.append((full_score,e,(full_score,peak,raw,vol,direction,sampled_score)))
  candidates=sorted(reviewed,key=lambda x:x[0],reverse=True);ev,sc=full_ev,full_sc;reviewed_count=len(candidates)
 selected=[];ranked=[];months={}
 for score,e,fit in candidates:
  if months.get(fit[1],0)>=g["max_factors_per_peak_month"]:continue
  direction=fit[4];r=(direction*ev.eval(e).loc[sc.f.index]).groupby(sc.f.trade_date).rank(pct=True).astype("float32")
  if r.notna().sum()<100 or float(r.std())<1e-8:continue
  if any(abs(r.corr(x))>=g["corr_prune_threshold"] for x in ranked):continue
  ranked.append(r);months[fit[1]]=months.get(fit[1],0)+1;selected.append({"name":f"factor_{len(selected)+1:02d}","expression":e,"expression_text":text(e),"direction":direction,"category":expression_category(e),"score":score,"peak_month":fit[1],"depth":depth(e),"nodes":nodes(e),"sampled_search_score":fit[5] if len(fit)>5 else score})
  if len(selected)>=g["library_size"]:break
 out=Path(out_dir);out.mkdir(parents=True,exist_ok=True);payload={"options":asdict(opt),"factors":selected,"search_rows":int(ctx._search.sum()),"unique_expressions":len(cache),"full_cross_section_candidates_reviewed":reviewed_count,"seconds":time.time()-start};(out/"factor_library.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");return payload,ctx
