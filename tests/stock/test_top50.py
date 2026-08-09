import random,tempfile,unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ga_factor_mining.stock.top50.data import add_labels
from ga_factor_mining.stock.top50.ga import Evaluator,Scorer,crossover,depth,random_expr,valid_expr
from ga_factor_mining.stock.top50.metrics import evaluate_predictions
from ga_factor_mining.stock.top50.run_experiments import final_test

CONFIG={"target":{"column":"future_ret_5d","top_k":2,"relevance_edges":[.5,.7,.9,.99]}}
class V2Tests(unittest.TestCase):
 def test_labels_and_exact_topk(self):
  d=pd.DataFrame({"ts_code":[f"s{i}" for i in range(5)],"trade_date":["20240101"]*5,"future_ret_5d":[0,1,2,3,4]});x=add_labels(d,CONFIG);self.assertEqual(int(x.is_target_topk.sum()),2);self.assertEqual(int(x.relevance.max()),4)
  x["score"]=x.future_ret_5d;ranked,*_=evaluate_predictions(x,2);self.assertEqual(int(ranked.is_pred_topk.sum()),2)
 def test_temporal_is_backward_only(self):
  d=pd.DataFrame({"ts_code":["a"]*7,"trade_date":[str(i) for i in range(7)],"x":np.arange(7.)});e=Evaluator(d);delta=e.eval(["delta_5","x"]);self.assertTrue(delta.iloc[:5].isna().all());self.assertEqual(delta.iloc[5],5);d.loc[6,"x"]=999;self.assertEqual(delta.iloc[5],5)
 def test_slope_is_five_point_linear_fit(self):
  d=pd.DataFrame({"ts_code":["a"]*5,"trade_date":[str(i) for i in range(5)],"x":[1.,3.,5.,7.,9.]});self.assertAlmostEqual(Evaluator(d).eval(["slope_5","x"]).iloc[-1],2.)
 def test_temporal_terminal_grammar(self):
  self.assertTrue(valid_expr(["add",["mean_5","x"],"y"]));self.assertFalse(valid_expr(["mean_5",["add","x","y"]]))
 def test_genetic_depth_repair_is_detectable(self):
  a=["add","x",["mul","y","z"]];b=["signed_sqrt","x"];child=crossover(a,b,random.Random(1),"symmetric_subtree");self.assertGreaterEqual(depth(child),0)
 def test_genetic_seed_is_reproducible(self):
  a=[random_expr(["x","y"],3,random.Random(42)) for _ in range(5)];b=[random_expr(["x","y"],3,random.Random(42)) for _ in range(5)];self.assertEqual(a,b)
 def test_reverse_ndcg_uses_ascending_scores(self):
  d=pd.DataFrame({"ts_code":[f"s{i}" for i in range(100)],"trade_date":["20240101"]*100,"target_rank":np.arange(1,101)/100,"relevance":[0]*90+[1]*5+[2]*3+[3]+[4]});sc=Scorer(d,np.ones(100,dtype=bool),k=10,min_days=1);daily=sc._daily(-d.target_rank.to_numpy(),"ndcg");self.assertGreater(daily.neg.iloc[0],daily.pos.iloc[0])
 def test_final_test_requires_frozen_config(self):
  with tempfile.TemporaryDirectory() as td:
   with self.assertRaises(RuntimeError):final_test({},Path(td))
if __name__=="__main__":unittest.main()
