"""Verify published measured results without model downloads or LLM execution."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import binomtest
HERE=Path(__file__).resolve().parent;OUT=HERE/'results';EXP=HERE/'reproduction/experiments/clinc150_8shot'
sys.path.insert(0,str(HERE/'reproduction'))
from src.label_parsing import parse_intent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s]
def holm(values):
 v=np.asarray(values,float);order=np.argsort(v,kind='stable');out=np.empty_like(v);maximum=0.
 for rank,i in enumerate(order):maximum=max(maximum,(len(v)-rank)*v[i]);out[i]=min(1,maximum)
 return out

def main():
 manifest=read(HERE/'PACKAGING_MANIFEST.json')
 for f in manifest['files']:assert sha(HERE/f['published'])==f['published_sha256'],f['published']
 names=read(EXP/'data/label_mapping.json')['id_to_label'];assert len(set(names))==150 and 'oos' not in names
 pool=set(pd.read_csv(EXP/'data/demonstration_pool.csv').example_id)
 meta=pd.read_csv(EXP/'data/meta_training.csv',keep_default_na=False).set_index('example_id')
 raw=lines(OUT/'meta_label_scores.jsonl');assert len(raw)==15000
 scores={(r['query_id'],r['strategy']):r for r in raw};assert len(scores)==15000
 strategies=['similar','diverse','balanced']
 train=pd.read_csv(OUT/'meta_training_dataset.csv').set_index('query_id');assert len(train)==5000
 for qid,row in train.iterrows():
  group=[scores[qid,s] for s in strategies];a=np.array([r['mean_log_likelihood'] for r in group])
  target=next(i for i in [0,2,1] if a.max()-a[i]<=1e-6)
  assert target==row.target_class and np.isfinite(a).all()
  for r in group:
   assert r['target_label']==meta.loc[qid,'label_name'] and r['includes_eos'] is False
   assert np.isclose(r['sum_log_likelihood']/r['label_token_count'],r['mean_log_likelihood'],atol=1e-5)
 methods=['always_similar','always_diverse','always_balanced','logistic_regression','tabpfn_v2','oracle_retrospective']
 checked=[]
 for split,n in [('validation',3000),('test',4500)]:
  gold=pd.read_csv(EXP/f'data/{split}.csv',keep_default_na=False).set_index('example_id');assert len(gold)==n
  raw=lines(OUT/f'{split}_fixed_predictions.jsonl');assert len(raw)==3*n
  ix={(r['query_id'],r['strategy']):r for r in raw};assert len(ix)==3*n
  for r in raw:
   assert not {'gold','gold_label','target_label'}&set(r)
   assert parse_intent(r['raw_output'],names)['predicted_label']==r['predicted_label']
   assert parse_intent(r['raw_output'],names)['valid']==r['valid']
   assert len(set(r['demo_ids']))==8 and set(r['demo_ids'])<=pool
  qids=gold.index.tolist();y=gold.label_name.to_numpy();fixed=np.array([[ix[q,s]['predicted_label'] or '__INVALID__' for s in strategies] for q in qids])
  hit=fixed==y[:,None];choices={m:np.full(n,j) for j,m in enumerate(methods[:3])}
  for m in methods[3:5]:
   f=pd.read_csv(OUT/f'{m}_{split}_choices.csv').set_index('query_id');assert f.index.is_unique and set(f.index)==set(qids)
   choices[m]=f.loc[qids].strategy_class.to_numpy(int)
  oracle=np.zeros(n,int);oracle[~hit[:,0]&hit[:,1]]=1;oracle[~hit[:,0]&hit[:,2]]=2;choices[methods[5]]=oracle
  metrics=pd.read_csv(OUT/f'{split}_metrics.csv').set_index('method');correct={}
  for m,actions in choices.items():
   pred=fixed[np.arange(n),actions];c=pred==y;correct[m]=c
   assert np.isclose(accuracy_score(y,pred),metrics.loc[m,'accuracy'],atol=1e-12)
   assert np.isclose(f1_score(y,pred,labels=names,average='macro',zero_division=0),metrics.loc[m,'macro_f1'],atol=1e-12)
   assert int(c.sum())==metrics.loc[m,'correct_count']
   assert int(np.sum(~correct['always_similar']&c))==metrics.loc[m,'beneficial']
   assert int(np.sum(correct['always_similar']&~c))==metrics.loc[m,'harmful']
   checked.append({'split':split,'method':m,'accuracy':float(c.mean()),'correct':int(c.sum())})
  assert np.array_equal(correct['oracle_retrospective'],hit.any(1))
  comparisons=pd.read_csv(OUT/f'{split}_paired_comparisons.csv');pvals=[];deploy=[]
  for _,r in comparisons.iterrows():
   a,b=correct[r.left],correct[r.right];ab=int((a&~b).sum());ba=int((~a&b).sum())
   p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.
   assert ab==r.left_only_correct and ba==r.right_only_correct and np.isclose(p,r.mcnemar_exact_p)
   if not r.analysis_only_oracle:pvals.append(p);deploy.append(r)
  for r,p in zip(deploy,holm(pvals)):assert np.isclose(p,r.holm_p)
 print('PASS: package hashes, 15000 likelihood records, 22500 fixed predictions, all 12 metric rows, switching counts, Oracle and McNemar/Holm tests.')
 for r in checked:print(f"{r['split']:10s} {r['method']:22s} {r['accuracy']:.6%} ({r['correct']})")
 print('No model loaded. Historical cache/682-file checks remain evidence of the original run, not requirements for this repository snapshot.')

if __name__=='__main__':main()
