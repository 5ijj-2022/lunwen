"""Render paper figures from request-level events.jsonl."""
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('events',type=Path);p.add_argument('--out',type=Path,default=Path('figures'));a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 rows=[json.loads(x) for x in a.events.read_text().splitlines() if x.strip()]
 import matplotlib.pyplot as plt
 phases=sorted(set(r.get('phase','') for r in rows)); rates=[]
 for phase in phases:
  x=[r for r in rows if r.get('phase')==phase];rates.append(sum(r.get('target_error',False) for r in x)/len(x))
 fig,ax=plt.subplots(figsize=(6,4));ax.bar(phases,rates);ax.set_ylabel('Target error rate');ax.set_ylim(0,1);fig.tight_layout();fig.savefig(a.out/'target_error_by_phase.png',dpi=200);plt.close(fig)
 fig,ax=plt.subplots(figsize=(6,4));ax.scatter([r.get('similarity',0) for r in rows],[int(r.get('hit',False)) for r in rows]);ax.set_xlabel('Cache similarity');ax.set_ylabel('Cache hit (1/0)');fig.tight_layout();fig.savefig(a.out/'similarity_hits.png',dpi=200);plt.close(fig)
if __name__=='__main__':main()
