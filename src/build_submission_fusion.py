"""
build_submission_fusion.py — Part of the MAC 2026 Track 2 pipeline. See README.md (Complete Pipeline Guide).
Run `python src/build_submission_fusion.py --help` for options where applicable.
"""
import sys, csv, time
sys.path.insert(0,'/tmp')
from sweep import load_norm, fuse
PRED="predictions"
CAT="./data/MMA-52/Annotations/adatad/category_idx.txt"
cats=[l.strip() for l in open(CAT) if l.strip()]; c2i={c:i for i,c in enumerate(cats)}
def trunc(r,k=1200): return {v:sorted(p,key=lambda x:-x['score'])[:k] for v,p in r.items()}
L1=trunc(load_norm('large_test.json')); L2=trunc(load_norm('large2_test.json'))
print(f'loaded test {len(L1)} vids',flush=True)
def build(name,w,sigma,tk=1000):
    t=time.time(); fused=fuse([L1,L2],w,sigma=sigma,top_k=tk); rows=[]; rid=0
    for vid in sorted(fused):
        for p in sorted(fused[vid],key=lambda x:-x['score'])[:tk]:
            i=c2i.get(p['label'],-1)
            if i<0: continue
            rows.append([rid,vid,round(p['segment'][0],6),round(p['segment'][1],6),i,round(p['score'],6)]); rid+=1
    out=f'{PRED}/{name}'
    with open(out,'w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['ID','video-id','t-start','t-end','label','score']); wr.writerows(rows)
    print(f'{name:<40} w={w} s={sigma} tk={tk} rows={len(rows):,} ({time.time()-t:.0f}s)',flush=True)
build('submission_TUNED_4753_s07_top1000.csv',[0.47,0.53],0.7,1000)   # primary, val 22.10
build('submission_HEDGE_4555_s07_top1000.csv',[0.45,0.55],0.7,1000)   # milder hedge, val 21.96
