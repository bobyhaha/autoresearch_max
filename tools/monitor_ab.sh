#!/usr/bin/env bash
# Concurrent A/B verdict: {TAG}_treat vs {TAG}_ctrl, paired per seed. Usage: monitor_ab.sh TAG [NSEEDS] [SIGMA]
set -u
WD=/home/user/ai4ai/temp_autoresearch_baiyu_noisefloor
TAG=$1; NEED=${2:-3}; SIGMA=${3:-0.00087}
if [ "$NEED" -lt 3 ]; then
  echo "ABORT: effect verdicts require at least 3 paired seeds" >&2
  exit 2
fi
for i in $(seq 1 220); do
  n=$(grep -l "EXIT_CODE=0" $WD/${TAG}_treat_seed*.log 2>/dev/null | wc -l)
  m=$(grep -l "EXIT_CODE=0" $WD/${TAG}_ctrl_seed*.log 2>/dev/null | wc -l)
  [ "$n" -ge "$NEED" ] && [ "$m" -ge "$NEED" ] && break; sleep 20
done
echo "=== $TAG: treat vs ctrl (concurrent A/B; sigma=$SIGMA) ==="
/home/user/ph/autoresearch/.venv/bin/python - "$WD" "$TAG" "$SIGMA" "$NEED" <<'PY'
import re,glob,sys,os,statistics as st
WD,TAG,SIGMA=sys.argv[1],sys.argv[2],float(sys.argv[3])
NEED=int(sys.argv[4]) if len(sys.argv)>4 else 3
def pick(prefix,seed):
    fs=[f for f in glob.glob(f"{WD}/{prefix}_seed{seed}_gpu*.log") if re.search(r"EXIT_CODE=0",open(f).read())]
    return max(fs,key=os.path.getmtime) if fs else None
def vbpb(prefix,seed):
    f=pick(prefix,seed)
    if not f: return None
    ms=re.findall(r"^val_bpb:\s*([0-9.]+)",open(f).read(),re.M); return float(ms[-1]) if ms else None
deltas=[]; tv=[]
for seed in (42,43,44,45,46,47):
    c=vbpb(f"{TAG}_ctrl",seed); t=vbpb(f"{TAG}_treat",seed)
    if c is None and t is None: continue
    d=None if (c is None or t is None) else round(t-c,6)
    print(f"seed{seed}: ctrl={c} treat={t} delta={d}")
    if t is not None: tv.append(t)
    if c is not None and t is not None: deltas.append(t-c)
if len(deltas)<NEED:
    print(f"--- INCONCLUSIVE: {len(deltas)} valid pairs (need {NEED}) ---"); sys.exit(2)
mean=st.mean(deltas); sd=st.stdev(deltas)
same=all(d<0 for d in deltas) or all(d>0 for d in deltas)
se=sd/(len(deltas)**0.5) if sd>0 else 1e-9; tstat=mean/se
thr=2*(2**0.5)*SIGMA/(len(deltas)**0.5)
# one-sided 95% t-crit by dof
tcrit={1:6.31,2:2.92,3:2.35,4:2.13,5:2.02}.get(len(deltas)-1,2.0)
strong=same and abs(tstat)>tcrit and abs(mean)>=thr
sugg=same and abs(mean)>=thr and not strong
print(f"treat_mean={st.mean(tv):.4f}  n={len(deltas)}")
print(f"mean_delta={mean:+.5f} std={sd:.5f} paired_t={tstat:+.2f} (crit={tcrit}) thr={thr:.5f} same_sign={same}")
v=("SIGNIFICANT "+("BETTER" if mean<0 else "WORSE")) if strong else \
  ("SUGGESTIVE "+("better" if mean<0 else "worse")+" -> add seeds") if sugg else "NULL (within noise)"
print(f"--- VERDICT: {v} ---")
PY
