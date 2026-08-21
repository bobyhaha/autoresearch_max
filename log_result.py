#!/usr/bin/env python3
"""Append one experiment result to campaign_log.jsonl and results.tsv.
Usage: python log_result.py exp=exp001 val_bpb=1.02 vram_mb=45060 steps=671 status=keep \
       desc="matrix_lr 0.05" phase=1 ts_start=... ts_end=... commit=abc"""
import json, sys, os, datetime
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(ROOT, "campaign_log.jsonl")
TSV = os.path.join(ROOT, "results.tsv")
kv = {}
for a in sys.argv[1:]:
    k, v = a.split("=", 1)
    kv[k] = v
recs = []
if os.path.exists(LOG):
    for line in open(LOG):
        line = line.strip()
        if line:
            recs.append(json.loads(line))
exp_num = 0 if kv.get("status") == "baseline" else max([r.get("exp_num", 0) for r in recs], default=0) + 1
def num(x):
    return None if x in (None, "", "NA") else x
def iso2epoch(s):
    return datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
vb = num(kv.get("val_bpb")); vr = num(kv.get("vram_mb")); st = num(kv.get("steps"))
rec = {
    "exp_num": exp_num, "exp": kv["exp"],
    "val_bpb": float(vb) if vb else None,
    "vram_gb": round(float(vr) / 1024, 1) if vr else None,
    "steps": int(st) if st else None,
    "status": kv["status"], "desc": kv.get("desc", ""), "phase": kv.get("phase", ""),
    "milestone": kv.get("milestone", ""),
    "ts_start": kv.get("ts_start", ""), "ts_end": kv.get("ts_end", ""),
    "ts_end_epoch": iso2epoch(kv["ts_end"]) if kv.get("ts_end") else None,
    "commit": kv.get("commit", ""),
}
with open(LOG, "a") as f:
    f.write(json.dumps(rec) + "\n")
with open(TSV, "a") as f:
    f.write("\t".join([
        kv.get("commit", "")[:7],
        f'{rec["val_bpb"]:.6f}' if rec["val_bpb"] else "0.000000",
        f'{rec["vram_gb"]:.1f}' if rec["vram_gb"] else "0.0",
        rec["status"], rec["desc"],
    ]) + "\n")
print(f"logged exp_num={exp_num} {rec['exp']} bpb={rec['val_bpb']} status={rec['status']}")
