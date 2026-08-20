#!/usr/bin/env python3
"""val_bpb over experiment number. ONE baseline (shown as a reference line + its measured
noise band); the numbered series contains ONLY runs that change something vs baseline."""
import glob
import html
import json
import math
import pathlib
import statistics

REPO = pathlib.Path(__file__).resolve().parent.parent
BASE_CFG = {"dbs": 128, "tbs": 19, "depth": 8, "dim": 512,
            "mlp": 4, "ve": 2, "win": "SSSL", "swdiv": 2}
BASELINE_SOLO = 1.012836        # the cleanest baseline measurement: solo, uncontended

baseline_runs=[]; exps=[]

# --- banked harness controls: all unedited train.py -> baseline, never experiments
for f in glob.glob(str(REPO/".autoresearch/records/result_bundle/*.json")):
    d=json.loads(pathlib.Path(f).read_text())
    for a in d["payload"]["arms"]:
        m=a.get("metrics") or {}
        if "val_bpb" in m: baseline_runs.append({"t":a["ended_at"],"bpb":m["val_bpb"],"steps":int(m.get("num_steps",0))})

# --- manual probe waves, in run order. "ctl" is the unedited config -> baseline.
PROBES=[("ctl",539,1.037566,True),("LLLL",539,1.039666,False),("prefetch",543,1.036580,False),
        ("bigbatch B=256",611,1.027293,False),("bb_ref",685,1.018231,False),("bb_LLLL",685,1.019852,False),
        ("MLP 6x",684,1.005793,False),("depth 11",677,1.000244,False),("depth 11 rep",694,0.998079,False),
        ("depth 12",672,0.997525,False),("d11+MLP5",664,0.997569,False),("MLP 8x",685,0.999040,False),
        ("bb_ref ep",687,1.017759,False),("bigbatch ep",688,1.017611,False),("MLP 6x ep",691,1.005065,False),
        ("depth 11 ep",677,0.999833,False),("B=256 solo",909,0.998145,False),
        ("MLP 6x solo",852,0.989563,False),("depth 11 solo",758,0.990926,False)]
for n,s,b,isbase in PROBES:
    (baseline_runs if isbase else exps).append(
        {"t":"","bpb":b,"steps":s,"name":n,"kind":"probe"} if not isbase else {"t":"","bpb":b,"steps":s})

# --- 24h sweep
for f in sorted(glob.glob(str(REPO/"runs/sweep/results/*.json"))):
    r=json.loads(pathlib.Path(f).read_text()); m=r.get("metrics") or {}; cfg=r.get("cfg") or {}
    isbase = cfg and all(cfg.get(k)==v for k,v in BASE_CFG.items())
    if r.get("ok") and "val_bpb" in m:
        rec={"t":r["ended"],"bpb":m["val_bpb"],"steps":int(m.get("num_steps",0)),"name":r["name"],"kind":"sweep"}
        (baseline_runs if isbase else exps).append(rec)
    elif not isbase:
        exps.append({"t":r.get("ended",0),"bpb":None,"steps":0,"name":r["name"],"kind":"failed"})

exps.sort(key=lambda r:(r["t"] if isinstance(r["t"],(int,float)) else 0))
for i,r in enumerate(exps,1): r["n"]=i
ok=[r for r in exps if r["bpb"] is not None]
bvals=[r["bpb"] for r in baseline_runs]
# min/max over heterogeneous controls is not a noise estimate -- it is a range statistic
# inflated by contention. Use +-2 sigma of the control set and report n and sigma.
_mu=statistics.mean(bvals); _sd=statistics.pstdev(bvals) if len(bvals)>1 else 0.0
bmin,bmax=_mu-2*_sd,_mu+2*_sd
band_n, band_sd, band_mu = len(bvals), _sd, _mu
best=min(ok,key=lambda r:r["bpb"]) if ok else None

W,H=980,440; L,Rp,T,B=78,128,22,50
CLIP = 1.1     # plotted y-range only. Runs above it keep their index and are marked INVALID.
for r in exps:
    r["invalid"] = r["bpb"] is not None and r["bpb"] >= CLIP
excluded=[r for r in exps if r["invalid"]]      # kept in the series, drawn as x at the top
ok=[r for r in exps if r["bpb"] is not None and not r["invalid"]]
best=min(ok,key=lambda r:r["bpb"]) if ok else None
lo=min([r["bpb"] for r in ok]+[bmin])-0.003; hi=max([r["bpb"] for r in ok]+[bmax])+0.003
def X(n): return L+(n-0.5)/max(len(exps),1)*(W-L-Rp)
def Y(b): return (H-B)-(b-lo)/(hi-lo)*(H-T-B)      # reversed: bpb ascends, improvement descends
p=[]
v=math.floor(lo*100)/100
while v<=hi:
    if lo<=v<=hi:
        p.append(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{W-Rp}" y2="{Y(v):.1f}"/>')
        p.append(f'<text class="tick" x="{L-9}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
        p.append(f'<line class="tickmark" x1="{L}" y1="{Y(v):.1f}" x2="{L-5}" y2="{Y(v):.1f}"/>')
    v=round(v+0.01,3)
stepn=max(1,len(exps)//10)
for n in range(stepn,len(exps)+1,stepn):
    p.append(f'<text class="tick" x="{X(n):.1f}" y="{H-B+18}" text-anchor="middle">{n}</text>')
    p.append(f'<line class="tickmark" x1="{X(n):.1f}" y1="{H-B}" x2="{X(n):.1f}" y2="{H-B+5}"/>')
# baseline noise band + the clean solo baseline line
p.append(f'<rect class="band" x="{L}" y="{Y(bmax):.1f}" width="{W-Rp-L}" height="{abs(Y(bmin)-Y(bmax)):.1f}"/>')
p.append(f'<line class="baserule" x1="{L}" y1="{Y(BASELINE_SOLO):.1f}" x2="{W-Rp}" y2="{Y(BASELINE_SOLO):.1f}"/>')
p.append(f'<text class="baselab" x="{W-Rp+7}" y="{Y(BASELINE_SOLO)+4:.1f}">baseline {BASELINE_SOLO:.4f}</text>')
p.append(f'<text class="bandlab" x="{W-Rp+7}" y="{Y((bmin+bmax)/2)+18:.1f}">control ±2σ</text>')
p.append(f'<text class="bandlab" x="{W-Rp+7}" y="{Y((bmin+bmax)/2)+31:.1f}">n={band_n}, σ={band_sd:.4f}</text>')
# running best
pts=[];cur=None
for r in exps:
    if r["bpb"] is None or r.get("invalid"): continue
    cur=r["bpb"] if cur is None else min(cur,r["bpb"]); pts.append((X(r["n"]),Y(cur)))
if pts: p.append('<path class="frontier" d="M '+" L ".join(f"{x:.1f} {y:.1f}" for x,y in pts)+'"/>')
for r in exps:
    cx=X(r["n"])
    if r.get("invalid"):
        cy=Y(hi-0.002)
        p.append(f'<path class="xmark" d="M {cx-4:.1f} {cy-4:.1f} L {cx+4:.1f} {cy+4:.1f} M {cx+4:.1f} {cy-4:.1f} L {cx-4:.1f} {cy+4:.1f}" '
                 f'tabindex="0" data-t="#{r["n"]} {html.escape(r["name"])} - val_bpb {r["bpb"]:.4f} INVALID (off scale)"/>')
        continue
    if r["bpb"] is None:
        cy=Y(hi-0.002)
        p.append(f'<path class="xmark" d="M {cx-4:.1f} {cy-4:.1f} L {cx+4:.1f} {cy+4:.1f} M {cx+4:.1f} {cy-4:.1f} L {cx-4:.1f} {cy+4:.1f}" tabindex="0" data-t="#{r["n"]} {html.escape(r["name"])} — FAILED"/>')
        continue
    cls="c3" if r["kind"]=="sweep" else "c2"
    tip=f'#{r["n"]} {r["name"]} — val_bpb {r["bpb"]:.6f}, {r["steps"]} steps'
    p.append(f'<circle class="dot {cls}" cx="{cx:.1f}" cy="{Y(r["bpb"]):.1f}" r="4.5" tabindex="0" data-t="{html.escape(tip,quote=True)}"/>')
p.append(f'<rect class="spine" x="{L}" y="{T}" width="{W-Rp-L}" height="{H-B-T}"/>')
p.append(f'<text class="alab" x="{(L+W-Rp)/2:.0f}" y="{H-10}" text-anchor="middle">experiment number (runs that change something vs baseline)</text>')
p.append(f'<text class="alab" transform="rotate(-90 17 {(T+H-B)/2:.0f})" x="17" y="{(T+H-B)/2:.0f}" text-anchor="middle">val_bpb  (lower is better)</text>')
if best: p.append(f'<text class="dlab" x="{X(best["n"]):.1f}" y="{Y(best["bpb"])+19:.1f}" text-anchor="middle">{best["bpb"]:.5f}</text>')

SL=0.080269
def _resid(r):
    return r["bpb"]+SL*math.log(r["steps"]) if (r["bpb"] and r["steps"]) else None
_bres=[_resid(b) for b in baseline_runs if b.get("steps")]
_bres=[x for x in _bres if x]
REF=statistics.mean(_bres) if _bres else None
rows="".join(
  f'<tr><td class="num">{r["n"]}</td><td>{html.escape(r["name"])}</td>'
  f'<td class="num">{"—" if r["bpb"] is None else format(r["bpb"],".6f")}</td>'
  f'<td class="num">{r["steps"] or ""}</td>'
  f'<td class="num">{"" if _resid(r) is None else format(_resid(r),".6f")}</td>'
  f'<td class="num">{"" if (_resid(r) is None or REF is None) else format(_resid(r)-REF,"+.6f")}</td>'
  f'<td>{"INVALID" if r.get("invalid") else ("failed" if r["bpb"] is None else r["kind"])}</td></tr>'
  for r in exps)
nf=len(exps)-len(ok)
offnote = ("<div class=\"note\" style=\"border-left-color:var(--c4)\"><strong>"
           f"{len(excluded)} run(s) excluded from this chart</strong> for scoring at or above "
           f"val_bpb {CLIP}: " + ", ".join(f"{html.escape(r['name'])} ({r['bpb']:.3f})" for r in excluded)
           + ". All are a broken EMA implementation (the average was seeded at random init and never warmed), "
             "so they measure nothing about weight averaging. Recorded here so the exclusion is visible."
             "</div>") if excluded else ""
(REPO/"runs/campaign_progress.html").write_text(f"""<title>OPHIS v3 — val_bpb over experiment number</title>
<style>
:root{{color-scheme:light;--plane:#fcfcfb;--surface:#fff;--ink:#141414;--ink2:#454545;--muted:#7d7d7d;
--grid:#e6e6e3;--axis:#3a3a3a;--border:#d8d8d4;--c2:#9467bd;--c3:#1f77b4;--c4:#e07b00;--best:#d62728;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{color-scheme:dark;--plane:#131315;--surface:#1b1b1e;
--ink:#f2f2f0;--ink2:#c2c2bf;--muted:#8d8d8a;--grid:#2e2e32;--axis:#9a9a97;--border:#33333a;
--c2:#a074cb;--c3:#4a90d9;--c4:#c8811f;--best:#e0605c;}}}}
:root[data-theme="dark"]{{color-scheme:dark;--plane:#131315;--surface:#1b1b1e;--ink:#f2f2f0;--ink2:#c2c2bf;
--muted:#8d8d8a;--grid:#2e2e32;--axis:#9a9a97;--border:#33333a;--c2:#a074cb;--c3:#4a90d9;--c4:#c8811f;--best:#e0605c;}}
body{{background:var(--plane);color:var(--ink);margin:0;}}
.wrap{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;max-width:1040px;margin:0 auto;padding:26px 18px 56px;}}
h1{{font-size:19px;font-weight:600;margin:0 0 6px;}}
.sub{{color:var(--ink2);font-size:13px;line-height:1.55;max-width:82ch;margin:0 0 16px;}}
code{{font-family:ui-monospace,Menlo,monospace;font-size:12px;}}
.tiles{{display:flex;flex-wrap:wrap;gap:0;margin-bottom:14px;}}
.tile{{padding:0 30px 0 0;}} .k{{font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);}}
.v{{font-size:19px;font-weight:600;margin-top:2px;font-variant-numeric:tabular-nums;}}
.d{{font-size:11.5px;color:var(--ink2);}}
.scroll{{overflow-x:auto;}} svg{{display:block;min-width:{W}px;}}
.grid{{stroke:var(--grid);stroke-width:.7;}}
.spine{{fill:none;stroke:var(--axis);stroke-width:.9;}} .tickmark{{stroke:var(--axis);stroke-width:.9;}}
.tick,.alab{{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums;}}
.band{{fill:var(--muted);opacity:.13;}}
.baserule{{stroke:var(--axis);stroke-width:1.3;stroke-dasharray:6 3;}}
.baselab{{fill:var(--ink2);font-size:11.5px;font-weight:600;}}
.bandlab,.dlab{{fill:var(--muted);font-size:10.5px;}}
.dlab{{fill:var(--ink2);font-size:11.5px;font-weight:600;}}
.frontier{{fill:none;stroke:var(--best);stroke-width:2;stroke-linejoin:round;}}
.dot{{stroke:var(--surface);stroke-width:1.5;cursor:pointer;}}
.xmark{{stroke:var(--c4);stroke-width:2;fill:none;stroke-linecap:round;}}
.offscale{{stroke:var(--c4);stroke-width:2.2;fill:none;stroke-linejoin:round;}}
.c2{{fill:var(--c2);}} .c3{{fill:var(--c3);}}
.legend{{display:flex;gap:18px;flex-wrap:wrap;padding:9px 4px 2px;font-size:12px;color:var(--ink2);}}
.li{{display:inline-flex;align-items:center;gap:6px;}}
.sw{{width:10px;height:10px;border-radius:50%;display:inline-block;}}
.sw.c2{{background:var(--c2);}} .sw.c3{{background:var(--c3);}}
.sw.ln{{width:16px;height:0;border-top:2px solid var(--best);border-radius:0;}}
.sw.bd{{background:var(--muted);opacity:.35;border-radius:2px;width:16px;height:10px;}}
table{{width:100%;border-collapse:collapse;font-size:12.5px;margin-top:18px;}}
th,td{{text-align:left;padding:5px 9px;border-bottom:1px solid var(--border);}}
th{{color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;}}
td.num{{text-align:right;font-variant-numeric:tabular-nums;}}
.tt{{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface);color:var(--ink);
border:1px solid var(--border);border-radius:7px;padding:8px 10px;font-size:12px;max-width:320px;
box-shadow:0 6px 22px rgba(0,0,0,.2);z-index:9;}}
.note{{font-size:12px;color:var(--ink2);border-left:3px solid var(--best);padding:2px 0 2px 12px;margin-top:14px;line-height:1.55;}}
</style>
<div class="wrap">
<h1>OPHIS v3 — val_bpb over experiment number</h1>
<p class="sub">One point per 300-second run that <strong>changes something</strong> versus the unedited baseline,
in the order it was run. Frozen scope <code>karpathy_228791f_original_bpb_h200_zpnc71_300s_v3</code>.
The dashed line is the baseline measured solo; the grey band is the spread of {len(baseline_runs)} re-runs of that
same unedited code under host contention — <strong>anything inside the band is indistinguishable from changing nothing</strong>.</p>
<div class="tiles">
<div class="tile"><div class="k">Experiments</div><div class="v">{len(exps)}</div><div class="d">changed vs baseline</div></div>
<div class="tile"><div class="k">Baseline</div><div class="v">{BASELINE_SOLO:.5f}</div><div class="d">unedited, solo</div></div>
<div class="tile"><div class="k">Best</div><div class="v">{best["bpb"]:.5f}</div><div class="d">{html.escape(best["name"])} — #{best["n"]}</div></div>
<div class="tile"><div class="k">Gain</div><div class="v">{best["bpb"]-BASELINE_SOLO:+.5f}</div><div class="d">vs baseline</div></div>
<div class="tile"><div class="k">Failed</div><div class="v">{nf}</div><div class="d">OOM / runtime</div></div>
</div>
<div class="scroll"><svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
aria-label="val_bpb by experiment number, reversed axis, with baseline band and running-best">{''.join(p)}</svg></div>
<div class="legend"><span class="li"><span class="sw c2"></span>manual probe</span>
<span class="li"><span class="sw c3"></span>24h sweep</span>
<span class="li"><span class="sw ln"></span>running best</span>
<span class="li"><span class="sw bd"></span>baseline noise band</span></div>
{offnote}
<div class="note">Baseline control re-runs are deliberately <em>not</em> numbered experiments — they change
nothing and exist only to measure noise. All points shown are out-of-harness probes at n=1, seed 42, so the
running best is <strong>not SOTA</strong>.</div>
<table><thead><tr><th class="num">#</th><th>experiment</th><th class="num">val_bpb</th><th class="num">steps</th><th class="num">step-ctrl residual</th><th class="num">vs control resid</th><th>status</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class="tt" id="tt"></div>
<script>
const tt=document.getElementById('tt');
for(const el of document.querySelectorAll('.dot,.xmark')){{
 const s=()=>{{tt.textContent=el.dataset.t;tt.style.opacity=1;const r=el.getBoundingClientRect();
  tt.style.left=Math.min(innerWidth-330,r.left+14)+'px';tt.style.top=(r.top-8)+'px';}};
 el.addEventListener('mouseenter',s);el.addEventListener('focus',s);
 el.addEventListener('mouseleave',()=>tt.style.opacity=0);el.addEventListener('blur',()=>tt.style.opacity=0);}}
</script>""")
print(f"{len(exps)} experiments plotted; {len(baseline_runs)} baseline re-runs folded into the band "
      f"[{bmin:.6f}, {bmax:.6f}]; best {best['bpb']:.6f} ({best['name']})")
