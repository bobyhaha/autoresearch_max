#!/usr/bin/env python3
"""Regenerate the post-origin val_bpb charts as a self-contained HTML artifact.

The immutable ledgers remain complete, but the displayed research series starts
at the recovered experiment-501 result (0.927183), as requested by the operator.
Later chart-admissible governed RunRecords are appended chronologically. Typed
evidence and terminal-invalid experiment updates are applied fail-closed to the
derived chart only. Every point has a hover tooltip; no source record is modified
or deleted.
"""
import json, os, datetime, html, re

ROOT = os.path.dirname(os.path.abspath(__file__))
LEGACY_LOG = os.path.join(ROOT, "campaign_log.jsonl")
RUNS_LOG = os.path.join(ROOT, "research", "experiments", "runs", "runs.jsonl")
RUN_EVIDENCE_LOG = os.path.join(
    ROOT, "research", "knowledge", "internal", "run_evidence.jsonl"
)
EVIDENCE_UPDATES_LOG = os.path.join(
    ROOT, "research", "refinement", "evidence_updates.jsonl"
)
OUT = os.environ.get("CHART_OUT") or os.path.join(ROOT, "charts", "campaign.html")
ANCHOR_EXP_NUM = 501
ANCHOR_VAL_BPB = 0.927183

# Display window, env-driven. Defaults reproduce the historical behaviour exactly:
# the series starts at the anchor and runs to the end of the ledger.
#   CHART_START_EXP  first legacy exp_num to display (default: the anchor, 501)
#   CHART_END_EXP    last legacy exp_num to display  (default: unbounded)
# Setting CHART_START_EXP below the anchor shows the RUN-UP to 0.927183, which is
# the only way to see that the anchor is the minimum of a cluster rather than a
# frontier. Time is then measured from the first displayed point, not the anchor.
WINDOW_START = int(os.environ.get("CHART_START_EXP", ANCHOR_EXP_NUM))
_end = os.environ.get("CHART_END_EXP", "").strip()
WINDOW_END = int(_end) if _end else None

def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def iso_epoch(value):
    if not value:
        return 0
    return datetime.datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    ).timestamp()

def normalize_legacy(record, source_seq):
    # Copy each row before adding chart-only metadata: campaign_log.jsonl remains
    # an immutable input ledger.
    normalized = dict(record)
    normalized["_source"] = "legacy_campaign"
    normalized["_source_seq"] = source_seq
    if not normalized.get("ts_end_epoch") and normalized.get("ts_end"):
        normalized["ts_end_epoch"] = iso_epoch(normalized["ts_end"])
    return normalized

def normalize_run(record, source_seq):
    outcomes = record.get("outcome_values") or {}
    experiment_id = record.get("experiment_id", "")
    arm_id = record.get("arm_id", "")
    seed = record.get("seed")
    return {
        "exp": record.get("run_id", ""),
        "run_id": record.get("run_id", ""),
        "experiment_id": experiment_id,
        "arm_id": arm_id,
        "role": record.get("role", ""),
        "seed": seed,
        "val_bpb": outcomes.get("val_bpb"),
        "steps": outcomes.get("num_steps"),
        "status": record.get("status", ""),
        "phase": "governed",
        "ts_start": record.get("started_at", ""),
        "ts_end": record.get("ended_at", ""),
        "ts_end_epoch": iso_epoch(record.get("ended_at")),
        "desc": f"{experiment_id} · {arm_id} · seed {seed}",
        "change": f"{experiment_id} · {arm_id} · seed {seed}",
        "milestone": record.get("milestone", ""),
        "_source": "run_record",
        "_source_seq": source_seq,
    }

def governed_chart_policy(run_evidence, evidence_updates):
    """Resolve typed chart exclusions and explicit governed-SOTA authority."""
    excluded_run_ids = set()
    excluded_experiment_ids = set()
    governed_sota_run_ids = set()
    governed_sota_experiment_ids = set()

    for evidence in run_evidence:
        facts = evidence.get("facts") or {}
        run_ids = {
            run_id
            for run_id in (evidence.get("run_ids") or [])
            if isinstance(run_id, str) and run_id
        }
        experiment_id = evidence.get("experiment_id")

        if facts.get("chart_point_permitted") is False:
            if run_ids:
                excluded_run_ids.update(run_ids)
            elif isinstance(experiment_id, str) and experiment_id:
                excluded_experiment_ids.add(experiment_id)

        # A governed SOTA marker is opt-in. Historical milestone strings and
        # chart-origin status never create this authority.
        if facts.get("sota_claim_permitted") is True:
            if run_ids:
                governed_sota_run_ids.update(run_ids)
            elif isinstance(experiment_id, str) and experiment_id:
                governed_sota_experiment_ids.add(experiment_id)

    for update in evidence_updates:
        if str(update.get("result", "")).lower() != "invalid":
            continue
        experiment_id = update.get("experiment_id")
        if isinstance(experiment_id, str) and experiment_id:
            excluded_experiment_ids.add(experiment_id)

    return {
        "excluded_run_ids": excluded_run_ids,
        "excluded_experiment_ids": excluded_experiment_ids,
        "governed_sota_run_ids": governed_sota_run_ids,
        "governed_sota_experiment_ids": governed_sota_experiment_ids,
    }

legacy_recs = [
    normalize_legacy(record, i)
    for i, record in enumerate(read_jsonl(LEGACY_LOG))
]
chart_policy = governed_chart_policy(
    read_jsonl(RUN_EVIDENCE_LOG),
    read_jsonl(EVIDENCE_UPDATES_LOG),
)
run_recs = []
for i, record in enumerate(read_jsonl(RUNS_LOG)):
    run_id = record.get("run_id")
    experiment_id = record.get("experiment_id")
    if (
        run_id in chart_policy["excluded_run_ids"]
        or experiment_id in chart_policy["excluded_experiment_ids"]
    ):
        continue
    normalized = normalize_run(record, i)
    normalized["_governed_sota"] = (
        run_id in chart_policy["governed_sota_run_ids"]
        or experiment_id in chart_policy["governed_sota_experiment_ids"]
    )
    run_recs.append(normalized)
recs = legacy_recs + run_recs

def epoch(r):
    return r.get("ts_end_epoch") or 0

def sort_key(r):
    exp_num = r.get("exp_num")
    return (
        epoch(r),
        0 if r["_source"] == "legacy_campaign" else 1,
        exp_num if isinstance(exp_num, (int, float)) else r["_source_seq"],
    )

recs.sort(key=sort_key)
anchor = next(
    (
        r
        for r in legacy_recs
        if r.get("exp_num") == ANCHOR_EXP_NUM
        and r.get("val_bpb") == ANCHOR_VAL_BPB
    ),
    None,
)
if anchor is None:
    raise RuntimeError(
        f"missing chart anchor exp {ANCHOR_EXP_NUM} / val_bpb {ANCHOR_VAL_BPB}"
    )
anchor_epoch = epoch(anchor)

# Keep the historical ledger intact and filter only this derived visualization.
# Legacy points begin at exp 501; governed rows begin at the same wall-clock
# anchor. This prevents an earlier, superseded baseline from defining the chart.
_n_all = len(recs)

def _in_window(exp_num):
    if not isinstance(exp_num, (int, float)):
        return False
    if exp_num < WINDOW_START:
        return False
    return WINDOW_END is None or exp_num <= WINDOW_END

# Time origin is the first DISPLAYED legacy point, so a window that includes the
# run-up does not produce negative minutes.
_legacy_in = [
    r for r in recs
    if r["_source"] == "legacy_campaign" and _in_window(r.get("exp_num")) and epoch(r)
]
window_start_epoch = min(epoch(r) for r in _legacy_in) if _legacy_in else anchor_epoch
window_end_epoch = (
    max(epoch(r) for r in _legacy_in)
    if (_legacy_in and WINDOW_END is not None) else None
)

recs = [
    r
    for r in recs
    if (
        r["_source"] == "legacy_campaign"
        and _in_window(r.get("exp_num"))
    )
    or (
        r["_source"] == "run_record"
        and epoch(r) >= window_start_epoch
        and (window_end_epoch is None or epoch(r) <= window_end_epoch)
    )
]
last_legacy_exp_num = max(
    int(r["exp_num"])
    for r in recs
    if r["_source"] == "legacy_campaign"
)
governed_index = 0
for r in recs:
    if r["_source"] == "legacy_campaign":
        r["_chart_exp_num"] = int(r["exp_num"])
    else:
        r["_chart_exp_num"] = last_legacy_exp_num + 1 + governed_index
        governed_index += 1
    r["_chart_anchor"] = r is anchor
t0 = window_start_epoch

def short_change(r):
    if r.get("change"):
        return r["change"]
    d = r.get("desc", "") or r.get("exp", "")
    d = re.sub(r"^(refine|explore)\s+", "", d)
    d = re.sub(r"\s*\(.*?\)\s*", "", d)
    return d[:40]

baseline_bpb = ANCHOR_VAL_BPB
n_hidden = _n_all - len(recs)

best = None
for r in recs:
    r["_is_sota"] = bool(r.get("_governed_sota"))
    if r.get("val_bpb") and r["status"] not in {"crash", "invalid"}:
        if best is None or r["val_bpb"] < best - 1e-9:
            best = r["val_bpb"]
    r["_best_so_far"] = best

n_exp = len([r for r in recs if r["status"] != "baseline"])
n_ms = len([r for r in recs if r.get("_is_sota")])
n_crash = len([r for r in recs if r["status"] in {"crash", "invalid"}])
n_governed = len([r for r in recs if r["_source"] == "run_record"])
n_governed_bpb = len([
    r for r in recs
    if r["_source"] == "run_record" and isinstance(r.get("val_bpb"), (int, float))
])
improve = (baseline_bpb - best) if (baseline_bpb and best) else 0.0
updated = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
anchor_time = datetime.datetime.fromtimestamp(
    anchor_epoch, datetime.timezone.utc
).strftime("%Y-%m-%d %H:%M:%S UTC")
_shown = sorted(
    int(r["exp_num"]) for r in recs
    if r["_source"] == "legacy_campaign" and isinstance(r.get("exp_num"), (int, float))
)
if not _shown:
    window_text = "No experiments in the selected window"
elif WINDOW_START >= ANCHOR_EXP_NUM and WINDOW_END is None:
    window_text = f"Displayed series starts at experiment {ANCHOR_EXP_NUM} / {ANCHOR_VAL_BPB}"
else:
    _after = max(0, _shown[-1] - ANCHOR_EXP_NUM)
    window_text = (
        f"Window: experiments {_shown[0]}\u2013{_shown[-1]} "
        f"({len(_shown)} points) \u00b7 the run-up to 0.927183 and {_after} experiments after it"
    )

data = [{
    "seq": r["_chart_exp_num"], "exp": r.get("exp"),
    "t_min": round((epoch(r) - t0) / 60.0, 2) if epoch(r) and t0 is not None else None,
    "val_bpb": r.get("val_bpb"), "best": r.get("_best_so_far"),
    "status": r.get("status"), "phase": r.get("phase", ""), "steps": r.get("steps"),
    "desc": r.get("desc", ""), "change": short_change(r),
    "milestone": (
        "0.927183 historical chart origin"
        if r.get("_chart_anchor")
        else (
            r.get("milestone", "") or "post-origin governed SOTA"
            if r.get("_is_sota")
            else ""
        )
    ),
    "sota": r.get("_is_sota"),
    "anchor": r.get("_chart_anchor", False),
    "source": r["_source"], "run_id": r.get("run_id", ""),
    "experiment_id": r.get("experiment_id", ""), "arm_id": r.get("arm_id", ""),
    "role": r.get("role", ""), "seed": r.get("seed"),
} for r in recs]

payload = json.dumps(data)
tmpl = r"""<!-- generated -->
<title>Autoresearch Reimpl — val_bpb campaign</title>
<style>
:root{
 --bg:#f6f7f9; --card:#ffffff; --line:#e7eaef; --ink:#111725; --ink2:#5b6675; --ink3:#8a94a4;
 --accent:#0e9f6e; --accentsoft:rgba(14,159,110,.14); --base:#e02424; --dot:#94a3b8; --keep:#3b6ef5; --governed:#7c3aed;
 --tt-bg:#0f1626; --tt-fg:#f3f5f8; --tt-accent:#5eead4;
}
@media (prefers-color-scheme:dark){:root{
 --bg:#0a0f1a; --card:#111927; --line:#20293a; --ink:#eef2f7; --ink2:#9aa6b6; --ink3:#61708a;
 --accent:#34d399; --accentsoft:rgba(52,211,153,.16); --base:#f76d6d; --dot:#5c6b83; --keep:#6aa0ff; --governed:#c4b5fd;
 --tt-bg:#1b2536; --tt-fg:#f3f5f8; --tt-accent:#5eead4;
}}
:root[data-theme=dark]{
 --bg:#0a0f1a; --card:#111927; --line:#20293a; --ink:#eef2f7; --ink2:#9aa6b6; --ink3:#61708a;
 --accent:#34d399; --accentsoft:rgba(52,211,153,.16); --base:#f76d6d; --dot:#5c6b83; --keep:#6aa0ff; --governed:#c4b5fd;
 --tt-bg:#1b2536; --tt-fg:#f3f5f8; --tt-accent:#5eead4;}
:root[data-theme=light]{
 --bg:#f6f7f9; --card:#ffffff; --line:#e7eaef; --ink:#111725; --ink2:#5b6675; --ink3:#8a94a4;
 --accent:#0e9f6e; --accentsoft:rgba(14,159,110,.14); --base:#e02424; --dot:#94a3b8; --keep:#3b6ef5; --governed:#7c3aed;
 --tt-bg:#0f1626; --tt-fg:#f3f5f8; --tt-accent:#5eead4;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 48px}
h1{font-size:19px;font-weight:660;letter-spacing:-.01em;margin:0 0 3px}
.sub{color:var(--ink2);font-size:12.5px;margin-bottom:22px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:24px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.tile .v{font-size:24px;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.tile .k{color:var(--ink3);font-size:11.5px;margin-top:2px;text-transform:uppercase;letter-spacing:.04em}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 18px 10px;margin-bottom:18px}
.card h2{font-size:13px;font-weight:600;margin:0 0 2px;color:var(--ink)}
.card .csub{font-size:11.5px;color:var(--ink3);margin-bottom:6px}
.chart{position:relative;width:100%;overflow:visible}.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart svg circle.pt{cursor:pointer}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:10px 4px 2px;font-size:11.5px;color:var(--ink2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.dotm{width:9px;height:9px;border-radius:50%;display:inline-block}.linem{width:16px;height:0;border-top:2px solid;display:inline-block}
#tt{position:fixed;z-index:60;pointer-events:none;opacity:0;transform:translateY(2px);transition:opacity .1s,transform .1s;
 background:var(--tt-bg);color:var(--tt-fg);border-radius:10px;padding:10px 12px;font-size:12px;min-width:170px;max-width:290px;
 box-shadow:0 8px 30px rgba(2,8,20,.35);line-height:1.5}
#tt .e{font-weight:680;font-size:12.5px;margin-bottom:1px}
#tt .c{color:var(--tt-accent);margin-bottom:7px;font-weight:600;font-size:11.5px}
#tt .r{display:flex;justify-content:space-between;gap:16px}#tt .k{opacity:.62}#tt b{font-variant-numeric:tabular-nums;font-weight:620}
</style>
<div class=wrap>
<h1>Autoresearch reimplementation — val&#95;bpb campaign</h1>
<div class=sub>__WINDOW_TEXT__ · elapsed time is measured from the first displayed experiment · the 0.927183 historical chart origin (exp 501, __ANCHOR_TIME__) is drawn as a dashed reference · updated __UPDATED__ · hover any point for details</div>
<div class=tiles>
<div class=tile><div class=v>__BASELINE__</div><div class=k>historical chart origin</div></div>
<div class=tile><div class=v style="color:var(--accent)">__BEST__</div><div class=k>best displayed val_bpb</div></div>
<div class=tile><div class=v style="color:var(--accent)">__IMPROVE__</div><div class=k>delta below origin</div></div>
<div class=tile><div class=v>__NEXP__</div><div class=k>post-origin experiments / runs<span style="opacity:.65"> · __NHID__ pre-origin points omitted</span></div></div>
<div class=tile><div class=v style="color:var(--governed)">__NGOV__</div><div class=k>governed RunRecords<span style="opacity:.65"> · __NGOVBPB__ with val_bpb</span></div></div>
<div class=tile><div class=v>__NMS__</div><div class=k>post-origin governed SOTA milestones</div></div>
<div class=tile><div class=v>__NCRASH__</div><div class=k>failed / invalid</div></div>
</div>
<div class=card><h2>val&#95;bpb over time</h2><div class=csub>lower is better · minutes since the first displayed experiment</div>
<div class=chart id=chartTime></div>
<div class=legend><span><span class=linem style="border-color:var(--accent)"></span>running best in window</span><span><span class=dotm style="background:var(--governed)"></span>governed RunRecord</span><span><span class=dotm style="background:var(--keep)"></span>legacy kept experiment</span><span><span class=dotm style="background:var(--dot)"></span>legacy discarded / pilot</span><span><span class=dotm style="background:var(--accent)"></span>governed SOTA</span><span><span class=linem style="border-color:var(--base);border-top-style:dashed"></span>0.927183 historical chart origin</span></div>
</div>
<div class=card><h2>val&#95;bpb by experiment / run</h2><div class=csub>experiment 501 onward, with governed runs in chronological sequence</div>
<div class=chart id=chartExp></div>
<div class=legend><span><span class=dotm style="background:var(--governed)"></span>governed RunRecord</span><span><span class=dotm style="background:var(--keep)"></span>legacy kept experiment</span><span><span class=dotm style="background:var(--dot)"></span>legacy discarded / pilot</span><span><span class=dotm style="background:var(--accent)"></span>governed SOTA</span><span><span class=dotm style="background:var(--base)"></span>0.927183 historical chart origin</span></div>
</div>
</div>
<div id=tt></div>
<script>
const DATA=__PAYLOAD__;
const BASE=__BASELINE_NUM__;
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim()}
function esc(s){return (s||'').toString().replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function att(s){return esc(s).replace(/"/g,'&quot;')}
function build(xkey,xlabel){
 const W=1000,H=380,m={l:64,r:120,t:26,b:38};
 const pts=DATA.filter(d=>d.val_bpb&&d[xkey]!=null&&d.status!=='crash'&&d.status!=='invalid');
 if(!pts.length) return '<svg viewBox="0 0 '+W+' '+H+'"><text x="500" y="190" text-anchor="middle" fill="'+css('--ink3')+'">no data yet</text></svg>';
 const xs=pts.map(d=>d[xkey]),ys=pts.map(d=>d.val_bpb);
 let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 if(BASE){ymax=Math.max(ymax,BASE);ymin=Math.min(ymin,BASE)}
 const yr=(ymax-ymin)||0.02;ymin-=yr*0.12;ymax+=yr*0.12;if(xmax===xmin){xmax=xmin+1;xmin-=1}
 const xpad=(xmax-xmin)*0.02;xmin-=xpad;xmax+=xpad;
 const X=v=>m.l+(v-xmin)/(xmax-xmin)*(W-m.l-m.r);
 const Y=v=>m.t+(1-(v-ymin)/(ymax-ymin))*(H-m.t-m.b);
 const INK3=css('--ink3'),LINE=css('--line'),ACC=css('--accent'),BASEC=css('--base'),DOT=css('--dot'),KEEP=css('--keep'),GOV=css('--governed'),BG=css('--card'),INK=css('--ink'),INK2=css('--ink2');
 let s='<svg viewBox="0 0 '+W+' '+H+'" preserveAspectRatio="xMidYMid meet" font-family="inherit">';
 s+='<defs><linearGradient id="ag" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="'+ACC+'" stop-opacity="0.16"/><stop offset="1" stop-color="'+ACC+'" stop-opacity="0"/></linearGradient></defs>';
 // horizontal gridlines only (recessive)
 const nT=4;for(let i=0;i<=nT;i++){const yv=ymin+(ymax-ymin)*i/nT,y=Y(yv);
  s+='<line x1="'+m.l+'" y1="'+y.toFixed(1)+'" x2="'+(W-m.r)+'" y2="'+y.toFixed(1)+'" stroke="'+LINE+'" stroke-width="1"/>';
  s+='<text x="'+(m.l-10)+'" y="'+(y+4).toFixed(1)+'" text-anchor="end" font-size="11" fill="'+INK3+'">'+yv.toFixed(3)+'</text>';}
 const nX=5;for(let i=0;i<=nX;i++){const xv=xmin+(xmax-xmin)*i/nX,x=X(xv);
  s+='<text x="'+x.toFixed(1)+'" y="'+(H-m.b+18)+'" text-anchor="middle" font-size="11" fill="'+INK3+'">'+xv.toFixed(0)+'</text>';}
 s+='<text x="'+((m.l+W-m.r)/2)+'" y="'+(H-4)+'" text-anchor="middle" font-size="11" fill="'+INK3+'">'+xlabel+'</text>';
 // best-so-far step path + area fill
 const bpts=DATA.filter(d=>d.best&&d[xkey]!=null).sort((a,b)=>a[xkey]-b[xkey]);
 if(bpts.length){let dstr='',prev=null,first=null,lastx=null;
  bpts.forEach(d=>{const x=X(d[xkey]),y=Y(d.best);if(prev===null){dstr+='M'+x+' '+y;first=x;}else{dstr+=' L'+x+' '+prev+' L'+x+' '+y;}prev=y;lastx=x;});
  const area=dstr+' L'+lastx+' '+Y(ymin)+' L'+first+' '+Y(ymin)+' Z';
  s+='<path d="'+area+'" fill="url(#ag)" stroke="none"/>';
  s+='<path d="'+dstr+'" fill="none" stroke="'+ACC+'" stroke-width="2.5" stroke-linejoin="round"/>';}
 // historical chart-origin reference
 if(BASE){const y=Y(BASE);s+='<line x1="'+m.l+'" y1="'+y.toFixed(1)+'" x2="'+(W-m.r)+'" y2="'+y.toFixed(1)+'" stroke="'+BASEC+'" stroke-dasharray="4 4" stroke-width="1.4" opacity="0.85"/>';
  s+='<text x="'+(W-m.r+6)+'" y="'+(y+4).toFixed(1)+'" font-size="10.5" fill="'+BASEC+'">historical origin '+BASE.toFixed(6)+'</text>';}
 // points
 pts.forEach(d=>{const x=X(d[xkey]),y=Y(d.val_bpb);const isBase=d.anchor;
  const c=isBase?BASEC:(d.sota?ACC:((d.source==='run_record')?GOV:((d.status==='keep')?KEEP:DOT)));const r=(d.sota||isBase)?5.5:3.4;
  s+='<circle cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="'+r+'" fill="'+c+'" stroke="'+BG+'" stroke-width="1.2"/>';
  s+='<circle class="pt" cx="'+x.toFixed(1)+'" cy="'+y.toFixed(1)+'" r="12" fill="transparent" '
    +'data-exp="'+att(d.exp)+'" data-change="'+att(d.change||d.desc)+'" data-bpb="'+d.val_bpb+'" '
    +'data-steps="'+(d.steps||'')+'" data-status="'+att(d.status)+'" data-ms="'+att(d.milestone||'')+'" '
    +'data-source="'+att(d.source)+'" data-role="'+att(d.role)+'" data-arm="'+att(d.arm_id)+'" data-seed="'+att(d.seed)+'"/>';});
 // milestone callouts with connector
 const seen=[];pts.filter(d=>d.sota).forEach(d=>{const x=X(d[xkey]),y=Y(d.val_bpb);
  let ly=y-22;while(seen.some(v=>Math.abs(v-ly)<14)){ly-=15;}seen.push(ly);
  const anchor=(x>W-m.r-140)?'end':'start';const lx=anchor==='end'?x-8:x+8;
  s+='<line x1="'+x+'" y1="'+(y-6)+'" x2="'+x+'" y2="'+(ly+3)+'" stroke="'+ACC+'" stroke-width="1"/>';
  s+='<text x="'+lx.toFixed(1)+'" y="'+ly.toFixed(1)+'" font-size="11" font-weight="650" text-anchor="'+anchor+'" fill="'+INK+'">'+esc(d.milestone)+'</text>';
  s+='<text x="'+lx.toFixed(1)+'" y="'+(ly+12).toFixed(1)+'" font-size="10" text-anchor="'+anchor+'" fill="'+INK2+'">'+d.val_bpb.toFixed(4)+'</text>';});
 s+='</svg>';return s;
}
const tt=document.getElementById('tt');
function showTip(c,ev){
 const bpb=parseFloat(c.getAttribute('data-bpb'));const dl=BASE?(BASE-bpb):null;const ms=c.getAttribute('data-ms');
 const dcol=dl>=0?css('--accent'):css('--base');
 let h='<div class="e">'+esc(c.getAttribute('data-exp'))+'</div>';
 h+='<div class="c">'+esc(c.getAttribute('data-change'))+(ms?(' · ★ '+esc(ms)):'')+'</div>';
 h+='<div class="r"><span class="k">val_bpb</span><b>'+bpb.toFixed(6)+'</b></div>';
 if(dl!==null)h+='<div class="r"><span class="k">vs historical chart origin</span><b style="color:'+dcol+'">'+(dl>=0?'−':'+')+Math.abs(dl).toFixed(4)+'</b></div>';
 h+='<div class="r"><span class="k">steps</span><b>'+(c.getAttribute('data-steps')||'–')+'</b></div>';
 h+='<div class="r"><span class="k">status</span><b>'+esc(c.getAttribute('data-status'))+'</b></div>';
 h+='<div class="r"><span class="k">source</span><b>'+esc(c.getAttribute('data-source'))+'</b></div>';
 if(c.getAttribute('data-role'))h+='<div class="r"><span class="k">role / arm</span><b>'+esc(c.getAttribute('data-role'))+' / '+esc(c.getAttribute('data-arm'))+'</b></div>';
 if(c.getAttribute('data-seed'))h+='<div class="r"><span class="k">seed</span><b>'+esc(c.getAttribute('data-seed'))+'</b></div>';
 tt.innerHTML=h;tt.style.opacity='1';tt.style.transform='translateY(0)';moveTip(ev);
}
function moveTip(ev){const pad=16;let x=ev.clientX+pad,y=ev.clientY+pad;const w=tt.offsetWidth,hh=tt.offsetHeight;
 if(x+w>innerWidth-8)x=ev.clientX-w-pad;if(y+hh>innerHeight-8)y=ev.clientY-hh-pad;tt.style.left=x+'px';tt.style.top=y+'px';}
function hideTip(){tt.style.opacity='0';tt.style.transform='translateY(2px)';}
function wire(id){const el=document.getElementById(id);
 el.addEventListener('pointerover',e=>{const c=e.target.closest('circle.pt');if(c)showTip(c,e);});
 el.addEventListener('pointermove',e=>{if(tt.style.opacity==='1')moveTip(e);});
 el.addEventListener('pointerout',e=>{if(e.target.closest('circle.pt'))hideTip();});}
function redraw(){document.getElementById('chartTime').innerHTML=build('t_min','minutes since 0.927183 historical chart origin');
 document.getElementById('chartExp').innerHTML=build('seq','experiment / run number (starts at 501)');wire('chartTime');wire('chartExp');}
redraw();
new MutationObserver(redraw).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});
</script>
"""
out = (tmpl
    .replace("__UPDATED__", html.escape(updated))
    .replace("__WINDOW_TEXT__", html.escape(window_text))
    .replace("__ANCHOR_TIME__", html.escape(anchor_time))
    .replace("__BASELINE__", f"{baseline_bpb:.4f}" if baseline_bpb else "—")
    .replace("__BEST__", f"{best:.4f}" if best else "—")
    .replace("__IMPROVE__", f"{improve:+.4f}" if improve else "—")
    .replace("__NEXP__", str(n_exp))
    .replace("__NHID__", str(n_hidden))
    .replace("__NGOV__", str(n_governed))
    .replace("__NGOVBPB__", str(n_governed_bpb))
    .replace("__NMS__", str(n_ms))
    .replace("__NCRASH__", str(n_crash))
    .replace("__PAYLOAD__", payload)
    .replace("__BASELINE_NUM__", f"{baseline_bpb}" if baseline_bpb else "null"))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(out)
print(f"wrote {OUT} ({n_exp} exp, {n_ms} milestones, best={best}, baseline={baseline_bpb}, improve={improve:+.4f})")
