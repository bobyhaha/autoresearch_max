#!/usr/bin/env python3
"""Per-challenge val_bpb progress report for one catalogued decision frame.

Challenge IDs, order, scopes, budgets, and the sticky default selection come
from the structured challenge/setup records. This tool never mixes challenges.

Challenge membership of a run:
  * a run tagged `challenge_<id>` belongs to that challenge;
  * legacy `frame_<id>` tags are resolved through the catalog;
  * an untagged legacy run belongs to the catalog entry for the top-level scope;
  * a run tagged `diagnostic_offbudget_*` belongs to NO frame -- it deliberately
    varied compute and can never be an adopt input.

Usage:
  python tools/frame_progress.py
  python tools/frame_progress.py --challenge fixed_steps_2000 --html out.html
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibeautoresearch.registry import ResearchRegistry  # noqa: E402


def _load(rel: str) -> list[dict]:
    path = ROOT / "research" / rel
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("{")
    ]


def challenge_of(
    run: dict, *, decision_to_challenge: dict[str, str], primary_challenge: str
) -> str | None:
    """Return the challenge a run belongs to, or None for diagnostics."""
    tags = run.get("tags") or []
    for tag in tags:
        if tag.startswith("diagnostic_offbudget"):
            return None
        if tag.startswith("challenge_"):
            return tag[len("challenge_") :]
        if tag.startswith("frame_"):
            return decision_to_challenge.get(tag[len("frame_") :])
    return primary_challenge


def collect(
    challenge: str,
    *,
    decision_to_challenge: dict[str, str],
    primary_challenge: str,
) -> dict:
    runs = _load("experiments/runs/runs.jsonl")
    experiments = {e["experiment_id"]: e for e in _load("experiments/gated/experiments.jsonl")}
    interventions = {i["intervention_id"]: i for i in _load("toolkit/available/interventions.jsonl")}

    kept, excluded = [], {"other_frame": 0, "no_frame": 0, "incomplete": 0, "no_metric": 0}
    for run in runs:
        if run.get("status") != "complete":
            excluded["incomplete"] += 1
            continue
        if "val_bpb" not in (run.get("outcome_values") or {}):
            excluded["no_metric"] += 1
            continue
        rf = challenge_of(
            run,
            decision_to_challenge=decision_to_challenge,
            primary_challenge=primary_challenge,
        )
        if rf is None:
            excluded["no_frame"] += 1
            continue
        if rf != challenge:
            excluded["other_frame"] += 1
            continue
        kept.append(run)

    by_exp: dict[str, dict[str, list]] = {}
    for run in kept:
        slot = by_exp.setdefault(run["experiment_id"], {"treatment": [], "baseline": [], "when": []})
        slot.setdefault(run.get("role", "treatment"), []).append(run["outcome_values"]["val_bpb"])
        slot["when"].append(run.get("started_at", ""))

    points, best = [], None
    for eid, slot in sorted(by_exp.items(), key=lambda kv: min(kv[1]["when"] or [""])):
        treat = slot.get("treatment") or []
        base = slot.get("baseline") or []
        if not treat:
            continue
        t_mean = st.mean(treat)
        b_mean = st.mean(base) if base else None
        best = t_mean if best is None else min(best, t_mean)
        exp = experiments.get(eid, {})
        arm = next((a for a in exp.get("arms", []) if a.get("role") == "treatment"), {})
        iv = arm.get("intervention_id", "")
        points.append(
            {
                "i": len(points) + 1,
                "experiment_id": eid,
                "intervention_id": iv,
                "lever": (interventions.get(iv, {}) or {}).get("description", "")[:110]
                or exp.get("title", ""),
                "stage": exp.get("stage", ""),
                "seeds": len(treat),
                "val_bpb": round(t_mean, 6),
                "baseline": round(b_mean, 6) if b_mean is not None else None,
                "delta": round(t_mean - b_mean, 6) if b_mean is not None else None,
                "running_best": round(best, 6),
            }
        )
    return {
        "frame": challenge,
        "points": points,
        "excluded": excluded,
        "n_runs_used": len(kept),
    }


def render_html(report: dict, floor: float, baseline: float | None) -> str:
    """Standalone chart. Frame is stated in the title so it can never be misread."""
    pts = report["points"]
    frame = report["frame"]
    if not pts:
        body = (
            f"<p class='empty'>No completed runs recorded in frame <code>{frame}</code> yet. "
            "Once the baseline is measured and experiments run, this chart fills in.</p>"
        )
        return _PAGE.replace("__FRAME__", frame).replace("__BODY__", body).replace("__DATA__", "[]")
    return (
        _PAGE.replace("__FRAME__", frame)
        .replace("__BODY__", "<div id='chart'></div><div class='tip' id='tip'></div>")
        .replace("__DATA__", json.dumps(pts, separators=(",", ":")))
        .replace("__FLOOR__", repr(floor))
        .replace("__BASE__", repr(baseline if baseline is not None else 0))
    )


_PAGE = """<title>OPHIS — val_bpb progress · __FRAME__</title>
<style>
:root{--bg:#fcfcfb;--panel:#fff;--ink:#0b0b0b;--soft:#52514e;--muted:#7a7873;--line:#e0dfda;
--win:#2a78d6;--hurt:#eb6834;--mid:#b9b7b1;--band:#e8e7e3;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
@media(prefers-color-scheme:dark){:root:where(:not([data-theme=light])){--bg:#1a1a19;--panel:#212120;--ink:#fff;
--soft:#c3c2b7;--muted:#8e8d85;--line:#33332f;--win:#3987e5;--hurt:#d95926;--mid:#6f6e68;--band:#2c2c29}}
:root[data-theme=dark]{--bg:#1a1a19;--panel:#212120;--ink:#fff;--soft:#c3c2b7;--muted:#8e8d85;
--line:#33332f;--win:#3987e5;--hurt:#d95926;--mid:#6f6e68;--band:#2c2c29}
body{margin:0;padding:30px 26px 60px;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.6}
.wrap{max-width:1040px;margin:0 auto}
.eyebrow{font-family:var(--mono);font-size:10.5px;letter-spacing:.17em;text-transform:uppercase;color:var(--muted);font-weight:600}
h1{font-size:30px;letter-spacing:-.028em;margin:8px 0 10px;font-weight:700}
.frametag{display:inline-block;font-family:var(--mono);font-size:11px;padding:3px 10px;border-radius:20px;
border:1px solid var(--win);color:var(--win);font-weight:600;margin-bottom:14px}
p{color:var(--soft);max-width:70ch}
.box{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px 14px;margin:20px 0;position:relative}
svg{width:100%;height:auto;display:block;overflow:visible}
.gl{stroke:var(--line);stroke-width:1}
.axl{fill:var(--muted);font-family:var(--mono);font-size:10px}
.sota{fill:none;stroke:var(--ink);stroke-width:2}
.base{stroke:var(--muted);stroke-width:1.5;stroke-dasharray:5 4}
.pt{stroke:var(--panel);stroke-width:2;cursor:pointer}
.tip{position:absolute;opacity:0;pointer-events:none;background:var(--panel);border:1px solid var(--line);
border-radius:7px;padding:10px 12px;font-size:12.4px;max-width:300px;box-shadow:0 6px 22px rgba(0,0,0,.16)}
.tip b{font-family:var(--mono)}
.empty{font-style:italic;color:var(--muted)}
code{font-family:var(--mono);font-size:.87em}
table{border-collapse:collapse;width:100%;font-size:12.6px;margin-top:8px}
th{text-align:left;font-family:var(--mono);font-size:9.5px;letter-spacing:.1em;text-transform:uppercase;
color:var(--muted);padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:6px 10px;border-bottom:1px solid var(--line)}
td.n{text-align:right;font-family:var(--mono)}
</style>
<div class="wrap">
<div class="eyebrow">OPHIS · decision frame</div>
<h1>val_bpb progress</h1>
<div class="frametag">frame: __FRAME__</div>
<p>Only runs recorded in this frame are shown. Results from the other decision frame are
excluded by construction — the two answer different questions and are never compared.</p>
<div class="box">__BODY__</div>
<div class="box"><table id="tbl"></table></div>
</div>
<script>
var D=__DATA__, FLOOR=__FLOOR__, BASE=__BASE__;
if(D.length){
 var W=980,H=400,L=64,R=24,T=20,B=54,iw=W-L-R,ih=H-T-B;
 var vs=D.map(function(p){return p.val_bpb}).concat(BASE?[BASE]:[]);
 var lo=Math.min.apply(null,vs),hi=Math.max.apply(null,vs),pad=(hi-lo)*0.15||0.002;
 var y0=lo-pad,y1=hi+pad;
 var xs=function(i){return D.length<2?L+iw/2:L+(i-1)/(D.length-1)*iw};
 var ys=function(v){return T+(1-(v-y0)/(y1-y0))*ih};
 var col=function(d){return d===null?'var(--mid)':(d<=-FLOOR?'var(--win)':(d>=FLOOR?'var(--hurt)':'var(--mid)'))};
 var s=['<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="val_bpb by experiment in frame __FRAME__">'];
 if(BASE){s.push('<rect x="'+L+'" y="'+ys(BASE+FLOOR)+'" width="'+iw+'" height="'+(ys(BASE-FLOOR)-ys(BASE+FLOOR))+'" fill="var(--band)"/>');
  s.push('<line class="base" x1="'+L+'" y1="'+ys(BASE)+'" x2="'+(L+iw)+'" y2="'+ys(BASE)+'"/>');}
 for(var k=0;k<=4;k++){var v=y0+(y1-y0)*k/4,yy=ys(v);
  s.push('<line class="gl" x1="'+L+'" y1="'+yy+'" x2="'+(L+iw)+'" y2="'+yy+'"/>');
  s.push('<text class="axl" x="'+(L-8)+'" y="'+(yy+3.5)+'" text-anchor="end">'+v.toFixed(4)+'</text>');}
 var d='';D.forEach(function(p,k){var x=xs(p.i),y=ys(p.running_best);
  d+=k===0?'M'+x+','+y:'L'+x+','+ys(D[k-1].running_best)+'L'+x+','+y;});
 s.push('<path class="sota" d="'+d+'"/>');
 D.forEach(function(p){s.push('<circle class="pt" data-i="'+p.i+'" cx="'+xs(p.i)+'" cy="'+ys(p.val_bpb)+'" r="5.5" fill="'+col(p.delta)+'"/>');
  if(p.i===1||p.i%5===0)s.push('<text class="axl" x="'+xs(p.i)+'" y="'+(T+ih+18)+'" text-anchor="middle">'+p.i+'</text>');});
 s.push('</svg>');
 document.getElementById('chart').innerHTML=s.join('');
 var tip=document.getElementById('tip'),box=document.querySelector('.box');
 box.addEventListener('mousemove',function(e){var t=e.target.closest('[data-i]');
  if(!t){tip.style.opacity=0;return;}var p=D[+t.getAttribute('data-i')-1];
  tip.innerHTML='<div><b>'+p.i+'.</b> '+p.lever+'</div><div>val_bpb <b>'+p.val_bpb.toFixed(6)+'</b></div>'+
   (p.delta!==null?'<div>Δ <b>'+(p.delta>0?'+':'')+p.delta.toFixed(6)+'</b></div>':'')+
   '<div>seeds <b>'+p.seeds+'</b> · '+p.stage+'</div>';
  var r=box.getBoundingClientRect();tip.style.left=(e.clientX-r.left+14)+'px';
  tip.style.top=(e.clientY-r.top+14)+'px';tip.style.opacity=1;});
 box.addEventListener('mouseleave',function(){tip.style.opacity=0});
 document.getElementById('tbl').innerHTML='<tr><th>#</th><th>lever</th><th>val_bpb</th><th>Δ</th><th>seeds</th></tr>'+
  D.map(function(p){return '<tr><td class="n">'+p.i+'</td><td>'+p.lever+'</td><td class="n">'+p.val_bpb.toFixed(6)+
  '</td><td class="n">'+(p.delta===null?'—':(p.delta>0?'+':'')+p.delta.toFixed(6))+'</td><td class="n">'+p.seeds+'</td></tr>';}).join('');
}
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--challenge",
        default=None,
        help="challenge ID/alias; omit for the sticky active challenge",
    )
    ap.add_argument("--html", default="", help="write a standalone chart here")
    ap.add_argument("--json", default="", help="write the raw report here")
    args = ap.parse_args()

    registry = ResearchRegistry(ROOT / "research")
    setup = registry.setup_reconciliation()
    catalog = registry.challenge_catalog()
    selected = (
        catalog.resolve(registry.challenge_events()[-1].challenge_id)
        if args.challenge is None
        else catalog.resolve(args.challenge)
    )
    challenge_id = str(selected["challenge_id"])
    scope_id = str(selected["scope_id"])
    decision_to_challenge = {
        str(entry["decision_frame"]): str(entry["challenge_id"])
        for entry in catalog.challenges
    }
    primary_challenge = str(catalog.resolve("")["challenge_id"])
    if not scope_id:
        floor = (
            setup.baseline["effective_sigma"]
            * setup.baseline["tolerance_sigma"]
        )
        baseline = setup.baseline["expected"]
    else:
        entry = setup.scope_for(scope_id)
        base = entry.get("baseline") or {}
        if not base:
            print(
                json.dumps(
                    {
                        "frame": challenge_id,
                        "status": entry.get("status"),
                        "note": "frame has no measured baseline yet -- no floor to judge against",
                    },
                    indent=2,
                )
            )
        floor = base.get("effective_sigma", 0) * base.get("tolerance_sigma", 2) or 0.0
        baseline = base.get("expected")

    report = collect(
        challenge_id,
        decision_to_challenge=decision_to_challenge,
        primary_challenge=primary_challenge,
    )
    report["floor"] = floor
    report["frame_baseline"] = baseline
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.html:
        Path(args.html).write_text(render_html(report, floor, baseline), encoding="utf-8")
    print(
        json.dumps(
            {
                "frame": report["frame"],
                "experiments": len(report["points"]),
                "runs_used": report["n_runs_used"],
                "excluded": report["excluded"],
                "baseline": baseline,
                "floor": floor,
                "html": args.html or None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
