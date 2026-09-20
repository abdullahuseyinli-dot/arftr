"""Build a label-free review index from the frozen body-witness crop manifest."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = ROOT / ".runs/research_20260913/body_witness_pilot_v1"


def build(run: Path) -> str:
    rows = json.loads((run / "crop_manifest.json").read_text(encoding="utf-8"))
    if len(rows) != 128 or any(set(row["crops"]) != {"1.0", "1.25"} for row in rows):
        raise RuntimeError("Review index requires all 256 frozen crops")
    cards = []
    for row in rows:
        sample = html.escape(row["sample_id"])
        images = []
        for extent, caption in (("1.0", "primary 1.0"), ("1.25", "stability 1.25")):
            relative = html.escape(row["crops"][extent]["file"].removeprefix("blinded_review/"))
            images.append(f"<figure><img src='{relative}'><figcaption>{caption}</figcaption></figure>")
        cards.append(
            f"<section><h2>{int(row['selection_index']):03d} {sample}</h2>"
            + "".join(images)
            + "</section>"
        )
    return """<!doctype html><meta charset='utf-8'><title>Blinded body review</title>
<style>body{font:14px system-ui;background:#eee}section{background:white;margin:1rem;padding:1rem}
figure{display:inline-block;margin:.5rem}img{max-width:520px;min-width:260px;image-rendering:auto}
h2{font-size:14px}figcaption{text-align:center}</style>
<h1>Blinded body-landmark feasibility review</h1>
<p>Reviewers work independently. Do not consult action labels, ARFTR outputs, error membership,
or pose estimates. Mark visibility and primary-crop pixel coordinates in the assigned CSV. A hidden
or unlocalizable landmark receives visible=0 and blank coordinates. Resolve disagreements before
pose estimates are revealed.</p>
""" + "\n".join(cards)


def build_interactive(run: Path) -> str:
    """Build a label-blind click-to-annotate review app.

    Review state remains inside the reviewer's browser until they export the
    assigned CSV.  Only sample identifiers and frozen crop paths enter the app.
    """

    manifest = json.loads((run / "crop_manifest.json").read_text(encoding="utf-8"))
    if len(manifest) != 128 or any(set(row["crops"]) != {"1.0", "1.25"} for row in manifest):
        raise RuntimeError("Interactive review requires all 256 frozen crops")
    rows = [
        {
            "selection_index": int(row["selection_index"]),
            "sample_id": row["sample_id"],
            "primary": row["crops"]["1.0"]["file"].removeprefix("blinded_review/"),
            "stability": row["crops"]["1.25"]["file"].removeprefix("blinded_review/"),
        }
        for row in manifest
    ]
    row_json = json.dumps(rows, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Blinded body-witness review</title>
<style>
:root{font:15px system-ui;color:#18212b;background:#eef1f4}body{margin:0}.top{position:sticky;top:0;z-index:5;background:#18212b;color:white;padding:.65rem 1rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap}.top button,.top select{font:inherit}.app{max-width:1450px;margin:auto;padding:1rem}.warning{background:#fff5cc;border-left:5px solid #c88a00;padding:.8rem;margin-bottom:1rem}.grid{display:grid;grid-template-columns:minmax(380px,1fr) minmax(300px,.75fr);gap:1rem}.panel{background:white;border-radius:8px;padding:1rem;box-shadow:0 2px 10px #0001}.image-wrap{position:relative;display:block}.image-wrap img{display:block;width:100%;height:100%;image-rendering:auto}.image-wrap canvas{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}.stability img{width:min(100%,360px);height:auto;max-height:36vh;image-rendering:auto}.joints{display:grid;grid-template-columns:repeat(2,minmax(170px,1fr));gap:.45rem}.joint{display:grid;grid-template-columns:1fr auto;gap:.35rem;align-items:center;border:2px solid #dce2e8;border-radius:6px;padding:.45rem;background:#f8fafb}.joint.active{border-color:#1565c0;background:#e8f2ff}.joint.done{box-shadow:inset 5px 0 #2e7d32}.joint.hidden{box-shadow:inset 5px 0 #777}.joint button{font:inherit}.actions{display:flex;gap:.5rem;flex-wrap:wrap;margin:.8rem 0}.actions button{font:inherit;padding:.5rem .7rem}.primary-action{background:#1565c0;color:white;border:0;border-radius:4px}.complete{background:#2e7d32;color:white;border:0;border-radius:4px}.complete:disabled{background:#9aa4ad}.status{font-weight:650}.small{font-size:.87rem;color:#53606d}textarea{width:100%;box-sizing:border-box;min-height:55px}kbd{background:#eee;border:1px solid #bbb;border-radius:3px;padding:0 .2rem}@media(max-width:900px){.grid{grid-template-columns:1fr}.joints{grid-template-columns:1fr}}
</style></head><body>
<div class="top"><strong>Blinded review</strong><label>Reviewer <select id="reviewer"><option value="reviewer_a">A</option><option value="reviewer_b">B</option></select></label><span id="progress"></span><button id="previous">Previous</button><button id="next">Next</button><button id="export">Export assigned CSV</button></div>
<main class="app"><div class="warning"><strong>Stay blinded:</strong> do not open action labels, ARFTR outputs, error membership, or pose estimates. Reviewers A and B must work independently. Click landmarks only on the <strong>primary 1.0 crop</strong>; use 1.25 only to judge stability. This geometry-corrected reviewer uses a new browser-state namespace.</div>
<h1 id="title"></h1><div class="grid"><section class="panel"><h2>Primary 1.0 — click here</h2><div class="image-wrap"><img id="primary" alt="Primary actor crop"><canvas id="overlay"></canvas></div><p class="small">Select a landmark at right, then click its location. Coordinates are stored in original crop pixels.</p></section>
<section class="panel"><h2>Landmarks</h2><p id="activeHint" class="status"></p><div id="joints" class="joints"></div><div class="actions"><button id="notVisible">Mark selected not visible</button><button id="clear">Clear selected</button></div><label>Notes<textarea id="notes"></textarea></label><div class="actions"><button id="complete" class="complete">Mark example complete</button></div><p id="completeHint" class="small"></p><div class="stability"><h2>Stability reference 1.25 — do not click</h2><img id="stability" alt="Wider stability crop"></div></section></div>
<p class="small">Keyboard: <kbd>←</kbd>/<kbd>→</kbd> previous/next; number keys 1–8 select a landmark. Browser-local progress is separated by reviewer. Export frequently.</p></main>
<script>
const rows=""" + row_json + """;
const landmarks=[['left_shoulder','#e53935'],['right_shoulder','#d81b60'],['left_hip','#8e24aa'],['right_hip','#5e35b1'],['left_knee','#3949ab'],['right_knee','#1e88e5'],['left_ankle','#00897b'],['right_ankle','#43a047']];
const fields=['sample_id','reviewer_id','review_complete','notes',...landmarks.flatMap(([n])=>[n+'_visible',n+'_x_crop',n+'_y_crop'])];
let reviewer='reviewer_a',position=0,active=0,state={};
const $=id=>document.getElementById(id); const canvas=$('overlay'),ctx=canvas.getContext('2d');
function key(){return 'hac-body-witness-review-v2-'+reviewer}
function load(){try{state=JSON.parse(localStorage.getItem(key())||'{}')}catch{state={}}}
function save(){localStorage.setItem(key(),JSON.stringify(state))}
function record(){const id=rows[position].sample_id;if(!state[id])state[id]={review_complete:false,notes:'',marks:{}};return state[id]}
function mark(name){return record().marks[name]||{visible:null,x:'',y:''}}
function completed(){return rows.filter(r=>state[r.sample_id]?.review_complete).length}
function fitPrimary(){const img=$('primary'),wrap=img.parentElement,panel=wrap.parentElement;if(!img.naturalWidth)return;const style=getComputedStyle(panel),available=panel.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight),scale=Math.min(Math.min(640,available)/img.naturalWidth,innerHeight*.70/img.naturalHeight);wrap.style.width=(img.naturalWidth*scale)+'px';wrap.style.height=(img.naturalHeight*scale)+'px'}
function draw(){const img=$('primary');if(!img.naturalWidth)return;fitPrimary();canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.font=Math.max(11,canvas.width/30)+'px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';landmarks.forEach(([name,color],i)=>{const m=mark(name);if(m.visible===true){ctx.beginPath();ctx.arc(Number(m.x),Number(m.y),Math.max(4,canvas.width/80),0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='white';ctx.fillText(String(i+1),Number(m.x),Number(m.y));}})}
function renderJoints(){const box=$('joints');box.innerHTML='';landmarks.forEach(([name,color],i)=>{const m=mark(name),d=document.createElement('div');d.className='joint'+(i===active?' active':'')+(m.visible===true?' done':m.visible===false?' hidden':'');const b=document.createElement('button');b.textContent=(i+1)+' '+name.replaceAll('_',' ');b.style.color=color;b.onclick=()=>{active=i;render()};const s=document.createElement('span');s.className='small';s.textContent=m.visible===true?`(${m.x}, ${m.y})`:m.visible===false?'not visible':'unset';d.append(b,s);box.append(d)})}
function render(){const row=rows[position],r=record();$('title').textContent=String(row.selection_index).padStart(3,'0')+' · '+row.sample_id;$('primary').src=row.primary;$('stability').src=row.stability;$('notes').value=r.notes||'';$('activeHint').textContent='Selected: '+landmarks[active][0].replaceAll('_',' ');const all=landmarks.every(([n])=>mark(n).visible!==null);$('complete').disabled=!all;$('complete').textContent=r.review_complete?'Example complete ✓':'Mark example complete';$('completeHint').textContent=all?'All eight visibility decisions are set.':'Set visible/not-visible for all eight landmarks.';$('progress').textContent=`${position+1}/128 · ${completed()} complete`;renderJoints();draw()}
$('primary').onload=draw;canvas.onclick=e=>{const rect=canvas.getBoundingClientRect(),img=$('primary'),name=landmarks[active][0];if(Math.abs(rect.width/rect.height-img.naturalWidth/img.naturalHeight)>1e-4)throw new Error('Image/canvas aspect ratio mismatch');record().marks[name]={visible:true,x:((e.clientX-rect.left)*canvas.width/rect.width).toFixed(2),y:((e.clientY-rect.top)*canvas.height/rect.height).toFixed(2)};record().review_complete=false;save();render();active=Math.min(7,active+1);render()};
$('notVisible').onclick=()=>{record().marks[landmarks[active][0]]={visible:false,x:'',y:''};record().review_complete=false;save();active=Math.min(7,active+1);render()};$('clear').onclick=()=>{delete record().marks[landmarks[active][0]];record().review_complete=false;save();render()};
$('notes').oninput=e=>{record().notes=e.target.value;save()};$('complete').onclick=()=>{if(landmarks.every(([n])=>mark(n).visible!==null)){record().review_complete=true;save();if(position<127)position++;render()}};
function move(delta){position=Math.max(0,Math.min(127,position+delta));active=0;render()} $('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);
$('reviewer').onchange=e=>{reviewer=e.target.value;load();position=0;active=0;render()};
function csvCell(v){const s=String(v??'');return /[",\\n\\r]/.test(s)?'"'+s.replaceAll('"','""')+'"':s}
$('export').onclick=()=>{const lines=[fields.join(',')];rows.forEach(row=>{const r=state[row.sample_id]||{review_complete:false,notes:'',marks:{}},out=[row.sample_id,reviewer,r.review_complete?1:0,r.notes||''];landmarks.forEach(([name])=>{const m=r.marks?.[name]||{visible:null,x:'',y:''};out.push(m.visible===null?'':m.visible?1:0,m.visible===true?m.x:'',m.visible===true?m.y:'')});lines.push(out.map(csvCell).join(','))});const blob=new Blob([lines.join('\\r\\n')+'\\r\\n'],{type:'text/csv;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=reviewer+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.addEventListener('keydown',e=>{if(e.target.tagName==='TEXTAREA')return;if(e.key==='ArrowLeft')move(-1);else if(e.key==='ArrowRight')move(1);else if(/^[1-8]$/.test(e.key)){active=Number(e.key)-1;render()}});window.addEventListener('resize',draw);load();render();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    output = args.run.resolve() / "blinded_review/index_v6_interactive.html"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(build_interactive(args.run.resolve()))
    print(json.dumps({"status": "BLINDED_INTERACTIVE_REVIEW_COMPLETE", "path": str(output)}))


if __name__ == "__main__":
    main()
