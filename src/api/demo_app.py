"""
api/demo_app.py

Simple browser UI for testing the two-layer voice authorisation flow end to
end, using your own microphone. Separate from main.py (which is the
streaming/WebSocket API) so neither disturbs the other.

    .venv/bin/uvicorn src.api.demo_app:app --port 8900
    open http://127.0.0.1:8900

Flow mirrors the SecureVoice product:
  1. Enrol   -- record a few samples of your voice -> ECAPA-TDNN centroid
  2. Order   -- record one sample + pick an amount
  3. Decide  -- Layer 2 (is it the right voice?) + Layer 2.5 (is it human?)
                -> approve / deny, shown with and without the anti-spoofing
                layer so the difference is visible.

Single in-memory enrolment; this is a test harness, not a product.
"""

import io
import json
from pathlib import Path

import av
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

from ..detector.aasist_l import AasistLDetector

SAMPLE_RATE = 16000
SIMILARITY_HARD_DENY = 0.50
RISK_TIERS = [(500, 0), (2000, 50), (10000, 70), (10**9, 85)]

app = FastAPI(title="SecureVoice — two-layer demo")

_state = {"centroid": None, "n_enrolled": 0}
_models = {}


def _decode(data: bytes) -> np.ndarray:
    """Decode any browser/uploaded audio (webm/opus, wav, m4a...) to 16k mono."""
    container = av.open(io.BytesIO(data))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            chunks.append(rf.to_ndarray().reshape(-1))
    container.close()
    if not chunks:
        raise ValueError("no audio decoded")
    return np.concatenate(chunks).astype(np.float32)


def _ecapa():
    if "ecapa" not in _models:
        from speechbrain.inference.speaker import EncoderClassifier
        _models["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="models/ecapa",
            run_opts={"device": "cpu"})
    return _models["ecapa"]


def _aasist():
    if "aasist" not in _models:
        det = AasistLDetector(device="cpu")
        baseline = Path("results/aasist_baseline.json")
        if baseline.exists():
            with open(baseline) as f:
                det.threshold = json.load(f)["threshold"]
        _models["aasist"] = det
    return _models["aasist"]


ROBUST_CHECKPOINT = Path("models/aasist_l_domain_adapted.pth")


def _aasist_robust():
    """The fine-tuned model (see tests/finetune_robust.py), trained with
    room/noise/mic augmentation so it doesn't rely on studio-clean audio to
    recognise 'real'. Loaded lazily and only if the checkpoint exists yet --
    the UI degrades gracefully to single-model mode while training runs."""
    if not ROBUST_CHECKPOINT.exists():
        return None
    key = f"aasist_robust:{ROBUST_CHECKPOINT.stat().st_mtime}"
    if key not in _models:
        for stale in [k for k in list(_models) if k.startswith("aasist_robust:")]:
            del _models[stale]
        det = AasistLDetector(checkpoint_path=str(ROBUST_CHECKPOINT), device="cpu")
        det.threshold = _aasist().threshold  # same operating point for a fair comparison
        _models[key] = det
    return _models[key]


def _embed(audio: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        e = _ecapa().encode_batch(torch.from_numpy(audio).unsqueeze(0))
    v = e.squeeze().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)


def _min_trust_for(amount: float) -> int:
    for ceiling, min_trust in RISK_TIERS:
        if amount < ceiling:
            return min_trust
    return 85


@app.post("/api/enroll")
async def enroll(files: list[UploadFile] = File(...)):
    embeddings = []
    for f in files:
        embeddings.append(_embed(_decode(await f.read())))
    centroid = np.mean(embeddings, axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    _state["centroid"] = centroid
    _state["n_enrolled"] = len(embeddings)
    return {"enrolled": True, "n_samples": len(embeddings)}


@app.post("/api/reset")
async def reset():
    _state["centroid"] = None
    _state["n_enrolled"] = 0
    return {"reset": True}


@app.post("/api/verify")
async def verify(file: UploadFile = File(...), amount: float = Form(1500)):
    if _state["centroid"] is None:
        return JSONResponse({"error": "Enrol a voice first."}, status_code=400)

    audio = _decode(await file.read())
    similarity = float(np.dot(_state["centroid"], _embed(audio)))
    trust = similarity * 100
    min_trust = _min_trust_for(amount)

    det = _aasist()
    result = det.score(audio)
    is_fake = bool(result.score > det.threshold)

    voice_ok = similarity >= SIMILARITY_HARD_DENY
    approved_sv_only = bool(voice_ok and trust >= min_trust)
    approved_with_antispoof = bool(approved_sv_only and not is_fake)

    robust = _aasist_robust()
    robust_block = None
    if robust is not None:
        r2 = robust.score(audio)
        robust_block = {
            "spoof_score": r2.score,
            "is_fake": bool(r2.score > robust.threshold),
            "latency_ms": r2.latency_ms,
        }

    return {
        "similarity": similarity,
        "trust": trust,
        "min_trust": min_trust,
        "voice_matches": bool(voice_ok),
        "spoof_score": result.score,
        "spoof_threshold": det.threshold,
        "is_fake": is_fake,
        "latency_ms": result.latency_ms,
        "approved_sv_only": approved_sv_only,
        "approved_with_antispoof": approved_with_antispoof,
        "caught_only_by_antispoof": bool(approved_sv_only and not approved_with_antispoof),
        "duration_s": len(audio) / SAMPLE_RATE,
        "robust": robust_block,
    }


PAGE = """
<!doctype html><html><head><meta charset="utf-8">
<title>SecureVoice — two-layer test</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;max-width:760px;margin:32px auto;
      padding:0 18px;color:#1a1a1a;line-height:1.5}
 h1{font-size:22px;margin-bottom:4px} .sub{color:#666;margin-bottom:26px;font-size:14px}
 .card{border:1px solid #ddd;border-radius:10px;padding:18px;margin-bottom:16px}
 .card h2{font-size:15px;margin:0 0 10px;text-transform:uppercase;letter-spacing:.5px;color:#444}
 button{font-size:14px;padding:9px 15px;border-radius:7px;border:1px solid #bbb;
        background:#fff;cursor:pointer;margin-right:7px}
 button.rec{background:#d33;color:#fff;border-color:#d33}
 button:disabled{opacity:.45;cursor:not-allowed}
 .status{font-size:13px;color:#666;margin-top:9px}
 .row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #f0f0f0}
 .row span:last-child{font-variant-numeric:tabular-nums;font-weight:600}
 .ok{color:#0a7d33} .bad{color:#c00}
 .verdict{padding:13px;border-radius:8px;margin-top:11px;font-weight:600;text-align:center}
 .approved{background:#e6f6ea;color:#0a7d33} .denied{background:#fdeaea;color:#c00}
 .highlight{background:#fff6e0;border:1px solid #f0c36d;padding:11px;border-radius:8px;
            margin-top:11px;font-size:14px}
 select,input{font-size:14px;padding:7px;border-radius:6px;border:1px solid #bbb}
 .muted{font-size:12px;color:#888;margin-top:8px}
</style></head><body>
<h1>SecureVoice — two-layer voice authorisation</h1>
<div class="sub">Layer 2: speaker verification (ECAPA-TDNN) &nbsp;·&nbsp;
 Layer 2.5: anti-spoofing (AASIST-L)</div>

<div class="card">
 <h2>Step 1 — Enrol your voice</h2>
 <div>Record 3 short clips (~4s each). Say anything.</div>
 <div style="margin-top:11px">
  <button id="enrollBtn">Record enrolment clip</button>
  <button id="doEnroll" disabled>Enrol (0 clips)</button>
  <button id="resetBtn">Reset</button>
 </div>
 <div class="status" id="enrollStatus">Not enrolled yet.</div>
</div>

<div class="card">
 <h2>Step 2 — Place an order by voice</h2>
 <div>Order amount:
  <select id="amount">
   <option value="300">Rs.300 (auto-approve tier)</option>
   <option value="1500" selected>Rs.1,500 (needs trust 50)</option>
   <option value="3500">Rs.3,500 (needs trust 70)</option>
   <option value="25000">Rs.25,000 (needs trust 85)</option>
  </select>
 </div>
 <div style="margin-top:11px">
  <button id="testBtn" disabled>Record & check</button>
  <label class="muted">or upload a file:
   <input type="file" id="upload" accept="audio/*" disabled></label>
 </div>
 <div class="status" id="testStatus"></div>
</div>

<div class="card" id="results" style="display:none">
 <h2>Result</h2>
 <div id="out"></div>
</div>

<div class="muted">Tip: to see the anti-spoofing layer fire, upload a synthetic/
 AI-generated clip, or a spoof file from data/raw/asvspoof2019_la/.</div>

<script>
let enrolled=[], mediaRecorder, chunks=[];
const $=id=>document.getElementById(id);

async function record(seconds=4){
  const stream=await navigator.mediaDevices.getUserMedia({audio:true});
  return new Promise(res=>{
    chunks=[]; mediaRecorder=new MediaRecorder(stream);
    mediaRecorder.ondataavailable=e=>chunks.push(e.data);
    mediaRecorder.onstop=()=>{stream.getTracks().forEach(t=>t.stop());
      res(new Blob(chunks,{type:'audio/webm'}));};
    mediaRecorder.start();
    setTimeout(()=>mediaRecorder.stop(), seconds*1000);
  });
}

function countdown(el,btn,secs,label){
  btn.classList.add('rec'); let n=secs;
  el.textContent=label+' '+n+'s...';
  const t=setInterval(()=>{n--; if(n>0){el.textContent=label+' '+n+'s...';}
    else{clearInterval(t); btn.classList.remove('rec');}},1000);
}

$('enrollBtn').onclick=async()=>{
  $('enrollBtn').disabled=true;
  countdown($('enrollStatus'),$('enrollBtn'),4,'Recording enrolment clip —');
  const blob=await record(4);
  enrolled.push(blob);
  $('enrollStatus').textContent=enrolled.length+' clip(s) recorded.';
  $('doEnroll').textContent='Enrol ('+enrolled.length+' clips)';
  $('doEnroll').disabled=enrolled.length<1;
  $('enrollBtn').disabled=false;
};

$('doEnroll').onclick=async()=>{
  const fd=new FormData();
  enrolled.forEach((b,i)=>fd.append('files',b,'e'+i+'.webm'));
  $('enrollStatus').textContent='Enrolling...';
  const r=await fetch('/api/enroll',{method:'POST',body:fd});
  const j=await r.json();
  $('enrollStatus').textContent='Enrolled with '+j.n_samples+' clip(s). Ready to test.';
  $('testBtn').disabled=false; $('upload').disabled=false;
};

$('resetBtn').onclick=async()=>{
  enrolled=[]; await fetch('/api/reset',{method:'POST'});
  $('enrollStatus').textContent='Not enrolled yet.';
  $('doEnroll').textContent='Enrol (0 clips)'; $('doEnroll').disabled=true;
  $('testBtn').disabled=true; $('upload').disabled=true;
  $('results').style.display='none';
};

$('testBtn').onclick=async()=>{
  $('testBtn').disabled=true;
  countdown($('testStatus'),$('testBtn'),4,'Recording order —');
  const blob=await record(4);
  await send(blob,'order.webm');
  $('testBtn').disabled=false;
};

$('upload').onchange=async e=>{
  if(e.target.files[0]) await send(e.target.files[0], e.target.files[0].name);
};

async function send(blob,name){
  $('testStatus').textContent='Analysing...';
  const fd=new FormData();
  fd.append('file',blob,name); fd.append('amount',$('amount').value);
  const r=await fetch('/api/verify',{method:'POST',body:fd});
  const j=await r.json();
  if(j.error){$('testStatus').textContent=j.error; return;}
  $('testStatus').textContent='';
  render(j);
}

function render(j){
  const pct=x=>(x*100).toFixed(1);
  $('out').innerHTML=`
   <div class="row"><span>Layer 2 — voice similarity</span>
     <span class="${j.voice_matches?'ok':'bad'}">${j.similarity.toFixed(3)}</span></div>
   <div class="row"><span>Trust score (needs ≥ ${j.min_trust})</span>
     <span class="${j.trust>=j.min_trust?'ok':'bad'}">${j.trust.toFixed(0)}</span></div>
   <div class="row"><span>Layer 2.5 — P(synthetic speech)</span>
     <span class="${j.is_fake?'bad':'ok'}">${j.spoof_score.toFixed(4)}</span></div>
   <div class="row"><span>Anti-spoofing verdict</span>
     <span class="${j.is_fake?'bad':'ok'}">${j.is_fake?'SYNTHETIC':'genuine human'}</span></div>
   <div class="row"><span>Inference latency</span>
     <span>${j.latency_ms.toFixed(0)} ms</span></div>
   <div class="verdict ${j.approved_sv_only?'approved':'denied'}">
     Speaker verification only → ${j.approved_sv_only?'ORDER APPROVED':'ORDER DENIED'}</div>
   <div class="verdict ${j.approved_with_antispoof?'approved':'denied'}">
     With anti-spoofing layer → ${j.approved_with_antispoof?'ORDER APPROVED':'ORDER DENIED'}</div>
   ${j.caught_only_by_antispoof?'<div class="highlight">This passed speaker '+
     'verification and was stopped <b>only</b> by the anti-spoofing layer.</div>':''}
   ${j.robust?renderRobust(j):
     '<div class="muted" style="margin-top:10px">Robustness-improved model not '+
     'trained yet — run tests/finetune_domain_adapt.py, then reload this page.</div>'}`;
  $('results').style.display='block';
}

function renderRobust(j){
  const r=j.robust;
  const changed = j.is_fake !== r.is_fake;
  return `
   <div class="row" style="margin-top:14px;border-top:2px solid #ddd;padding-top:14px">
     <span><b>Original model</b> — P(synthetic)</span>
     <span class="${j.is_fake?'bad':'ok'}">${j.spoof_score.toFixed(4)} (${j.is_fake?'SYNTHETIC':'genuine'})</span></div>
   <div class="row"><span><b>Robustness-improved model</b> — P(synthetic)</span>
     <span class="${r.is_fake?'bad':'ok'}">${r.spoof_score.toFixed(4)} (${r.is_fake?'SYNTHETIC':'genuine'})</span></div>
   ${changed?'<div class="highlight">The two models DISAGREE on this clip — '+
     (r.is_fake?'the fine-tuned model is now catching something the original missed.':
     'the fine-tuned model corrected a false alarm from the original model. '+
     'This is the fix for the "100% of real recordings flagged as fake" finding.')+
     '</div>':'<div class="muted" style="margin-top:8px">Both models agree on this clip.</div>'}`;
}
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
