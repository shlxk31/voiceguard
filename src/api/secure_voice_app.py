"""
api/secure_voice_app.py

SecureVoice — explainable multi-layer voice authorization demo.

    .venv/bin/uvicorn src.api.secure_voice_app:app --port 8901
    open http://127.0.0.1:8901

WHAT MAKES THIS DIFFERENT FROM demo_app.py
The earlier demo showed two numbers and a verdict. This one shows the
*reasoning*: every layer's evidence in calibrated log-likelihood-ratio form,
how those combine, the cost-derived threshold for the requested amount, and
the margin by which the decision was made. Nothing on screen is a magic
number -- each traces to a model output or a formula (see risk/policy.py).
"""

import io
import json
import uuid
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from ..detector.aasist_l import AasistLDetector
from ..detector.liveness_l2 import LivenessDetector
from ..risk.calibration import load_calibrators
from ..risk.policy import RiskPolicy, CostModel
from ..risk.engine import DecisionEngine, LAYER_INFO
from ..nlu.understand import CommandUnderstanding
from ..assistant.groq_client import GroqAssistant
from ..assistant.respond import ResponseComposer
from ..assistant.passphrase import verify as verify_passphrase, normalise as norm_phrase
from ..context.l4 import ContextLayer
from ..context.profile import ProfileStore
from ..risk.stake import estimate_stake
from ..risk.pricing import PriceEstimator

SAMPLE_RATE = 16000
ROOT = Path(__file__).resolve().parent.parent.parent

app = FastAPI(title="SecureVoice — explainable voice authorization")

#: Whose session this browser is in. Enrollment and history are per-user
#: because L4 asks "is this normal FOR THIS USER", which is meaningless
#: against a single global voiceprint.
_state = {"user_id": None}
_models = {}

#: Synthesised replies are written here and served back by /api/audio.
AUDIO_DIR = ROOT / "data" / "tts_cache"

#: Every clip captured through the browser is archived here, verbatim, in the
#: exact container the browser produced.
#:
#: This is not a convenience. L3 flags this machine's browser-recorded speech
#: as synthetic with P = 1.0 while the same speaker's phone recordings score
#: 0.000000 -- a capture-domain gap that can only be closed by adapting on
#: audio from THIS capture path. An earlier version computed the enrolment
#: embedding and dropped the waveform, so a recording session produced no
#: reusable training data at all. Keeping the bytes costs nothing and is the
#: difference between a session being useful and being wasted.
#:
#: The audio never leaves the machine: ASR is local faster-whisper, and only
#: the transcript is sent to Groq.
CAPTURE_DIR = ROOT / "data" / "captures"


def _archive(user_id: str, kind: str, filename: str, data: bytes) -> str:
    """Store one captured clip. Never raises -- archiving must not block auth."""
    try:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
        ext = Path(filename or "").suffix or ".webm"
        d = CAPTURE_DIR / (user_id or "_anon") / kind
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{ts}{ext}"
        p.write_bytes(data)
        return str(p.relative_to(ROOT))
    except Exception:
        return ""


# ------------------------------------------------------------------ audio --
def _decode(data: bytes) -> np.ndarray:
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


# ----------------------------------------------------------------- models --
def _ecapa():
    if "ecapa" not in _models:
        from speechbrain.inference.speaker import EncoderClassifier
        _models["ecapa"] = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(ROOT / "models" / "ecapa"), run_opts={"device": "cpu"})
    return _models["ecapa"]


def _aasist():
    """Domain-adapted checkpoint when present, else the original."""
    if "aasist" not in _models:
        adapted = ROOT / "models" / "aasist_l_domain_adapted.pth"
        if adapted.exists():
            det = AasistLDetector(checkpoint_path=str(adapted), device="cpu")
            _models["aasist_variant"] = "domain-adapted"
        else:
            det = AasistLDetector(device="cpu")
            _models["aasist_variant"] = "original (benchmark)"
        _models["aasist"] = det
    return _models["aasist"]


def _liveness():
    if "liveness" not in _models:
        _models["liveness"] = LivenessDetector()
    return _models["liveness"]


def _engine():
    if "engine" not in _models:
        cals = load_calibrators(ROOT / "models" / "calibrators.json")
        _models["engine"] = DecisionEngine(cals, RiskPolicy(CostModel()))
    return _models["engine"]


def _nlu():
    if "nlu" not in _models:
        _models["nlu"] = CommandUnderstanding(model_size="base", device="cpu")
    return _models["nlu"]


def _profiles() -> ProfileStore:
    if "profiles" not in _models:
        _models["profiles"] = ProfileStore()
    return _models["profiles"]


def _assistant() -> GroqAssistant:
    if "assistant" not in _models:
        _models["assistant"] = GroqAssistant()
    return _models["assistant"]


def _l4() -> ContextLayer:
    if "l4" not in _models:
        _models["l4"] = ContextLayer(_assistant())
    return _models["l4"]


def _pricer() -> PriceEstimator:
    if "pricer" not in _models:
        _models["pricer"] = PriceEstimator(_assistant())
    return _models["pricer"]


def _composer() -> ResponseComposer:
    if "composer" not in _models:
        _models["composer"] = ResponseComposer(_assistant(), AUDIO_DIR)
    return _models["composer"]


def _current():
    """The logged-in user's profile, or None."""
    uid = _state.get("user_id")
    return _profiles().get_user(uid) if uid else None


def _embed(audio: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        e = _ecapa().encode_batch(torch.from_numpy(audio).unsqueeze(0))
    v = e.squeeze().cpu().numpy()
    return v / (np.linalg.norm(v) + 1e-9)


# --------------------------------------------------------------- endpoints --
@app.get("/api/config")
async def config():
    eng = _engine()
    amounts = [100, 300, 1000, 3000, 10000, 30000, 100000]
    u = _current()
    return {
        "layers": [{"id": k, **v} for k, v in LAYER_INFO.items()],
        "calibrated_layers": sorted(eng.calibrators.keys()),
        "costs": eng.policy.costs.to_dict(),
        "threshold_curve": eng.policy.curve(amounts),
        "assistant_available": _assistant().available,
        "has_passphrase": bool(u and u.passphrase),
        "users": _profiles().list_users(),
        "user": None if u is None else {
            "user_id": u.user_id, "display_name": u.display_name,
            "enrolled": u.enrolled, "n_enroll": u.n_enroll,
            "n_real": u.n_real, "n_simulated": u.n_simulated,
        },
    }


# --------------------------------------------------------------- accounts --
@app.post("/api/login")
async def login(user_id: str = Form(...), display_name: str = Form(""),
                seed_history: bool = Form(True)):
    """
    Select or create a user. Seeding gives L4's behavioural sub-signal a
    baseline to compare against; it is DECLARED SYNTHETIC and the UI says so.
    """
    uid = user_id.strip().lower().replace(" ", "_")
    if not uid:
        return JSONResponse({"error": "user_id required"}, status_code=400)
    store = _profiles()
    existing = store.get_user(uid)
    if existing is None:
        store.create_user(uid, display_name.strip() or uid)
        if seed_history:
            store.seed_history(uid, n=40)
    _state["user_id"] = uid
    u = store.get_user(uid)
    return {"user_id": u.user_id, "display_name": u.display_name,
            "enrolled": u.enrolled, "n_enroll": u.n_enroll,
            "n_real": u.n_real, "n_simulated": u.n_simulated}


@app.post("/api/enroll")
async def enroll(files: list[UploadFile] = File(...)):
    u = _current()
    if u is None:
        return JSONResponse({"error": "Sign in first."}, status_code=400)
    embs = []
    for f in files:
        data = await f.read()
        _archive(u.user_id, "enroll", f.filename, data)
        embs.append(_embed(_decode(data)))
    c = np.mean(embs, axis=0)
    _profiles().set_embedding(u.user_id, c / (np.linalg.norm(c) + 1e-9), len(embs))
    return {"enrolled": True, "n_samples": len(embs), "user_id": u.user_id}


@app.post("/api/enroll_passphrase")
async def enroll_passphrase(file: UploadFile = File(...)):
    """
    Record the challenge phrase. Whatever the user says becomes their phrase,
    transcribed and normalised; verification later is fuzzy (see
    assistant/passphrase.py) because ASR never returns the same string twice.
    """
    u = _current()
    if u is None:
        return JSONResponse({"error": "Sign in first."}, status_code=400)
    data = await file.read()
    _archive(u.user_id, "passphrase", file.filename, data)
    heard = _nlu().understand(_decode(data)).transcript
    phrase = norm_phrase(heard)
    if len(phrase.split()) < 2:
        return JSONResponse(
            {"error": f"Heard only {heard!r}. Use a phrase of at least two "
                      f"words so it is hard to guess."}, status_code=400)
    _profiles().set_passphrase(u.user_id, phrase)
    return {"passphrase_set": True, "heard": heard, "normalised": phrase,
            "n_words": len(phrase.split())}


#: Labels the data-collection endpoint accepts.
#:
#: "adapt"           -- genuine speech through THIS capture path, for closing
#:                      the L3 capture-domain gap.
#: "live"/"playback" -- the paired corpus L2 needs. The pairing is the whole
#:                      point: the previous liveness experiment reached 100%
#:                      accuracy by learning that live clips were iPhone .m4a
#:                      and playback clips were MacBook .wav, i.e. it
#:                      classified the RECORDING DEVICE, not liveness. Both
#:                      members of a pair must come from one microphone in one
#:                      session, which is why the UI records them back to back
#:                      rather than trusting anyone to remember.
CAPTURE_LABELS = ("adapt", "live", "playback")


@app.post("/api/capture")
async def capture(file: UploadFile = File(...), label: str = Form("adapt")):
    """
    Archive a training clip and do nothing else.

    Deliberately runs no models: no ASR, no Groq call, no decision. Collecting
    a corpus should not cost rate limit or wall-clock time, and a capture that
    fails because an API was throttled is a capture lost.
    """
    u = _current()
    if u is None:
        return JSONResponse({"error": "Sign in first."}, status_code=400)
    if label not in CAPTURE_LABELS:
        return JSONResponse(
            {"error": f"label must be one of {CAPTURE_LABELS}"}, status_code=400)
    data = await file.read()
    path = _archive(u.user_id, label, file.filename, data)
    return {"saved": bool(path), "path": path, "label": label,
            "bytes": len(data), "counts": _capture_counts(u.user_id)}


def _capture_counts(user_id: str) -> dict:
    out = {}
    for label in CAPTURE_LABELS:
        d = CAPTURE_DIR / user_id / label
        out[label] = len(list(d.glob("*"))) if d.exists() else 0
    return out


@app.get("/api/captures")
async def captures():
    u = _current()
    if u is None:
        return {"counts": {k: 0 for k in CAPTURE_LABELS}}
    return {"counts": _capture_counts(u.user_id), "user_id": u.user_id}


@app.post("/api/reset")
async def reset():
    """Clear this user's voiceprint. History is kept."""
    u = _current()
    if u is not None:
        _profiles().set_embedding(u.user_id, np.zeros(0), 0)
        with _profiles()._conn() as c:
            c.execute("UPDATE users SET embedding = NULL, n_enroll = 0 "
                      "WHERE user_id = ?", (u.user_id,))
    return {"reset": True}


@app.get("/api/audio/{name}")
async def audio(name: str):
    """Serve a synthesised reply. Name is constrained to our own files."""
    p = AUDIO_DIR / Path(name).name
    if not p.exists() or p.suffix != ".wav":
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(p, media_type="audio/wav")


# ---------------------------------------------------------- authorization --
#: Level the anti-spoofing input is normalised to before scoring.
#:
#: The domain-adapted checkpoint was fine-tuned on one speaker's browser
#: captures, which all sat near RMS 0.03. It therefore learned "quiet" as a
#: cue for "human": a second speaker recording at RMS 0.04-0.08 was scored
#: P(synthetic) = 0.95-1.00 on entirely genuine speech, while the same
#: speaker's quieter clips passed. Normalising the level before scoring
#: removed five of six of those false positives and left the first speaker's
#: scores unchanged.
#:
#: This is applied ONLY on the live path. The benchmark figures reported for
#: this project come from the original checkpoint on unmodified ASVspoof
#: audio, and normalising there measurably hurts (6.0% -> 16.7% EER on a
#: 200-file subset) because that corpus is peak-normalised already.
L3_TARGET_RMS = 0.030


def _level_normalise(audio: np.ndarray, target_rms: float = L3_TARGET_RMS) -> np.ndarray:
    """Scale to a fixed RMS, with headroom so the waveform cannot clip."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 1e-6:
        return audio
    out = audio * (target_rms / rms)
    peak = float(np.max(np.abs(out)))
    return out * (0.95 / peak) if peak > 0.95 else out


def _run_layers(audio_arr: np.ndarray, centroid: np.ndarray, history: list[dict]):
    """Score every implemented layer on one utterance."""
    understanding = _nlu().understand(audio_arr)

    similarity = float(np.dot(centroid, _embed(audio_arr)))     # L1
    live = _liveness().score(audio_arr)                         # L2
    spoof = _aasist().score(_level_normalise(audio_arr))        # L3
    ctx = _l4().score(understanding.transcript,                 # L4
                      amount=understanding.amount, history=history)

    raw = {"L1": similarity,
           "L2": live.score if live.available else None,
           "L3": spoof.score,
           "L4": ctx.normality}
    return understanding, raw, ctx, spoof


@app.post("/api/authorize")
async def authorize(file: UploadFile = File(...),
                     speak: bool = Form(True)):
    """
    Risk is set by the STAKE of what was asked, not by a rupee figure alone.

    There is no longer a numeric fallback. A flat default (it was Rs.1,500)
    forced every unpriced request to the same threshold, so "what is the
    weather" demanded exactly as much evidence as a Rs.1,500 payment. When no
    amount is spoken, the stake now comes from the action's category and, for
    purchases, the estimated price of the item -- both of which are actually
    about the request, which a constant never was.
    """
    u = _current()
    if u is None or not u.enrolled:
        return JSONResponse({"error": "Sign in and enrol a voice first."},
                            status_code=400)

    data = await file.read()
    capture = _archive(u.user_id, "authorize", file.filename, data)
    audio_arr = _decode(data)
    understanding, raw, ctx, spoof = _run_layers(audio_arr, u.embedding, u.history)

    amount = understanding.amount
    amount_origin = ("detected from speech" if amount is not None
                     else "no amount spoken - stake derived from the action")

    # Price lookup only for things that are actually bought; asking the model
    # to price "turn off the lights" wastes the rate-limited token budget.
    price = _pricer().estimate(understanding.transcript) \
        if ctx.intent.category == "purchase" else None

    # The threshold is set by the STAKE, not the rupee figure alone: a request
    # with no amount ("unlock the front door") would otherwise receive the
    # lowest bar the policy can issue. See risk/stake.py.
    stake = estimate_stake(amount, ctx.intent.category,
                           irreversible=ctx.intent.irreversible,
                           illegitimate=ctx.intent.illegitimate,
                           estimated_price=price.stake_value if price else 0.0,
                           price_item=price.item if price else "")
    decision = _engine().decide(raw, stake.stake)

    rid = uuid.uuid4().hex[:10]
    reply = _composer().compose(decision, recipient=ctx.intent.recipient,
                                transcript=understanding.transcript,
                                speak=speak, request_id=rid,
                                has_passphrase=bool(u.passphrase),
                                spoken_amount=understanding.amount,
                                item=(price.item if price else ""),
                                action=ctx.intent.action)

    _profiles().record(u.user_id, amount=stake.stake, recipient=ctx.intent.recipient,
                       action=ctx.intent.action, transcript=understanding.transcript,
                       outcome=decision.outcome, fused_llr=decision.fused_llr,
                       simulated=False)

    out = decision.to_dict()
    out["request_id"] = rid
    out["understanding"] = understanding.to_dict()
    out["amount_origin"] = amount_origin
    out["spoken_amount"] = amount
    out["stake"] = stake.to_dict()
    out["price"] = price.to_dict() if price else None
    out["context"] = ctx.to_dict()
    out["assistant"] = reply.to_dict()
    out["user"] = {"user_id": u.user_id, "display_name": u.display_name,
                   "n_real": u.n_real, "n_simulated": u.n_simulated}
    out["meta"] = {
        "duration_s": len(audio_arr) / SAMPLE_RATE,
        "latency_ms": spoof.latency_ms,
        "aasist_variant": _models.get("aasist_variant", "unknown"),
        "capture": capture,
    }
    return out


@app.post("/api/challenge")
async def challenge(file: UploadFile = File(...),
                    amount: float = Form(...),
                    prior_llr: float = Form(0.0),
                    speak: bool = Form(True)):
    """
    Second-utterance re-verification, the action behind the CHALLENGE outcome.

    The new utterance is scored by the same layers and its evidence is ADDED
    to the evidence from the first. That is not an ad-hoc bonus: calibrated
    LLRs are log-odds, so accumulating independent evidence is a sum -- the
    same rule the engine uses to combine layers, applied across time.

    The amount is carried over from the original request rather than re-read
    from the passphrase, so an attacker cannot use the challenge step to
    quietly restate a smaller amount and lower the bar.
    """
    u = _current()
    if u is None or not u.enrolled:
        return JSONResponse({"error": "Sign in and enrol a voice first."},
                            status_code=400)

    data = await file.read()
    _archive(u.user_id, "challenge", file.filename, data)
    audio_arr = _decode(data)
    understanding, raw, ctx, spoof = _run_layers(audio_arr, u.embedding, u.history)

    # The enrolled phrase must come back before the voice evidence counts.
    # A challenge is a request for a SECOND, INDEPENDENT factor; accepting a
    # second utterance of anything at all would only re-sample the same one.
    check = verify_passphrase(u.passphrase, understanding.transcript)

    decision = _engine().decide(raw, amount)
    if check.enrolled and not check.matched:
        # Wrong words: the challenge is failed outright, regardless of how
        # good the voice match is. Nothing is added to the prior evidence.
        combined = float(prior_llr)
        outcome, thresholds = RiskPolicy.DENY, _engine().policy.thresholds(amount)
    else:
        combined = float(prior_llr) + decision.fused_llr
        outcome, thresholds = _engine().policy.decide(combined, amount)

    rid = uuid.uuid4().hex[:10]
    final = decision
    final.outcome, final.fused_llr, final.thresholds = outcome, combined, thresholds
    reply = _composer().compose(final, recipient=ctx.intent.recipient,
                                transcript=understanding.transcript,
                                speak=speak, request_id=rid,
                                has_passphrase=bool(u.passphrase),
                                spoken_amount=understanding.amount,
                                action=ctx.intent.action)

    out = final.to_dict()
    out["request_id"] = rid
    out["is_challenge_response"] = True
    out["passphrase"] = check.to_dict()
    out["prior_llr"] = float(prior_llr)
    out["this_utterance_llr"] = float(combined - float(prior_llr))
    out["understanding"] = understanding.to_dict()
    out["context"] = ctx.to_dict()
    out["assistant"] = reply.to_dict()
    out["meta"] = {"duration_s": len(audio_arr) / SAMPLE_RATE,
                   "latency_ms": spoof.latency_ms}
    return out


PAGE = r"""
<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SecureVoice</title>
<style>
:root{
  --bg:#0f1117; --panel:#171a23; --panel2:#1e222d; --line:#2a2f3d;
  --tx:#e7e9ee; --dim:#9aa2b4; --faint:#6b7386;
  --ok:#35c88f; --bad:#ff6b6b; --warn:#f5a524; --acc:#5b8def;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
.wrap{max-width:1080px;margin:0 auto;padding:28px 20px 60px}
header{display:flex;align-items:baseline;gap:12px;margin-bottom:6px}
h1{font-size:21px;margin:0;letter-spacing:-.3px}
.sub{color:var(--dim);font-size:13px;margin-bottom:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;margin-bottom:16px}
.card h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--faint);margin:0 0 14px}
button{font:inherit;font-weight:500;padding:9px 14px;border-radius:8px;border:1px solid var(--line);
  background:var(--panel2);color:var(--tx);cursor:pointer;transition:.15s}
button:hover:not(:disabled){border-color:var(--acc);color:#fff}
button:disabled{opacity:.4;cursor:not-allowed}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}
button.rec{background:var(--bad);border-color:var(--bad);color:#fff}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.status{color:var(--dim);font-size:13px;margin-top:10px}
input[type=range]{width:100%;accent-color:var(--acc)}
.amt{font-size:26px;font-weight:600;letter-spacing:-.5px}
.muted{color:var(--faint);font-size:12px}
/* verdict */
.verdict{border-radius:12px;padding:18px 20px;margin-bottom:16px;border:1px solid;display:flex;
  justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
.v-ALLOW{background:rgba(53,200,143,.09);border-color:rgba(53,200,143,.35)}
.v-CHALLENGE{background:rgba(245,165,36,.09);border-color:rgba(245,165,36,.35)}
.v-DENY{background:rgba(255,107,107,.09);border-color:rgba(255,107,107,.35)}
.v-label{font-size:23px;font-weight:700;letter-spacing:-.4px}
.v-ALLOW .v-label{color:var(--ok)} .v-CHALLENGE .v-label{color:var(--warn)} .v-DENY .v-label{color:var(--bad)}
.v-sum{color:var(--dim);font-size:13px;max-width:560px}
/* evidence */
.ev{border:1px solid var(--line);border-radius:10px;padding:13px 14px;margin-bottom:10px;background:var(--panel2)}
.ev.off{opacity:.45}
.ev-top{display:flex;justify-content:space-between;gap:10px;align-items:baseline}
.ev-name{font-weight:600}
.ev-model{color:var(--faint);font-size:11.5px;margin-top:1px}
.ev-threat{color:var(--dim);font-size:12px;margin-top:7px}
.ev-threat b{color:var(--tx);font-weight:500}
.llr{font-variant-numeric:tabular-nums;font-weight:700;font-size:15px}
.pos{color:var(--ok)} .neg{color:var(--bad)} .neu{color:var(--faint)}
.bar{height:7px;background:var(--line);border-radius:4px;margin-top:9px;position:relative;overflow:hidden}
.bar i{position:absolute;top:0;bottom:0;left:50%;background:var(--ok);border-radius:4px}
.bar i.n{background:var(--bad)}
.mid{position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--faint);opacity:.6}
.reading{font-size:12px;color:var(--dim);margin-top:7px}
/* scale */
.scale{position:relative;height:60px;margin:18px 0 6px}
.track{position:absolute;top:26px;left:0;right:0;height:6px;background:var(--line);border-radius:3px}
.zone{position:absolute;top:26px;height:6px;border-radius:3px}
.tick{position:absolute;top:12px;width:2px;height:34px;border-radius:1px}
.tlabel{position:absolute;top:-2px;font-size:10.5px;color:var(--faint);transform:translateX(-50%);white-space:nowrap}
.marker{position:absolute;top:6px;width:3px;height:46px;background:#fff;border-radius:2px;box-shadow:0 0 0 3px var(--panel)}
.mlabel{position:absolute;top:50px;font-size:11px;font-weight:600;transform:translateX(-50%);white-space:nowrap}
/* derivation */
.deriv{font:12px/1.85 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim);
  background:#12141b;border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto}
.deriv div:last-child,.deriv div:nth-last-child(2){color:var(--tx)}
details summary{cursor:pointer;color:var(--dim);font-size:12.5px;margin-bottom:8px}
.tag{display:inline-block;font-size:10.5px;padding:2px 7px;border-radius:20px;
  background:var(--panel2);border:1px solid var(--line);color:var(--dim);margin-left:6px}
</style></head><body><div class="wrap">

<header><h1>SecureVoice</h1><span class="tag" id="variant">loading…</span></header>
<div class="sub">Continuous voice authorization. Four evidence layers, one cost-derived threshold.</div>

<div class="card">
  <h2>Account</h2>
  <div class="row">
    <select id="uSel"><option value="">— new user —</option></select>
    <input id="uNew" placeholder="new user id" style="width:150px">
    <button id="bLogin" class="primary">Sign in</button>
  </div>
  <div class="status" id="sUser">Not signed in.</div>
</div>

<div class="grid">
  <div class="card">
    <h2>Enrolment</h2>
    <div class="row">
      <button id="bRec" disabled>Record clip</button>
      <button id="bEnrol" class="primary" disabled>Enrol (0)</button>
      <button id="bPhrase" disabled>Passphrase</button>
      <button id="bReset">Reset</button>
    </div>
    <div class="status" id="sEnrol">Sign in first.</div>
    <div class="status" id="sPhrase"></div>
  </div>

  <div class="card">
    <h2>Request</h2>
    <div class="row">
      <button id="bTest" class="primary" disabled>Speak</button>
      <select id="dur">
        <option value="6">6 s</option>
        <option value="8" selected>8 s</option>
        <option value="12">12 s</option>
        <option value="20">20 s</option>
      </select>
      <label class="muted">or <input type="file" id="up" accept="audio/*" disabled></label>
    </div>
    <div class="status" id="sTest"></div>
  </div>
</div>

<div id="out" style="display:none">

  <div class="verdict" id="verdict">
    <div>
      <div class="muted" style="margin-bottom:2px">L5 · Risk policy decides</div>
      <div class="v-label" id="vLabel"></div>
      <div class="v-sum" id="vSum"></div>
    </div>
    <div style="text-align:right">
      <div class="muted">weakest layer</div>
      <div class="llr" id="vLlr" style="font-size:24px"></div>
    </div>
  </div>

  <div class="card" id="replyCard" style="display:none">
    <h2>Assistant</h2>
    <div id="replyText" style="font-size:16px;line-height:1.6"></div>
    <audio id="replyAudio" controls style="width:100%;margin-top:12px;display:none"></audio>
    <div class="row" id="chalRow" style="display:none;margin-top:12px">
      <button id="bChal" class="primary">Answer challenge</button>
    </div>
    <div class="status" id="sChal"></div>
  </div>

  <div class="card" id="heardCard">
    <h2>Request understood</h2>
    <div id="heard"></div>
  </div>

  <div class="card">
    <h2>Evidence · layers 1–4</h2>
    <div id="evs"></div>
  </div>

  <div class="card">
    <h2>L5 · Risk policy</h2>
    <div class="ev" style="margin-bottom:6px">
      <div class="ev-top">
        <div><div class="ev-name">L5 · Risk policy</div>
        <div class="ev-model">Sets the bar for this request, then decides</div></div>
        <div class="llr" id="l5Out"></div>
      </div>
      <div class="reading" id="l5Read"></div>
    </div>
    <div class="scale" id="scale"></div>
    <details style="margin-top:14px">
      <summary>How the threshold was derived</summary>
      <div class="deriv" id="deriv"></div>
      <div class="muted" style="margin-top:10px;line-height:1.6">
        Each layer's raw score is calibrated on held-out data into a log-likelihood ratio.
        The layers combine by weakest link, since an attacker must defeat all of them.
        The threshold minimises expected cost (Kinnunen et al., Odyssey 2018).
      </div>
    </details>
  </div>

  <div class="card" id="ctxCard" style="display:none">
    <h2>Context detail</h2>
    <div id="ctxBody"></div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
let clips=[];
const fmt=n=>'₹'+Math.round(n).toLocaleString('en-IN');

async function rec(sec=4){
  // Echo cancellation, noise suppression and auto gain control are ON by
  // default in every browser. They are aggressive DSP, and this is forensic
  // audio: L3 asks whether a real vocal tract produced this waveform, so
  // handing it a signal that has already been spectrally reshaped is exactly
  // the wrong input. Measured effect of simulating them on a clip that scores
  // P(synthetic)=0.000000 raw: 0.029 with both applied -- a 75x push toward a
  // false "synthetic" verdict, on genuine speech.
  const st=await navigator.mediaDevices.getUserMedia({audio:{
    echoCancellation:false, noiseSuppression:false, autoGainControl:false}});
  return new Promise(res=>{let c=[];const m=new MediaRecorder(st);
    m.ondataavailable=e=>c.push(e.data);
    m.onstop=()=>{st.getTracks().forEach(t=>t.stop());res(new Blob(c,{type:'audio/webm'}))};
    m.start();setTimeout(()=>m.stop(),sec*1000);});
}
function countdown(el,btn,s,txt){btn.classList.add('rec');let n=s;el.textContent=`${txt} ${n}s…`;
  const t=setInterval(()=>{n--;n>0?el.textContent=`${txt} ${n}s…`:(clearInterval(t),btn.classList.remove('rec'))},1000);}

let USER=null, LAST=null;

function paintUser(u){
  USER=u;
  if(!u){$('sUser').textContent='Not signed in.';return;}
  const hist=[]; if(u.n_real)hist.push(u.n_real+' real');
  if(u.n_simulated)hist.push(u.n_simulated+' simulated');
  $('sUser').innerHTML=`Signed in as <b>${u.display_name}</b>. `+
    (u.enrolled?`Voiceprint enrolled (${u.n_enroll} clips).`:'No voiceprint yet — enrol below.')+
    (hist.length?` History: ${hist.join(' + ')} transactions.`:'');
  $('bRec').disabled=false;$('bPhrase').disabled=false;
  $('bEnrol').disabled=clips.length===0;
  $('bTest').disabled=!u.enrolled; $('up').disabled=!u.enrolled;
  $('sEnrol').textContent=u.enrolled?`Enrolled with ${u.n_enroll} clip(s).`:'Not enrolled.';
}

$('bLogin').onclick=async()=>{
  const id=($('uNew').value||$('uSel').value||'').trim();
  if(!id){$('sUser').textContent='Pick an existing user or type a new id.';return;}
  $('sUser').textContent='Signing in…';
  const fd=new FormData();fd.append('user_id',id);fd.append('display_name',id);
  const j=await(await fetch('/api/login',{method:'POST',body:fd})).json();
  if(j.error){$('sUser').textContent=j.error;return;}
  clips=[];$('bEnrol').textContent='Enrol (0)';
  paintUser(j);
  const c=await(await fetch('/api/config')).json();
  $('uSel').innerHTML='<option value="">— new user —</option>'+
    c.users.map(u=>`<option value="${u.user_id}" ${u.user_id===j.user_id?'selected':''}>${u.display_name}</option>`).join('');
  $('uNew').value='';
};

$('bRec').onclick=async()=>{$('bRec').disabled=true;
  countdown($('sEnrol'),$('bRec'),4,'Recording');
  clips.push(await rec(4));
  $('sEnrol').textContent=`${clips.length} clip(s) recorded.`;
  $('bEnrol').textContent=`Enrol (${clips.length})`;$('bEnrol').disabled=false;$('bRec').disabled=false;};

$('bPhrase').onclick=async()=>{
  $('bPhrase').disabled=true;
  countdown($('sPhrase'),$('bPhrase'),5,'Say your passphrase —');
  const blob=await rec(5);
  $('sPhrase').textContent='Transcribing…';
  const fd=new FormData();fd.append('file',blob,'phrase.webm');
  const j=await(await fetch('/api/enroll_passphrase',{method:'POST',body:fd})).json();
  $('bPhrase').disabled=false;
  $('sPhrase').textContent=j.error?j.error:`Passphrase set (${j.n_words} words): “${j.heard}”`;
};

$('bEnrol').onclick=async()=>{const fd=new FormData();
  clips.forEach((b,i)=>fd.append('files',b,`e${i}.webm`));
  $('sEnrol').textContent='Enrolling…';
  const j=await(await fetch('/api/enroll',{method:'POST',body:fd})).json();
  $('sEnrol').textContent=`Enrolled with ${j.n_samples} clip(s).`;
  $('bTest').disabled=false;$('up').disabled=false;};

$('bReset').onclick=async()=>{clips=[];await fetch('/api/reset',{method:'POST'});
  $('sEnrol').textContent='Not enrolled.';$('bEnrol').textContent='Enrol (0)';$('bEnrol').disabled=true;
  $('bTest').disabled=true;$('up').disabled=true;$('out').style.display='none';};

$('bTest').onclick=async()=>{$('bTest').disabled=true;
  const secs=+$('dur').value;
  countdown($('sTest'),$('bTest'),secs,'Recording');
  await send(await rec(secs),'req.webm');$('bTest').disabled=false;};
$('up').onchange=e=>{if(e.target.files[0])send(e.target.files[0],e.target.files[0].name)};

async function send(blob,name){
  $('sTest').textContent='Transcribing and analysing…';
  const fd=new FormData();fd.append('file',blob,name);
  const j=await(await fetch('/api/authorize',{method:'POST',body:fd})).json();
  if(j.error){$('sTest').textContent=j.error;return;}
  $('sTest').textContent='';LAST=j;render(j);
}

// The CHALLENGE outcome, acted on: a second utterance whose calibrated
// evidence is ADDED to the first (log-odds accumulate). The amount is carried
// over server-side so the challenge cannot be used to restate a smaller one.
$('bChal').onclick=async()=>{
  $('bChal').disabled=true;
  countdown($('sChal'),$('bChal'),6,'Listening');
  const blob=await rec(6);
  $('sChal').textContent='Re-verifying…';
  const fd=new FormData();fd.append('file',blob,'chal.webm');
  fd.append('amount',LAST.amount);fd.append('prior_llr',LAST.fused_llr);
  const j=await(await fetch('/api/challenge',{method:'POST',body:fd})).json();
  $('bChal').disabled=false;
  if(j.error){$('sChal').textContent=j.error;return;}
  const pw=j.passphrase||{};
  const pwMsg=pw.enrolled
    ? (pw.matched?'Passphrase matched. ':`Passphrase failed — heard “${pw.heard}”. `)
    : 'No passphrase enrolled. ';
  $('sChal').textContent=pwMsg+`Total confidence now ${j.fused_llr>=0?'+':''}${j.fused_llr.toFixed(2)}.`;
  LAST=j;render(j,true);
};

function render(j,isChallenge){
  $('out').style.display='block';

  // ---- assistant reply -----------------------------------------------------
  const a=j.assistant||{};
  if(a.text){
    $('replyCard').style.display='block';
    $('replyText').textContent='“'+a.text+'”';
    const au=$('replyAudio');
    if(a.audio_url){au.src=a.audio_url;au.style.display='block';au.play().catch(()=>{});}
    else{au.style.display='none';}

    $('chalRow').style.display=(j.outcome==='CHALLENGE')?'flex':'none';
    if(j.outcome!=='CHALLENGE')$('sChal').textContent='';
  } else {$('replyCard').style.display='none';}

  // ---- layer 4 detail ------------------------------------------------------
  const c=j.context;
  if(c && c.normality!=null){
    const it=c.intent||{};
    const sig=(v,label,note)=>v==null
      ? `<div class="ev off"><div class="ev-name">${label}</div><div class="reading">unavailable</div></div>`
      : `<div class="ev"><div class="ev-top"><div class="ev-name">${label}</div>
         <div class="llr ${v>0.7?'pos':v<0.4?'neg':'neu'}">${v.toFixed(3)}</div></div>
         <div class="reading">${note}</div></div>`;
    $('ctxCard').style.display='block';
    $('ctxBody').innerHTML=
      `<div class="muted" style="margin-bottom:12px">Three signals, multiplied → <b>${c.normality.toFixed(3)}</b></div>`+
      sig(c.s_injection,'Prompt injection <span class="ev-model">Prompt Guard 2</span>',
          c.injection_prob==null?'':'P(injection) = '+c.injection_prob.toFixed(4))+
      sig(c.s_semantic,'Social engineering <span class="ev-model">gpt-oss-120b</span>',
          it.available?`urgency ${it.urgency.toFixed(2)} · secrecy ${it.secrecy.toFixed(2)} · authority ${it.authority.toFixed(2)} · coercion ${it.coercion.toFixed(2)}`:'')+
      sig(c.s_behaviour,'Behavioural fit <span class="ev-model">user history</span>',
          (c.behaviour_detail&&c.behaviour_detail.n_history)
            ? `vs ${c.behaviour_detail.n_history} past transactions (median ${fmt(c.behaviour_detail.median_amount)})`+
              (c.behaviour_detail.amount_z!=null?` · this amount is ${c.behaviour_detail.amount_z.toFixed(1)}σ above typical`:'')
            : (c.behaviour_detail&&c.behaviour_detail.reason)||'')+
      (it.summary?`<div class="muted" style="margin-top:10px">Parsed intent: <b>${it.action}</b>`+
        (it.recipient?` → <b>${it.recipient}</b>`:'')+`. ${it.summary.replace(/</g,'&lt;')}</div>`:'')+
      `<div class="muted" style="margin-top:8px">Behavioural history is partly synthetic in this demo.</div>`;
  } else {$('ctxCard').style.display='none';}

  // what the system heard
  const u=j.understanding||{};
  const detected=u.amount!=null;
  const lv=(j.stake&&j.stake.level)||'';
  const lvColor={Minimal:'var(--faint)',Low:'var(--ok)',Moderate:'var(--warn)',
                 High:'var(--bad)',Critical:'var(--bad)'}[lv]||'var(--dim)';
  $('heard').innerHTML=`
    <div style="font-size:16px;line-height:1.6;margin-bottom:14px">
      ${u.transcript?('“'+u.transcript.replace(/</g,'&lt;')+'”')
        :'<span class="muted">'+(u.asr_available===false?'Transcription unavailable':'(nothing transcribed)')+'</span>'}
    </div>
    <div class="row" style="gap:30px;align-items:baseline">
      <div><div class="muted">risk</div>
        <div style="font-size:19px;font-weight:600;color:${lvColor}">${lv}</div></div>
      <div><div class="muted">${j.stake&&j.stake.is_monetary?'value':'action'}</div>
        <div style="font-size:${j.stake&&j.stake.is_monetary?'19px':'13px'};font-weight:500">
          ${j.stake?j.stake.headline.replace(/Rs\./g,'₹'):'—'}</div></div>
    </div>
    ${(u.candidates||[]).filter(c=>c.rejected_because).map(c=>
      `<div class="muted" style="margin-top:10px">Ignored “${c.source_text}” — ${c.rejected_because}.</div>`).join('')}
    ${u.note?`<div class="muted" style="margin-top:8px">${u.note}</div>`:''}
    <details style="margin-top:14px"><summary>Stake calculation</summary>
      <div class="deriv">${(j.stake?j.stake.explain:[]).map(x=>
        `<div>${x.replace(/</g,'&lt;').replace(/Rs\./g,'₹')}</div>`).join('')}</div></details>`;

  $('verdict').className='verdict v-'+j.outcome;
  $('vLabel').textContent=j.outcome;
  $('vSum').textContent=j.summary;
  const l=$('vLlr');l.textContent=(j.fused_llr>=0?'+':'')+j.fused_llr.toFixed(2);
  l.className='llr '+(j.fused_llr>0?'pos':j.fused_llr<0?'neg':'neu');

  // evidence cards
  $('evs').innerHTML=j.evidence.map(e=>{
    if(!e.available) return `<div class="ev off"><div class="ev-top">
      <div><div class="ev-name">${e.layer_id} · ${e.name}</div>
      <div class="ev-model">${e.threat}</div></div>
      <div class="llr neu">—</div></div></div>`;
    const mag=Math.min(Math.abs(e.llr)/5,1)*50, neg=e.llr<0;
    return `<div class="ev"><div class="ev-top">
      <div><div class="ev-name">${e.layer_id} · ${e.name}</div>
      <div class="ev-model">${e.threat}</div></div>
      <div class="llr ${e.llr>0?'pos':e.llr<0?'neg':'neu'}">${e.llr>=0?'+':''}${e.llr.toFixed(2)}</div></div>
      <div class="bar"><span class="mid"></span>
        <i class="${neg?'n':''}" style="${neg?`right:50%;width:${mag}%;left:auto`:`left:50%;width:${mag}%`}"></i></div>
      <div class="reading">${e.reading}</div></div>`;
  }).join('');

  // scale
  const lo=Math.min(j.thresholds.tau_deny,j.fused_llr)-2, hi=Math.max(j.thresholds.tau_allow,j.fused_llr)+2;
  const P=v=>((v-lo)/(hi-lo))*100;
  const pd=P(j.thresholds.tau_deny), pa=P(j.thresholds.tau_allow), pf=P(j.fused_llr);
  $('scale').innerHTML=`<div class="track"></div>
    <div class="zone" style="left:0;width:${pd}%;background:rgba(255,107,107,.32)"></div>
    <div class="zone" style="left:${pd}%;width:${pa-pd}%;background:rgba(245,165,36,.32)"></div>
    <div class="zone" style="left:${pa}%;right:0;background:rgba(53,200,143,.32)"></div>
    <div class="tick" style="left:${pd}%;background:var(--bad)"></div>
    <div class="tlabel" style="left:${pd}%">deny ${j.thresholds.tau_deny.toFixed(2)}</div>
    <div class="tick" style="left:${pa}%;background:var(--ok)"></div>
    <div class="tlabel" style="left:${pa}%">allow ${j.thresholds.tau_allow.toFixed(2)}</div>
    <div class="marker" style="left:${pf}%"></div>
    <div class="mlabel" style="left:${pf}%">evidence ${j.fused_llr>=0?'+':''}${j.fused_llr.toFixed(2)}</div>`;

  const l5=$('l5Out');
  l5.textContent=j.outcome;
  l5.className='llr '+(j.outcome==='ALLOW'?'pos':j.outcome==='DENY'?'neg':'neu');
  // The bar is described by RISK LEVEL, not by the rupee stake behind it.
  // For anything the user did not actually say an amount for, the figure is
  // our own estimate, and putting it on screen invites an argument about a
  // number that only sets strictness. The full arithmetic stays one click
  // away under "How the threshold was derived".
  $('l5Read').textContent=
    `Evidence ${j.fused_llr>=0?'+':''}${j.fused_llr.toFixed(2)} vs `+
    `${j.thresholds.tau_allow.toFixed(2)} needed to allow, `+
    `${j.thresholds.tau_deny.toFixed(2)} below which it is denied. `+
    `Bar set by ${((j.stake&&j.stake.level)||'the').toLowerCase()} risk for this request.`;

  $('deriv').innerHTML=j.derivation.map(d=>`<div>${d.replace(/</g,'&lt;')}</div>`).join('');
}

fetch('/api/config').then(r=>r.json()).then(c=>{
  $('variant').textContent=c.calibrated_layers.join(', ')+' calibrated'+
    (c.assistant_available?' · assistant online':' · assistant offline');
  $('uSel').innerHTML='<option value="">— new user —</option>'+
    c.users.map(u=>`<option value="${u.user_id}">${u.display_name}</option>`).join('');
  if(c.user){$('uSel').value=c.user.user_id;paintUser(c.user);
    if(c.has_passphrase)$('sPhrase').textContent='Passphrase enrolled.';}
});
$('uSel').onchange=()=>{if($('uSel').value)$('uNew').value='';};
</script></div></body></html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
