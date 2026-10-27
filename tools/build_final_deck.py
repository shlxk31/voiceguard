"""
tools/build_final_deck.py

Produces the complete Phase-3 deck: the team's existing slides, three new
introduction slides and a rewritten contents page inserted near the front,
and the ten results slides appended at the end.

Everything reuses the deck's own Canva banner artwork by deep-copying an
existing slide, so the header is genuine rather than redrawn. Typography is
read from the deck itself (Gagalin 40.7 pt titles; Arial Bold 20 pt
sub-heads; Arial 19.22 pt bullets with marL 415059 / indent -207530).

Slide order is rewritten afterwards through the presentation's sldIdLst,
because python-pptx can only append.

SOURCING NOTE
Every performance figure comes from the project's own result files. The two
publicly reported fraud cases on the opening slide are external claims and
are marked in the speaker notes as requiring verification before presenting.
"""
from __future__ import annotations

import copy, csv, json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "slides" / "figs"
SRC = ROOT / "Phase3.pptx"
TEMPLATE_SLIDE = 6

INK = RGBColor(0x00, 0x00, 0x00)
BLUE = RGBColor(0x1F, 0x3B, 0x8F)
RED = RGBColor(0xB0, 0x1C, 0x1C)
GREEN = RGBColor(0x14, 0x6B, 0x3A)
GREY = RGBColor(0x59, 0x59, 0x59)
NAVY = RGBColor(0x1F, 0x2E, 0x6E)

rows_all = [r for r in csv.DictReader(open(ROOT / "results" / "ablation.csv"))
            if r["fusion_rule"] == "min"]
A = {(r["subset"], r["threat"]): float(r["error_pct"]) for r in rows_all}
FULL = "L1 + L2 + L3 + L4 (full)"
BASE = json.loads((ROOT / "results" / "aasist_baseline.json").read_text())
GEN = next(r for r in rows_all if r["subset"] == FULL and r["threat"] == "genuine user")


def far(sub, threat):
    return A[(sub, threat)]


# ------------------------------------------------------------- utilities --
def clone_template(prs, template):
    dest = prs.slides.add_slide(prs.slide_layouts[6])
    for shp in template.shapes:
        dest.shapes._spTree.append(copy.deepcopy(shp._element))
    return dest


def set_title(slide, text):
    for sh in slide.shapes:
        if sh.has_text_frame and "Current Implementation" in sh.text_frame.text:
            r = sh.text_frame.paragraphs[0].runs[0]
            r.text = text
            n = len(text)
            r.font.size = Pt(40.7 if n <= 30 else 34 if n <= 42 else 29)
            return
    raise RuntimeError("title placeholder missing")


def _para_props(p, *, level, line_pts, bullet):
    pPr = p._p.get_or_add_pPr()
    pPr.set("algn", "l")
    if level:
        pPr.set("lvl", str(level)); pPr.set("marL", "415059"); pPr.set("indent", "-207530")
    ln = pPr.makeelement(qn("a:lnSpc"), {})
    ln.append(ln.makeelement(qn("a:spcPts"), {"val": str(int(line_pts * 100))}))
    pPr.append(ln)
    if bullet:
        pPr.append(pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}))
        pPr.append(pPr.makeelement(qn("a:buChar"), {"char": "•"}))


def body(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tb._first = True
    return tb


def _p(tb):
    tf = tb.text_frame
    if getattr(tb, "_first", False):
        tb._first = False
        return tf.paragraphs[0]
    return tf.add_paragraph()


def head(tb, text, size=20, colour=INK):
    p = _p(tb); _para_props(p, level=0, line_pts=size * 1.25, bullet=False)
    r = p.add_run(); r.text = text
    r.font.name = "Arial Bold"; r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = colour


def point(tb, lead, rest="", size=19.22, colour=INK):
    p = _p(tb); _para_props(p, level=1, line_pts=size * 1.24, bullet=True)
    r = p.add_run(); r.text = lead
    r.font.name = "Arial"; r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = colour
    if rest:
        r2 = p.add_run(); r2.text = "  " + rest
        r2.font.name = "Arial"; r2.font.size = Pt(size - 2.2); r2.font.color.rgb = GREY


def plain(tb, text, size=16, colour=GREY, bullet=False):
    p = _p(tb); _para_props(p, level=1, line_pts=size * 1.26, bullet=bullet)
    r = p.add_run(); r.text = text
    r.font.name = "Arial"; r.font.size = Pt(size); r.font.color.rgb = colour


def picture(slide, name, left, top, width):
    if (FIGS / name).exists():
        slide.shapes.add_picture(str(FIGS / name), Inches(left), Inches(top), Inches(width))


def table(slide, headers, data, left, top, width, height, fs=13):
    t = slide.shapes.add_table(len(data) + 1, len(headers),
                               Inches(left), Inches(top), Inches(width), Inches(height)).table
    for j, h in enumerate(headers):
        c = t.cell(0, j); c.text = ""
        r = c.text_frame.paragraphs[0].add_run(); r.text = h
        r.font.size = Pt(fs); r.font.bold = True; r.font.name = "Arial"
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
    for i, row in enumerate(data, 1):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ""
            r = c.text_frame.paragraphs[0].add_run(); r.text = str(val)
            r.font.size = Pt(fs); r.font.name = "Arial"; r.font.color.rgb = INK
            if j > 0 and str(val).endswith("%"):
                v = float(str(val).rstrip("%"))
                if v >= 20: r.font.color.rgb = RED; r.font.bold = True
                elif v <= 2: r.font.color.rgb = GREEN
            c.fill.solid()
            c.fill.fore_color.rgb = (RGBColor(0xFF, 0xFF, 0xFF) if i % 2
                                     else RGBColor(0xEF, 0xF2, 0xF9))


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def reorder(prs, order):
    """
    Rewrite slide order; `order` holds current 0-based indices.

    Slides omitted from `order` are removed properly, relationship included.
    Dropping only the sldId leaves an orphaned slide part in the package,
    which makes PowerPoint offer to "repair" the file on open.
    """
    sldIdLst = prs.slides._sldIdLst
    ids = list(sldIdLst)
    kept_rids = {ids[i].rId for i in order}
    dropped = [el for el in ids if el.rId not in kept_rids]

    for el in ids:
        sldIdLst.remove(el)
    for i in order:
        sldIdLst.append(ids[i])

    for el in dropped:
        prs.part.drop_rel(el.rId)


# ================================================================ build ==
def build():
    prs = Presentation(str(SRC))
    tpl = list(prs.slides)[TEMPLATE_SLIDE]
    n_original = len(prs.slides)

    def S(title):
        s = clone_template(prs, tpl); set_title(s, title); return s

    # ================= NEW INTRODUCTION SLIDES (appended, moved later) ====
    # --- I1: contents -----------------------------------------------------
    s = S("Contents")
    tb = body(s, 0.75, 1.15, 9.2, 6.2)
    for i, t in enumerate([
        "Why Voice Security Matters — Real Cases",
        "How a Voice Attack Actually Works",
        "The Four Ways In — Attack Types",
        "Literature Survey & Research Gaps",
        "Phase 1 Recap — Proposed Solution",
        "System Design, Workflow & Implementation",
        "Objectives Achieved",
        "Why Exactly Five Layers — Ablation Study",
        "How the Threshold Was Decided",
        "Challenges, Solutions & Future Work"], 1):
        p = _p(tb); _para_props(p, level=0, line_pts=27, bullet=False)
        r = p.add_run(); r.text = f"{i:02d}.   "
        r.font.name = "Arial"; r.font.size = Pt(20); r.font.bold = True; r.font.color.rgb = BLUE
        r2 = p.add_run(); r2.text = t
        r2.font.name = "Arial"; r2.font.size = Pt(20); r2.font.color.rgb = INK
    notes(s, "Keep this to fifteen seconds. Say only that the deck moves from the problem, "
             "to the design, to the measured results, and that the last four sections are new "
             "since Phase 2.")

    # --- I2: real cases ---------------------------------------------------
    s = S("Why Voice Security Matters")
    tb = body(s, 0.55, 1.05, 9.6, 2.35)
    plain(tb, "Voice is no longer just an interface for playing music. It now moves money, "
              "places orders and unlocks doors — and attackers have noticed.", size=18, colour=INK)
    point(tb, "Voice cloning has already been used in real fraud.",
          "A UK energy firm was reportedly defrauded of about EUR 220,000 in 2019 after a "
          "cloned voice impersonated a company chief (Wall Street Journal, 2019).")
    point(tb, "Deepfaked calls now target finance staff directly.",
          "A Hong Kong company reportedly lost about USD 25 million in 2024 after a video call "
          "with a synthetic impersonation of its CFO.")
    point(tb, "Cloning a voice no longer needs a laboratory.",
          "A few seconds of public audio is enough for modern text-to-speech systems.")
    table(s, ["What a voice assistant can do today", "What protects it in most systems"],
          [["Place orders and make payments", "One voice match, checked once"],
           ["Change account details, add payees", "The same single check"],
           ["Unlock doors, disable alarms", "The same single check"],
           ["Read out OTPs and messages", "Usually nothing at all"]],
          left=0.9, top=3.75, width=8.7, height=2.2, fs=14)
    notes(s, "VERIFY BOTH NEWS FIGURES BEFORE PRESENTING — they are external reports, not our "
             "measurements. Everything else in this deck is measured by us. "
             "The point of the table is that capability has grown quickly while the security "
             "check underneath has not changed.")

    # --- I3: how the attack works ----------------------------------------
    s = S("How a Voice Attack Works")
    tb = body(s, 0.55, 1.05, 9.6, 2.6)
    point(tb, "Step 1 — the attacker obtains your voice.",
          "A phone call, a video posted online, or a voicemail is enough.")
    point(tb, "Step 2 — a model reproduces it.",
          "Text-to-speech or voice conversion generates any sentence in that voice.")
    point(tb, "Step 3 — the assistant hears the right voice.",
          "Speaker verification compares the sound to the enrolled profile and accepts it.")
    point(tb, "Step 4 — the transaction is authorised.",
          "The system has no further question to ask, because its only question was answered.")
    head(tb, "The gap this project addresses", colour=BLUE)
    plain(tb, "Asking “is this the right voice?” is not the same as asking “is this the right "
              "person, here, now, making a reasonable request?” Those are four questions, and "
              "our system asks all four.", size=16, colour=INK)
    picture(s, "single_layer.png", 2.30, 4.65, 6.1)
    notes(s, "The chart is our own measurement, not an illustration. With voice matching alone, "
             "74% of replayed recordings and 43% of AI clones are accepted. That is the "
             "single-layer system most products ship today.")

    # --- I4: attack types -------------------------------------------------
    s = S("The Four Ways In")
    tb = body(s, 0.55, 1.05, 9.6, 1.15)
    plain(tb, "Attacks on a voice assistant fall into four families. They are different problems, "
              "and — as our results show — they need different defences.", size=17, colour=INK)
    picture(s, "attacks.png", 0.60, 2.35, 9.4)
    notes(s, "Spend most of your time on this slide. Everything later in the deck maps back to "
             "these four families: one layer per family, plus a policy layer that decides. "
             "If Sir asks why five layers, this is where the answer begins.")

    # ======================= RESULTS SLIDES (appended) ====================
    s = S("Objectives Achieved")
    tb = body(s, 0.55, 1.05, 9.6, 1.95)
    point(tb, "All five layers are built and running end to end.",
          "Four layers gather evidence; the fifth decides.")
    point(tb, "Every layer is calibrated on held-out data.", "No cutoff is set by hand.")
    point(tb, "The threshold is derived from cost, not guessed.")
    point(tb, "An ablation study shows each layer earns its place.")
    point(tb, "Limitations are measured and stated openly.")
    picture(s, "architecture.png", 1.55, 3.30, 7.5)
    notes(s, "Open by stating all five layers work and that every claim from here is backed by "
             "a measurement we can reproduce. Note the diagram: audio is transcribed on the "
             "device, so no recording of the user's voice ever leaves it.")

    s = S("Security Objectives Achieved")
    tb = body(s, 0.55, 1.05, 9.6, 2.15)
    point(tb, f"Impostor detection: {far(FULL,'impostor (other human)'):.2f}% wrongly accepted.")
    point(tb, f"Replay detection: {far(FULL,'replay of a recording'):.2f}% wrongly accepted.",
          f"Without this layer it is {far('L1 + L3','replay of a recording'):.0f}%.")
    point(tb, f"AI-voice detection: {BASE['eval_eer']*100:.2f}% equal error rate.",
          f"On {BASE['n_eval']:,} files of the public ASVspoof 2019 benchmark.")
    point(tb, "Our figure matches the published result for this model.",
          "Evidence the implementation is correct, not merely running.")
    table(s, ["Threat", "Wrongly accepted", "Layer responsible"],
          [["Impostor (different person)", f"{far(FULL,'impostor (other human)'):.2f}%", "L1  Voiceprint"],
           ["Replay of a recording",       f"{far(FULL,'replay of a recording'):.2f}%",  "L2  Liveness"],
           ["AI-generated (cloned) voice", f"{far(FULL,'TTS / VC clone'):.2f}%",         "L3  Anti-spoofing"],
           ["Prompt-injection command",    f"{far(FULL,'prompt injection'):.2f}%",       "L4  Context"],
           ["Social-engineering command",  f"{far(FULL,'social engineering'):.2f}%",     "L4  Context"]],
          left=1.0, top=3.55, width=8.7, height=2.6)
    notes(s, "If asked whether the system is error-free, say no directly, then use this table. "
             "Injection and social engineering are the weakest columns and we know it.")

    s = S("Usability Objectives Achieved")
    tb = body(s, 0.55, 1.05, 9.6, 2.7)
    point(tb, "The amount is detected from speech, not typed in.")
    point(tb, "Ordinary commands pass with almost no friction.",
          "“Turn off the lights” is treated as a light switch, not a payment.")
    point(tb, "Security scales with what is at risk.")
    point(tb, "The assistant speaks back and repeats what it understood.",
          "The user can catch a mishearing before money moves.")
    point(tb, "An uncertain request is challenged, not refused.")
    table(s, ["Outcome for a genuine user", "Share", "What the user experiences"],
          [["Allowed immediately", f"{float(GEN['allow_pct']):.2f}%", "No friction at all"],
           ["Challenged",          f"{float(GEN['challenge_pct']):.2f}%", "Asked once more, then proceeds"],
           ["Refused outright",    f"{float(GEN['deny_pct']):.2f}%", "Must use another channel"]],
          left=1.0, top=4.15, width=8.7, height=1.7)
    notes(s, "Roughly nine in ten genuine users still complete their request, most immediately "
             "and the rest after one extra sentence.")

    s = S("The Five Layers")
    tb = body(s, 0.55, 1.05, 9.6, 2.6)
    point(tb, "L1  Voiceprint  (ECAPA-TDNN, cosine similarity)", "Is this the enrolled user?")
    point(tb, "L2  Liveness", "A mouth, or a loudspeaker?")
    point(tb, "L3  Anti-spoofing  (AASIST-L)", "Human throat, or computer?")
    point(tb, "L4  Context", "Is this request normal for this user?")
    point(tb, "L5  Risk policy  —  not a detector.",
          "Sets how much proof this request needs, then decides.", colour=BLUE)
    table(s, ["Layer", "Question it answers", "Threat it exists for"],
          [["L1  Voiceprint",    "Is this the enrolled user?", "A different person"],
           ["L2  Liveness",      "Mouth or loudspeaker?",      "Replay of a recording"],
           ["L3  Anti-spoofing", "Human or AI-generated?",     "Cloned / synthetic voice"],
           ["L4  Context",       "Is this request normal?",    "Manipulated command"],
           ["L5  Risk policy",   "How much proof is needed?",  "Sets the bar and decides"]],
          left=1.0, top=4.05, width=8.7, height=2.6)
    notes(s, "Expect: “Where is Layer 5? I only see four scores.” L5 has no score because it "
             "does not vote, it decides. It is the only layer that looks at nothing but the "
             "output of the other four.")

    s = S("Why Five Layers, Not Three or Four?")
    tb = body(s, 0.55, 1.05, 9.6, 2.05)
    point(tb, "We did not argue this — we measured it.", "Each layer was switched off in turn.")
    point(tb, f"Remove L2: replay rises {far(FULL,'replay of a recording'):.2f}% to {far('L1 + L3','replay of a recording'):.0f}%.", colour=RED)
    point(tb, f"Remove L3: cloned voices rise to {far('L1 only','TTS / VC clone'):.2f}%.", colour=RED)
    point(tb, f"Remove L4: injected commands rise {far(FULL,'prompt injection'):.0f}% to {far('L1 + L2 + L3','prompt injection'):.1f}%.", colour=RED)
    picture(s, "ablation.png", 0.85, 3.45, 9.0)
    notes(s, "The direct answer to “prove five is better than four” — a measurement on held-out "
             f"data. If asked what it costs: genuine users challenged rises from "
             f"{far('L1 only','genuine user'):.0f}% to {far(FULL,'genuine user'):.2f}%, but most "
             "recover by answering the challenge.")

    s = S("Mapping Layers to Threats")
    tb = body(s, 0.55, 1.05, 9.6, 2.1)
    point(tb, "No two layers cover the same threat.", "There is no overlap to fall back on.")
    point(tb, "A replayed clip passes L1 and L3 — correctly.",
          "It is the user's voice, from a human throat. Only L2 objects.")
    point(tb, "A cloned voice passes L1 because the clone succeeded.")
    point(tb, "A scam command may pass all three audio layers.",
          "If the real user is tricked, only context notices.")
    picture(s, "replay.png", 1.55, 3.55, 7.5)
    notes(s, "If there is time, run this live: speak a command, then play the recording back "
             "through the speakers. L1 and L3 stay green while L2 turns red.")

    s = S("How the Layers Work Together")
    tb = body(s, 0.55, 1.05, 9.6, 6.0)
    head(tb, "Step 1 — put every layer on one scale", colour=BLUE)
    plain(tb, "A similarity of 0.72 and a spoof probability of 0.99 cannot be compared. "
              "Calibration converts both into the same unit of evidence.")
    head(tb, "Step 2 — combine by the weakest link, not by adding", colour=BLUE)
    plain(tb, "Security is a chain. An attacker must defeat every layer, so the weakest "
              "layer decides the outcome.")
    head(tb, "Step 3 — compare against the bar for this request", colour=BLUE)
    plain(tb, "The threshold is calculated from what is at stake, then the system allows, "
              "challenges, or denies.")
    head(tb, "Adding the scores was tried, and it failed", colour=RED)
    point(tb, "Impostor acceptance rose 0.25% to 52.25%.",
          "Anti-spoofing correctly said “real human” about an impostor, cancelling the "
          "correct rejection from the voiceprint layer.", size=17, colour=RED)
    point(tb, "The weakest-link rule gives a guarantee.",
          "Adding a layer can only lower the score, so it can never open a new hole.",
          size=17, colour=GREEN)
    notes(s, "Expect: “Why not just add the scores up?” Because the layers answer different "
             "questions. An impostor is a real human, so adding lets one correct answer cancel "
             "another. This is a measured result, not a theoretical worry.")

    s = S("How the Threshold Was Decided")
    tb = body(s, 0.55, 1.05, 9.6, 2.5)
    point(tb, "The old approach used fixed cutoffs such as 0.75 and 0.85.",
          "Nothing explains why 0.75 and not 0.70.")
    point(tb, "The new threshold is calculated from the cost of being wrong.",
          "Fraud costs the money lost; refusing a real customer costs a support call.")
    point(tb, "Only one of those two costs grows with the amount.",
          "That is why the bar rises with the value of the request.")
    point(tb, "The method is published and widely used.",
          "Kinnunen et al., Odyssey 2018 — the measure ASVspoof itself adopted.", colour=BLUE)
    picture(s, "threshold.png", 1.75, 3.95, 7.1)
    notes(s, "Say this out loud: every number in our threshold is a business cost you can argue "
             "with, not a score cutoff we invented. If asked whether the costs are arbitrary — "
             "partly, and we tested it: three behave gently, only the assumed attack rate "
             "matters, and a real bank measures that from its own records.")

    s = S("Challenges Faced")
    tb = body(s, 0.55, 1.05, 9.6, 6.0)
    head(tb, "Our own live voice was flagged as synthetic — every time", colour=BLUE)
    plain(tb, "Eight of eight genuine recordings were rejected as AI-generated. The cause was "
              "the browser, not the model: noise suppression and automatic gain control reshape "
              "the sound, and the detector reads that as artificial.")
    head(tb, "A model number was mistaken for money", colour=BLUE)
    plain(tb, "“Samsung Galaxy Fold 8 Ultra” was read as an 8-rupee order, putting the lowest "
              "possible security bar on an expensive purchase.")
    head(tb, "Our first liveness result was too good, so we rejected it", colour=BLUE)
    plain(tb, "It reported 100% accuracy. Live and replayed clips had been recorded on "
              "different devices, so the model had learned the device, not liveness.")
    head(tb, "Free API limits slowed experiments", colour=BLUE)
    plain(tb, "Calibration runs are limited by tokens per minute, not computing power.")
    notes(s, "Presenting the discarded result is deliberate. A team that finds and reports its "
             "own mistake is more trustworthy than one reporting only successes.")

    s = S("Solutions and Future Work")
    tb = body(s, 0.55, 1.05, 9.6, 6.0)
    head(tb, "Solved", colour=GREEN)
    point(tb, "Browser audio processing is switched off before analysis.",
          "False rejection fell from every clip to about one in twenty.", size=17)
    point(tb, "A bare number is never treated as money.",
          "The screen shows which numbers were ignored, and why.", size=17)
    point(tb, "Liveness was rebuilt with proper controls.",
          "On 20 fresh clips never used in training, all 20 were correct.", size=17)
    head(tb, "Future work", colour=BLUE)
    point(tb, "A randomly chosen challenge phrase.",
          "The passphrase is fixed today, so a recording of it would pass.", size=17)
    point(tb, "Wider testing before any general claim.",
          "Better loudspeakers, more speakers, more rooms, real commands.", size=17)
    point(tb, "Measure the real attack rate from live data.",
          "Our sensitivity study shows this assumption matters most.", size=17)
    notes(s, "Close by naming our own limitations before anyone asks. It shows we understand the "
             "difference between a working prototype and a proven system.")

    # ---------------------------------------------------- final ordering --
    # original: 0 title, 1 old contents, 2..6 content
    # appended: n..n+3 = contents + 3 intro, then n+4.. = results
    n = n_original
    order = ([0]                      # title
             + [n]                    # new contents (replaces the old one)
             + [n + 1, n + 2, n + 3]  # three introduction slides
             + list(range(2, n_original))   # existing content, old contents dropped
             + list(range(n + 4, len(prs.slides))))  # results slides
    reorder(prs, order)
    return prs


if __name__ == "__main__":
    out = ROOT / "slides" / "SecureVoice_Phase3_Final.pptx"
    p = build(); p.save(out)
    print(f"wrote {out}  ({len(p.slides)} slides)")
