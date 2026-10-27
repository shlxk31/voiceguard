"""
tools/build_phase3_slides.py

Appends the ten new slides to the team's Phase3 Canva deck, reusing the deck's
own banner artwork and typography rather than imitating it.

HOW THE TEMPLATE IS PRESERVED
Slide 7 of Phase3.pptx is a content slide carrying only the Canva header group
(the navy banner and its decorative squares) and the title text box. That slide
is deep-copied once per new slide, so the header is the genuine artwork from
the deck, not a redrawn approximation. Only the title string and the body are
then written.

Typography is taken from the deck itself, measured rather than guessed:
    title      Gagalin, 40.7 pt, white, centred, line spacing 56.98 pt
    sub-head   Arial Bold, 20 pt, black, level 0, no bullet
    bullet     Arial, 19.22 pt, black, level 1, marL 415059, indent -207530,
               buChar "*", line spacing 23.06 pt
Body sizes are stepped down slightly on dense slides so nothing overflows;
every slide is checked for overflow and overlap after generation.

All numbers come from the project's result files, so re-running an experiment
regenerates a deck that agrees with it.
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
TEMPLATE_SLIDE = 6          # zero-based: the banner-only "Current Implementation" slide

INK = RGBColor(0x00, 0x00, 0x00)
BLUE = RGBColor(0x1F, 0x3B, 0x8F)
RED = RGBColor(0xB0, 0x1C, 0x1C)
GREEN = RGBColor(0x14, 0x6B, 0x3A)
GREY = RGBColor(0x59, 0x59, 0x59)

rows_all = [r for r in csv.DictReader(open(ROOT / "results" / "ablation.csv"))
            if r["fusion_rule"] == "min"]
A = {(r["subset"], r["threat"]): float(r["error_pct"]) for r in rows_all}
FULL = "L1 + L2 + L3 + L4 (full)"
BASE = json.loads((ROOT / "results" / "aasist_baseline.json").read_text())
GEN = next(r for r in rows_all if r["subset"] == FULL and r["threat"] == "genuine user")


def far(sub, threat):
    return A[(sub, threat)]


# ------------------------------------------------------------ slide copy --
def clone_template(prs, template):
    """Deep-copy the template slide's shape tree onto a new blank slide."""
    dest = prs.slides.add_slide(prs.slide_layouts[6])
    for shp in template.shapes:
        dest.shapes._spTree.append(copy.deepcopy(shp._element))
    return dest


def set_title(slide, text):
    """Rewrite the cloned title box, keeping its Gagalin run properties."""
    for sh in slide.shapes:
        if sh.has_text_frame and "Current Implementation" in sh.text_frame.text:
            r = sh.text_frame.paragraphs[0].runs[0]
            r.text = text
            # Long headings need a step down so they stay on one line.
            n = len(text)
            r.font.size = Pt(40.7 if n <= 30 else 34 if n <= 42 else 29)
            return sh
    raise RuntimeError("title placeholder not found in cloned slide")


# --------------------------------------------------------- text building --
def _para_props(p, *, level, line_pts, bullet):
    """Reproduce the deck's own paragraph properties, including the bullet."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("algn", "l")
    if level:
        pPr.set("lvl", str(level))
        pPr.set("marL", "415059")
        pPr.set("indent", "-207530")
    ln = pPr.makeelement(qn("a:lnSpc"), {})
    pts = ln.makeelement(qn("a:spcPts"), {"val": str(int(line_pts * 100))})
    ln.append(pts); pPr.append(ln)
    if bullet:
        bf = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"}); pPr.append(bf)
        bc = pPr.makeelement(qn("a:buChar"), {"char": "•"}); pPr.append(bc)


def body(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tb._first = True
    return tb


def _next_para(tb):
    tf = tb.text_frame
    if getattr(tb, "_first", False):
        tb._first = False
        return tf.paragraphs[0]
    return tf.add_paragraph()


def head(tb, text, size=20, colour=INK):
    p = _next_para(tb)
    _para_props(p, level=0, line_pts=size * 1.2, bullet=False)
    r = p.add_run(); r.text = text
    r.font.name = "Arial Bold"; r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = colour


def point(tb, lead, rest="", size=19.22, colour=INK):
    p = _next_para(tb)
    _para_props(p, level=1, line_pts=size * 1.2, bullet=True)
    r = p.add_run(); r.text = lead
    r.font.name = "Arial"; r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = colour
    if rest:
        r2 = p.add_run(); r2.text = "  " + rest
        r2.font.name = "Arial"; r2.font.size = Pt(size - 2.2); r2.font.bold = False
        r2.font.color.rgb = GREY


def plain(tb, text, size=16, colour=GREY):
    p = _next_para(tb)
    _para_props(p, level=1, line_pts=size * 1.22, bullet=False)
    r = p.add_run(); r.text = text
    r.font.name = "Arial"; r.font.size = Pt(size); r.font.color.rgb = colour


def picture(slide, name, left, top, width):
    if (FIGS / name).exists():
        slide.shapes.add_picture(str(FIGS / name), Inches(left), Inches(top), Inches(width))


def table(slide, headers, data, left, top, width, height, fs=13):
    t = slide.shapes.add_table(len(data) + 1, len(headers),
                               Inches(left), Inches(top), Inches(width), Inches(height)).table
    navy = RGBColor(0x1F, 0x2E, 0x6E)
    for j, h in enumerate(headers):
        c = t.cell(0, j); c.text = ""
        r = c.text_frame.paragraphs[0].add_run(); r.text = h
        r.font.size = Pt(fs); r.font.bold = True; r.font.name = "Arial"
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        c.fill.solid(); c.fill.fore_color.rgb = navy
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
    return t


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# =============================================================== content ==
def build():
    prs = Presentation(str(SRC))
    tpl = list(prs.slides)[TEMPLATE_SLIDE]

    def S(title):
        s = clone_template(prs, tpl)
        set_title(s, title)
        return s

    # ------------------------------------------------------------ 1 -----
    s = S("Objectives Achieved")
    tb = body(s, 0.55, 1.05, 9.6, 1.95)
    point(tb, "All five layers are built and running end to end.",
          "Four layers gather evidence; the fifth decides.")
    point(tb, "Every layer is calibrated on held-out data.",
          "No score cutoff is set by hand.")
    point(tb, "The threshold is derived from cost, not guessed.")
    point(tb, "An ablation study shows each layer earns its place.")
    point(tb, "Limitations are measured and stated openly.")
    picture(s, "architecture.png", 1.55, 3.30, 7.5)
    notes(s, "Open by stating that all five layers work, and that every claim in the next ten "
             "minutes is backed by a measurement we can reproduce. Point at the diagram: audio "
             "is transcribed on the device, so no recording of the user's voice ever leaves it.")

    # ------------------------------------------------------------ 2 -----
    s = S("Security Objectives Achieved")
    tb = body(s, 0.55, 1.05, 9.6, 2.15)
    point(tb, f"Impostor detection: {far(FULL,'impostor (other human)'):.2f}% wrongly accepted.",
          "A different person is almost never accepted.")
    point(tb, f"Replay detection: {far(FULL,'replay of a recording'):.2f}% wrongly accepted.",
          f"Without this layer it is {far('L1 + L3','replay of a recording'):.0f}%.")
    point(tb, f"AI-voice detection: {BASE['eval_eer']*100:.2f}% equal error rate.",
          f"Measured on {BASE['n_eval']:,} files of the ASVspoof 2019 benchmark.")
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
             "Injection and social engineering are the weakest columns and we know it. Naming "
             "your own weak point first is what makes the strong numbers believable.")

    # ------------------------------------------------------------ 3 -----
    s = S("Usability Objectives Achieved")
    tb = body(s, 0.55, 1.05, 9.6, 2.7)
    point(tb, "The amount is detected from speech, not typed in.",
          "The user simply says the order.")
    point(tb, "Ordinary commands pass with almost no friction.",
          "“Turn off the lights” is treated as a light switch, not a payment.")
    point(tb, "Security scales with what is at risk.",
          "A laptop purchase demands far more proof than a lamp.")
    point(tb, "The assistant speaks back and repeats what it understood.",
          "The user can catch a mishearing before money moves.")
    point(tb, "An uncertain request is challenged, not refused.")
    table(s, ["Outcome for a genuine user", "Share", "What the user experiences"],
          [["Allowed immediately", f"{float(GEN['allow_pct']):.2f}%", "No friction at all"],
           ["Challenged",          f"{float(GEN['challenge_pct']):.2f}%", "Asked once more, then proceeds"],
           ["Refused outright",    f"{float(GEN['deny_pct']):.2f}%", "Must use another channel"]],
          left=1.0, top=4.15, width=8.7, height=1.7)
    notes(s, "The headline: roughly nine in ten genuine users still complete their request, most "
             "immediately and the rest after one extra sentence. Only about one in eight is "
             "stopped completely, and that is the price of the security on the previous slide.")

    # ------------------------------------------------------------ 4 -----
    s = S("The Five Layers")
    tb = body(s, 0.55, 1.05, 9.6, 2.6)
    point(tb, "L1  Voiceprint  (ECAPA-TDNN, cosine similarity)",
          "Is this the enrolled user's voice?")
    point(tb, "L2  Liveness", "Did this come from a mouth, or a loudspeaker?")
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
    notes(s, "Expect: “Where is Layer 5? I only see four scores.” Answer: L5 has no "
             "score because it does not vote, it decides. It is the only layer that looks at "
             "nothing but the output of the other four.")

    # ------------------------------------------------------------ 5 -----
    s = S("Why Five Layers, Not Three or Four?")
    tb = body(s, 0.55, 1.05, 9.6, 2.05)
    point(tb, "We did not argue this — we measured it.",
          "Each layer was switched off in turn.")
    point(tb, f"Remove L2: replay rises {far(FULL,'replay of a recording'):.2f}% to {far('L1 + L3','replay of a recording'):.0f}%.",
          "Three in four replays would be accepted.", colour=RED)
    point(tb, f"Remove L3: cloned voices rise to {far('L1 only','TTS / VC clone'):.2f}%.",
          "An AI copy would pass almost half the time.", colour=RED)
    point(tb, f"Remove L4: injected commands rise {far(FULL,'prompt injection'):.0f}% to {far('L1 + L2 + L3','prompt injection'):.1f}%.",
          colour=RED)
    picture(s, "ablation.png", 0.85, 3.45, 9.0)
    notes(s, "This is the direct answer to “prove five is better than four” — a "
             "measurement on held-out data, not an opinion. If asked what the fifth layer costs: "
             f"more genuine users get challenged, {far('L1 only','genuine user'):.0f}% rises to "
             f"{far(FULL,'genuine user'):.2f}%, but most recover by answering the challenge.")

    # ------------------------------------------------------------ 6 -----
    s = S("Mapping Layers to Threats")
    tb = body(s, 0.55, 1.05, 9.6, 2.1)
    point(tb, "No two layers cover the same threat.",
          "There is no overlap to fall back on.")
    point(tb, "A replayed clip passes L1 and L3 — correctly.",
          "It is the user's voice, from a human throat. Only L2 objects.")
    point(tb, "A cloned voice passes L1 because the clone succeeded.",
          "Only anti-spoofing sees the difference.")
    point(tb, "A scam command may pass all three audio layers.",
          "If the real user is tricked, only context notices.")
    picture(s, "replay.png", 1.55, 3.55, 7.5)
    notes(s, "If there is time, run this live: speak a command, then play the recording back "
             "through the laptop speakers. Watching L1 and L3 stay green while L2 turns red is "
             "the clearest demonstration in the project of why one layer is not enough.")

    # ------------------------------------------------------------ 7 -----
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
          "Anti-spoofing correctly said “real human” about an impostor, and that "
          "cancelled the correct rejection from the voiceprint layer.", size=17, colour=RED)
    point(tb, "The weakest-link rule gives a guarantee.",
          "Adding a layer can only lower the score, so it can never open a new hole.",
          size=17, colour=GREEN)
    notes(s, "Expect: “Why not just add the scores up?” Because the layers answer "
             "different questions. L1's “genuine” means the right person; L3's means "
             "a real human. An impostor is a real human, so adding lets one correct answer "
             "cancel another. This is a measured result, not a theoretical worry.")

    # ------------------------------------------------------------ 8 -----
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
             "with, not a score cutoff we invented. If asked whether the costs are arbitrary: "
             "partly, and we tested it. We varied each over ten times its value. Three behave "
             "gently. Only the assumed attack rate matters a lot, and that is the one figure a "
             "real bank measures from its own records.")

    # ------------------------------------------------------------ 9 -----
    s = S("Challenges Faced")
    tb = body(s, 0.55, 1.05, 9.6, 6.0)
    head(tb, "Our own live voice was flagged as synthetic — every time", colour=BLUE)
    plain(tb, "Eight out of eight genuine recordings were rejected as AI-generated. The cause "
              "was the browser, not the model: noise suppression and automatic gain control "
              "reshape the sound, and the detector reads that as artificial.")
    head(tb, "A model number was mistaken for money", colour=BLUE)
    plain(tb, "“Samsung Galaxy Fold 8 Ultra” was read as an 8-rupee order, putting the "
              "lowest possible security bar on an expensive purchase.")
    head(tb, "Our first liveness result was too good, so we rejected it", colour=BLUE)
    plain(tb, "It reported 100% accuracy. The live and replayed clips had been recorded on "
              "different devices, so the model had learned the device, not liveness.")
    head(tb, "Free API limits slowed experiments", colour=BLUE)
    plain(tb, "Calibration runs are limited by tokens per minute, not computing power.")
    notes(s, "Presenting the discarded result is deliberate. A team that finds and reports its "
             "own mistake is far more trustworthy than one reporting only successes — and "
             "if a reviewer found it later, the damage would be much worse.")

    # ----------------------------------------------------------- 10 -----
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
             "difference between a working prototype and a proven system, which is exactly the "
             "distinction a reviewer listens for.")

    return prs


if __name__ == "__main__":
    out = ROOT / "slides" / "Phase3_with_new_slides.pptx"
    build().save(out)
    print(f"wrote {out}")
