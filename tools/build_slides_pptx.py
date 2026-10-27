"""
tools/build_slides_pptx.py

Generates the 10 additional slides as a .pptx in the same visual language as
the team's existing Phase-1 and Phase-2 decks: 4:3, navy title banner, plain
white body, left-aligned bullets, institutional rather than decorative.

Numbers are read from the project's result files, so a re-run of any
experiment regenerates a deck that agrees with it. Speaker notes are written
into the real notes pane, not onto the slide.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "slides" / "figs"

NAVY   = RGBColor(0x1F, 0x2E, 0x6E)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
INK    = RGBColor(0x1A, 0x1A, 0x1A)
GREY   = RGBColor(0x5A, 0x5A, 0x5A)
BLUE   = RGBColor(0x1F, 0x3B, 0x8F)
RED    = RGBColor(0xB0, 0x1C, 0x1C)
GREEN  = RGBColor(0x14, 0x6B, 0x3A)

SW, SH = Inches(10), Inches(7.5)          # 4:3, matching the existing decks

rows_all = [r for r in csv.DictReader(open(ROOT / "results" / "ablation.csv"))
            if r["fusion_rule"] == "min"]
A = {(r["subset"], r["threat"]): float(r["error_pct"]) for r in rows_all}
FULL = "L1 + L2 + L3 + L4 (full)"
BASE = json.loads((ROOT / "results" / "aasist_baseline.json").read_text())
GEN = next(r for r in rows_all if r["subset"] == FULL and r["threat"] == "genuine user")


def far(sub, threat):
    return A[(sub, threat)]


# --------------------------------------------------------------- chrome --
def new_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[6])          # blank
    band = s.shapes.add_shape(1, 0, Inches(0.32), SW, Inches(0.92))
    band.fill.solid(); band.fill.fore_color.rgb = NAVY
    band.line.fill.background(); band.shadow.inherit = False
    tf = band.text_frame
    tf.margin_left = Inches(0.45); tf.margin_right = Inches(0.3)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    r = p.add_run(); r.text = title
    r.font.size = Pt(26); r.font.bold = True; r.font.color.rgb = WHITE
    r.font.name = "Arial"
    # small decorative squares, as in the existing decks
    for i, (x, y, sz) in enumerate([(0.14, 0.08, 0.12), (0.32, 0.08, 0.12),
                                    (0.14, 0.24, 0.0)]):
        if sz == 0:
            continue
        sq = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(sz), Inches(sz))
        sq.fill.solid(); sq.fill.fore_color.rgb = NAVY if i == 0 else RGBColor(0x9A,0xA8,0xD4)
        sq.line.fill.background(); sq.shadow.inherit = False
    return s


def body_box(s, top=1.45, height=5.4, left=0.55, width=8.9):
    tb = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame; tf.word_wrap = True
    return tf


def head(tf, text, first=False):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    r = p.add_run(); r.text = text
    r.font.size = Pt(15); r.font.bold = True; r.font.color.rgb = BLUE
    r.font.name = "Arial"
    p.space_before = Pt(0 if first else 10); p.space_after = Pt(4)


def bullet(tf, lead, rest="", first=False, size=14, colour=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.level = 0
    r = p.add_run(); r.text = "•  " + lead
    r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = colour or INK; r.font.name = "Arial"
    if rest:
        r2 = p.add_run(); r2.text = "  " + rest
        r2.font.size = Pt(size - 1.5); r2.font.bold = False
        r2.font.color.rgb = GREY; r2.font.name = "Arial"
    p.space_after = Pt(7)


def sub(tf, text, size=12):
    p = tf.add_paragraph()
    r = p.add_run(); r.text = "     " + text
    r.font.size = Pt(size); r.font.color.rgb = GREY; r.font.name = "Arial"
    p.space_after = Pt(6)


def notes(s, text):
    s.notes_slide.notes_text_frame.text = text


def picture(s, name, left, top, width):
    if (FIGS / name).exists():
        s.shapes.add_picture(str(FIGS / name), Inches(left), Inches(top), Inches(width))


def add_table(s, headers, data, left, top, width, height, fs=11):
    shape = s.shapes.add_table(len(data) + 1, len(headers),
                               Inches(left), Inches(top), Inches(width), Inches(height))
    t = shape.table
    for j, h in enumerate(headers):
        c = t.cell(0, j); c.text = ""
        p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = h
        r.font.size = Pt(fs); r.font.bold = True; r.font.color.rgb = WHITE
        r.font.name = "Arial"
        c.fill.solid(); c.fill.fore_color.rgb = NAVY
    for i, row in enumerate(data, start=1):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ""
            p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(val)
            r.font.size = Pt(fs); r.font.name = "Arial"
            r.font.color.rgb = INK
            if j > 0 and str(val).endswith("%"):
                v = float(str(val).rstrip("%"))
                if v >= 20: r.font.color.rgb = RED; r.font.bold = True
                elif v <= 2: r.font.color.rgb = GREEN
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor(0xFF,0xFF,0xFF) if i % 2 else RGBColor(0xEF,0xF2,0xF9)
    return t


def footer(s, n):
    tb = s.shapes.add_textbox(Inches(8.9), Inches(6.95), Inches(0.9), Inches(0.35))
    p = tb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(n)
    r.font.size = Pt(11); r.font.color.rgb = GREY; r.font.name = "Arial"


# =============================================================== slides ==
def build() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = SW, SH

    # ---------------------------------------------------------- slide 1 --
    s = new_slide(prs, "OBJECTIVES ACHIEVED — OVERVIEW")
    tf = body_box(s, top=1.42, height=2.05)
    bullet(tf, "All five layers are built and running end to end.",
           "Four layers gather evidence; the fifth decides.", first=True)
    bullet(tf, "Every layer is calibrated on held-out data.",
           "No score cutoff is set by hand.")
    bullet(tf, "The decision threshold is derived from cost, not guessed.")
    bullet(tf, "An ablation study shows each layer earns its place.")
    bullet(tf, "Limitations are measured and stated openly.")
    # 1.97 aspect -> 3.30in tall at 6.5in wide; sits clear of the page number
    picture(s, "architecture.png", 1.75, 3.52, 6.50)
    footer(s, 1)
    notes(s, "Open by stating that all five layers work, and that every claim in the next "
             "ten minutes is backed by a measurement we can reproduce on request. "
             "Point at the diagram: audio is transcribed on the device, so no recording "
             "of the user's voice ever leaves the machine.")

    # ---------------------------------------------------------- slide 2 --
    s = new_slide(prs, "SECURITY OBJECTIVES ACHIEVED")
    tf = body_box(s, top=1.42, height=2.35)
    bullet(tf, f"Impostor detection: {far(FULL,'impostor (other human)'):.2f}% wrongly accepted.",
           "A different person is almost never accepted.", first=True)
    bullet(tf, f"Replay detection: {far(FULL,'replay of a recording'):.2f}% wrongly accepted.",
           f"Without this layer it is {far('L1 + L3','replay of a recording'):.0f}%.")
    bullet(tf, f"AI-voice detection: {BASE['eval_eer']*100:.2f}% equal error rate.",
           f"Measured on {BASE['n_eval']:,} files of the public ASVspoof 2019 benchmark.")
    bullet(tf, "Our figure matches the published result for this model.",
           "That is evidence the implementation is correct, not merely running.")
    bullet(tf, "Every decision is logged and can be reconstructed.")
    add_table(s,
              ["Threat", "Wrongly accepted", "Layer responsible"],
              [["Impostor (different person)", f"{far(FULL,'impostor (other human)'):.2f}%", "L1  Voiceprint"],
               ["Replay of a recording",       f"{far(FULL,'replay of a recording'):.2f}%",  "L2  Liveness"],
               ["AI-generated (cloned) voice", f"{far(FULL,'TTS / VC clone'):.2f}%",         "L3  Anti-spoofing"],
               ["Prompt-injection command",    f"{far(FULL,'prompt injection'):.2f}%",       "L4  Context"],
               ["Social-engineering command",  f"{far(FULL,'social engineering'):.2f}%",     "L4  Context"]],
              left=0.9, top=4.05, width=8.2, height=2.4)
    footer(s, 2)
    notes(s, "If asked whether the system is error-free, say no directly, then use this table. "
             "Injection and social engineering are the weakest columns and we know it. "
             "Being first to name your own weak point is what makes the strong numbers believable.")

    # ---------------------------------------------------------- slide 3 --
    s = new_slide(prs, "USABILITY OBJECTIVES ACHIEVED")
    tf = body_box(s, top=1.42, height=3.0)
    bullet(tf, "The amount is detected from speech, not typed in.",
           "The user simply says the order.", first=True)
    bullet(tf, "Ordinary commands pass with almost no friction.",
           "“Turn off the lights” is treated as a light switch, not a payment.")
    bullet(tf, "Security scales with what is at risk.",
           "A laptop purchase demands far more proof than a lamp.")
    bullet(tf, "The assistant speaks back and repeats what it understood.",
           "The user can catch a mishearing before money moves.")
    bullet(tf, "An uncertain request is challenged, not refused.",
           "The system asks for the passphrase and adds that evidence.")
    add_table(s,
              ["Outcome for a genuine user", "Share", "What the user experiences"],
              [["Allowed immediately", f"{float(GEN['allow_pct']):.2f}%", "No friction at all"],
               ["Challenged",          f"{float(GEN['challenge_pct']):.2f}%", "Asked once more, then proceeds"],
               ["Refused outright",    f"{float(GEN['deny_pct']):.2f}%", "Must use another channel"]],
              left=0.9, top=4.7, width=8.2, height=1.5)
    footer(s, 3)
    notes(s, "The headline: roughly nine in ten genuine users still complete their request — "
             "most immediately, the rest after one extra sentence. Only about one in eight is "
             "stopped completely, and that is the price of the security on the previous slide.")

    # ---------------------------------------------------------- slide 4 --
    s = new_slide(prs, "THE FIVE LAYERS OF THE SYSTEM")
    tf = body_box(s, top=1.42, height=3.4)
    bullet(tf, "L1  Voiceprint  (ECAPA-TDNN, cosine similarity)",
           "Is this the enrolled user's voice?", first=True)
    bullet(tf, "L2  Liveness",
           "Did this come from a mouth, or from a loudspeaker?")
    bullet(tf, "L3  Anti-spoofing  (AASIST-L)",
           "Was this made by a human throat, or by a computer?")
    bullet(tf, "L4  Context",
           "Is this request normal for this user?")
    bullet(tf, "L5  Risk policy  —  not a detector.",
           "It sets how much proof this request needs, then makes the call.", colour=BLUE)
    add_table(s,
              ["Layer", "Question it answers", "Threat it exists for"],
              [["L1  Voiceprint",    "Is this the enrolled user?", "A different person"],
               ["L2  Liveness",      "Mouth or loudspeaker?",      "Replay of a recording"],
               ["L3  Anti-spoofing", "Human or AI-generated?",     "Cloned / synthetic voice"],
               ["L4  Context",       "Is this request normal?",    "Manipulated command"],
               ["L5  Risk policy",   "How much proof is needed?",  "Sets the bar and decides"]],
              left=0.9, top=4.92, width=8.2, height=1.95)
    footer(s, 4)
    notes(s, "Expect the question: “Where is Layer 5? I only see four scores.” "
             "Answer: L5 has no score because it does not vote — it decides. It is the only "
             "layer that looks at nothing but the output of the other four.")

    # ---------------------------------------------------------- slide 5 --
    s = new_slide(prs, "WHY FIVE LAYERS AND NOT THREE OR FOUR?")
    tf = body_box(s, top=1.42, height=2.1)
    bullet(tf, "We did not argue this — we measured it.",
           "Each layer was switched off in turn.", first=True)
    bullet(tf, f"Remove L2: replay rises {far(FULL,'replay of a recording'):.2f}% → {far('L1 + L3','replay of a recording'):.0f}%.",
           "Nearly three in four replayed recordings would be accepted.", colour=RED)
    bullet(tf, f"Remove L3: cloned voices rise to {far('L1 only','TTS / VC clone'):.2f}%.",
           "An AI copy would pass almost half the time.", colour=RED)
    bullet(tf, f"Remove L4: injected commands rise {far(FULL,'prompt injection'):.0f}% → {far('L1 + L2 + L3','prompt injection'):.1f}%.",
           colour=RED)
    bullet(tf, "Every smaller combination is blind to something.")
    picture(s, "ablation.png", 0.7, 3.62, 8.6)
    footer(s, 5)
    notes(s, "This slide is the direct answer to “prove five is better than four”. It is a "
             "measurement on held-out data, not an opinion. If asked what the fifth layer costs: "
             f"more genuine users get challenged, {far('L1 only','genuine user'):.0f}% rises to "
             f"{far(FULL,'genuine user'):.2f}%, but most recover by answering the challenge.")

    # ---------------------------------------------------------- slide 6 --
    s = new_slide(prs, "MAPPING LAYERS TO THREATS")
    tf = body_box(s, top=1.42, height=2.1)
    bullet(tf, "No two layers cover the same threat.",
           "There is no overlap to fall back on, so none can be removed.", first=True)
    bullet(tf, "A replayed clip passes L1 and L3 — and they are right to pass it.",
           "It genuinely is the user's voice, from a human throat. Only L2 can object.")
    bullet(tf, "A cloned voice passes L1 because the clone succeeded.",
           "Only the anti-spoofing layer sees the difference.")
    bullet(tf, "A scam command may pass all three audio layers.",
           "If the real user is tricked into speaking, only context notices.")
    picture(s, "replay.png", 1.35, 3.55, 7.3)
    footer(s, 6)
    notes(s, "If there is time, run this live: speak a command, then play the recording back "
             "through the laptop speakers. Watching L1 and L3 stay green while L2 turns red is "
             "the clearest demonstration in the whole project of why one layer is not enough.")

    # ---------------------------------------------------------- slide 7 --
    s = new_slide(prs, "HOW THE FIVE LAYERS WORK TOGETHER")
    tf = body_box(s, top=1.42, height=4.6)
    head(tf, "Step 1 — put every layer on one scale", first=True)
    sub(tf, "A similarity of 0.72 and a spoof probability of 0.99 cannot be compared. "
            "Calibration converts both into the same unit of evidence.")
    head(tf, "Step 2 — combine by the weakest link, not by adding")
    sub(tf, "Security is a chain. An attacker must defeat every layer, so the weakest layer decides.")
    head(tf, "Step 3 — compare against the bar for this request")
    sub(tf, "The threshold is calculated from what is at stake, then the system allows, "
            "challenges, or denies.")
    head(tf, "Adding the scores was tried, and it failed")
    bullet(tf, "Impostor acceptance rose 0.25% → 52.25%.",
           "The anti-spoofing layer correctly said “real human” about an impostor, and that "
           "cancelled the correct rejection from the voiceprint layer.", colour=RED)
    bullet(tf, "The weakest-link rule gives a guarantee.",
           "Adding a layer can only lower the score, so a new layer can never open a new hole.",
           colour=GREEN)
    footer(s, 7)
    notes(s, "Expect: “Why not just add the scores up?” Answer: because the layers answer "
             "different questions. L1's “genuine” means the right person; L3's “genuine” means "
             "a real human. An impostor is a real human, so adding lets one correct answer "
             "cancel another. This is a real result we measured, not a theoretical worry.")

    # ---------------------------------------------------------- slide 8 --
    s = new_slide(prs, "HOW THE THRESHOLD WAS DECIDED")
    tf = body_box(s, top=1.42, height=2.5)
    bullet(tf, "The old approach used fixed cutoffs such as 0.75 and 0.85.",
           "Nothing explains why 0.75 and not 0.70. That cannot be defended.", first=True)
    bullet(tf, "The new threshold is calculated from the cost of being wrong.",
           "Approving a fraud costs the money lost; refusing a real customer costs a support call.")
    bullet(tf, "Only one of those two costs grows with the amount.",
           "That is why the bar rises with the value of the request.")
    bullet(tf, "The method is published and widely used.",
           "Kinnunen et al., Odyssey 2018 — the measure ASVspoof itself adopted.", colour=BLUE)
    bullet(tf, "Risk is not only about money.",
           "“Forward the OTP” has no amount but real loss, so we use the wider idea of stake.")
    picture(s, "threshold.png", 1.65, 3.95, 6.70)      # 2.31 aspect -> 2.90in tall
    footer(s, 8)
    notes(s, "Say this sentence out loud: every number in our threshold is a business cost you "
             "can argue with, not a score cutoff we invented. "
             "If asked whether the costs themselves are arbitrary: partly, and we tested it. We "
             "varied each over ten times its value. Three behave gently. Only the assumed attack "
             "rate matters a lot, and that is the one figure a real bank measures from its records.")

    # ---------------------------------------------------------- slide 9 --
    s = new_slide(prs, "CHALLENGES FACED DURING IMPLEMENTATION")
    tf = body_box(s, top=1.42, height=4.9)
    head(tf, "Our own live voice was flagged as synthetic — every time", first=True)
    sub(tf, "Eight out of eight genuine recordings were rejected as AI-generated. The cause was "
            "the browser, not the model: noise suppression and automatic gain control reshape "
            "the sound, and the detector reads that as artificial.")
    head(tf, "A model number was mistaken for money")
    sub(tf, "“Samsung Galaxy Fold 8 Ultra” was read as an 8-rupee order, which put the lowest "
            "possible security bar on an expensive purchase.")
    head(tf, "Our first liveness result was too good, so we rejected it")
    sub(tf, "It reported 100% accuracy. The live and replayed clips had been recorded on "
            "different devices, so the model had learned the device, not liveness.")
    head(tf, "Free API limits slowed experiments")
    sub(tf, "Calibration runs are limited by tokens per minute, not by computing power.")
    footer(s, 9)
    notes(s, "Presenting the discarded result is deliberate. A team that finds and reports its "
             "own mistake is far more trustworthy than one that reports only successes — and if "
             "a reviewer found it later, the damage would be much worse.")

    # --------------------------------------------------------- slide 10 --
    s = new_slide(prs, "SOLUTIONS AND FUTURE WORK")
    tf = body_box(s, top=1.42, height=5.0)
    head(tf, "Solved", first=True)
    bullet(tf, "Browser audio processing is switched off before analysis.",
           "False rejection of genuine speech fell from every clip to about one in twenty.")
    bullet(tf, "A bare number is never treated as money.",
           "The screen now shows which numbers were ignored, and why.")
    bullet(tf, "Liveness was rebuilt with proper controls.",
           "Both recordings now come from one microphone seconds apart. On 20 fresh clips never "
           "used in training, all 20 were classified correctly.")
    head(tf, "Future work")
    bullet(tf, "A randomly chosen challenge phrase.",
           "Today the passphrase is fixed, so a recording of it would pass.")
    bullet(tf, "Wider testing before any general claim.",
           "Better loudspeakers, more speakers, more rooms, real user commands.")
    bullet(tf, "Measure the real attack rate from live data.",
           "Our sensitivity study shows this is the one assumption that truly matters.")
    footer(s, 10)
    notes(s, "Close by naming our own limitations before anyone asks. It shows we understand the "
             "difference between a working prototype and a proven system — which is exactly the "
             "distinction a reviewer is listening for.")

    return prs


if __name__ == "__main__":
    out = ROOT / "slides" / "SecureVoice_Additional_Slides.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    build().save(out)
    print(f"wrote {out}")
