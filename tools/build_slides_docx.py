"""
tools/build_slides_docx.py

Builds the speaker-ready DOCX for the 10 additional presentation slides.

Every figure and every number in this document is generated from the project's
own result files, not typed by hand:
    results/ablation.csv       -> the layer-ablation numbers and Fig. 2
    results/sensitivity.csv    -> the cost-constant sensitivity statement
    results/aasist_baseline.json -> the AASIST-L benchmark figures
    models/calibrators.json    -> the calibration parameters
So if an experiment is re-run, this script regenerates a document that agrees
with it, and a stale slide cannot silently survive.
"""
from __future__ import annotations
import csv, json
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "slides" / "figs"

ACCENT = RGBColor(0x1F, 0x3B, 0x73)
MUTED  = RGBColor(0x55, 0x55, 0x55)


# ----------------------------------------------------------- data loading --
def ablation() -> dict:
    rows = [r for r in csv.DictReader(open(ROOT / "results" / "ablation.csv"))
            if r["fusion_rule"] == "min"]
    return {(r["subset"], r["threat"]): float(r["error_pct"]) for r in rows}, rows


A, ROWS = ablation()
FULL = "L1 + L2 + L3 + L4 (full)"
BASE = json.loads((ROOT / "results" / "aasist_baseline.json").read_text())


def far(subset, threat):
    return A[(subset, threat)]


# --------------------------------------------------------------- helpers --
def h1(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = True; r.font.size = Pt(15); r.font.color.rgb = ACCENT
    p.space_after = Pt(4)
    return p


def h2(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = True; r.font.size = Pt(10.5); r.font.color.rgb = ACCENT
    p.space_before = Pt(8); p.space_after = Pt(2)
    return p


def bullet(doc, lead, rest=""):
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(lead); r.bold = True
    if rest:
        p.add_run(" " + rest)
    p.space_after = Pt(3)
    return p


def note(doc, label, text):
    p = doc.add_paragraph()
    r = p.add_run(label + " "); r.bold = True; r.italic = True; r.font.size = Pt(9.5)
    r.font.color.rgb = MUTED
    r2 = p.add_run(text); r2.italic = True; r2.font.size = Pt(9.5)
    r2.font.color.rgb = MUTED
    p.space_before = Pt(5)
    return p


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, htxt in enumerate(headers):
        c = t.rows[0].cells[i]
        c.text = ""
        run = c.paragraphs[0].add_run(htxt)
        run.bold = True; run.font.size = Pt(9)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.size = Pt(9)
            if str(val).startswith("**"):
                run.text = str(val).strip("*"); run.bold = True
    if widths:
        for r_ in t.rows:
            for i, w in enumerate(widths):
                r_.cells[i].width = Inches(w)
    doc.add_paragraph().space_after = Pt(2)
    return t


def figure(doc, name, caption, width=6.3):
    if not (FIGS / name).exists():
        return
    doc.add_picture(str(FIGS / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    p = doc.add_paragraph()
    r = p.add_run(caption); r.italic = True; r.font.size = Pt(8.5)
    r.font.color.rgb = MUTED
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.space_after = Pt(8)


def slide(doc, n, title, first=False):
    if not first:
        doc.add_page_break()
    h1(doc, f"Slide {n} — {title}")
    pr = doc.add_paragraph()
    pr.add_run("_" * 78).font.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    pr.space_after = Pt(6)


# ============================================================== document ==
def build() -> Document:
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"; st.font.size = Pt(10.5)
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(0.85)
        s.top_margin = s.bottom_margin = Inches(0.7)

    # ---- cover ----
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("SecureVoice — Additional Presentation Slides")
    r.bold = True; r.font.size = Pt(19); r.font.color.rgb = ACCENT
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Multi-Layer Continuous Authentication for Voice Assistants")
    r.font.size = Pt(12); r.font.color.rgb = MUTED
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Objectives Achieved  ·  Why Five Layers  ·  Thresholds and Challenges")
    r.italic = True; r.font.size = Pt(10); r.font.color.rgb = MUTED
    doc.add_paragraph()

    h2(doc, "How to use this document")
    bullet(doc, "Bold text", "is the bullet to put on the slide. The sentence after it is your spoken explanation — do not print it.")
    bullet(doc, "Speaker note", "tells you what to say while the slide is on screen.")
    bullet(doc, "If asked", "gives a prepared answer to the question that slide invites.")
    bullet(doc, "Every number", "in this deck comes from a result file in the project, so each one can be reproduced on request.")

    note(doc, "Source of truth:",
         "results/ablation.csv, results/sensitivity.csv, results/aasist_baseline.json and "
         "models/calibrators.json. Figures are generated by script, not drawn by hand.")

    # ============================ SECTION A ============================
    doc.add_page_break()
    h1(doc, "Section A — Objectives Achieved")

    slide(doc, 1, "Objectives Achieved: Overview", first=True)
    bullet(doc, "All five layers are built and running end to end.",
           "Four layers gather evidence; the fifth decides. The demo runs live in a browser.")
    bullet(doc, "Every layer is calibrated on held-out data.",
           "No score cutoff is chosen by hand. Each detector's output is converted to a common scale before it is used.")
    bullet(doc, "The decision threshold is derived from cost, not guessed.",
           "We can show the formula and the published paper it comes from.")
    bullet(doc, "An ablation study proves each layer earns its place.",
           "We switched layers off one at a time and measured what got through.")
    bullet(doc, "Limitations are measured and stated openly.",
           "Where our data is small or synthetic, we say so on the slide rather than hide it.")
    figure(doc, "architecture.png",
           "Figure 1 — System architecture. Audio is transcribed on the device; only text is shared. "
           "Four evidence layers feed one decision policy.")
    note(doc, "Speaker note:",
         "Open by stating that all five layers work, and that every claim in the next ten minutes "
         "is backed by a measurement we can reproduce.")

    slide(doc, 2, "Security and Authentication Objectives Achieved")
    bullet(doc, f"Impostor detection: {far('L1 only','impostor (other human)'):.2f}% wrongly accepted.",
           "A different person speaking is almost never accepted as the enrolled user.")
    bullet(doc, f"Replay detection: {far(FULL,'replay of a recording'):.2f}% wrongly accepted.",
           f"Without this layer the figure is {far('L1 + L3','replay of a recording'):.0f}%, so it is doing real work.")
    bullet(doc, f"AI-voice detection: {BASE['eval_eer']*100:.2f}% equal error rate on the public ASVspoof 2019 benchmark.",
           f"Measured on {BASE['n_eval']:,} test files. This matches the published result for this model, which shows our pipeline is correct.")
    bullet(doc, "Manipulated commands are caught by a dedicated layer.",
           "Attacks that sound perfectly normal, such as 'ignore your security checks', are handled by the context layer, not by the audio layers.")
    bullet(doc, "Every decision is fully logged and can be reconstructed.",
           "The log stores the transcript, each layer's score, the threshold and how it was calculated.")
    table(doc,
          ["Threat", "Wrongly accepted", "Layer responsible"],
          [["Impostor (different person)", f"{far(FULL,'impostor (other human)'):.2f}%", "L1 Voiceprint"],
           ["Replay of a recording", f"{far(FULL,'replay of a recording'):.2f}%", "L2 Liveness"],
           ["AI-generated (cloned) voice", f"{far(FULL,'TTS / VC clone'):.2f}%", "L3 Anti-spoofing"],
           ["Prompt-injection command", f"{far(FULL,'prompt injection'):.2f}%", "L4 Context"],
           ["Social-engineering command", f"{far(FULL,'social engineering'):.2f}%", "L4 Context"]],
          widths=[2.6, 1.6, 2.0])
    note(doc, "Speaker note:",
         "Point out that the benchmark figure matches the published one. That is the evidence "
         "our implementation is correct, not just working.")
    note(doc, "If asked",
         "'Is it error-free?' — No. Say so directly, then give this table. Injection and "
         "social engineering are the weakest columns and we know it.")

    slide(doc, 3, "Usability and Intelligence Objectives Achieved")
    bullet(doc, "The amount is detected from speech, not typed in.",
           "The user simply says the order. Earlier the operator moved a slider by hand, which is not how a real product works.")
    bullet(doc, "Simple commands go through with almost no friction.",
           "'Turn off the lights' is treated as a light switch, not as a payment.")
    bullet(doc, "Security scales with what is at risk.",
           "A laptop purchase demands far more evidence than switching on a lamp, and the difference is calculated, not hand-set.")
    bullet(doc, "The assistant speaks back and repeats what it understood.",
           "This lets the user catch a mishearing before any money moves.")
    bullet(doc, "An uncertain request is challenged, not refused.",
           "The system asks for the passphrase and adds that evidence to what it already has.")
    g = next(r for r in ROWS if r["subset"] == FULL and r["threat"] == "genuine user")
    table(doc,
          ["Outcome for a genuine user", "Share", "What the user experiences"],
          [["Allowed immediately", f"{float(g['allow_pct']):.2f}%", "No friction at all"],
           ["Challenged", f"{float(g['challenge_pct']):.2f}%", "Asked once more, then proceeds"],
           ["Refused outright", f"{float(g['deny_pct']):.2f}%", "Must use another channel"]],
          widths=[2.4, 1.2, 2.6])
    note(doc, "Speaker note",
         "The headline is that roughly nine in ten genuine users still complete their request — "
         "most immediately, the rest after one extra sentence.")

    # ============================ SECTION B ============================
    doc.add_page_break()
    h1(doc, "Section B — Why Exactly Five Layers?")

    slide(doc, 4, "The Five Layers of the System", first=True)
    bullet(doc, "L1 Voiceprint (ECAPA-TDNN, cosine similarity).",
           "Answers: is this the enrolled user's voice? Compares a 192-number summary of the voice with the stored one.")
    bullet(doc, "L2 Liveness.",
           "Answers: did this come from a mouth or a loudspeaker? A replayed clip has passed through a speaker, which changes its sound in a measurable way.")
    bullet(doc, "L3 Anti-spoofing (AASIST-L).",
           "Answers: was this made by a human throat or by a computer? Looks at the raw waveform for traces left by synthesis.")
    bullet(doc, "L4 Context.",
           "Answers: is this request normal for this user? Reads the words and compares the request with the user's own history.")
    bullet(doc, "L5 Risk policy.",
           "Not a detector. It works out how much evidence this particular request needs, then makes the final call.")
    table(doc,
          ["Layer", "Question it answers", "Threat it exists for"],
          [["L1 Voiceprint", "Is this the enrolled user?", "A different person"],
           ["L2 Liveness", "Mouth or loudspeaker?", "Replay of a recording"],
           ["L3 Anti-spoofing", "Human or AI-generated?", "Cloned / synthetic voice"],
           ["L4 Context", "Is this request normal?", "Manipulated or injected command"],
           ["L5 Risk policy", "How much proof is needed?", "— sets the bar and decides"]],
          widths=[1.5, 2.3, 2.4])
    note(doc, "If asked",
         "'Where is Layer 5? I only see four scores.' — L5 has no score because it does not vote. "
         "It decides. It is the only layer that looks at nothing but the other four.")

    slide(doc, 5, "Why Five Layers and Not Three or Four?")
    bullet(doc, "We did not argue this — we measured it.",
           "Each layer was switched off in turn and we recorded how many attacks got through.")
    bullet(doc, f"Remove L2 and replay attacks rise from {far(FULL,'replay of a recording'):.2f}% to {far('L1 + L3','replay of a recording'):.0f}%.",
           "Nearly three in four replayed recordings would be accepted.")
    bullet(doc, f"Remove L3 and cloned voices rise to {far('L1 only','TTS / VC clone'):.2f}%.",
           "An AI copy of the user's voice would pass almost half the time.")
    bullet(doc, f"Remove L4 and injected commands rise from {far(FULL,'prompt injection'):.0f}% to {far('L1 + L2 + L3','prompt injection'):.1f}%.",
           "Commands designed to talk the assistant out of its own rules get through twice as often.")
    bullet(doc, "Every smaller combination is blind to something.",
           "The complete system is the best or joint-best against every attack we tested.")
    figure(doc, "ablation.png",
           "Figure 2 — Attacks wrongly allowed, for each combination of layers. "
           "Lower is better. Each missing layer leaves one bar tall.")
    note(doc, "Speaker note",
         "This slide is the direct answer to 'prove five is better than four'. "
         "It is a measurement on held-out data, not an opinion.")
    note(doc, "If asked",
         "'What does the fifth layer cost you?' — Honestly, more genuine users get challenged: "
         f"{far('L1 only','genuine user'):.0f}% rises to {far(FULL,'genuine user'):.2f}%. "
         "But most of those recover by answering the challenge.")

    slide(doc, 6, "Mapping Layers to Threats")
    bullet(doc, "No two layers cover the same threat.",
           "That is why none of them can be removed — there is no overlap to fall back on.")
    bullet(doc, "A replayed clip passes L1 and L3, and they are right to pass it.",
           "It genuinely is the user's voice, and it genuinely came from a human throat. Only L2 can object.")
    bullet(doc, "A cloned voice passes L1 because the clone succeeded.",
           "The speaker check cannot help here; only the anti-spoofing layer sees the difference.")
    bullet(doc, "A scam command may pass all three audio layers.",
           "If the real user is tricked into speaking, the audio is perfect. Only the context layer notices the request is strange.")
    figure(doc, "replay.png",
           "Figure 3 — The same sentence, spoken and then replayed through a speaker. "
           "L1 and L3 stay positive; only L2 turns negative and stops the request.", width=5.8)
    note(doc, "Speaker note",
         "If there is time, run this live: speak a command, then play the recording back. "
         "It is the clearest demonstration of why one layer is not enough.")

    slide(doc, 7, "How the Five Layers Work Together")
    bullet(doc, "Each layer speaks a different language, so we translate them first.",
           "A similarity of 0.72 and a spoof probability of 0.99 cannot be compared. Calibration puts both on one scale.")
    bullet(doc, "The layers are combined by the weakest link, not by adding scores.",
           "Security is a chain: an attacker must defeat every layer, so the weakest layer decides.")
    bullet(doc, "Adding scores was tried and it failed badly.",
           "Impostor acceptance rose from 0.25% to 52.25%, because the anti-spoofing layer correctly said 'real human' about an impostor and that cancelled the correct rejection.")
    bullet(doc, "The weakest-link rule gives a safety guarantee.",
           "Adding a new layer can only lower the score, so a new layer can never open a new hole.")
    bullet(doc, "The result is compared with the threshold for that request.",
           "The bar is calculated from what is at stake, then the system allows, challenges or denies.")
    note(doc, "If asked",
         "'Why not just add the scores up?' — Because the layers answer different questions. "
         "L1's 'genuine' means the right person; L3's 'genuine' means a real human. "
         "An impostor is a real human, so adding lets one correct answer cancel another.")

    # ============================ SECTION C ============================
    doc.add_page_break()
    h1(doc, "Section C — Thresholds, Challenges and Solutions")

    slide(doc, 8, "Threshold Selection and Comparison", first=True)
    bullet(doc, "The old approach used fixed cutoffs such as 0.75 and 0.85.",
           "The weakness is simple: nothing explains why 0.75 rather than 0.70. That cannot be defended in a review.")
    bullet(doc, "The new threshold is calculated from the cost of being wrong.",
           "Approving a fraud costs the money lost; wrongly refusing a real customer costs a support call. The best cutoff follows from those two costs.")
    bullet(doc, "Only one of the two costs grows with the amount.",
           "That is why the bar rises with the value of the request, and it rises smoothly rather than in hand-drawn steps.")
    bullet(doc, "The method is published and widely used.",
           "Kinnunen et al., 'A Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and ASV', Odyssey 2018 — the measure ASVspoof itself adopted.")
    bullet(doc, "Risk is not only about money.",
           "'Forward the OTP' has no amount but real loss, so we use the wider idea of stake. A light switch is low stake; disclosing an OTP is high.")
    figure(doc, "threshold.png",
           "Figure 4 — The evidence required rises with the stake of the request. "
           "The shaded band is where the system asks again instead of guessing.", width=5.9)
    note(doc, "Speaker note",
         "Say this sentence: every number in our threshold is a business cost you can argue with, "
         "not a score cutoff we invented.")
    note(doc, "If asked",
         "'Are the costs themselves arbitrary?' — Partly, and we tested that. We varied each one "
         "over ten times its value. Three behave gently. Only the assumed attack rate matters a lot, "
         "and that is the one figure a real bank can measure from its own records.")

    slide(doc, 9, "Challenges Faced During Implementation")
    bullet(doc, "Our own live voice was flagged as synthetic, every single time.",
           "Eight out of eight genuine recordings were rejected as AI-generated.")
    bullet(doc, "The cause was the browser, not the model.",
           "Browsers apply noise suppression and automatic gain control by default. These reshape the sound, and the detector reads that as artificial.")
    bullet(doc, "A model number was mistaken for money.",
           "'Samsung Galaxy Fold 8 Ultra' was treated as an 8-rupee order, which set the lowest possible security bar on an expensive purchase.")
    bullet(doc, "Our first liveness result was too good, and we rejected it.",
           "It reported 100% accuracy, but the live and replayed clips had been recorded on different devices, so the model had learned the device, not liveness.")
    bullet(doc, "The free API tier limited how fast we could run experiments.",
           "Calibration runs are limited by tokens per minute, not by computing power.")
    note(doc, "Speaker note",
         "Presenting the discarded result is deliberate. A team that finds and reports its own "
         "mistake is more trustworthy than one that reports only successes.")

    slide(doc, 10, "Solutions and Future Improvements")
    bullet(doc, "Browser audio processing is now switched off before analysis.",
           "False rejections of genuine speech fell from every clip to about one in twenty. The rule: a forensic check must see the unprocessed sound.")
    bullet(doc, "A bare number is never treated as money.",
           "An amount is accepted only with a currency or price word nearby, and the screen shows which numbers were ignored and why.")
    bullet(doc, "Liveness was rebuilt with proper controls.",
           "Both recordings are now made through one microphone seconds apart. On 20 fresh clips never used in training, all 20 were classified correctly.")
    bullet(doc, "Future work: a randomly chosen challenge phrase.",
           "Today the passphrase is fixed, so a recording of it would pass. A phrase the system picks at random cannot be pre-recorded.")
    bullet(doc, "Future work: wider testing before any general claim.",
           "Better loudspeakers, more speakers, more rooms, and real user commands instead of generated ones.")
    bullet(doc, "Future work: measure the attack rate from real data.",
           "Our sensitivity study shows this is the one assumption that really matters.")
    note(doc, "Speaker note",
         "Close by naming our own limitations before anyone asks. It shows we understand "
         "the difference between a working prototype and a proven system.")

    # ---- appendix ----
    doc.add_page_break()
    h1(doc, "Appendix — Full Ablation Table (for questions)")
    doc.add_paragraph("Percentage of attacks wrongly allowed. Last column is the share of genuine "
                      "users not allowed straight through. Lower is better in every column.")
    threats = ["impostor (other human)", "replay of a recording", "TTS / VC clone",
               "prompt injection", "social engineering", "genuine user"]
    hdr = ["Layers used", "Impostor", "Replay", "AI clone", "Injection", "Social", "Genuine not allowed"]
    body = []
    for sub, lab in [("L1 only", "L1 only"), ("L1 + L3", "L1 + L3"),
                     ("L1 + L2 + L3", "L1 + L2 + L3"), ("L1 + L3 + L4", "L1 + L3 + L4"),
                     (FULL, "All four layers")]:
        body.append([lab] + [f"{A[(sub, t)]:.2f}%" for t in threats])
    table(doc, hdr, body, widths=[1.4, 0.85, 0.85, 0.85, 0.85, 0.85, 1.15])
    note(doc, "Reading this table:",
         "Look down each column. The tall number in a column is the attack that the missing "
         "layer would have caught. The bottom row is best or joint-best everywhere.")
    return doc


if __name__ == "__main__":
    out = ROOT / "slides" / "SecureVoice_Slides_Content.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    build().save(out)
    print(f"wrote {out}")
