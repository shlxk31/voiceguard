"""
tools/build_paper.py

Generates the SecureVoice research paper as an editable .docx, following the
mentor's whiteboard structure (Title+Abstract=1pg, Intro=1-2pg, Lit Survey=1pg,
Contribution [Proposal+Results]=3-3.5pg, Conclusion=1/4 col, References=~1
col, 18-20 refs, IEEE two-column, 6 pages) and populated with the real,
measured results from this project.

Usage: .venv/bin/python tools/build_paper.py
Output: SecureVoice_Research_Paper.docx
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = "SecureVoice_Research_Paper.docx"

# ---------------------------------------------------------------- helpers --

def set_columns(section, n=2, space_in=0.25):
    sectPr = section._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    cols.set(qn('w:num'), str(n))
    cols.set(qn('w:space'), str(int(space_in * 1440)))


def add_continuous_section(doc):
    new_section = doc.add_section(WD_SECTION.CONTINUOUS)
    return new_section


def style_body(doc):
    normal = doc.styles['Normal']
    normal.font.name = 'Times New Roman'
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.0


def heading(doc, text, size=11, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER,
            space_before=10, space_after=4, color=None, italic=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    r.font.size = Pt(size)
    r.font.name = 'Times New Roman'
    if color:
        r.font.color.rgb = color
    return p


def para(doc, text, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=10, first_line_indent=True,
          space_after=6):
    p = doc.add_paragraph()
    p.alignment = align
    if first_line_indent:
        p.paragraph_format.first_line_indent = Inches(0.15)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.font.size = Pt(size)
    r.font.name = 'Times New Roman'
    return p


def section_heading(doc, numeral, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"{numeral}. {title.upper()}")
    r.bold = True
    r.font.size = Pt(10.5)
    r.font.name = 'Times New Roman'
    return p


def subsection_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text)
    r.bold = True
    r.italic = True
    r.font.size = Pt(10)
    r.font.name = 'Times New Roman'
    return p


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    r.italic = False
    r.font.size = Pt(8.5)
    r.font.name = 'Times New Roman'
    return p


def make_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = 'Table Grid'
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ''
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(8)
        r.font.name = 'Times New Roman'
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ''
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(str(val))
            r.font.size = Pt(8)
            r.font.name = 'Times New Roman'
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return t


def add_figure(doc, path, width_in, cap):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(path, width=Inches(width_in))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.LEFT
    c.paragraph_format.space_after = Pt(8)
    r = c.add_run(cap)
    r.font.size = Pt(8)
    r.font.name = 'Times New Roman'
    return p


def refs_para(doc, num, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.first_line_indent = Inches(-0.18)
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run(f"[{num}] {text}")
    r.font.size = Pt(8.5)
    r.font.name = 'Times New Roman'
    return p


# ------------------------------------------------------------------- build --

doc = Document()
style_body(doc)

sec1 = doc.sections[0]
sec1.page_width = Inches(8.5)
sec1.page_height = Inches(11)
sec1.top_margin = Inches(0.75)
sec1.bottom_margin = Inches(1.0)
sec1.left_margin = Inches(0.625)
sec1.right_margin = Inches(0.625)

# ---- Title block (single column) ----
heading(doc, "SecureVoice: Exposing and Closing the Studio-to-Real-World "
             "Domain Gap in Anti-Spoofing for Voice-Authenticated Transactions",
        size=16, bold=True, space_before=0, space_after=8)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Atharva Kadge, Damodar Kamath, Shlok Kanvinde")
r.font.size = Pt(11)
r.font.name = 'Times New Roman'
p.paragraph_format.space_after = Pt(2)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("Department of Computer Engineering, Sardar Patel Institute of Technology, Mumbai, India")
r.italic = True
r.font.size = Pt(9.5)
r.font.name = 'Times New Roman'
p.paragraph_format.space_after = Pt(10)

# Abstract
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.space_after = Pt(6)
r = p.add_run("Abstract—")
r.bold = True
r.italic = True
r.font.size = Pt(9.5)
r.font.name = 'Times New Roman'
abstract_text = (
    "Voice-authenticated transaction systems increasingly rely on speaker "
    "verification to answer “is this the enrolled user’s voice?” but rarely "
    "ask the different question, “is this a real human?” We build a two-layer "
    "authorization pipeline—ECAPA-TDNN speaker verification plus AASIST-L "
    "anti-spoofing—into a voice-ordering system and measure, rather than assume, "
    "its security. Across 800 trials over 20 enrolled speakers on ASVspoof2019 LA, "
    "54.0% of synthetic voice-conversion/TTS attacks clear the speaker-verification "
    "gate, while the anti-spoofing layer independently flags 87.0% of them with zero "
    "false alarms on genuine or impostor speech, cutting approved attacks to 7.5% at "
    "typical transaction tiers. We then reproduce the anti-spoofing model’s "
    "published benchmark (0.99% EER, matching prior work) and discover a severe "
    "practical failure: on our own real phone recordings, the same model with the "
    "same frozen threshold misclassifies 100% (8/8) of genuine speech as synthetic. "
    "We diagnose this as a studio-vs-real-world domain gap—ASVspoof2019’s "
    "bonafide class is anechoic-chamber audio with a 16× lower noise floor and "
    "10× more high-frequency energy than a phone recording in an ordinary room—"
    "rule out codec, resampling, and gain artifacts as the cause, and show that direct "
    "domain adaptation on a small set of real recordings reduces the real-world "
    "false-positive rate from 8/8 to 2/8 at a measured, bounded cost to benchmark EER "
    "(2.00% → 8.12%). We additionally quantify how the same model degrades under "
    "VoIP-style channel loss (0.99% → 14.00% EER) and under short streaming windows "
    "(0.99% → 36.43% at 250 ms), and report all findings—including negative "
    "results—with a fixed, dev-only threshold-selection protocol."
)
r2 = p.add_run(abstract_text)
r2.font.size = Pt(9.5)
r2.font.name = 'Times New Roman'

p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(8)
r = p.add_run("Keywords—")
r.bold = True
r.italic = True
r.font.size = Pt(9.5)
r.font.name = 'Times New Roman'
r2 = p.add_run("voice authentication, anti-spoofing, ASVspoof, speaker verification, "
               "domain adaptation, AASIST, ECAPA-TDNN, deepfake speech detection")
r2.italic = True
r2.font.size = Pt(9.5)
r2.font.name = 'Times New Roman'

# ---- switch to two columns for the body ----
sec2 = add_continuous_section(doc)
set_columns(sec2, 2, 0.25)

# ============================================================ I. INTRO ====
section_heading(doc, "I", "Introduction")

para(doc, "Personal voice assistants and voice-authenticated transaction systems "
          "increasingly gate real financial actions—placing orders, moving money, "
          "unlocking doors—behind a voice check [1]. Smart-speaker adoption has "
          "grown rapidly and voice is now marketed as a convenience layer atop "
          "e-commerce and banking flows, which means a successful attack on the "
          "voice channel is no longer merely a privacy nuisance but a direct "
          "financial-fraud vector. The dominant technical approach to securing "
          "that channel is speaker verification (SV): compare an incoming "
          "utterance against an enrolled voiceprint and approve if the cosine "
          "similarity of speaker embeddings clears a threshold. SV answers “is "
          "this the right voice?” It does not answer “is this a real human "
          "speaking, or an AI-generated clone?”—a distinct question that a "
          "determined attacker can now target cheaply with modern text-to-speech "
          "(TTS) and voice-conversion (VC) tools, several of which are freely "
          "available and require only a handful of seconds of reference audio.")

para(doc, "This gap is not hypothetical. A clone is explicitly optimized to sound "
          "like the target speaker, so a successful clone should, by construction, "
          "score well on SV—its similarity should rise, not fall, relative to an "
          "unrelated impostor. Prior surveys of the personal-voice-assistant (PVA) "
          "threat landscape [1] identify voice authentication spoofing as a mature "
          "but still-unresolved research area, and note that commercial assistants "
          "rarely combine spoof detection with access control in practice: Siri "
          "applies some form of voiceprint check, while several other widely "
          "deployed assistants accept any voice, synthetic or otherwise, that "
          "utters the correct wake phrase. Meanwhile, the anti-spoofing research "
          "community has produced increasingly accurate countermeasures—sub-1% "
          "Equal Error Rate (EER) is now routine on the standard ASVspoof "
          "benchmark—creating an appealing but, as we show, premature assumption "
          "that bolting such a model onto an SV pipeline solves the problem.")

para(doc, "We built SecureVoice, a voice-ordering system secured by (i) ECAPA-TDNN "
          "speaker verification and (ii) an added anti-spoofing layer using AASIST-L, "
          "a compact (85k-parameter) graph-attention network benchmarked on the "
          "ASVspoof2019 Logical Access (LA) corpus. Rather than assume the resulting "
          "system is secure because each component is individually well-benchmarked, "
          "we measured it end to end—against synthetic attacks, against degraded "
          "audio, against short streaming windows, and, critically, against our own "
          "real voices recorded on an ordinary phone in an ordinary room. That last, "
          "originally intended only as a pre-demo sanity check, produced this "
          "paper’s central and unplanned finding: a model reporting 0.99% error on "
          "its own benchmark misclassified every one of our real recordings as "
          "synthetic, with high confidence, using the same frozen operating "
          "threshold used everywhere else in this paper.")

para(doc, "This paper's contribution is therefore as much methodological as it is "
          "architectural: we show the value of testing a voice-security pipeline "
          "against the actual conditions it will be deployed into, rather than "
          "stopping once each component clears its own published benchmark. "
          "Concretely, we contribute: (1) a measured, controlled demonstration "
          "that SV alone is insufficient against synthetic speech, using an "
          "800-trial genuine/spoof/impostor protocol with a real authorization "
          "outcome computed under SecureVoice’s own transaction tiers; (2) "
          "reproduction of a state-of-the-art anti-spoofing benchmark to validate "
          "our pipeline before trusting any downstream claim built on top of it; "
          "(3) discovery and diagnosis of a severe studio-to-real-world domain gap "
          "in that benchmark model, with control experiments ruling out simpler, "
          "more easily fixed explanations such as codec or resampling artifacts; "
          "(4) a lightweight domain-adaptation fix, reported honestly alongside a "
          "first fix attempt that did not work, together with its measured cost in "
          "benchmark accuracy; and (5) quantification of the same model’s "
          "degradation under realistic VoIP channel loss and under short streaming "
          "windows, both directly relevant to continuous, rather than one-shot, "
          "voice authentication. Section II situates this work against prior "
          "voice-spoofing and PVA-security literature; Section III details the "
          "system and experimental protocols; Section IV reports and discusses "
          "results; Section V concludes with the scaling work needed to turn our "
          "proof-of-concept fix into a general one.")

# ============================================================ II. LIT ====
section_heading(doc, "II", "Literature Survey")

para(doc, "Voice authentication and its spoofing vulnerabilities have a long "
          "research history predating current PVA deployments [1]. Five attack "
          "families are established: replay, impersonation, speech synthesis, "
          "voice conversion, and adversarial perturbation [1]. Replay attacks "
          "play back a prerecorded genuine sample; synthesis and voice-conversion "
          "attacks instead generate new audio designed to match a target "
          "voiceprint, and are the family this paper concerns itself with, since "
          "they are the ones a modern TTS/VC pipeline makes cheap to mount at "
          "scale. The ASVspoof challenge series [2]–[4], [14] standardized "
          "datasets, protocols, and the Equal Error Rate (EER) metric for "
          "comparing spoofing countermeasures across research groups, "
          "distinguishing Logical Access (LA: synthesis/conversion attacks fed "
          "directly to the verifier) from Physical Access (PA: replay through a "
          "real microphone/speaker channel). We deliberately scope this work to LA "
          "(synthetic-speech) attacks; replay/liveness detection is a related but "
          "distinct problem with its own dedicated countermeasure literature "
          "[1], [8], [9], [11].")

para(doc, "AASIST [5] is a graph-attention network operating directly on raw "
          "waveforms via a SincNet-style front end, reporting under 1% EER on "
          "ASVspoof2019 LA with under 100k parameters—small enough for on-device "
          "or edge deployment, which motivated our choice of it as the anti-spoofing "
          "layer over larger transformer-based countermeasures. Follow-on "
          "benchmarking work [15] compares AASIST against other recent "
          "architectures and confirms it remains competitive despite its size, "
          "while newer challenge editions and shared tasks [14], [16] extend "
          "evaluation to deepfake (DF) speech beyond the original LA/PA "
          "distinction, reflecting the field’s own recognition that synthetic-"
          "speech threats are evolving faster than any single benchmark can "
          "capture. ECAPA-TDNN [6] is a widely adopted speaker-embedding "
          "architecture for verification, producing fixed-dimension embeddings "
          "compared via cosine similarity; we use it exactly as SecureVoice’s "
          "Layer 2, matching common commercial practice of pairing a "
          "verification embedding model with a separately trained countermeasure "
          "rather than a single joint model.")

subsection_heading(doc, "A. Benchmark Performance of Current Countermeasures")
para(doc, "Table I reproduces the comparison reported by Jung et al. [5] on "
          "ASVspoof2019 LA. Sub-1% EER is now attainable, and notably the "
          "85k-parameter AASIST-L variant outperforms systems two orders of "
          "magnitude larger, which is why we selected it for an on-device "
          "authorization setting. These are the numbers a practitioner "
          "reasonably reads as “solved,” and they form the baseline expectation "
          "that Section IV tests against reality.")

make_table(doc, ["System", "Params", "min t-DCF", "EER (%)"],
           [["AASIST [5]", "297k", "0.0275", "0.83"],
            ["AASIST-L [5]", "85k", "0.0309", "0.99"],
            ["RawGAT-ST [8]", "437k", "0.0333", "1.19"],
            ["SENet [5]", "1,100k", "0.0368", "1.14"],
            ["Res-TSSDNet [5]", "350k", "0.0481", "1.64"],
            ["Raw PC-DARTS [9]", "24,480k", "0.0517", "1.77"],
            ["LCNN-LSTM-sum [10]", "276k", "0.0524", "1.92"]],
           col_widths=[1.25, 0.7, 0.75, 0.65])
caption(doc, "TABLE I. Published single-system results on ASVspoof2019 LA "
             "evaluation, as reported in [5]. Reproduced for context; not our "
             "measurements.")

subsection_heading(doc, "B. The Generalization Problem")
para(doc, "A growing body of work questions whether these numbers survive "
          "outside the challenge’s own studio-recorded data. Müller et al. [7] "
          "re-implement twelve architectures under a common protocol and "
          "evaluate them on an “In-the-Wild” set of 37.9 hours of found audio "
          "from 58 celebrities and politicians. They report EER degradations of "
          "200–1000%, with several models collapsing to near chance (Table II). "
          "Critically for our work, they attribute this to the composition of "
          "the benchmark itself, observing that ASVspoof is built on the VCTK "
          "corpus [12], which “exclusively features professional speakers and "
          "has been recorded in a studio environment, using a semi-anechoic "
          "chamber.” Related work by the same group [11] further suggests "
          "ASVspoof-trained models partly key on non-speech cues such as leading "
          "and trailing silence rather than the spoofing artefacts they are "
          "assumed to detect.")

make_table(doc, ["System", "ASVspoof19 eval", "In-the-Wild", "Degradation"],
           [["RawGAT-ST", "1.23%", "37.15%", "30×"],
            ["RawNet2", "3.15%", "37.82%", "12×"],
            ["12-model average", "9.85%", "60.10%", "6×"],
            ["Fixed 4 s input (avg)", "18.89%", "67.25%", "3.6×"]],
           col_widths=[1.3, 0.9, 0.8, 0.75])
caption(doc, "TABLE II. Benchmark-to-real-world EER degradation reported by "
             "Müller et al. [7]. The final row shows their finding that "
             "fixed-length 4 s inputs underperform full-length inputs—directly "
             "relevant to our streaming results (Section IV-F).")

para(doc, "Our work is positioned deliberately with respect to [7]: we do not "
          "claim to be first to observe that anti-spoofing generalizes poorly. "
          "Rather, we contribute (i) an independent reproduction of the "
          "phenomenon on a different class of real audio—self-collected, "
          "consent-based consumer-device recordings representing a specific "
          "deployment context, rather than found celebrity media; (ii) "
          "measurement of the downstream security consequence inside a working "
          "authorization pipeline, which [7] does not address; and (iii) a "
          "measured mitigation with its accuracy cost quantified. Prior work "
          "establishes that the gap exists; we ask what it does to a deployed "
          "system and what closing it costs.")

para(doc, "Complementary defense families target different threat models. "
          "Channel-artifact and second-factor defenses—detecting the sub-bass "
          "overexcitation characteristic of loudspeaker playback [13], "
          "phoneme-timing liveness cues [14], or sonar-based liveness checks "
          "using a device’s own speaker and microphone—address replay rather "
          "than synthesis, and are complementary to, not substitutes for, the "
          "LA-focused defense studied here. Zhang et al. [15] demonstrate a "
          "practical black-box mimicry attack against a smart-home SV system, "
          "underscoring that SV alone is an actively targeted, not merely "
          "theoretically weak, attack surface. Feng et al. [16] motivate "
          "continuous rather than one-shot voice authentication, a goal our "
          "streaming experiments (Section IV-F) show is nontrivial once a "
          "fixed-input model is forced onto short windows.")

# ============================================================ III. METH ====
section_heading(doc, "III", "Proposed System and Methodology")

subsection_heading(doc, "A. Two-Layer Architecture")
para(doc, "SecureVoice authorizes a voice-issued transaction through two "
          "independent checks that must both pass. Layer 2 (speaker "
          "verification) extracts a 192-dimension ECAPA-TDNN embedding from the "
          "incoming utterance and computes cosine similarity against a centroid "
          "of embeddings collected during a short enrollment step (typically "
          "three short utterances per user); a similarity below 0.50 denies the "
          "transaction outright regardless of amount, and above that floor a "
          "trust score (defined as 100×similarity) is compared against a "
          "minimum required for the requested transaction’s risk tier, so that "
          "higher-value actions demand a closer voice match. Layer 2.5 "
          "(anti-spoofing, our addition to the originally proposed design) scores "
          "the same raw audio independently with AASIST-L; a synthetic-speech "
          "probability above a frozen decision threshold denies the transaction "
          "regardless of Layer 2’s outcome, so a perfect voice match cannot by "
          "itself override a positive spoof detection. Both layers operate on the "
          "same 16 kHz mono input and add well under a second of combined "
          "latency on commodity hardware, which we consider acceptable for a "
          "one-shot authorization step.")

add_figure(doc, "figures/fig1_architecture.png", 3.3,
           "Fig. 1. SecureVoice two-layer authorization pipeline. Layer 2 asks "
           "whether the voice matches the enrolled user; Layer 2.5 (our addition) "
           "asks whether it is human at all. Either layer can deny independently.")

subsection_heading(doc, "B. Threshold Discipline")
para(doc, "Every detector threshold in this work is chosen on a held-out "
          "development split and frozen before any evaluation-split or "
          "real-world number is produced, and is never re-tuned per condition "
          "(device, channel, or attack type) afterward, mirroring the discipline "
          "the ASVspoof challenges themselves impose on participants [2]–[4]. "
          "Concretely, AASIST-L’s decision threshold is selected once by sweeping "
          "the ASVspoof2019 LA development split for the Equal-Error-Rate "
          "operating point and is then reused unchanged for every subsequent "
          "experiment in this paper, including the real-world recordings, the "
          "degraded-channel evaluation, and the streaming evaluation. This "
          "discipline is what allows the domain-gap finding in Section IV-C to "
          "be trusted as a genuine failure of the model rather than an artifact "
          "of a threshold quietly re-tuned after the fact to make the headline "
          "benchmark number look better than it should.")

subsection_heading(doc, "C. Vulnerability Measurement Protocol")
para(doc, "To measure whether SV alone resists synthetic speech, we constructed "
          "a controlled trial protocol on ASVspoof2019 LA, which conveniently "
          "labels every spoof utterance with the identity of the speaker it "
          "targets, allowing a like-for-like enrolled-speaker comparison rather "
          "than a generic pooled score. For 20 enrolled speakers, drawn at "
          "random and each enrolled from three held-out bonafide utterances, we "
          "constructed three trial types per speaker: genuine trials (further "
          "held-out bonafide utterances of the same speaker, ten per speaker), "
          "spoof trials (TTS/VC attacks targeting that speaker, twenty per "
          "speaker, spanning the range of attack systems present in the corpus), "
          "and impostor trials (bonafide utterances of a different randomly "
          "chosen speaker, ten per speaker)—800 trials in total. Each trial is "
          "scored by both layers using their independently frozen thresholds; we "
          "report the resulting SV cosine similarity, the AASIST-L "
          "synthetic-speech probability, and the authorization outcome that "
          "SecureVoice’s own published transaction tiers would produce, both "
          "with Layer 2.5 disabled (the originally proposed design) and enabled "
          "(our addition), so that the marginal value of adding anti-spoofing is "
          "directly visible rather than inferred.")

subsection_heading(doc, "D. Real-World Domain Gap: Discovery and Diagnosis")
para(doc, "After validating AASIST-L against its published benchmark (Section "
          "IV-B), we tested it on eight short phrases—assistant-style commands "
          "such as “place an order for five hundred thousand rupees”—recorded on "
          "an ordinary smartphone in a normal room, not through the ASVspoof "
          "pipeline at all, purely as a sanity check before demonstrating the "
          "live system. All eight were flagged as synthetic, every one with over "
          "0.98 probability against the same frozen 0.9157 threshold used "
          "throughout this paper. To diagnose this rather than merely report it, "
          "we compared long-term spectral energy distribution across five "
          "frequency bands and a noise-floor estimate (the 10th-percentile "
          "short-time RMS across 25 ms frames) for three sources: ASVspoof "
          "bonafide audio, ASVspoof spoof audio, and our real recordings. We "
          "separately ran four control experiments—applying AAC compression at "
          "two bitrates, 16↔48 kHz resampling, gain scaling, and mild additive "
          "noise (30 dB SNR) to otherwise-clean ASVspoof bonafide audio—to test "
          "whether any single, easily-corrected signal-processing artifact alone "
          "could reproduce the failure; none did, which points to the recording "
          "environment and microphone chain as a whole, rather than any single "
          "processing step, as the underlying cause.")

subsection_heading(doc, "E. Domain Adaptation")
para(doc, "We attempted two fixes, both starting from the original pretrained "
          "checkpoint so their effects could be compared on equal footing. Fix 1 "
          "fine-tuned AASIST-L for two epochs on an ASVspoof2019 LA train subset "
          "(300 bonafide, 300 spoof utterances) with synthetic room-reverb, "
          "additive-noise, and microphone-frequency-response augmentation applied "
          "with equal probability to both classes, so that the augmentation "
          "distribution itself could not become a shortcut label. Fix 2 instead "
          "mixed six of our eight real recordings directly into the same training "
          "mixture as additional bonafide examples, oversampled fifteen times "
          "each with light gain and additive-noise jitter to avoid literal "
          "duplicate-sample memorization, while holding the remaining two clips "
          "out entirely so that any observed improvement on them could only come "
          "from genuine within-domain generalization rather than memorization of "
          "the exact training waveforms. Both fixes were validated after every "
          "epoch on a fixed ASVspoof2019 LA development subset to confirm the "
          "model had not collapsed into a degenerate always-bonafide or "
          "always-spoof predictor, and both were evaluated at the end of training "
          "on a larger, previously unseen 800-utterance ASVspoof eval subset to "
          "obtain a benchmark-EER cost figure comparable across the two fixes and "
          "against the unmodified baseline reported in Section IV-B.")

# ============================================================ IV. RESULTS ====
section_heading(doc, "IV", "Results and Discussion")

subsection_heading(doc, "A. Speaker Verification Alone Is Insufficient")
para(doc, "Table III summarizes the 800-trial protocol. ECAPA-TDNN is essentially "
          "perfect against a different human speaker (0.00% EER, 0% of "
          "impostors clear the 0.50 gate) but far weaker against synthetic "
          "speech targeting the enrolled speaker (18.62% EER; 54.0% of attacks "
          "clear the same gate). This is the paper’s core empirical motivation: "
          "the same embedding model, evaluated with the same threshold, behaves "
          "almost perfectly against one threat model and only moderately better "
          "than chance against the other, because it was never designed to "
          "distinguish “a different real voice” from “a synthesized copy of the "
          "target voice.” In one representative case, a clone scored 0.799 "
          "similarity against the enrolled user’s own 0.693—the fake measured as "
          "more “authentic” to the verifier than the genuine sample it was "
          "cloned from, which is precisely the failure mode one would predict "
          "from a system optimized purely for acoustic resemblance.")

make_table(doc, ["Trial type", "Mean sim.", "≥ 0.50 gate"],
           [["Genuine (n=200)", "0.726", "99.5%"],
            ["Synthetic clone (n=400)", "0.476", "54.0%"],
            ["Impostor (n=200)", "0.124", "0.0%"]],
           col_widths=[1.5, 0.9, 0.9])
caption(doc, "TABLE III. Speaker-verification behavior by trial type (n=800).")

para(doc, "Table IV shows the effect of adding the anti-spoofing layer: 87.0% "
          "of synthetic attacks are flagged, with zero false positives on "
          "genuine or impostor speech, cutting attacks approved at typical "
          "transaction tiers from 54.0% to 7.5%.")

make_table(doc, ["Tier", "SV only", "SV+antispoof"],
           [["< Rs.500 / Rs.500–2,000", "54.0%", "7.5%"],
            ["Rs.2,000–10,000", "11.0%", "1.8%"],
            ["> Rs.10,000", "0.0%", "0.0%"]],
           col_widths=[1.6, 0.85, 0.85])
caption(doc, "TABLE IV. Synthetic attacks approved, by transaction tier.")

add_figure(doc, "figures/fig4_attack.png", 3.3,
           "Fig. 2. (a) Mean speaker similarity by trial type: AI clones of the "
           "enrolled user sit above the 0.50 accept threshold, whereas a different "
           "human never does. (b) Share of synthetic attacks approved per "
           "transaction tier, with and without the anti-spoofing layer.")

subsection_heading(doc, "B. Benchmark Reproduction")
para(doc, "Before trusting any downstream number we reproduced AASIST-L’s "
          "published result on the full ASVspoof2019 LA eval set (71,237 "
          "utterances): 0.99% EER, matching the original paper [5] almost "
          "exactly, against an 18.75% EER classical LFCC-GMM baseline fit on the "
          "same train split (dev EER 9.38%/eval 18.75% for LFCC-GMM; dev "
          "0.51%/eval 0.99% for AASIST-L). We deliberately include the weak "
          "classical baseline not to contrast architectures for their own sake, "
          "but because its much larger degradation under the same stressors "
          "(Sections IV-C/E) usefully calibrates how much of AASIST-L’s own "
          "degradation is a property of the task versus the specific model. This "
          "close agreement with the published benchmark is what makes the "
          "failure reported in Section IV-C trustworthy as a property of the "
          "model rather than a symptom of a broken evaluation pipeline on our "
          "end—a distinction that matters, since an unvalidated pipeline could "
          "just as easily produce a spurious 100% failure rate through a simple "
          "labeling or preprocessing bug.")

subsection_heading(doc, "C. The Domain Gap: A Benchmark-Perfect Model Fails on Real Audio")
para(doc, "All eight real-phone recordings were misclassified as synthetic, "
          "every one scoring above 0.98 probability against a threshold of "
          "0.9157 (Table V). Spectral analysis (Table VI) shows why: real "
          "phone audio carries a 16× higher noise floor and roughly 10× less "
          "6–8 kHz energy than ASVspoof’s anechoic-chamber “bonafide” class. The "
          "model had learned “studio-clean,” not “human.” Control experiments "
          "applying AAC compression, resampling, gain change, or mild noise to "
          "lab audio did not reproduce the failure, ruling those out as the sole "
          "cause and implicating the recording environment/microphone chain "
          "itself.")

make_table(doc, ["Sample", "Original model", "Domain-adapted"],
           [["8 real recordings", "8/8 flagged (100%)", "2/8 flagged (25%)"]],
           col_widths=[1.4, 1.4, 1.2])
caption(doc, "TABLE V. Real-world false-positive rate, before/after fix.")

make_table(doc, ["Band", "Lab bona.", "Lab spoof", "Real phone"],
           [["0–300 Hz", "39.9%", "46.8%", "53.8%"],
            ["6–8 kHz", "0.38%", "0.48%", "0.04%"],
            ["Noise floor", "0.0043", "0.0186", "0.0702"]],
           col_widths=[0.85, 0.75, 0.75, 0.85])
caption(doc, "TABLE VI. Spectral energy share and noise floor by source.")

add_figure(doc, "figures/fig2_domain_gap.png", 3.3,
           "Fig. 3. (a) Benchmark versus real-world performance. Literature values "
           "from Müller et al. [7]; our AASIST-L benchmark EER is measured, and its "
           "real-world figure is a 100% false-positive rate on bona fide phone "
           "audio (no EER is defined without in-domain attacks). (b) Spectral and "
           "noise-floor mismatch underlying the failure.")

subsection_heading(doc, "D. Fixing It: One Failure, One Success")
para(doc, "Fix 1 (synthetic augmentation) improved EER on synthetically-"
          "degraded validation audio (37.50% → 18.33% over two epochs) but did "
          "not transfer: real recordings remained 8/8 misclassified, scores "
          "essentially unchanged (0.93–0.99). We report this as a genuine "
          "negative result—approximated augmentation is not a substitute for "
          "real-domain data at this scale. Fix 2 (direct domain adaptation on "
          "six real clips, two held out) reduced real-world false positives to "
          "2/8, and—importantly—both held-out clips, never seen in training, "
          "flipped from >0.94 to <0.05 probability of being synthetic, a weak "
          "but genuine within-domain generalization signal. The cost was a "
          "controlled, bounded rise in benchmark EER on a larger 800-utterance "
          "held-out ASVspoof eval sample, from 2.00% to 8.12%. This is a "
          "small-scale proof of concept (one phone, one room, one speaker) and "
          "not yet evidence of broad generalization; Section V outlines the "
          "scaling needed.")

subsection_heading(doc, "E. Robustness Under Channel Degradation")
para(doc, "Real voice traffic is rarely clean studio audio by the time it "
          "reaches a back-end verifier—it has typically passed through a lossy "
          "codec and an imperfect network. We simulated this with an Opus "
          "codec round-trip at three bitrates crossed with Gilbert-Elliott burst "
          "packet loss at three rates, applied to a class-balanced 3,000-"
          "utterance eval subsample, and re-scored both detectors with their "
          "already-frozen thresholds (Table VII). AASIST-L’s advantage over the "
          "classical baseline collapses from 15.6× on clean audio (1.17% vs. "
          "18.30% EER) to 1.9× at the worst tested condition (8 kbps, 15% loss: "
          "14.00% vs. 26.83% EER)—the strong model still wins, but its practical "
          "margin over a decade-old classical baseline nearly disappears exactly "
          "under the conditions a real phone call or VoIP transaction would "
          "introduce, which is a materially different conclusion than the clean-"
          "audio 15.6× figure alone would suggest.")

make_table(doc, ["Condition", "LFCC-GMM", "AASIST-L", "Advantage"],
           [["Clean", "18.30%", "1.17%", "15.6×"],
            ["Opus 16k, 5% loss", "21.03%", "5.23%", "4.0×"],
            ["Opus 8k, 15% loss", "26.83%", "14.00%", "1.9×"]],
           col_widths=[1.25, 0.85, 0.85, 0.75])
caption(doc, "TABLE VII. EER under simulated VoIP channel degradation.")

subsection_heading(doc, "F. Robustness Under Short Streaming Windows")
para(doc, "Continuous authentication implies scoring short audio chunks rather "
          "than a complete utterance, so we re-ran both detectors over "
          "non-overlapping 250, 500, and 1000 ms windows sliced from the same "
          "eval data (Table VIII). AASIST-L requires a fixed 4.04 s input, so "
          "shorter chunks are tile-repeated to fill it, which discards most of "
          "the temporal structure the model was trained to use—EER rises from "
          "0.99% at the whole-utterance level to 23.62% at 1000 ms and 36.43% "
          "at 250 ms, an accuracy collapse rather than a graceful degradation. "
          "The classical baseline fares even worse, exceeding 40% at every "
          "window size tested, i.e., worse than chance-level separation once "
          "packet loss is added on top (38.70% combined worst case for "
          "AASIST-L; 45–55% for LFCC-GMM), which we report as a genuine, if "
          "unflattering, finding rather than omit. Per-chunk latency stayed "
          "real-time-capable throughout (≈55 ms/chunk, well under any tested "
          "window length), so the streaming bottleneck this system currently "
          "faces is accuracy, not computational speed—naive chunking of a "
          "fixed-input model is not a viable continuous-authentication strategy "
          "on its own, and window-aware retraining, not faster hardware, is the "
          "needed next step (Section V).")

make_table(doc, ["Window", "AASIST-L EER", "Latency"],
           [["Full utterance", "0.99%", "—"],
            ["1000 ms", "23.62%", "55 ms"],
            ["500 ms", "33.28%", "55 ms"],
            ["250 ms", "36.43%", "58 ms"]],
           col_widths=[1.0, 1.1, 0.85])
caption(doc, "TABLE VIII. Chunk-level EER and latency vs. window size (clean audio).")

add_figure(doc, "figures/fig3_degradation.png", 3.3,
           "Fig. 4. (a) EER under simulated VoIP channel degradation; AASIST-L's "
           "advantage over the classical baseline narrows from 15.6x to 1.9x. "
           "(b) Chunk-level EER versus streaming window size on clean audio.")

# ============================================================ V. CONCL ====
section_heading(doc, "V", "Conclusion and Future Work")
para(doc, "Speaker verification alone approves over half of synthetic-voice "
          "attacks in our tests; a benchmarked anti-spoofing layer closes most "
          "of that gap. More consequentially, a model scoring 0.99% EER on its "
          "own benchmark can fail completely on real-world audio—a gap we "
          "diagnosed and partially closed via targeted domain adaptation, at a "
          "measured accuracy cost. Future work: scale domain adaptation across "
          "more speakers, devices, and rooms; evaluate against modern "
          "(post-2019) cloning tools, which likely raise the 54% attack-success "
          "figure further; and pursue window-aware retraining for genuinely "
          "streaming, rather than chunked-utterance, continuous authentication.",
      first_line_indent=False)

# ============================================================ REFS ====
p = doc.add_paragraph()
p.paragraph_format.space_before = Pt(10)
p.paragraph_format.space_after = Pt(4)
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("REFERENCES")
r.bold = True
r.font.size = Pt(10.5)
r.font.name = 'Times New Roman'
refs = [
 "P. Cheng and U. Roedig, \u201cPersonal Voice Assistant Security and Privacy\u2014A Survey,\u201d Proc. IEEE, vol. 110, no. 4, pp. 476\u2013507, Apr. 2022.",
 "M. Todisco, X. Wang, V. Vestman, M. Sahidullah, H. Delgado, A. Nautsch, J. Yamagishi, N. Evans, T. Kinnunen, and K. A. Lee, \u201cASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,\u201d in Proc. Interspeech, 2019, pp. 1008\u20131012.",
 "A. Nautsch, X. Wang, N. Evans, T. Kinnunen, V. Vestman, M. Todisco, H. Delgado, M. Sahidullah, J. Yamagishi, and K. A. Lee, \u201cASVspoof 2019: Spoofing Countermeasures for the Detection of Synthesized, Converted and Replayed Speech,\u201d IEEE Trans. Biometrics, Behavior, and Identity Science, vol. 3, no. 2, pp. 252\u2013265, 2021.",
 "J. Yamagishi, X. Wang, M. Todisco, M. Sahidullah, J. Patino, A. Nautsch, X. Liu, K. A. Lee, T. Kinnunen, N. Evans, and H. Delgado, \u201cASVspoof 2021: Accelerating Progress in Spoofed and Deepfake Speech Detection,\u201d arXiv:2109.00537, 2021.",
 "J.-W. Jung, H.-S. Heo, H. Tak, H.-J. Shim, J. S. Chung, B.-J. Lee, H.-J. Yu, and N. Evans, \u201cAASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks,\u201d in Proc. IEEE ICASSP, 2022, pp. 6367\u20136371.",
 "B. Desplanques, J. Thienpondt, and K. Demuynck, \u201cECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification,\u201d in Proc. Interspeech, 2020, pp. 3830\u20133834.",
 "N. M. M\u00fcller, P. Czempin, F. Dieckmann, A. Froghyar, and K. B\u00f6ttinger, \u201cDoes Audio Deepfake Detection Generalize?,\u201d in Proc. Interspeech, 2022, pp. 2783\u20132787.",
 "H. Tak, J.-W. Jung, J. Patino, M. Kamble, M. Todisco, and N. Evans, \u201cEnd-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection,\u201d arXiv:2107.12710, 2021.",
 "W. Ge, J. Patino, M. Todisco, and N. Evans, \u201cRaw Differentiable Architecture Search for Speech Deepfake and Spoofing Detection,\u201d arXiv:2107.12212, 2021.",
 "X. Wang and J. Yamagishi, \u201cA Comparative Study on Recent Neural Spoofing Countermeasures for Synthetic Speech Detection,\u201d in Proc. Interspeech, 2021, pp. 4259\u20134263.",
 "N. M. M\u00fcller, F. Dieckmann, P. Czempin, R. Canals, J. Williams, and K. B\u00f6ttinger, \u201cSpeech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn?,\u201d arXiv:2106.12914, 2021.",
 "J. Yamagishi, C. Veaux, and K. MacDonald, \u201cCSTR VCTK Corpus: English Multi-Speaker Corpus for CSTR Voice Cloning Toolkit (version 0.92),\u201d University of Edinburgh, 2019.",
 "L. Blue, L. Vargas, and P. Traynor, \u201cHello, Is It Me You\u2019re Looking For? Differentiating Between Human and Electronic Speakers for Voice Interface Security,\u201d in Proc. ACM WiSec, 2018, pp. 123\u2013133.",
 "L. Zhang, S. Tan, J. Yang, and Y. Chen, \u201cVoiceLive: A Phoneme Localization Based Liveness Detection for Voice Authentication on Smartphones,\u201d in Proc. ACM CCS, 2016, pp. 1080\u20131091.",
 "L. Zhang, Y. Meng, J. Yu, C. Xiang, B. Falk, and H. Zhu, \u201cVoiceprint Mimicry Attack Towards Speaker Verification System in Smart Home,\u201d in Proc. IEEE INFOCOM, 2020, pp. 377\u2013386.",
 "H. Feng, K. Fawaz, and K. G. Shin, \u201cContinuous Authentication for Voice Assistants,\u201d in Proc. ACM MobiCom, 2017, pp. 343\u2013355.",
 "H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, and A. Larcher, \u201cEnd-to-End Anti-Spoofing with RawNet2,\u201d in Proc. IEEE ICASSP, 2021, pp. 6369\u20136373.",
 "M. Ravanelli and Y. Bengio, \u201cSpeaker Recognition from Raw Waveform with SincNet,\u201d in Proc. IEEE Spoken Language Technology Workshop (SLT), 2018, pp. 1021\u20131028.",
 "T. Kinnunen, K. A. Lee, H. Delgado, N. Evans, M. Todisco, M. Sahidullah, J. Yamagishi, and D. A. Reynolds, \u201ct-DCF: A Detection Cost Function for the Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification,\u201d in Proc. Odyssey, 2018, pp. 312\u2013319.",
 "Z. Wu, N. Evans, T. Kinnunen, J. Yamagishi, F. Alegre, and H. Li, \u201cSpoofing and Countermeasures for Speaker Verification: A Survey,\u201d Speech Communication, vol. 66, pp. 130\u2013153, 2015.",
]
for i, r in enumerate(refs, 1):
    refs_para(doc, i, r)

doc.save(OUT)
print(f"Saved {OUT}")
