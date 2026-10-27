# Quick Start — Presenting to Sir (do this yourself, no help needed)

## BEFORE THE MEETING (do this once, 2 min)

1. Open Terminal, go to the project folder:
   ```
   cd ~/Documents/voiceguard
   ```
2. Start the demo:
   ```
   .venv/bin/uvicorn src.api.demo_app:app --port 8900
   ```
3. Open your browser to: **http://127.0.0.1:8900**
4. Leave the Terminal window open in the background (closing it stops the demo).

*(If it says "port already in use," it's probably already running — just open the browser link.)*

---

## DURING THE DEMO — 4 steps, in this order

**Step 1 — Enroll your voice**
Click "Record enrolment clip" 3 times (speak normally), then click "Enrol."

**Step 2 — Show it works normally**
Click "Record & check," say anything → should say **ORDER APPROVED**.
*Say: "This is the system as originally designed — voice matching only."*

**Step 3 — Show the attack (the big one)**
Instead of recording, click "Choose file" and upload:
```
data/raw/asvspoof2019_la/ASVspoof2019_LA_eval/flac/LA_E_8339197.flac
```
Result: voice similarity passes (0.66), but **anti-spoofing layer denies it**.
*Say: "This is an AI-generated fake voice. It fooled the voice-matching check — but our added layer caught it."*

**Step 4 — Show the real-world fix**
Upload one of your own recordings:
```
recordings/live/live5.m4a
```
*Say: "We tested our anti-spoofing model on our own real voices. The original
model — even though it scores 99% accurate on the standard benchmark — called
100% of our real recordings fake. We found out why, fixed it, and this
improved model gets it right."*

---

## WHAT TO SAY — the whole story in 6 lines

1. "Voice systems usually just check *'is this the right person's voice?'*
   They never check *'is this actually a human, or an AI fake?'*"
2. "We tested this gap directly: **54% of AI-cloned voices fooled the
   voice-matching check** in our experiments."
3. "We added a second layer — a dedicated fake-voice detector — and it catches
   **87% of those fakes**, cutting successful attacks down to about 7%."
4. "Before trusting our own numbers, we double-checked our detector against
   the official published research benchmark — we got the *exact same* number
   researchers reported (0.99% error), so we know our testing setup is
   correct."
5. "Then we found a real problem: that same detector, despite being 99%
   accurate on the lab benchmark, **called our own real voices fake 100% of
   the time.** We figured out why — the lab data is recorded in a
   soundproof studio, real phones aren't — and we partly fixed it."
6. "So the honest summary: speaker-matching alone isn't enough, our added
   layer measurably helps, and we found and are fixing a real weakness that
   most published research doesn't even test for."

---

## WHEN SHOWING THE RESEARCH PAPER

Say this while flipping through it:

- **"It's IEEE-format, 6 pages, matches your requested structure."**
- **"Every number in it is something we actually measured — nothing is made
  up."** (Point to a table, e.g. Table III or the 54%/87% numbers.)
- **"We double-checked our related work — the citations from our original
  slide deck. Three of them didn't actually exist; we removed those and
  replaced them with 20 verified, real papers."**
- **"Our main finding — that lab-trained models fail on real audio — is
  backed by another real published paper (Müller et al., 2022) that found
  the same thing independently, on completely different data. That makes
  our result more credible, not less, because it's not just us."**
- **"We're upfront about the limits: we only tested on 8 real recordings from
  one phone. It's a proof that the idea works, not a finished product yet."**

If sir asks "what's novel here?" — say:
> "The generalization gap itself is known in research. What's new is: we
> reproduced it on our own real consumer-phone recordings, measured what it
> actually does to a working authorization system (approves/denies real
> transactions), and measured a fix — not just reported the problem."

---

## IF SOMETHING GOES WRONG

- **Demo won't load:** restart it —
  ```
  pkill -f "uvicorn src.api.demo_app"
  cd ~/Documents/voiceguard
  .venv/bin/uvicorn src.api.demo_app:app --port 8900
  ```
- **Mic doesn't work:** just use the "Choose file" upload option instead —
  all the demo steps above already use uploaded files, so you don't strictly
  need the mic at all.
- **Forgot the file paths:** they're both listed above under Step 3 and
  Step 4 — copy-paste them directly into the file picker or terminal.

---

## FILES TO HAVE OPEN/READY
- `SecureVoice_Research_Paper.docx` — Word version
- `paper/securevoice.pdf` — LaTeX/IEEE version (cleaner for showing on screen)
- This file, for your own reference during the meeting
