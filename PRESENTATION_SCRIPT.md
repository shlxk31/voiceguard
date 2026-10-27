# Presentation Script — read or speak from this directly

Total time: ~7-8 minutes talking, then demo, then questions. Plain language,
say it in your own words — this is a guide, not something to memorize.

---

## PART 1 — Recap what Sir already knows (30 sec)

> "Sir, in Phase 1 we built SecureVoice — a voice-authenticated ordering
> system. A user enrolls their voice, and when they place an order, we use
> ECAPA-TDNN speaker verification to check if the voice matches, generate a
> trust score, and approve or deny based on the order amount — higher-value
> orders need a stronger voice match.
>
> That worked. But this phase, we asked a harder question: **what if the
> voice matches perfectly — but it's not actually a human speaking?**"

---

## PART 2 — The problem we found (1 min)

> "Speaker verification only answers one question: 'does this sound like the
> right person?' It was never built to answer: 'is this a real human, or an
> AI-generated fake?'
>
> That matters because a voice clone is *built* to sound like the target. So
> a good fake shouldn't fail the voice-match check — it should pass it.
>
> We tested this directly. We took our system, ran 800 test cases through
> it — real voices, different people's voices, and AI-cloned voices of our
> enrolled users — and measured what actually happened.
>
> Result: **54% of AI-cloned voices passed our voice-verification check.**
> Meanwhile, 0% of a *different real person's* voice ever passed. So the
> system is excellent at telling two humans apart — and much weaker at
> telling a human from a machine."

---

## PART 3 — What we added and how well it worked (1 min)

> "So we added a second, independent check — a dedicated AI-voice detector
> called AASIST-L, sitting alongside the voice-match layer. Both have to
> agree before an order goes through.
>
> With that layer added, the clones that get approved drop from 54% down to
> about **7.5%** — and it does this with **zero false alarms** on real users.
> It didn't just add security, it added security without annoying genuine
> customers.
>
> But before trusting any of these numbers, we did one more thing: we
> checked our detector against the actual published research benchmark for
> this model. The original paper reports 0.99% error rate. We measured
> **0.99%** ourselves, on the full 71,000-file test set. That match is
> important — it tells us our testing setup itself is correct, so everything
> after this is measured honestly, not just claimed."

---

## PART 4 — The real discovery (1.5 min) — this is the important part

> "Here's what we didn't expect. We took that same detector — 99% accurate
> on the official benchmark — and tested it on *our own real voices*,
> recorded on an ordinary phone.
>
> **It called 100% of our real recordings fake.** Every single one, with
> over 98% confidence.
>
> A model that's basically perfect in the lab was completely wrong on real
> audio. So we dug into why, instead of just noting it and moving on.
>
> The answer: the official training data is recorded in a professional,
> soundproof studio. Real phone audio in a normal room has 16 times more
> background noise, and much less high-frequency detail than studio audio.
> The model had learned to recognize 'studio-clean audio' as 'real human' —
> not actually 'human' itself. We ruled out simpler explanations too — file
> compression, resampling, volume — none of those alone caused it.
>
> And this isn't just us — a real published paper from 2022 found the exact
> same kind of gap on a completely different dataset. So this is a known,
> serious problem in this field — we just proved it happens in our own
> real-world setup too."

---

## PART 5 — The fix (1 min)

> "We tried two fixes. The first — training the model with artificially
> generated 'realistic' noise — didn't work. We're being upfront about
> that: it improved on our fake test data but made zero difference on our
> actual real recordings. A negative result, but an honest one.
>
> The second approach worked: we trained the model directly on a small
> number of our real recordings. Real-world false positives dropped from
> **8 out of 8, down to 2 out of 8** — and importantly, we tested this on
> recordings the model had *never seen before*, so it's not just memorizing.
>
> The cost: the official benchmark accuracy dropped a bit, from 2% error to
> about 8% error. We're stating that trade-off honestly instead of hiding
> it — because for a real product, a model that wrongly rejects real
> customers 100% of the time is a much bigger problem than a benchmark
> number moving a few points."

---

## PART 6 — Extra findings (30 sec, only if time allows)

> "Two more things we measured: how the system holds up on a bad phone
> connection — accuracy drops but it still clearly beats a basic method —
> and what happens if we try to check short 1-second clips instead of full
> sentences, for real-time detection. Short clips hurt accuracy a lot right
> now, which tells us that's the next real engineering problem to solve."

---

## PART 7 — Close (30 sec)

> "So to summarize: voice-matching alone isn't enough — we proved that.
> Adding an AI-voice detector measurably helps. And we found a real,
> serious weakness — that these detectors don't work outside a lab
> environment — diagnosed why, and partly fixed it. That's the paper, and
> that's what the demo will show live right now."

*(→ Now run the demo — see MENTOR_DEMO.md or QUICK_START_FOR_SIR.md for the
exact steps and files to use.)*

---

## IF SIR ASKS "what's actually new here, isn't this known?"

> "The general problem — lab models failing in the real world — is known in
> research, yes. What we added: we reproduced it ourselves on our own
> consumer-phone recordings, measured exactly what it does to a *working
> authorization system* — real approve/deny decisions — and measured a fix
> with its actual cost. Most papers report the problem; we also tried to
> close it and told the truth about how far we got."

## IF SIR ASKS ABOUT THE OLD SLIDE DECK'S REFERENCES

> "One more thing worth mentioning — while preparing citations for the new
> paper, we checked the references from our Phase 1 slides against actual
> published sources. Three of them — the Yadav & Rai, Liu et al., and Chen
> et al. entries — we could not find anywhere, under any search. They may
> have come from an AI drafting tool. We removed them and used 20 verified,
> real citations in the new paper instead."

---

## Files to have open
- `PRESENTATION_SCRIPT.md` (this file) — for you to glance at
- `paper/securevoice.pdf` — the paper, to show on screen
- Demo running at http://127.0.0.1:8900 — see QUICK_START_FOR_SIR.md
