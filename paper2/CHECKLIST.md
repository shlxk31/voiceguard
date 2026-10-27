# Researcher checklist — before submission

## Must be replaced
- [ ] **Author list and order.** Currently all three names, no corresponding
      author marked, no e-mail addresses. IEEE requires at least one.
- [ ] **Affiliation detail** — department, institute, city, country are
      present; add postcode/e-mail if the venue requires it.
- [ ] **Venue formatting.** Written for `\documentclass[conference]{IEEEtran}`.
      A journal submission needs `[journal]` and will re-flow to a different
      page count.

## Must be independently verified
- [ ] **Every reference.** Titles, authors, venues and page numbers were
      written from knowledge, not fetched from a database. **Verify all 24
      against IEEE Xplore / ACM DL / the publisher before submitting, and add
      DOIs.** DOIs were deliberately omitted rather than guessed, because a
      wrong DOI is worse than none. There are 23 references.
- [ ] **Do not reuse the reference list from the Phase-1/Phase-2 slide decks
      without checking.** Several entries there could not be located in any
      publication database during this project.

## Must be stated or obtained
- [ ] **Ethics / consent.** All live audio is from one project member. If any
      other person's voice is ever recorded, institutional ethics approval and
      written consent are required, and the paper must say so.
- [ ] **Hardware and software versions.** Add the machine (Apple silicon,
      CPU/GPU), Python version, PyTorch version, SpeechBrain and AASIST commit
      hashes, and the exact ASVspoof2019 LA protocol files used.
- [ ] **Language-model version.** The paper deliberately does not name a
      commercial model. If you name one, record the exact model identifier and
      the date, because these change without notice.
- [ ] **Data availability statement**, and whether the captured audio can be
      released (it is one member's biometric data — probably not).

## Results that are labelled as provisional and must be strengthened
- [ ] **L2 liveness (100% / 0% EER).** 15 pairs, one speaker, one room, one
      playback device. Repeat with a high-quality loudspeaker, at least two
      rooms and multiple speakers before this number is quoted anywhere
      without its scope.
- [ ] **L4 context layer.** Calibrated on a template-generated corpus. Collect
      genuine adversarial commands, or keep every L4 number explicitly marked
      as corpus-dependent separation rather than a deployment error rate.
- [ ] **Composite trials.** The ablation pairs independently measured
      conditions and assumes layer errors are independent. State this in any
      talk, and consider a joint-capture study.
- [ ] **Demographic evaluation.** All live results come from one speaker. No
      fairness claim can be made; the paper says so, and a reviewer will ask.
- [ ] **Voice conversion.** Under-tested relative to TTS. Prior work reports
      VC is the harder family.
- [ ] **Attack prior.** The sensitivity analysis shows the system depends on
      it heavily. Estimate it from real logs if any deployment data becomes
      available.

## Consistency pass already performed
- [x] Section numbering I–IX, sequential
- [x] All 23 citations referenced in text; no orphan entries
- [x] All equations numbered and referenced; symbols consistent
- [x] All 4 figures and 7 tables captioned and cited in text
- [x] No undefined references (`??`) in the compiled PDF
- [x] Every numeric claim cross-checked against its source result file
