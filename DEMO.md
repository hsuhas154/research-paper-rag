# Demo runbook

Two ways to show this system to someone, chosen because both are free and
neither depends on a hosting account. Hosted options were considered and
declined; see [DEPLOYMENT.md](DEPLOYMENT.md) for why.

1. **A recording.** Cannot break, works with no network, and shows the
   configuration the measured numbers actually describe.
2. **A temporary public link.** `share=True` tunnels to this machine, so a
   viewer uses the real system on the real GPU.

The recording is the one to make first. It is what you fall back on when
the venue wifi fails or a laptop refuses to wake.

## The temporary public link

```bash
SHARE=1 python app.py
```

Gradio prints a public URL valid for about 72 hours. It works only while
the process runs and only while this machine is awake.

**Point it at a throwaway library.** The link is public, and anyone holding
it can read what is indexed and add to it:

```bash
SHARE=1 CORPUS_DIR=data/demo-corpus python app.py
```

That starts an empty library in a separate directory, leaving the real
one untouched. Upload one or two papers you are happy for a stranger to
see.

Removing papers is disabled whenever `SHARE` is set. The button is hidden
and the handler refuses, because hiding a Gradio control takes it off the
page but leaves the function reachable to anyone with the link.

What is not verified: the tunnel itself has never been opened from this
machine. Gradio downloads a helper binary the first time, so do a dry run
before you need it rather than in front of someone.

## What to record

Record with **local generation**, not the hosted path. It is the honest
system, it needs no API key, and it is the configuration every number in
[EVALUATION.md](EVALUATION.md) was measured on.

```bash
ollama serve
```

```bash
python app.py
```

Roughly eight minutes, in this order:

**1. The library, about 30 seconds.** Open the Library tab on the real
corpus. Fifty papers, the chunk count, the filter box. Say that the
library is persistent and that the filter exists because fifty checkboxes
with no search is not a control anyone can use.

**2. One question, about 90 seconds.** Ask something that needs a specific
value out of a specific paper. Point out the order things happen in: the
sources appear in about a tenth of a second, then the answer streams in
behind them, and every claim carries the excerpt, page and section it came
from. Click through to a source and check one claim on camera. That single
act is the whole argument for the project.

**3. The retrieval modes, about two minutes.** This is the part that shows
engineering rather than plumbing, so do not rush it. Ask the *same*
question under dense, then BM25, then hybrid, then hybrid with
re-ranking, and read out how the retrieved passages change.

Have the reason ready: embedding search maps "1e4 cm2/s" and "1e7 cm2/s"
to nearly the same vector, so it is weakest exactly where a question asks
for a number, while BM25 matches rare terms and is blind to paraphrase.
Fusing them covers each one's blind spot. On the held out test split,
re-ranking moves Hit@5 from 0.737 to 0.803.

**4. A question the papers cannot answer, about 60 seconds.** Ask something
outside the corpus and let it say so. Most demos avoid this. Showing it
is more convincing than another good answer, because the failure mode
everyone expects from a RAG system is confident invention.

**5. The numbers, about 90 seconds.** On the held out test split: Hit@5
0.80, MRR@5 0.655, faithfulness 0.873 over 195 claims, citation validity
1.0, no contradictions. Say the word "held out" and mean it: the split is
paper level and stratified across 7 topic clusters, so no paper
contributes to both training and test.

Say plainly that the judge is the generating model and is lenient about
its own output, which makes faithfulness a floor rather than a
certificate. Volunteering a limitation is worth more than the number.

**6. What deployment cost, about 60 seconds.** Generation can move to a
hosted endpoint with one environment variable. Measured across the local
model, gpt-oss-20b and gpt-oss-120b on the same 40 questions with
retrieval and judge held fixed, faithfulness was statistically
indistinguishable: every paired bootstrap interval includes zero. So
hosting costs privacy and money rather than answer quality. See
[GENERATOR_COMPARISON.md](GENERATOR_COMPARISON.md).

## Two things worth saying out loud

Both are in the history and both land better volunteered than discovered.

**The benchmark was wrong twice before it was right.** An early version had
64.6% lexical overlap between questions and the chunks that answered them,
and used OCR typos as evidence terms. Both inverted the retrieval
conclusion, which means the first version of this project would have
reported the wrong result confidently.

**Mathematical characters were silently indexed as Korean.** A truncation
bug turned U+1D456 into U+D456 across 520 characters. Every automated
check passed through it. It was found by reading the output.

## Before recording

- `ollama serve` is up and the model is warm. The first question after
  startup pays a 30 to 40 second load, and that is not what the recording
  should open on. Ask one throwaway question first.
- The real corpus is in place, not a demo directory.
- Close anything with a paper title or a filename you do not want on
  camera.
