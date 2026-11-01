# Generator comparison

This document records what happens to answer quality when generation
moves off this machine. It exists because deployment introduced a second
way to generate answers, and the honest thing to do with a second
configuration is measure it rather than assume it behaves like the first.

**It does not replace [EVALUATION.md](EVALUATION.md).** That file records
the system as it was measured running entirely locally, and those numbers
stay as they are. This is a separate measurement of a separate
configuration, reported alongside rather than on top.

## What was held fixed

Only the model writing the answer changes. Everything upstream of it is
the same in all three runs:

- the same corpus of 50 indexed papers
- the same hybrid retrieval, dense plus BM25 fused by reciprocal rank
- the same fine-tuned cross encoder re-ranking the candidates
- the same 40 questions from the held out test split
- the same `top_k` of 5 passages per question
- the same judge

The judge deserves emphasis. Claim extraction and per claim judging in
`src/faithfulness.py` call `ollama.chat` directly rather than going
through the provider abstraction, so the judge was local `llama3.1:8b` at
temperature 0 in every run, including the two where generation was
remote. That was not a deliberate design decision for this experiment,
but it is the property that makes these three numbers comparable at all.

One asymmetry follows from it, and it matters when reading the table:
in the local run the judge is grading its own output, while in the two
hosted runs it is grading a different model's. A model is a lenient judge
of itself, so the local row carries an advantage the other two do not.

## Results

40 questions asked, 0 failed, in all three runs.

| generator | faithfulness | contradictions | citation validity | context use | claims | unsupported | wall clock |
|---|---|---|---|---|---|---|---|
| llama 3.1 8B, local | 0.8825 | 0.0 | 1.0 | 0.30 | 196 | 22 | 281 s |
| gpt-oss-20b, hosted | 0.8728 | 0.0 | 1.0 | 0.37 | 201 | 22 | 775 s |
| gpt-oss-120b, hosted | 0.8509 | 0.0 | 1.0 | 0.42 | 221 | 31 | 738 s |

Faithfulness is the mean of the per question supported claim fraction.
Unsupported is the claim count the judge declined to support.

## The ranking does not survive testing

Read as a league table, that looks like a finding: the local 8B model
tops it and the largest model comes last. It is not a finding. A paired
bootstrap over the per question scores, 20,000 resamples, puts zero
inside every interval:

| comparison | mean difference | 95% interval |
|---|---|---|
| local minus gpt-oss-20b | +0.0101 | [-0.0530, +0.0684] |
| local minus gpt-oss-120b | +0.0316 | [-0.0282, +0.0921] |
| gpt-oss-20b minus gpt-oss-120b | +0.0232 | [-0.0382, +0.0867] |

The one clean head to head is 20b against 120b, since neither model is
judging itself there. Question by question, 120b scored higher on 10,
lower on 11, and tied on 17. A sign test gives a two sided p of 1.000.

**On this benchmark the three generators are indistinguishable on
faithfulness.** Forty questions is not enough to separate models that
differ by two or three points, and the apparent ordering is noise.

### A prediction that did not hold up

Before these runs I predicted on the record that gpt-oss-120b would
score worse on faithfulness than gpt-oss-20b. The reasoning was a single
answer it had produced, which read better than the 20b equivalent while
introducing "depletion region" and "field opposes further diffusion",
phrases that appear nowhere in the retrieved excerpts, with citations
attached to them.

The aggregate moved in the predicted direction, 0.8509 against 0.8728.
The prediction is still not confirmed, because that gap does not survive
a paired test. One answer was never enough to generalise from, and the
direction matching is what a coin flip looks like half the time.

## What is measurably different

Two differences do hold up under the same paired test, both comparing
gpt-oss-120b against the local 8B model:

- **It makes more claims per answer.** Mean difference 0.625 claims,
  interval [0.075, 1.175]. 221 claims against 196 over the same
  questions.
- **It draws on more of the retrieved context.** Mean difference 0.120,
  interval [0.045, 0.205]. Context use rises from 0.30 to 0.42.

So the larger model is more thorough: it says more, and it reaches into
more of the five passages it was given. What it does not do is get a
larger share of what it says supported. The extra claims arrive with
roughly the same hit rate, which is why total unsupported claims rise
from 22 to 31 while the ratio stays flat.

Citation validity is 1.0 and contradiction rate 0.0 for all three. Every
excerpt number any model cited refers to an excerpt that was actually
supplied. This is worth stating because it was briefly untrue: the
citation pattern originally matched only ASCII brackets, and gpt-oss
writes its citations with full width CJK brackets, U+3010 and U+3011.
Before that was widened, these runs would have reported citation validity
of 0.0 for a model that was citing correctly throughout.

## What the wall clock does not tell you

The seconds column is not a generation speed benchmark and should not be
quoted as one. Every run pays for local claim extraction and local per
claim judging, which is several Ollama calls per question, and the hosted
runs add network round trips and any rate limit backoff on top. The
hosted runs being slower than the local one says almost nothing about
hosted generation speed. Measuring that would need the judge taken out of
the timing.

## Limitations

- **Forty questions.** Enough to detect a gross regression, not enough to
  rank models a few points apart. Every comparison here is underpowered.
- **The judge is an 8B model.** It is lenient, and it is lenient about
  its own output in particular, which is exactly the local row. Treat
  every faithfulness number as a floor rather than a certificate.
- **Single run per generator.** No seeds were varied, so run to run
  variation is unmeasured and is not in the intervals above.
- **Claim extraction is itself a model output.** Two generators phrasing
  the same fact differently can yield different claim counts, so claim
  counts are a property of the answer and the extractor together.

## Reproducing

Needs a Groq key in the calling shell, and Ollama available for the local
run and for the judge:

    read -s GROQ_KEY && export GROQ_KEY
    bash scripts/compare_generators.sh

Per model output lands in `eval/generator_comparison/`, as a JSON file
with the summary and per question rows, and a log of the full run. A run
that fails any question exits non zero and its results are kept as
`<model>.partial.json`, so an incomplete comparison cannot be mistaken
for a finished one.
