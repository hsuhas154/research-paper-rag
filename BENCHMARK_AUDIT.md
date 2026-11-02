# Benchmark audit

A record of what testing the system on a phone turned up, and what
measuring the fix then revealed about the benchmark itself.

None of this changes [EVALUATION.md](EVALUATION.md). Those numbers stay as
they were measured. This document says what they mean, which is not
the same thing as changing them.

## How it started

A public link was opened for a dry run and real questions were asked
through it. One question, about the drivers of biodiversity loss,
returned eight passages. **Five of them were bibliography**, pages of
nothing but author names and journal citations.

That is not something any automated check here reports. It was found by
reading the output, which is the second time on this project that the
thing worth finding was invisible to the test suite.

## The bug, and the two I added fixing it

`strip_references_section` was leaving 172 chunks of citation list in the
index, across 9 of the 50 papers, for two independent reasons:

- The heading pattern required the line to end right after "references",
  so Science's **"REFERENCES AND NOTES"** never matched, and neither did
  Annual Review's **"LITERATURE CITED"**.
- Any heading before the final 30% of a paper was ignored. A long
  bibliography starts earlier than that: in one paper the heading was
  found correctly at 63% of the document and then discarded.

Fixing it took three attempts, and the two failures are worth recording
because both were worse than the original problem.

**Lowering the threshold to 50% broke "Theory of Superconductivity".**
That paper is a poor OCR scan in which the sentence "...references to the
literature may be found" wraps so that `references` sits alone on a line,
at 62% of the document. Position cannot tell that from a real heading.
Accepting it deleted twelve pages of physics. The fix is that a published
heading is capitalised and wrapped prose is not.

**Truncating everything after the heading destroyed appendices.** Several
papers here put their appendices behind their bibliography, and this cost
the RAG paper its "Further Details on FEVER" and "Retrieval Collapse"
sections and ResNet its object-detection baselines. Benchmark questions
were being answered out of exactly that text. Pages after the heading are
now dropped only while they still read like citations: reference pages
score 7.6 to 53 citation markers per thousand characters and appendix
pages score 0.3 to 4.7, so the boundary is wide.

Bibliography chunks now stand at 10, all of them in the ViT paper, whose
references begin at 49.4% and are followed by appendices. Leaving those is
the safe direction.

## The fix changes no measurement

This was worth knowing and was not the expected answer. The prediction on
record was that Hit@5 would improve, because five wasted passage slots out
of eight looked like pure retrieval loss. It does not.

On the 40 test questions whose evidence is genuine prose, with no evidence
lost on either side:

| | Hit@5 | MRR@5 |
|---|---|---|
| before | 0.88 | 0.752 |
| after | 0.85 | 0.756 |

And on answer quality, split by question type:

| | faithfulness | context use |
|---|---|---|
| genuine questions, before | 0.911 | 0.325 |
| genuine questions, after | 0.877 | 0.325 |

Context utilisation is identical. The faithfulness difference sits well
inside the interval measured for this judge at this sample size, which
[GENERATOR_COMPARISON.md](GENERATOR_COMPARISON.md) puts at roughly plus or
minus 0.06 for 38 questions.

Hit@5 only requires one relevant passage in the top five. The
bibliographies were occupying the other four, which costs a reader real
context and costs the metric nothing. That is why the demo exposed this
and the benchmark never did.

So the fix is justified on correctness, not on scores. It is the right
behaviour, and it is measurement-neutral.

## What the measurement exposed instead

**36 of the 76 test queries are answered by bibliographies or front
matter.** Ten were checked by hand and nine are unambiguous artifacts.
Their questions read like this:

- "What types of research are being cited in the references listed?"
- "What specific research papers were published between 1971 and 1973?"
- "What sources of funding supported the research at the University of
  Alabama at Birmingham?"

with evidence terms such as `['anselm', 'icvgip']` and
`['18th', 'anshumali']`, which are author names lifted out of citation
entries. One query asks about the Poly-encoders model while its evidence
terms are `['billion-scale', 'edouard']`, so the question and its answer
key do not even correspond.

This is the third contamination of the same family found in this
benchmark, after the 64.6% question-to-chunk lexical overlap and the OCR
typos used as evidence terms. All three come from the same cause: the
questions were generated from chunks, and a chunk of citations yields a
question about citations.

### It runs the opposite way to the obvious guess

These queries are hard, because a retriever is bad at finding fragments of
a reference list, and it should be. So they drag the average down rather
than inflating it.

| test queries | Hit@5 | MRR@5 |
|---|---|---|
| all 76, as published | 0.80 | 0.655 |
| the 40 with genuine prose evidence | 0.88 | 0.752 |

**The published 0.80 understates retrieval on real questions.** The
honest reading of `EVALUATION.md` is that 0.80 is the figure for a test
set about half of which asks the system to do something it was never
built to do.

## What has not been done

The corpus at `data/corpus` has not been re-indexed. Everything above was
measured on a parallel index built in a scratch directory, and the
original index files were backed up first.

Re-indexing, and any revision of the numbers in `EVALUATION.md`, waits on
the benchmark. Rebuilding the index would change those numbers slightly
for reasons unrelated to the contamination, and publishing a second set of
figures drawn from a test set that is 47% artifact would not be an
improvement on the first. The benchmark is the thing to fix, and it is a
larger piece of work than this was.

`scripts/reindex_corpus.py` exists for when that happens. It rebuilds from
the archived PDFs and passes every title through explicitly, because
`eval/splits.json` matches papers by title fragment and a silently
re-extracted title would reassign a paper's split.
