# Phase 2 Evaluation - 16-Paper Corpus

Testing the Phase 2 system against a deliberately heterogeneous library:
16 papers spanning machine learning, planetary science, computational
chemistry, microelectronics, power systems, and clinical prediction, with
publication years from 2001 to 2026 and layouts ranging from clean modern
two-column PDFs to a scanned IEEE conference paper with inconsistent fonts.

**Corpus:** 16 papers · 185 pages · 802 chunks · indexed in 7.7 s
**Retrieval:** `hybrid+rerank` (BM25 + dense → Reciprocal Rank Fusion →
cross-encoder), top-5 passages
**Generation:** Llama 3.1 8B via Ollama, local

---

## Headline results

| Metric | Result | What it means |
|---|---|---|
| **Paper-level routing** | **39/39 (100%)** | Every question retrieved its top passage from the correct paper, out of 16 |
| Answer factually correct | 36/39 (92%) | Verified by hand against the source PDFs |
| **Hallucinations** | **0/39** | Every failure was an explicit "the context does not contain this" |
| Relevant passage in top 5 | 38/39 (97%) | The evidence reached the model |
| Top-1 passage relevant | 31/39 (79%) | The evidence was ranked first |
| Answers carrying citations | 38/39 (97%) | Claims traceable to an excerpt |
| Mean answer latency | 3.3 s | Warm model, RTX 5070 8 GB |

The 100% paper-level routing is the result that matters most for a
multi-document system: with 16 unrelated papers in one index, no question
was answered from the wrong paper. The gap between that and 79% top-1 is
entirely about *which* passage within the right paper ranked first - and
since the model receives all five, it rarely matters.

**Retrieval benchmark across all 16 papers** (39 labelled queries,
`python -m scripts.compare_retrieval --k 5`):

| mode | Hit@5 | MRR@5 | P@5 | sec/query |
|---|---|---|---|---|
| dense (Phase 1 behaviour) | 0.82 | 0.705 | 0.50 | 0.019 |
| bm25 | 0.95 | 0.813 | 0.54 | 0.001 |
| hybrid | 0.92 | 0.787 | 0.53 | 0.004 |
| **hybrid + rerank** | **0.97** | **0.881** | **0.58** | 0.247 |

Dense-only retrieval - the entire Phase 1 strategy - drops to 0.82 Hit@5 at
16-paper scale, while BM25 climbs to 0.95. Scaling from one paper to a real
library is exactly where the Phase 2 work pays off, and it is a much sharper
justification for hybrid retrieval than the single-paper test gave.

---

## Table 1 - Questions asked, and how the model performed

Legend: **T1** = top-ranked passage was the correct one · **Top5** = correct
passage within the five given to the model · **Ans** = answer factually
correct, verified by hand against the PDF.

| Paper | Questions | Results and Metrics |
|---|---|---|
| **Attention Is All You Need** (11p, 36 chunks) | 1. How many identical layers are in the Transformer's encoder stack?<br>2. What BLEU score did the big Transformer reach on English-to-German?<br>3. Why is the dot product scaled before the softmax? | **T1 3/3 · Top5 3/3 · Ans 3/3** ✅<br>All exact: "N = 6", "28.4 BLEU on WMT 2014", and the vanishing-gradient rationale for √dk. Clean citations throughout. Slowest paper (6.1 s avg) - first query of the run, cold model. |
| **Adam: A Method for Stochastic Optimization** (15p, 41c) | 1. What default values are recommended for the exponential decay rates?<br>2. Why does Adam need initialization bias correction?<br>3. What does the name Adam stand for? | **T1 2/3 · Top5 2/3 · Ans 2/3** ⚠️<br>Q1 exact (β₁=0.9, β₂=0.999); Q2 correct and well-reasoned. **Q3 failed**: the one chunk containing "adaptive moment estimation" was ranked 2nd by BM25 but demoted out of the top 5 by the cross-encoder. The model answered honestly - "the paper does not provide an expansion" - rather than inventing one. Retrieval failure, not a hallucination. |
| **Generative Adversarial Nets** (9p, 35c) | 1. What two-player game do the generator and discriminator play?<br>2. Which image datasets were the adversarial nets trained on?<br>3. What is the optimal discriminator for a fixed generator? | **T1 3/3 · Top5 3/3 · Ans 3/3** ✅<br>Reproduced the minimax value function and the full dataset list (MNIST, TFD, CIFAR-10). Q3's formula is right but printed as `G(x) =` instead of `D*_G(x) =` - a PDF glyph-extraction artifact faithfully carried through, not a model error. |
| **Deep Learning** (Nature review, 9p, 78c) | 1. How does backpropagation compute gradients through a multilayer network?<br>2. What are the four key ideas behind convolutional neural networks?<br>3. What do the authors expect unsupervised learning to become? | **T1 3/3 · Top5 3/3 · Ans 3/3** ✅<br>Q2 returned the exact four - local connections, shared weights, pooling, many layers. Notable because this paper marks headings by font size alone; before the Phase 2 fix below it yielded **1 section for 9 pages**, and now yields 11. |
| **A Few Useful Things to Know About Machine Learning** (10p, 55c) | 1. What three components does every learning algorithm consist of?<br>2. Why does generalization matter more than training set performance?<br>3. Is more data or a cleverer algorithm more valuable? | **T1 3/3 · Top5 3/3 · Ans 3/3** ✅<br>Q1 exact (representation, evaluation, optimization). Q3 quoted the paper's own line - "a dumb algorithm with lots and lots of data beats a clever one with modest amounts of it." This paper's title is set across five short lines at 50pt; it was the case that broke title detection and drove the fix. |
| **Dai et al. 2024** - Venusian atmospheric chemistry (13p, 66c) | 1. What eddy diffusion coefficient Kzz value was used in the cloud layer?<br>2. Which existing photochemical model is it built on?<br>3. What novel mechanism eliminates O₂ at 70 km? | **T1 2/3 · Top5 3/3 · Ans 3/3** ✅<br>Q1 gave the full profile (1.7×10⁴ → 5.3×10⁴ cm² s⁻¹) *and* flagged the competing 1×10⁸ value from a later study - the numeric-lookup case dense retrieval handles worst. Q2 correct (VULCAN) but only at rank 5. |
| **Jiang et al. 2024** - Iron-sulfur UV absorber (8p, 58c) | 1. Which iron-bearing species is proposed as the Venus UV absorber?<br>2. How were the candidate minerals synthesised and characterised? | **T1 2/2 · Top5 2/2 · Ans 2/2** ✅<br>Named both mineral phases with formulae (rhomboclase, acid ferric sulfate) and reproduced the synthesis protocol including masses and volumes. |
| **Spacek et al. 2026** - UV-blue absorbance model (13p, 65c) | 1. Over what wavelength range is the unknown Venus UV absorber active?<br>2. What does the model assume about absorbance in bulk liquid? | **T1 2/2 · Top5 2/2 · Ans 2/2** ✅<br>Q1 correctly distinguished the well-constrained 350-450 nm band from the wider 200-600 nm claims, attributing each to its source study rather than collapsing them. |
| **Mahieux et al. 2024** - Venus trace-gas upper limits (23p, 90c) | 1. Which instrument was used to obtain the spectra?<br>2. What upper limit was derived for formaldehyde? | **T1 1/2 · Top5 2/2 · Ans 2/2** ✅<br>Q1 exact (SOIR echelle spectrometer on Venus Express, 65-170 km). Q2 gave the right number densities but appended a self-contradicting "no explicit upper limit" caveat - the numbers are correct, the hedge is noise. Largest paper in the corpus. |
| **Trabelsi et al. 2026** - Chlorine-sulfur isomers (11p, 47c) | 1. Which Cl-S isomers are studied as parents of ClS₂ and SCl₂?<br>2. What bond dissociation energies were computed? | **T1 2/2 · Top5 2/2 · Ans 2/2** ✅<br>Listed all three isomers and the per-isomer BDEs in eV with the CCSD(T) level of theory. This paper's title was extractable only after stripping Private Use Area icon glyphs (see fixes below). |
| **64-channel ultrasound transducer amplifier** (5p, 19c) | 1. What is the die size of the preamplifier ASIC?<br>2. What preamplifier architecture was used? | **T1 1/2 · Top5 2/2 · Ans 2/2** ✅<br>Q1 exact ("4.91mm x 6.91mm"). Hardest document in the corpus: a 2003 scan whose font sizes wobble mid-paragraph, whose title is set *smaller* than its own affiliation lines, and which initially produced 13 sub-60-word chunks out of 28. |
| **Characterization and modeling of EDT leakage** (6p, 36c) | 1. What causes edge direct tunneling leakage in ultrathin gate oxide MOSFETs?<br>2. What analytic model describes electron direct tunneling? | **T1 1/2 · Top5 2/2 · Ans 1/2** ⚠️<br>Q1 correct and precise (electron tunneling from n⁺ poly to the n-type drain extension, dominant over GIDL/BTBT at thin oxides). **Q2 vague** - the model is defined by equations whose symbols do not survive text extraction, so the answer could only point at the excerpt. A PDF-extraction limit, not a retrieval or generation limit. |
| **A Sub-λ-Size Modulator** (12p, 37c) | 1. Which active material is used in the sub-wavelength modulator?<br>2. What is the efficiency-loss limit in electro-absorption modulators? | **T1 0/2 · Top5 2/2 · Ans 1/2** ⚠️<br>Q1 correct (indium-tin-oxide, with context on transparent conducting oxides) despite ranking 2nd. **Q2 honest miss**: the model said the excerpts never define the term - true, the phrase appears in the title and headings but the definition is spread across figures. Worst top-1 rate in the corpus, though the evidence still reached the model both times. |
| **Magnetically Aligned Anisotropic Conductive Adhesive** (8p, 35c) | 1. How are the conductive particles aligned?<br>2. What microwave application is it used for? | **T1 2/2 · Top5 2/2 · Ans 2/2** ✅<br>Both exact, Q2 including the 90 GHz figure and the solder-bump replacement claim. |
| **Modular Multilevel Converters** (8p, 34c) | 1. Who invented the MMC and in what year?<br>2. Why are MMCs used in HVdc transmission? | **T1 2/2 · Top5 2/2 · Ans 2/2** ✅<br>Q1 is a single-chunk needle across the whole 802-chunk corpus - "Prof. Rainer Marquardt, 2001" - found at rank 1. |
| **Predicting Sepsis Onset in ICUs** (24p, 70c) | 1. What GNN architecture predicts sepsis onset?<br>2. How is predictive uncertainty estimated?<br>3. How does it perform under BMI and age distribution shifts? | **T1 2/3 · Top5 3/3 · Ans 3/3** ✅<br>Q1 exact (spectral-normalized heterogeneous graph transformer + approximate Gaussian process head). Q2 reproduced the posterior mean/variance equations, though Unicode maths symbols came through mangled from the PDF. |

---

## Improvements made during this evaluation

Testing against these 16 papers exposed six defects that the original
single-paper test could never have surfaced. All were fixed as part of
Phase 2, with unit tests added for each.

| # | Defect found | Cause | Fix | Measured effect |
|---|---|---|---|---|
| 1 | `Deep Learning` and `A Few Useful Things...` each detected **1 section across 9-10 pages** | Heading detection was purely textual; these journals mark headings by font size alone, with no numbering and no conventional names | Carry per-line font size and weight through extraction; detect a heading when it is ≥8% larger than body text *or* bold, **and** has the shape of a heading | 1 → **11** and 1 → **18** sections; citations now name a real section |
| 2 | Titles wrong on 3 of 16 papers | Font heuristic used a 15-character floor per line, which rejected the 13-char title "Deep learning" and every line of a 5-line wrapped title; one paper's title is set *smaller* than its affiliations | Prefer the PDF's own metadata title (validated, markup-stripped); fall back to size-tiered typography with positional tie-break; fall back finally to the filename | 13/16 → **15/16** correct, 16/16 usable |
| 3 | One title extracted as `' Online '` | Publisher icon fonts map UI glyphs ("View Online", CrossMark) into the Unicode **Private Use Area**; invisible to every text filter | Strip PUA codepoints at extraction | Title correct; also removes junk glyphs from chunk text corpus-wide |
| 4 | **11% of chunks under 60 words** (46% on the scanned IEEE paper) | False font-based headings each forced a chunk boundary, shattering passages into context-free scraps | Fold any "section" with under 40 words of body back into the previous section - judge a heading by what follows it, not by how it looks | 90 → **40** tiny chunks; mean chunk on the worst paper 81 → 131 words |
| 5 | A false heading **deleted that line's text** from the document | Detected heading lines were skipped rather than kept | Keep the heading as the first words of the section it introduces | No text loss on misfires; real headings now add retrieval signal ("A novel mechanism to eliminate O₂ at 70 km" describes its section better than any sentence in it) |
| 6 | Re-indexing the uploads folder produced **17 documents from 16 files** | `add_pdf` archived each PDF back into the same folder used to stage sources, so a re-index picked up its own copies | Archive to `data/corpus/pdfs/`, separate from `data/uploads/` | Duplicate-free re-index; regression test added |

**Tuned, with evidence rather than intuition.** A sweep over candidate-pool
size and BM25 fusion weight (`pool ∈ {20,30,40} × w ∈ {1.0,1.5,2.0}`) showed
weighting BM25 higher never helps - so Reciprocal Rank Fusion stays
parameter-free - while widening the pool from 20 to 30 lifts MRR@5 from
0.861 to **0.881**, with 40 giving nothing further. The default is now 30.

**Deliberately not done.** Blending the fusion rank back into the re-ranked
order gained +0.004 MRR and did not fix the one failing query. Chasing a
single test-set query with extra machinery is overfitting, not engineering,
so it was rejected and the limitation documented instead.

Test suite: **82 unit tests**, all passing.

---

## Re-ranker study: does domain adaptation help?

The Phase 2 evaluation blamed the stock re-ranker, `cross-encoder/ms-marco-MiniLM-L-6-v2`,
for its single outright failure, and Phase 3 scoped a domain-adapted
replacement. That turned out to be the wrong diagnosis, and the work to
establish it is worth keeping.

### Step 1: do bigger off-the-shelf re-rankers help? No.

Every model re-ranks an identical candidate pool, so differences are the
model and nothing else (`python -m scripts.compare_rerankers`):

| model | params | Hit@5 | MRR@5 | P@5 | sec/query |
|---|---|---|---|---|---|
| no re-rank (fusion order) | - | 0.923 | 0.787 | 0.533 | 0.000 |
| **ms-marco-MiniLM-L-6-v2 (current)** | 22M | **0.974** | **0.881** | 0.585 | **0.273** |
| ms-marco-MiniLM-L-12-v2 | 33M | 0.974 | 0.876 | 0.595 | 0.932 |
| BAAI/bge-reranker-base | 278M | 0.974 | 0.876 | 0.600 | 4.513 |

A model twelve times larger is sixteen times slower for no gain in Hit@5
and a slightly worse MRR. Hit@5 is saturated at 0.974 across all three,
which says the ceiling is set by what the first stage proposes, not by
how well it is re-ranked.

### Step 2: does fine-tuning on this corpus help? Also no.

In-domain data was built from the corpus itself
(`scripts.build_rerank_dataset`): the local LLM reads each chunk and
writes a question that chunk answers, giving 565 positives across 12
papers, and the project's own retrievers mine 2,260 hard negatives, which
are by construction the passages the deployed system actually confuses
with the right answer.

**Four papers were held out of training entirely** (Attention, Dai,
Sepsis, EDT), covering 11 of the 39 benchmark queries. This is the part
that matters: training and testing on the same papers measures
memorisation.

First attempt, 2 epochs at lr 2e-5:

| split | stock | fine-tuned |
|---|---|---|
| trained papers (28 queries) | 0.964 / MRR 0.887 | **1.000 / MRR 0.890** |
| **held-out papers (11 queries)** | **1.000 / MRR 0.867** | **0.818 / MRR 0.758** |

Textbook overfitting. The model learned these twelve papers and got
materially worse on the four it had never seen. Reported on trained
papers alone it would have looked like a 3.6 point Hit@5 win.

Lowering the learning rate removes the damage but adds nothing. The best
configuration found, 1 epoch at lr 5e-6:

| split | stock | best fine-tuned |
|---|---|---|
| all (39) | 0.974 / **MRR 0.881** / 0.068 s | 0.974 / MRR 0.864 / 0.069 s |
| held-out (11) | 1.000 / **MRR 0.867** / 0.068 s | 1.000 / MRR 0.821 / 0.067 s |
| trained (28) | 0.964 / **MRR 0.887** / 0.068 s | 0.964 / MRR 0.881 / 0.068 s |

Identical Hit@5, identical precision, identical latency, consistently
worse MRR.

### Conclusion: the stock re-ranker ships

Fine-tuning is not adopted. `scripts/train_reranker.py` and the dataset
builder stay in the repository because the finding is only credible with
the apparatus that produced it, and because both become useful again at a
larger corpus size, but a locally trained model is **not** picked up
automatically. Doing so would silently ship a worse re-ranker to anyone
who ran the script. It has to be opted into with `RERANKER_MODEL`.

What this changes about the earlier diagnosis: the re-ranker was blamed
for the `adam-name` failure, but no re-ranker tested fixes it, including
one trained on this very corpus. Hit@5 sits at 0.974 for all of them.
The passage either reaches the candidate pool or it does not, and that is
a first-stage question. Retrieval breadth, not re-ranking quality, is
where the remaining accuracy is.

### A measurement bug worth recording

The first latency figures showed the fine-tuned model three to ten times
faster than stock, which is impossible: identical architecture, identical
weights shape. The re-ranker loads lazily on first use, and that load sat
inside the timed loop, so on an 11-query split several seconds of model
loading were being divided across 11 queries. The harness now forces the
load before timing, after which both models measure 0.068 s/query as
they should.

---

## Known limitations, stated plainly

1. **The cross-encoder occasionally demotes the one correct passage.** It is
   trained on MS MARCO web search, not scientific prose. When the answer term
   never appears in the question ("What does *Adam* stand for?" → "adaptive
   moment estimation"), it can rank the single relevant chunk out of the top
   5 even when BM25 put it 2nd. This is the cause of the only outright
   failure in 39 questions. **Scoped into Phase 3:** replace it with a
   domain-adapted re-ranker.
2. **Equation-heavy content degrades.** Symbols, subscripts, and Unicode
   maths do not survive PDF text extraction intact, so questions whose answer
   *is* an equation (EDT model, sepsis posterior) get vague or garbled
   answers. Affects the source text, not the retrieval logic.
   **Scoped into Phase 3:** dedicated equation, symbol, and table extraction.
3. **Comparative questions cannot be served by a single top-k budget.**
   "Compare Adam and the Transformer paper" fills all five passage slots
   from whichever paper matches more strongly, leaving nothing for the
   second, so the system correctly reports it cannot compare. Re-ranking
   cannot fix this: the passages that would answer it never enter the pool.
   Surfaced by Table 2, invisible to Table 1 because every Table 1 question
   targeted one paper. **Scoped into Phase 3:** per-document retrieval
   quotas when a question names or implies more than one paper.
4. **Cross-paper contamination when routing is ambiguous.** All three wrong
   answers in Table 2 came from the system retrieving a neighbouring paper
   and reporting its numbers, for example quoting Venus cloud aerosol
   particle sizes for a question about conductive adhesive. It is not
   invention, every number existed somewhere in the corpus, but it is the
   dominant error mode now that hallucination is absent. It tracks routing
   directly: every contaminated answer had a wrong-paper top-1 passage.
5. **Section detection remains heuristic** on scanned papers with unstable
   fonts. The 40-word floor suppresses most false headings, but the 2003
   ultrasound paper still yields some odd section labels.
6. **No answer-faithfulness metric.** Correctness in both tables was
   verified by hand against the PDFs. Automated groundedness scoring (RAGAS)
   is Phase 3.
7. **Single-turn only.** Follow-ups like "and at 60 km?" carry no context.

---

## Table 2 - Independent test set (for your own testing)

A completely separate set of questions, with no overlap with Table 1: none of
these target the same fact, section, or figure as any question in it. Run
against the same corpus, unscoped across all 16 papers, on `hybrid+rerank`
with top-5 passages, and graded by hand against the source PDFs.

**Results across the 58 questions**

| Metric | Result |
|---|---|
| Top-1 passage from the correct paper | 43/48 (90%) |
| Correct paper within top 5 | 46/48 (96%) |
| Answers fully correct | 35/48 (73%) |
| Answers partially correct | 10/48 (21%) |
| Answers wrong | 3/48 (6%) |
| Facts invented from outside the corpus | **0/58** |
| Should-fail questions correctly declined | **4/4** |
| Mean latency | 5.3 s |

Harder than Table 1 by design, and the drop from 92% to 73% fully correct is
mostly a change in question type rather than a regression. Table 1 asked for
headline facts; Table 2 asks for parameter values, method details, and figure
contents, which sit in exactly the equation-heavy and table-heavy text that
survives PDF extraction worst.

**The dominant failure mode has changed, and it is not hallucination.** All
three wrong answers are cross-paper contamination: the system retrieved a
passage from a neighbouring paper and reported its numbers. Not one answer
invented a fact that is absent from the corpus, and all four should-fail
questions were declined cleanly. Contamination tracks routing: every
contaminated answer came from a question whose top-1 passage was from the
wrong paper.

**Comparative questions are a newly exposed structural limit.** "Compare Adam
and the Transformer paper" and "the eddy diffusion coefficient according to
each paper that mentions it" both failed the same way: a single top-5 budget
gets filled by whichever paper matches most strongly, leaving nothing for the
second. No amount of re-ranking fixes this, because the passages that would
answer it are never all in the pool. It needs per-document retrieval quotas,
which is Phase 3 work.

**How to re-run these:** start the app (`python app.py`, see
[RUNNING.md](RUNNING.md)), leave **Search which papers** on all-selected so
each question has to find its own paper, keep the strategy on
`hybrid+rerank`, and compare each answer against the source PDF.

| Paper | Questions | Results and Metrics |
|---|---|---|
| **Attention Is All You Need** | 1. What is the computational complexity per layer of self-attention versus a recurrent layer?<br>2. How many attention heads does the base model use, and what is the dimension of each?<br>3. What positional encoding scheme is used, and why that one? | **3/3 correct · routing 3/3** ✅<br>Q1 exact: O(n²·d) for self-attention against O(n·d²) for recurrent. Q2 exact: h = 8 heads, d_k = d_v = d_model/h = 64. Q3 gave the sinusoidal scheme with the correct rationale, that PE(pos+k) is a linear function of PE(pos) so relative positions are easy to attend to. |
| **Adam: A Method for Stochastic Optimization** | 1. What is the regret bound proved for Adam in the convex case?<br>2. How does AdaMax differ from Adam?<br>3. Which datasets and model types were used in the experiments? | **2/3 correct, 1 partial · routing 3/3** ⚠️<br>Q1 exact: R(T) = O(√T). Q2 exact: AdaMax is the infinity-norm variant. **Q3 contaminated**: the logistic regression, MLP and CNN list is right, but it then appended TFD and the GAN generator details, which belong to the GANs paper. Cross-paper bleed, not invention. |
| **Generative Adversarial Nets** | 1. How many steps is the discriminator trained for per generator step?<br>2. What are the stated disadvantages of adversarial nets?<br>3. How was the log-likelihood of test data estimated? | **3/3 correct · routing 3/3** ✅<br>Q1 exact: k = 1, the least expensive option. Q2 named both stated disadvantages, including the Helvetica scenario by name. Q3 exact: Gaussian Parzen window with σ by cross-validation, and it carried through the paper's own caveat about high variance in high dimensions. |
| **Deep Learning** (Nature review) | 1. What problem do distributed representations solve that a classical n-gram model cannot?<br>2. What is LSTM and what problem does it address?<br>3. What role do the authors assign to attention mechanisms? | **2/3 correct, 1 wrong paper · routing 2/3** ⚠️<br>Q1 and Q2 exact, including the LSTM memory cell as a gated leaky accumulator. **Q3 answered from the wrong paper**: it retrieved Attention Is All You Need and described the Transformer. The content is accurate, just not this paper. Defensible given the question says "attention mechanisms" and the corpus holds a paper by that name, but it is a genuine routing miss. |
| **A Few Useful Things to Know About Machine Learning** | 1. What are the three sources of error that bias and variance decompose?<br>2. Why does the author say intuition fails in high dimensions?<br>3. What is said about theoretical guarantees in machine learning? | **2/3 correct, 1 honest miss · routing 3/3** ⚠️<br>Q2 and Q3 exact, including that theoretical guarantees drive algorithm design rather than practical decisions. **Q1 declined**: it defined bias and variance correctly from the paper but said the three-way decomposition is not stated. Correct behaviour, the paper does not present it that way. |
| **Dai et al. 2024** - Venusian atmospheric chemistry | 1. What is the vertical resolution and grid structure of the model?<br>2. How is the CO abundance profile compared against observations?<br>3. What boundary conditions were set at the lower boundary? | **3/3 correct · routing 2/3** ✅<br>Q1 exact: 2 km vertical resolution, surface to 112 km, and it said plainly that grid structure is not described. Q2 exact: agreement below 90 km, larger than observations near 100 km by a factor of about 2. Q3 exact: fixed VMRs at the bottom boundary, zero flux for unlisted species, NO at 5.5 ppb. |
| **Jiang et al. 2024** - Iron-sulfur UV absorber | 1. What sulfuric acid concentrations were tested?<br>2. What thermodynamic calculations support the mineral stability?<br>3. How does the proposed absorber compare with previously suggested candidates? | **2/3 correct, 1 partial · routing 2/3** ⚠️<br>Q2 and Q3 correct, Q3 quantitatively so, giving 1278 cm⁻¹ at 375 nm against the 5 to 73 g/L needed by known efficient absorbers. **Q1 partial**: it returned the chloride concentration table and said the H₂SO₄ concentrations are not stated there, missing the 70 to 98 wt % range that appears elsewhere in the paper. |
| **Spacek et al. 2026** - UV-blue absorbance model | 1. What organic compounds were compared against the model?<br>2. How is the absorber's concentration in the cloud droplets constrained?<br>3. What does the paper conclude about biological versus abiotic origins? | **2/3 correct, 1 partial · routing 3/3** ⚠️<br>Q1 exact: croconic acid, with the Hartley and Colmenero attributions. Q2 exact, including the molar absorptivity threshold used to bound the concentration. **Q3 partial**: it reported the abiotic-pathway discussion and then said there is no explicit conclusion on biological versus abiotic origin. |
| **Mahieux et al. 2024** - Venus trace-gas upper limits | 1. What is the spectral resolving power of the instrument?<br>2. How is the noise level of a spectrum computed?<br>3. What upper limit was derived for ammonia? | **3/3 correct · routing 3/3** ✅<br>Q1 exact: resolution 0.11 to 0.21 cm⁻¹, resolving power about 21,000. Q2 correct on photon random noise plus signal drift. Q3 exact: 28.4 ± 0.6 ppt at 65 km, with number densities. Largest paper in the corpus at 90 chunks. |
| **Trabelsi et al. 2026** - Chlorine-sulfur isomers | 1. What level of theory was used for the electronic structure calculations?<br>2. What are the predicted vibrational frequencies of SSCl₂?<br>3. What are the astrophysical implications for detection on Venus? | **2/3 correct, 1 partial · routing 3/3** ⚠️<br>Q1 exact and complete: CCSD(T)-F12 for ground state, CASSCF then MRCI for excited states. Q3 correct on the near-UV absorber implication. **Q2 partial**: it gave ν₃ at 701.3 cm⁻¹ and 2ν₁ at 1395.8 cm⁻¹ from Table III rather than the full frequency set. |
| **64-channel ultrasound transducer amplifier** | 1. What CMOS process technology was the ASIC fabricated in?<br>2. What is the power consumption per channel?<br>3. How was the chip packaged and tested on the PCB? | **3/3 correct · routing 3/3** ✅<br>Q1 exact: AMI 0.5 µm double poly, triple level metal CMOS. Q2 did arithmetic the paper does not state, dividing 960 mW across 64 channels to give 15 mW per channel, correctly shown. Q3 exact: 144-pin PQFP with the coax and probe test setup. Oldest and worst-scanned paper in the corpus. |
| **Characterization and modeling of EDT leakage** | 1. What gate oxide thicknesses were characterised?<br>2. How does EDT leakage vary with gate voltage?<br>3. What is the Franz-type dispersion relation used for? | **2/3 correct, 1 honest miss · routing 3/3** ⚠️<br>Q1 correct: 1.4 to 2.4 nm. Q3 correct, including the 6.25 nm tunnelling path width from the gate edge. **Q2 declined**: the gate-voltage dependence is carried in figures and equations whose symbols do not survive extraction, so it said so rather than guessing. |
| **A Sub-λ-Size Modulator** | 1. What extinction ratio does the modulator achieve?<br>2. What is the device's energy consumption per bit?<br>3. How does the plasmonic mode confinement work? | **1/3 correct, 1 partial, 1 contaminated · routing 1/3** ❌<br>Weakest paper in the set. Q1 correct, 6 dB extinction, though printed as "6 dB=m" from a broken glyph. Q3 partial on the MOS-gap mode confinement. **Q2 contaminated**: it pulled 960 mW from the ultrasound paper before admitting it had no answer. Routing failed here more than generation did. |
| **Magnetically Aligned Anisotropic Conductive Adhesive** | 1. What particle size and volume fraction were used?<br>2. What insertion loss was measured, and over what frequency band?<br>3. How does the aligned adhesive compare against solder bumping? | **2/3 correct, 1 contaminated · routing 2/3** ⚠️<br>Q2 correct: 1.33 dB at 50 GHz. Q3 correct, comparing against solder bumps with per-formulation losses to 30 GHz. **Q1 contaminated**: it reported r_eff = 0.43 µm and v_eff = 0.52, which are Venus cloud aerosol size parameters from the Spacek paper, not adhesive particles. It did hedge first, but the numbers are from the wrong paper. |
| **Modular Multilevel Converters** | 1. What is a half-bridge submodule and how does it differ from a full-bridge?<br>2. What is meant by black-start capability?<br>3. Which real-world HVdc projects are cited? | **2/3 correct, 1 partial · routing 3/3** ✅<br>Q3 was the strongest answer in the whole set: six real projects with voltage and power ratings, Trans Bay Cable, Nan'ao, Skagerrak 4, Kriegers Flak, Rudong, Zhangbei. Q2 inferred black-start correctly from context. **Q1 partial**: it said the half-bridge is not explicitly defined and reasoned from the figure caption. |
| **Predicting Sepsis Onset in ICUs** | 1. Which clinical datasets were used for training and evaluation?<br>2. What baseline methods was the model compared against?<br>3. How far in advance of onset can sepsis be predicted, and at what performance? | **3/3 correct · routing 3/3** ✅<br>Q1 correct: MIMIC-IV with the Sepsis-3 derived groups. Q2 exact: GCN and GRU baselines, with the stated reason for each. Q3 exact: 1, 2 and 3 hour horizons with the relative AUC ordering. |

### Suggested cross-paper questions

These deliberately span papers, to test whether the system correctly draws on
more than one source or correctly declines:

| Question | Results and Metrics |
|---|---|
| Which papers propose a candidate for the Venus UV absorber, and do they agree? | **Partial** ⚠️ Drew on all three Venus absorber papers (Jiang, Spacek, Trabelsi) and described the iron-sulfur candidate well, but did not actually adjudicate the "do they agree" half. |
| Compare how Adam and the Transformer paper describe their optimizer settings. | **Failed, honestly** ❌ All five passages came from the Adam paper, so it correctly reported that it could not compare. The clearest instance of the comparative-question limitation: one top-5 budget cannot cover two papers. |
| What do the machine learning papers say about the role of unsupervised learning? | **Correct** ✅ Spanned the Nature review and Domingos, covering layerwise unsupervised pre-training and autoencoders. |
| Which papers use a Gaussian process, and for what purpose? | **Correct** ✅ Identified the sepsis paper only, which is right. It did not mistake the GAN paper's Gaussian Parzen window for a Gaussian process. |
| What is the eddy diffusion coefficient on Venus according to each paper that mentions it? | **Partial** ⚠️ Gave the Dai profile exactly (1.7×10⁴ to 5.3×10⁴ cm² s⁻¹) but only reached that one paper, so the per-paper comparison the question asks for was not delivered. |
| Do any two papers in this library disagree with each other, and about what? | **Partial** ⚠️ Found no cross-paper disagreement and instead surfaced a within-paper nuance about overfitting and noise. Honest, but the question is beyond what top-5 retrieval can support. |

### Questions that *should* fail

Worth testing that the system declines rather than invents. A good answer
here is an explicit "the context does not contain this":

| Question | Results and Metrics |
|---|---|
| What is the capital of France? | **Correct refusal** ✅ Declined and named the unrelated topics it had retrieved instead. |
| What learning rate did the sepsis paper use for the Transformer architecture in the attention paper? | **Correct refusal** ✅ Declined and additionally flagged that the premise conflates two papers. |
| What does this library say about quantum error correction? | **Correct refusal** ✅ Declined and listed the actual topics of the retrieved passages. |
| Who won the 2024 Nobel Prize in Physics? | **Correct refusal** ✅ Declined; no attempt to answer from pretrained knowledge. |
