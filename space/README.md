---
title: Research Paper Q&A
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Ask questions about your own research papers, with citations
---

# Research Paper Q&A

Upload a research paper and ask questions about it. Answers are built only
from passages retrieved out of that paper, and every claim cites the
excerpt, page and section it came from, so you can check it.

**The library starts empty on purpose.** Upload your own PDF to try it.
Nothing is shipped with this Space, because the corpus it was developed
against is made of copyrighted journal articles that are not mine to
redistribute.

## How it works

1. Text is extracted page by page, with section headings detected
2. Split into overlapping chunks that never cross a section boundary
3. Indexed twice: dense embeddings in FAISS, and a BM25 lexical index
4. A question queries both, the rankings are fused, and a cross-encoder
   re-ranks the survivors
5. The top passages go to a language model told to answer only from them

Hybrid retrieval exists because the two halves fail differently. Embedding
search maps "1e4 cm2/s" and "1e7 cm2/s" to nearly the same vector, so it
is weak exactly where a question asks for a specific value. BM25 matches
rare terms and is blind to paraphrase. Fusing them covers each one's blind
spot.

## This demo differs from the measured system

The numbers in the project's evaluation were produced by a different
configuration, running entirely on one machine. They do not describe this
Space.

| | measured locally | this demo |
|---|---|---|
| generation | Llama 3.1 8B on a local GPU | a hosted model |
| re-ranker | fine-tuned on a 50 paper corpus | the stock model |
| corpus | 50 papers, 3,658 passages | whatever you upload |
| privacy | nothing leaves the machine | retrieved passages are sent to the model host |

Retrieval on the held-out test split scored Hit@5 0.80 and answer
faithfulness 0.873 **in the local configuration**. Expect this Space to do
somewhat worse, mostly because the fine-tuned re-ranker is not here.

**Do not upload anything confidential.** Unlike the local version, this
demo sends the retrieved passages of your document to a third-party model
provider in order to write the answer.

## Running it properly

The full version runs offline with no external calls:
[github.com/hsuhas154/research-paper-rag](https://github.com/hsuhas154/research-paper-rag)
