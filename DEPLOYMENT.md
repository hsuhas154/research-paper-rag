# Deployment

Three ways to run this, from least to most moving parts. Everything below
runs the same code; what changes is where it runs and who can reach it.

**What was actually chosen is section 4.** Sections 2 and 3 describe
shapes that fit, not shapes that were built.

**Verification note, stated up front.** The container files were written
and reviewed but **not built or run**, because Docker is not available in
the WSL distribution this project is developed in. The Python
configuration they depend on, `CORPUS_DIR` and `OLLAMA_HOST`, was tested
directly by running the app with those variables set and a corpus written
to a separate directory. Treat the compose stack as a first run that will
probably need a fix or two, not as something known to work.

---

## 1. Local, no containers

What the project has used throughout. See [RUNNING.md](RUNNING.md).

```bash
ollama serve
```

```bash
python -m src.api
```

The browser client is then at **http://127.0.0.1:8000** and the API docs at
**http://127.0.0.1:8000/docs**. The Gradio app, `python app.py`, remains as
the developer-facing interface.

---

## 2. Containers on one machine

```bash
docker compose up --build
```

With an NVIDIA GPU and the container toolkit installed:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open **http://localhost:8000**.

First start pulls the language model, about 4.9 GB, into a named volume.
That happens once. The app waits for Ollama to be healthy and for the pull
to finish before it starts.

### What is where

| Service | Purpose | GPU |
|---|---|---|
| `app` | retrieval, API, browser client | no |
| `ollama` | language model runtime | yes, with the override |
| `model-puller` | pulls the model once, then exits | no |

| Volume | Holds | Lose it and |
|---|---|---|
| `corpus` | the index and archived PDFs | every paper must be re-indexed |
| `model-cache` | embedder and re-ranker weights, ~200 MB | they re-download on first use |
| `ollama-models` | the language model, ~4.9 GB | it re-pulls |

Ollama is not published to the host. Nothing outside the compose network
needs to reach it, and exposing it would put an unauthenticated inference
endpoint on the machine.

### Loading papers

The corpus volume starts empty. Either upload PDFs through the web client,
or copy them in and index from inside the container:

```bash
docker compose cp ./papers/. app:/data/uploads/
```

```bash
docker compose exec app sh -c 'for f in /data/uploads/*.pdf; do python -m src.corpus add "$f"; done'
```

### The re-ranker gap, which matters more than it looks

The image does **not** contain the fine-tuned re-ranker. Its weights are
not in git, so a container built from a clean checkout falls back to the
stock model and scores **Hit@5 0.737 instead of 0.803** on the held-out
test split. Nothing warns you; retrieval simply gets worse.

Two ways to close it. Mount a model you trained locally:

```yaml
services:
  app:
    volumes:
      - ./models/reranker-domain:/models/reranker-domain:ro
    environment:
      RERANKER_MODEL: /models/reranker-domain
```

Or train it inside the running stack, which needs the corpus indexed
first and takes roughly 40 minutes:

```bash
docker compose exec app python -m scripts.build_rerank_dataset --out /data/rerank_train.jsonl
```

```bash
docker compose exec app python -m scripts.tune_reranker --data /data/rerank_train.jsonl
```

Confirm which model is live:

```bash
docker compose exec app python -c "from src.reranker import default_reranker_model; print(default_reranker_model())"
```

---

## 3. Cloud

Nothing here has been deployed to a cloud account. These are the shapes
that fit, with the costs that drive the choice.

### The decision that dominates everything

**Does the GPU travel?**

Generation is the only component that wants one. On an 8 GB card Llama
3.1 8B answers in 2 to 4 seconds. On CPU the same model takes roughly 30
to 90 seconds per answer, depending on cores. Everything else, embedding
and re-ranking included, is fast on CPU and stays that way.

So the real question is whether 30-second answers are acceptable. If they
are, deployment is cheap and ordinary. If they are not, the bill is
dominated by a GPU instance that sits idle between questions.

| Shape | Roughly | Answer latency | Fits when |
|---|---|---|---|
| CPU VM, 4 vCPU and 8 GB | tens of dollars a month | 30 to 90 s | demos, a handful of users, cost matters most |
| GPU VM, one T4 or A10 | hundreds a month | 2 to 5 s | interactive use, the GPU is busy enough to justify it |
| Split: app on CPU, generation on a hosted API | pay per token | 1 to 3 s | willing to send paper text to a third party |
| Local only | nothing | 2 to 4 s | the honest answer for a single-user research tool |

The third row is the one that changes the project's character. Every
answer would send retrieved passages to an external provider, which gives
up the property this system was built around: that nothing leaves the
machine. That is a real trade, not a detail.

What it costs in answer quality was measured rather than assumed, across
the local model and two hosted ones on the same 40 test questions with
retrieval and judge held fixed. The short version is that faithfulness is
indistinguishable between them at this sample size, so the trade is about
privacy and cost rather than groundedness. See
[GENERATOR_COMPARISON.md](GENERATOR_COMPARISON.md), which is kept separate
from [EVALUATION.md](EVALUATION.md) so the locally measured numbers stay
as they were recorded.

### Before exposing it to anyone else

The API has **no authentication**. It was built for a single user on
localhost, and it is not safe on a public address as it stands:

- `POST /documents` accepts uploads from anyone who can reach it.
- `DELETE /documents/{id}` deletes from the library, with no undo.
- `POST /ask/stream` runs model inference, which is the expensive path
  and an easy way to exhaust a machine.

Put it behind something that authenticates before it is reachable from a
network you do not control. For a single user, an SSH tunnel is simpler
and safer than anything else here:

```bash
ssh -N -L 8000:localhost:8000 user@host
```

### Other things that will bite

**Storage must be persistent.** On a platform with an ephemeral
filesystem, every restart loses the index. Mount real disk for `/data`,
or accept re-indexing on each boot: 50 papers take about 40 seconds.

**Startup is slow by design.** The app loads an embedding model, a
re-ranker, and the corpus before serving. Allow about 3 minutes before an
orchestrator decides it is unhealthy; the Dockerfile sets
`start-period=180s` for exactly that.

**Scaling out needs shared state.** Two app containers each hold their own
in-memory FAISS index built from the same volume. Reads are fine. Writes
are not: a paper indexed by one container is invisible to the other until
it restarts. One writer, or a real vector database, before running more
than one replica.

---

## 4. What was chosen, and why

**A recording plus a temporary public link.** Neither costs anything and
neither needs a hosting account. The runbook for both is in
[DEMO.md](DEMO.md).

Hosting was considered properly and declined on cost. For the record, so
the reasoning survives:

- **Hugging Face Spaces.** Creating a Gradio Space requires a paid plan,
  about 9 USD a month, even though the CPU Basic hardware it then runs on
  is free. Free accounts may host 2 Gradio Spaces on ZeroGPU instead, but
  that needs an account older than 30 days and gives 5 GPU-minutes a day,
  which one curious visitor exhausts.
- **Cloud Run.** Free tier and a clean URL, but the container files here
  have never been built, Docker is unavailable in the development
  environment, and the first build would have to be debugged in the cloud
  with billing enabled.

The measurement in [GENERATOR_COMPARISON.md](GENERATOR_COMPARISON.md) is
what made this easy to decide. Faithfulness was statistically
indistinguishable across local Llama 3.1 8B, gpt-oss-20b and gpt-oss-120b,
so moving generation off this machine buys no answer quality. It costs
privacy and money. With nothing to gain on quality, the cheap routes lose
nothing that matters.

### The public link

```bash
SHARE=1 CORPUS_DIR=data/demo-corpus python app.py
```

`SHARE` is read from the environment rather than a command line flag
because the Gradio interface is built at import time, before argument
parsing would run.

Two deliberate behaviours while it is set:

**Removal is refused, not merely hidden.** The Remove control is taken off
the page, and `delete_document` also returns early. A hidden Gradio
component is still a reachable endpoint for anyone holding the link, so
the layout alone would not protect a library that took hours to build.

**Uploading stays open.** Answering a visitor's own paper is the point of
the demo. `CORPUS_DIR` is the guard instead: point it at a throwaway
directory and the real library is never in reach.

**Not verified:** the tunnel has never been opened from this machine.
Gradio downloads a helper binary on first use, so this needs a dry run
before it is needed in front of someone.

