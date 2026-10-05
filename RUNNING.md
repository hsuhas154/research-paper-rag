# Running the Project

Everything here runs locally on your machine. Nothing is sent to an external
service, and no API key is needed.

---

## 1. One-time setup

You only do this section once.

### 1.1 The Python environment

This project runs in the **`ultimate_dl`** conda environment, not `base`.
`base` does not have the dependencies, and the `python` on your default PATH
is `base`.

```bash
conda activate ultimate_dl
```

Check it worked - you should see `(ultimate_dl)` at the start of your prompt,
and this should print a torch version and `True`:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Why this environment matters:** `ultimate_dl` holds a CUDA 12.8 nightly
> build of PyTorch, needed for your RTX 5070 (Blackwell) GPU. Installing
> packages that pull their own torch can break it. If you ever need a new
> package here, install it with `--no-deps` or check first that it will not
> touch torch.

### 1.2 Install Ollama and pull the model

Ollama runs the language model locally.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

```bash
ollama pull llama3.1:8b
```

That download is about 4.9 GB and only happens once.

---

## 2. Starting the project

There are two things to start, in this order: **Ollama**, then **the app**.

### Step 1 - Start Ollama

Ollama must be running before you ask any questions, because it serves the
language model that writes the answers.

Open a terminal and run:

```bash
ollama serve
```

Leave that terminal open. It will keep printing log lines - that is normal.

**Or**, to run it in the background and get your terminal back:

```bash
nohup ollama serve > /tmp/ollama.log 2>&1 &
```

**Check it is running** (this should print a JSON list containing
`llama3.1:8b`):

```bash
curl -s http://localhost:11434/api/tags
```

If it prints nothing, Ollama is not running yet.

> **Note:** on some systems Ollama is already running as a background service
> after install. If `curl` above already returns JSON, skip this step - you do
> not need to start it again. Running `ollama serve` twice gives you an
> `address already in use` error, which is harmless.

### Step 2 - Start the app

In a **second** terminal:

```bash
conda activate ultimate_dl
```

```bash
cd ~/research-paper-rag && python app.py
```

Wait for it to print something like:

```
Starting up: loading models and corpus...
Ready. Corpus holds 16 document(s), 802 chunks.
* Running on local URL:  http://127.0.0.1:7860
```

Open **http://127.0.0.1:7860** in your browser.

> The first startup after a fresh install also downloads the embedding model
> (~90 MB) and the re-ranking model (~90 MB). Later startups are fast.

---

## 3. Using the app

The interface has two tabs.

### Library tab

- **Add a paper:** choose a PDF, click **Index paper**. A 10-page paper takes
  a second or two. The table below shows everything currently indexed.
- **Remove a paper:** pick it from the dropdown, click **Remove**.

The library is **persistent** - papers stay indexed after you close and
reopen the app. You do not need to re-upload them.

### Ask tab

- Type a question and press **Ask** (or hit Enter).
- **Search which papers** - tick the papers to search. All are ticked by
  default, which searches the whole library.
- **Retrieval strategy** - leave this on `hybrid+rerank` for normal use. The
  other options exist so you can see the difference:
  - `dense` - meaning-based search only
  - `bm25` - exact keyword search only
  - `hybrid` - both, fused together
  - `hybrid+rerank` - both, then re-scored by a more accurate model *(best)*
- **Passages to retrieve** - how much source material to give the model.
  5 is a good default; raise it for broad questions.

Below each answer you get the **source passages** it was built from, each
labelled with its paper, page, and section, so you can check any claim.

> **The first question is slow** (~30-40 seconds) while the language model
> loads into GPU memory. Every question after that is much faster. This is
> normal and not a bug.

---

## 4. Stopping the project

### Stop the app

In the terminal running `python app.py`, press:

```
Ctrl + C
```

### Stop Ollama

If you started it with `ollama serve` in a terminal, press `Ctrl + C` in that
terminal.

If you started it in the background with `nohup`, stop it with:

```bash
pkill -f "ollama serve"
```

**Check it actually stopped** - this should print nothing:

```bash
pgrep -af "ollama serve"
```

> Stopping Ollama frees your GPU memory. You can leave it running if you plan
> to come back soon; it idles cheaply and unloads the model from VRAM by
> itself after a few minutes.

### Do I lose my indexed papers?

No. The library lives on disk in `data/corpus/` and survives restarts,
reboots, and shutdowns.

---

## 5. Command line usage

You can do everything without the web interface. Always
`conda activate ultimate_dl` first.

### Manage the library

```bash
python -m src.corpus list
```

```bash
python -m src.corpus add "data/uploads/Attention is All you Need.pdf"
```

```bash
python -m src.corpus remove <doc_id>
```

(`<doc_id>` is the 12-character ID shown by `list`.)

### Ask a question

```bash
python -m src.rag_pipeline "What BLEU score did the Transformer achieve?"
```

Add a retrieval mode as a second argument to compare strategies:

```bash
python -m src.rag_pipeline "What BLEU score did the Transformer achieve?" dense
```

### Run the tests

```bash
pytest tests/ -q
```

### Run the retrieval benchmark

```bash
python -m scripts.compare_retrieval --k 5 --verbose
```

### Check a test set is answerable

```bash
python -m scripts.validate_queries eval/corpus_queries.json
```

### Inspect one stage in isolation

Each module runs on its own, which is useful for debugging a specific paper:

```bash
python -m src.pdf_processor "data/uploads/GANs.pdf"
```

```bash
python -m src.chunker "data/uploads/GANs.pdf"
```

```bash
python -m src.bm25 "data/uploads/GANs.pdf"
```

---

## 6. Adding papers in bulk

To index a folder of PDFs at once, put them in `data/uploads/` and run:

```bash
for f in data/uploads/*.pdf; do python -m src.corpus add "$f"; done
```

This is safe to interrupt; papers already added stay added. Adding the same
file twice creates a second copy in the library, so remove the duplicate with
`python -m src.corpus remove <doc_id>` if you do it by accident.

---

## 7. Starting the library over

To wipe every indexed paper and start clean:

```bash
rm -rf data/corpus
```

Your original PDFs in `data/uploads/` are **not** touched. Re-index them with
the bulk command in section 6.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Could not reach the local LLM ... via Ollama` | Ollama is not running | Run `ollama serve` (section 2, step 1) |
| `ModuleNotFoundError: No module named 'torch'` (or faiss, gradio) | Wrong conda environment | `conda activate ultimate_dl` |
| `address already in use` from `ollama serve` | Ollama is already running | Nothing to do - it is fine |
| Port 7860 already in use | An older app instance is still running | `pkill -f "python app.py"`, then start again |
| First question hangs for ~40 seconds | Model loading into VRAM | Normal; only the first one |
| `No text could be extracted ... scanned PDF` | The PDF is page images with no text layer | Not supported; use a text-based PDF |
| Answers cite the wrong paper | Too many papers in scope | Untick irrelevant papers under **Search which papers** |
| App starts but library is empty | `data/corpus/` was deleted | Re-index (section 6) |

### Checking what is running

```bash
pgrep -af "ollama serve"; pgrep -af "python app.py"
```

### Freeing the GPU

If something else needs your 8 GB of VRAM:

```bash
pkill -f "ollama serve"
```

```bash
nvidia-smi
```

---

## 9. Running the HTTP API

The same pipeline is also available as an HTTP service, so something other
than the Gradio UI can use it.

Start Ollama first (section 2, step 1), then:

```bash
cd ~/research-paper-rag && python -m src.api
```

Or with uvicorn directly, which is what you want if you need a different port
or auto-reload while developing:

```bash
cd ~/research-paper-rag && uvicorn src.api:app --port 8000
```

Open **http://127.0.0.1:8000/docs** for interactive documentation where you
can try every endpoint from the browser.

### Endpoints

| Method | Path | Does |
|---|---|---|
| GET | `/health` | Whether models are loaded, and library size |
| GET | `/documents` | List indexed papers |
| POST | `/documents` | Upload and index a PDF |
| DELETE | `/documents/{doc_id}` | Remove a paper |
| POST | `/ask` | Ask a question |

### Examples

```bash
curl -s http://127.0.0.1:8000/health
```

```bash
curl -s -X POST http://127.0.0.1:8000/ask -H 'Content-Type: application/json' -d '{"question":"How many attention heads does the base model use?"}'
```

```bash
curl -s -X POST http://127.0.0.1:8000/documents -F 'file=@data/uploads/GANs.pdf'
```

```bash
curl -s -X DELETE http://127.0.0.1:8000/documents/<doc_id>
```

### Stopping it

Press `Ctrl + C` in its terminal, or if you backgrounded it:

```bash
pkill -f "src.api:app"
```

### What the status codes mean

| Code | Meaning |
|---|---|
| 409 | The library is empty; add a PDF first |
| 415 | The uploaded file is not a PDF |
| 422 | Bad request: unknown retrieval mode, out-of-range `top_k`, or a PDF with no text layer |
| 404 | Unknown document id |
| 502 | Ollama is unreachable; start it and retry |
| 503 | The server is still loading models |

---

## 10. Re-ranker experiments

The re-ranker was studied rather than assumed. These are the commands that
produced the [re-ranker study](EVALUATION.md), if you want to repeat it or
run it against a different corpus.

### Compare re-ranking models

Every model re-ranks an identical candidate pool, so what you see is the
model and nothing else:

```bash
python -m scripts.compare_rerankers
```

Restrict to papers that were held out of training, which is the honest test
of whether a fine-tuned model generalises:

```bash
python -m scripts.compare_rerankers --split heldout
```

### Build in-domain training data

The local LLM writes a question for each chunk, and hard negatives are mined
with the project's own retrievers. Needs Ollama running. Takes about 7
minutes for this corpus:

```bash
python -m scripts.build_rerank_dataset
```

The output is gitignored on purpose: it contains thousands of verbatim
passages from copyrighted papers, which is also why the PDFs are not
committed.

### Fine-tune a re-ranker

```bash
python -m scripts.train_reranker --epochs 1 --lr 5e-6
```

Training takes well under a minute on an 8 GB GPU.

### Use a different re-ranker

A fine-tuned model is **not** picked up automatically, because on this
corpus it measured worse than the stock one. Opt in explicitly:

```bash
RERANKER_MODEL=models/reranker-domain python app.py
```

The same variable accepts any Hugging Face cross-encoder:

```bash
RERANKER_MODEL=BAAI/bge-reranker-base python app.py
```
