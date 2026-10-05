"""
train_reranker.py

Fine-tunes a cross-encoder re-ranker on in-domain data built by
scripts.build_rerank_dataset.

Why fine-tune at all: the stock re-ranker is
cross-encoder/ms-marco-MiniLM-L-6-v2, trained on MS MARCO, which is web
search queries against web passages. Research paper prose is a different
distribution in every way that matters - terminology density, sentence
length, the fact that the answer term often never appears in the question
("what does Adam stand for?" against "adaptive moment estimation"). That
mismatch produced the only outright failure in the Phase 2 evaluation and
several in Table 2.

Training is binary classification over (question, passage) pairs: the
generated question against its own chunk is a positive, and against the
passages the deployed retrievers rank highest for it is a negative. That
is the same objective the stock model was trained with, so this is domain
adaptation rather than a change of task.

Usage:
    python -m scripts.train_reranker
    python -m scripts.train_reranker --epochs 2 --base cross-encoder/ms-marco-MiniLM-L-6-v2
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_OUT = Path("models/reranker-domain")


def load_pairs(path: Path, seed: int = 0):
    """Flattens the mined examples into labelled (question, passage) pairs."""
    positives, negatives = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        positives.append((row["question"], row["positive"], 1.0))
        for negative in row["negatives"]:
            negatives.append((row["question"], negative, 0.0))

    random.Random(seed).shuffle(negatives)
    return positives, negatives


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("eval/rerank_train.jsonl"))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--negatives-per-positive", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.data.exists():
        print(f"No training data at {args.data}. Run scripts.build_rerank_dataset first.")
        return 1

    import torch
    from datasets import Dataset
    from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    positives, negatives = load_pairs(args.data, args.seed)
    # Cap the negative ratio. Mined negatives outnumber positives several
    # to one, and left unbalanced the model learns to say "not relevant"
    # to everything, which scores well on the loss and badly on retrieval.
    keep = min(len(negatives), len(positives) * args.negatives_per_positive)
    pairs = positives + negatives[:keep]
    random.Random(args.seed).shuffle(pairs)

    print(f"base model     : {args.base}")
    print(f"training pairs : {len(pairs)} ({len(positives)} positive, {keep} negative)")

    dataset = Dataset.from_dict({
        "query": [p[0] for p in pairs],
        "passage": [p[1] for p in pairs],
        "label": [p[2] for p in pairs],
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossEncoder(args.base, num_labels=1, device=device)
    loss = BinaryCrossEntropyLoss(model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    training_args = CrossEncoderTrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = CrossEncoderTrainer(
        model=model, args=training_args, train_dataset=dataset, loss=loss
    )
    trainer.train()

    model.save_pretrained(str(args.out))
    print(f"\nsaved fine-tuned re-ranker to {args.out}")
    print("Evaluate it with:")
    print(f"  python -m scripts.compare_rerankers --models {args.base} {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
