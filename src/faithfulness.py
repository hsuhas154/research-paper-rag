"""
faithfulness.py

Measures whether a generated answer is actually supported by the passages
it was given.

Retrieval metrics say whether the evidence reached the model. They say
nothing about what the model then did with it, and the two can come apart:
an answer can cite [Excerpt 2] for a claim that [Excerpt 2] does not make.
Phase 2 verified this by hand across 97 questions, which does not scale
and cannot run in CI.

Three measures, each cheap and each answering a different question:

  faithfulness  of the claims in the answer, what fraction are supported
                by the retrieved context. The RAGAS formulation: split the
                answer into claims, judge each against the context alone.
  citation      of the [Excerpt N] markers, what fraction point at an
  validity      excerpt that was actually supplied, and does every claim
                carry one.
  context       of the passages supplied, what fraction the answer
  utilisation   actually used. Low values mean top_k is larger than it
                needs to be.

The judge is the same local model that generates. That is a real
limitation, since a model is a lenient judge of its own output, and it is
recorded in the report rather than hidden. It is still far better than no
measurement, and the interface takes a model name so a stronger judge can
be substituted.

RAGAS itself is deliberately not a dependency: it pulls a large stack
including its own LLM clients, and this project's environment is a
pinned CUDA nightly that is not worth risking for three prompts.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import ollama

from src.llm import LLM_MODEL_NAME
from src.vector_store import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[Excerpt\s+(\d+)\]", re.IGNORECASE)

CLAIM_PROMPT = (
    "Break the following answer into its distinct factual claims.\n\n"
    "Rules:\n"
    "- one claim per line, no numbering or bullets\n"
    "- each claim must stand alone and be checkable\n"
    "- drop hedging, citations, and anything that is not a factual assertion\n"
    "- if the answer makes no factual claims, reply with NONE\n\n"
    "Answer:\n{answer}\n\nClaims:"
)

VERDICT_PROMPT = (
    "You are checking whether a claim is supported by the given context.\n\n"
    "Context:\n{context}\n\n"
    "Claim: {claim}\n\n"
    "Is the claim supported by the context? Consider only the context, not "
    "your own knowledge. Reply with exactly one word: SUPPORTED, "
    "UNSUPPORTED, or CONTRADICTED."
)


@dataclass
class FaithfulnessReport:
    """Per-answer groundedness measurement."""
    claims: List[str] = field(default_factory=list)
    verdicts: List[str] = field(default_factory=list)
    n_citations: int = 0
    n_valid_citations: int = 0
    n_excerpts_supplied: int = 0
    n_excerpts_cited: int = 0
    judge_model: str = LLM_MODEL_NAME

    @property
    def faithfulness(self) -> Optional[float]:
        """Fraction of claims supported by the context, or None if no claims."""
        if not self.verdicts:
            return None
        return sum(v == "SUPPORTED" for v in self.verdicts) / len(self.verdicts)

    @property
    def contradiction_rate(self) -> Optional[float]:
        if not self.verdicts:
            return None
        return sum(v == "CONTRADICTED" for v in self.verdicts) / len(self.verdicts)

    @property
    def citation_validity(self) -> Optional[float]:
        """Fraction of [Excerpt N] markers pointing at a supplied excerpt."""
        if not self.n_citations:
            return None
        return self.n_valid_citations / self.n_citations

    @property
    def context_utilisation(self) -> Optional[float]:
        """Fraction of supplied passages the answer actually cited."""
        if not self.n_excerpts_supplied:
            return None
        return self.n_excerpts_cited / self.n_excerpts_supplied

    def as_dict(self) -> Dict[str, object]:
        return {
            "faithfulness": self.faithfulness,
            "contradiction_rate": self.contradiction_rate,
            "citation_validity": self.citation_validity,
            "context_utilisation": self.context_utilisation,
            "n_claims": len(self.claims),
            "n_citations": self.n_citations,
            "judge_model": self.judge_model,
        }


def _ask(prompt: str, model: str, num_predict: int) -> str:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_ctx": 8192, "num_predict": num_predict},
    )
    return response["message"]["content"].strip()


def extract_claims(answer: str, model: str = LLM_MODEL_NAME) -> List[str]:
    """Splits an answer into standalone factual claims."""
    if not answer.strip():
        return []
    try:
        raw = _ask(CLAIM_PROMPT.format(answer=answer[:4000]), model, 400)
    except Exception:
        return []
    if raw.strip().upper().startswith("NONE"):
        return []

    claims = []
    for line in raw.split("\n"):
        line = re.sub(r"^\s*[-*•]?\s*\d*[.)]?\s*", "", line).strip()
        line = CITATION_PATTERN.sub("", line).strip()
        if len(line) > 15:
            claims.append(line)
    return claims


def judge_claim(claim: str, context: str, model: str = LLM_MODEL_NAME) -> str:
    """Returns SUPPORTED, UNSUPPORTED, or CONTRADICTED."""
    try:
        verdict = _ask(VERDICT_PROMPT.format(context=context[:9000], claim=claim), model, 10)
    except Exception:
        return "UNSUPPORTED"

    upper = verdict.upper()
    for label in ("CONTRADICTED", "UNSUPPORTED", "SUPPORTED"):
        if label in upper:
            return label
    return "UNSUPPORTED"


def check_citations(answer: str, n_supplied: int) -> Dict[str, int]:
    """
    Counts citation markers and how many point at a real excerpt.

    A marker outside the supplied range is a fabricated citation: the
    answer looks sourced and is not, which is more dangerous than an
    uncited claim because it defeats the reader's own check.
    """
    cited = [int(n) for n in CITATION_PATTERN.findall(answer)]
    valid = [n for n in cited if 1 <= n <= n_supplied]
    return {
        "n_citations": len(cited),
        "n_valid_citations": len(valid),
        "n_excerpts_supplied": n_supplied,
        "n_excerpts_cited": len(set(valid)),
    }


def evaluate_answer(
    answer: str,
    sources: Sequence[RetrievedChunk],
    model: str = LLM_MODEL_NAME,
    max_claims: int = 8,
) -> FaithfulnessReport:
    """
    Measures how well an answer is grounded in the passages it was given.

    Args:
        answer: the generated answer
        sources: the passages supplied to the generator
        model: the judge. Defaults to the generating model, which is
            lenient about its own output; pass a stronger one if available.
        max_claims: claims judged per answer, to bound cost. Answers with
            more claims are sampled from the front, where the substantive
            assertions usually are.
    """
    context = "\n\n".join(
        f"[Excerpt {i}]\n{s.chunk.text}" for i, s in enumerate(sources, start=1)
    )
    report = FaithfulnessReport(judge_model=model)
    report.__dict__.update(check_citations(answer, len(sources)))

    report.claims = extract_claims(answer, model)[:max_claims]
    report.verdicts = [judge_claim(c, context, model) for c in report.claims]
    return report


def aggregate(reports: Sequence[FaithfulnessReport]) -> Dict[str, object]:
    """Corpus-level means, skipping answers a measure does not apply to."""
    def mean(values):
        present = [v for v in values if v is not None]
        return round(sum(present) / len(present), 4) if present else None

    return {
        "n_answers": len(reports),
        "faithfulness": mean([r.faithfulness for r in reports]),
        "contradiction_rate": mean([r.contradiction_rate for r in reports]),
        "citation_validity": mean([r.citation_validity for r in reports]),
        "context_utilisation": mean([r.context_utilisation for r in reports]),
        "total_claims": sum(len(r.claims) for r in reports),
        "judge_model": reports[0].judge_model if reports else None,
    }
