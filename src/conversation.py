"""
conversation.py

Multi-turn conversation support: carrying history into generation, and
rewriting follow-up questions so retrieval can act on them.

The hard part is not the history, it is the retrieval. "And at 60 km?" is
a perfectly clear follow-up to a human and a useless query to a retriever:
it has no content words to match lexically and embeds to nothing in
particular. Passing history to the generator does not help either, because
by then retrieval has already returned the wrong passages.

So a follow-up is rewritten into a standalone question before it reaches
the retriever, using the conversation so far. The original wording is kept
for display and for the generator, since the rewrite is a retrieval aid
rather than what the user actually asked.

Rewriting is skipped when the question already stands on its own, which is
most of the time. A rewrite costs an LLM call and can only do damage to a
question that was already complete.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import ollama

from src.llm import LLM_MODEL_NAME

# How many previous turns to consider. Two is enough for the pronouns and
# ellipsis that follow-ups actually use, and keeps the rewrite prompt
# small enough to stay fast.
HISTORY_TURNS_FOR_REWRITE = 2

# How many turns to pass to the generator as conversational context.
HISTORY_TURNS_FOR_GENERATION = 3

REWRITE_PROMPT = (
    "Rewrite the follow-up question as a standalone question that can be "
    "understood without the conversation.\n\n"
    "Rules:\n"
    "- resolve pronouns and references using the conversation\n"
    "- keep the user's intent exactly; do not answer it, expand it, or add detail\n"
    "- if it already stands alone, repeat it unchanged\n"
    "- reply with the question only, one line, no preamble\n\n"
    "{history}\n"
    "Follow-up: {question}\n\n"
    "Standalone question:"
)

# A question with no pronoun, no leading conjunction, and a reasonable
# number of content words almost certainly stands alone already.
_DEPENDENT_OPENER = re.compile(
    r"^\s*(and|but|so|then|what about|how about|why|and what|ok|okay|also)\b",
    re.IGNORECASE,
)
_REFERRING_EXPRESSION = re.compile(
    r"\b(it|its|it's|they|them|their|that|those|this|these|he|she|his|her|"
    r"the same|the former|the latter|there|then)\b",
    re.IGNORECASE,
)


@dataclass
class Turn:
    """One exchange, kept for both rewriting and generation."""
    question: str
    answer: str
    # The standalone form actually sent to the retriever, when it differed.
    retrieval_question: Optional[str] = None


@dataclass
class Conversation:
    """
    An ordered list of turns, with the history formatting both the
    rewriter and the generator need.
    """
    turns: List[Turn] = field(default_factory=list)

    def add(self, question: str, answer: str, retrieval_question: Optional[str] = None) -> None:
        self.turns.append(Turn(question, answer, retrieval_question))

    def clear(self) -> None:
        self.turns.clear()

    def is_empty(self) -> bool:
        return not self.turns

    def as_messages(self, limit: int = HISTORY_TURNS_FOR_GENERATION) -> List[Dict[str, str]]:
        """
        Recent turns as chat messages for the generator.

        Answers are truncated: the generator needs to know what was
        discussed, not to re-read entire previous answers, and a long
        history crowds the retrieved passages out of the context window.
        """
        messages: List[Dict[str, str]] = []
        for turn in self.turns[-limit:]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.answer[:600]})
        return messages

    def as_transcript(self, limit: int = HISTORY_TURNS_FOR_REWRITE) -> str:
        """Recent turns as plain text, for the rewrite prompt."""
        lines = []
        for turn in self.turns[-limit:]:
            lines.append(f"Q: {turn.question}")
            lines.append(f"A: {turn.answer[:300]}")
        return "\n".join(lines)


def looks_self_contained(question: str) -> bool:
    """
    Whether a question can go to the retriever as written.

    Deliberately biased towards saying yes. A needless rewrite spends an
    LLM call and risks distorting a question that was already fine; a
    missed rewrite leaves behaviour exactly as it was before this module
    existed.
    """
    stripped = question.strip()
    if _DEPENDENT_OPENER.match(stripped):
        return False
    if _REFERRING_EXPRESSION.search(stripped):
        return False
    # Very short questions rarely carry enough on their own.
    return len(stripped.split()) >= 5


def rewrite_follow_up(
    question: str,
    conversation: Conversation,
    model: str = LLM_MODEL_NAME,
) -> str:
    """
    Returns a standalone form of `question`, or the question unchanged.

    Falls back to the original on any failure, including an unreachable
    LLM: a degraded follow-up is better than no answer at all.
    """
    if conversation.is_empty() or looks_self_contained(question):
        return question

    prompt = REWRITE_PROMPT.format(
        history=conversation.as_transcript(), question=question.strip()
    )
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 2048, "num_predict": 80},
        )
    except Exception:
        return question

    rewritten = response["message"]["content"].strip().split("\n")[0]
    rewritten = re.sub(r"^(standalone question|question)\s*[:\-]\s*", "",
                       rewritten, flags=re.IGNORECASE).strip().strip('"')

    # Guard against the model answering instead of rewriting, or padding a
    # short question into an essay.
    if not rewritten or len(rewritten) > 300 or "?" not in rewritten:
        return question
    return rewritten
