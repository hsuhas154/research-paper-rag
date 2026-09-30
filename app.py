"""
app.py

Gradio UI for the Research Paper Q&A RAG system (Phase 2).

Two tabs:
  Library - upload papers, see what is indexed, remove papers. The
            library is persistent, so papers indexed in one session are
            still there the next time the app starts.
  Ask     - ask a question of any subset of the library, with the
            retrieval strategy selectable so the difference between
            dense, lexical, hybrid, and re-ranked retrieval is visible
            rather than buried in a config file.

Run with: python app.py
"""

import gradio as gr

from src.rag_pipeline import RAGPipeline
from src.retriever import (
    MODE_BM25,
    MODE_DENSE,
    MODE_HYBRID,
    MODE_HYBRID_RERANK,
    RETRIEVAL_MODES,
)

# Loaded once at startup, not per-request. Loading the embedding model
# and restoring the corpus is the expensive part; reusing one instance
# across the whole session keeps every subsequent interaction fast.
print("Starting up: loading models and corpus...")
pipeline = RAGPipeline(top_k=5)
print(f"Ready. Corpus holds {len(pipeline.documents)} document(s), "
      f"{len(pipeline.corpus.store)} chunks.")


MODE_DESCRIPTIONS = {
    MODE_DENSE: "Semantic similarity only (Phase 1 behaviour).",
    MODE_BM25: "Exact keyword matching only. Strong on names and numbers, blind to paraphrase.",
    MODE_HYBRID: "Dense and lexical results fused by Reciprocal Rank Fusion.",
    MODE_HYBRID_RERANK: "Hybrid retrieval, then cross-encoder re-ranking. Most accurate, slowest.",
}


# ----------------------------------------------------------------------
# Library tab
# ----------------------------------------------------------------------

def library_table() -> list:
    """Rows for the library table, newest paper last."""
    return [
        [d.doc_id, d.title, d.n_pages, d.n_chunks, d.added_at.replace("T", " ")[:16]]
        for d in pipeline.documents
    ]


def document_choices() -> list:
    """(label, value) pairs for the document pickers."""
    return [(d.short_title, d.doc_id) for d in pipeline.documents]


def library_status() -> str:
    if pipeline.corpus.is_empty():
        return "Library is empty. Upload a paper to get started."
    return (
        f"**{len(pipeline.documents)} paper(s) indexed**, "
        f"{len(pipeline.corpus.store)} chunks searchable."
    )


def refresh_library_views(message: str):
    """
    Every library change has to update the table, both document pickers,
    and the status line together - returning them as one tuple keeps
    them from drifting out of sync.
    """
    choices = document_choices()
    return (
        message,
        library_table(),
        library_status(),
        # Selecting everything by default means a question searches the
        # whole library unless the user deliberately narrows it.
        gr.update(choices=choices, value=[c[1] for c in choices]),
        gr.update(choices=choices, value=None),
    )


def add_document(pdf_file):
    """Indexes an uploaded PDF into the persistent library."""
    if pdf_file is None:
        return refresh_library_views("Please choose a PDF first.")

    try:
        record = pipeline.add_document(pdf_file)
    except Exception as error:
        return refresh_library_views(f"❌ Could not index the PDF: {error}")

    return refresh_library_views(
        f"✅ Indexed **{record.title}**: {record.n_pages} pages, {record.n_chunks} chunks."
    )


def delete_document(doc_id):
    """Removes a paper and everything derived from it."""
    if not doc_id:
        return refresh_library_views("Select a paper to remove.")

    title = pipeline.corpus.title_for(doc_id)
    removed = pipeline.remove_document(doc_id)
    message = f"🗑️ Removed **{title}**." if removed else "That paper is no longer in the library."
    return refresh_library_views(message)


# ----------------------------------------------------------------------
# Ask tab
# ----------------------------------------------------------------------

def format_sources(answer) -> str:
    """
    Renders the retrieved passages with the same excerpt numbers the
    model was told to cite, so a reader can follow a [Excerpt 3] in the
    answer straight to the text it came from.
    """
    blocks = []
    for i, source in enumerate(answer.sources, start=1):
        title = answer.document_titles.get(source.chunk.doc_id)
        blocks.append(
            f"**[Excerpt {i}]** · {source.chunk.citation(title)} · score {source.score:.3f}\n\n"
            f"{source.chunk.text}\n\n---"
        )
    return "\n\n".join(blocks)


def ask_question(question, selected_doc_ids, mode, top_k):
    """
    Answers a question and returns the answer, the source passages, and
    a one-line description of how retrieval reached them.
    """
    if not question or not question.strip():
        return "Please enter a question.", "", ""

    if pipeline.corpus.is_empty():
        return "The library is empty. Upload a paper on the Library tab first.", "", ""

    if not selected_doc_ids:
        return "Select at least one paper to search.", "", ""

    # Searching every paper is the same as not filtering at all, and
    # skipping the filter lets FAISS search without an id selector.
    doc_ids = None if len(selected_doc_ids) == len(pipeline.documents) else selected_doc_ids

    try:
        answer = pipeline.answer_question(
            question, doc_ids=doc_ids, mode=mode, top_k=int(top_k)
        )
    except Exception as error:
        return f"❌ {error}", "", ""

    scope = "all papers" if doc_ids is None else f"{len(selected_doc_ids)} paper(s)"
    stats = f"*Searched {scope} · {answer.retrieval_summary()}*"

    return answer.text, format_sources(answer), stats


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

with gr.Blocks(title="Research Paper Q&A") as demo:
    gr.Markdown("# Research Paper Q&A System")
    gr.Markdown(
        "Ask natural-language questions across a library of research papers. "
        "Answers are grounded in the papers' actual text using retrieval-augmented "
        "generation, and every retrieved passage is shown with its page and section "
        "so you can check the answer against the source."
    )

    with gr.Tabs():
        with gr.Tab("Ask"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Question",
                        placeholder="e.g. What eddy diffusion coefficient was used in the cloud layer?",
                        lines=3,
                    )
                    ask_btn = gr.Button("Ask", variant="primary")

                    gr.Markdown(
                        "*The first question after startup takes longer (~30-40s) while "
                        "the local LLM loads into GPU memory. Later questions are faster.*"
                    )

                    answer_output = gr.Markdown(label="Answer")
                    retrieval_stats = gr.Markdown()

                with gr.Column(scale=1):
                    scope_input = gr.CheckboxGroup(
                        label="Search which papers",
                        choices=document_choices(),
                        value=[d.doc_id for d in pipeline.documents],
                    )
                    mode_input = gr.Radio(
                        label="Retrieval strategy",
                        choices=RETRIEVAL_MODES,
                        value=MODE_HYBRID_RERANK,
                    )
                    mode_help = gr.Markdown(f"*{MODE_DESCRIPTIONS[MODE_HYBRID_RERANK]}*")
                    top_k_input = gr.Slider(
                        label="Passages to retrieve",
                        minimum=1, maximum=10, value=5, step=1,
                    )

            gr.Markdown("## Source passages")
            sources_output = gr.Markdown()

        with gr.Tab("Library"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Add a paper")
                    pdf_input = gr.File(label="PDF", file_types=[".pdf"], type="filepath")
                    add_btn = gr.Button("Index paper", variant="primary")

                with gr.Column():
                    gr.Markdown("### Remove a paper")
                    delete_input = gr.Dropdown(
                        label="Paper", choices=document_choices(), value=None
                    )
                    delete_btn = gr.Button("Remove", variant="stop")

            library_message = gr.Markdown()
            library_summary = gr.Markdown(library_status())
            library_view = gr.Dataframe(
                headers=["ID", "Title", "Pages", "Chunks", "Added (UTC)"],
                value=library_table(),
                interactive=False,
                wrap=True,
            )

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    library_outputs = [
        library_message, library_view, library_summary, scope_input, delete_input
    ]

    add_btn.click(fn=add_document, inputs=pdf_input, outputs=library_outputs)
    delete_btn.click(fn=delete_document, inputs=delete_input, outputs=library_outputs)

    mode_input.change(
        fn=lambda mode: f"*{MODE_DESCRIPTIONS[mode]}*",
        inputs=mode_input,
        outputs=mode_help,
    )

    ask_inputs = [question_input, scope_input, mode_input, top_k_input]
    ask_outputs = [answer_output, sources_output, retrieval_stats]
    ask_btn.click(fn=ask_question, inputs=ask_inputs, outputs=ask_outputs)
    question_input.submit(fn=ask_question, inputs=ask_inputs, outputs=ask_outputs)


if __name__ == "__main__":
    demo.launch()
