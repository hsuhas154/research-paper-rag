"""
app.py

Gradio UI for the Research Paper Q&A RAG system.

Two tabs:
  Library - upload papers, see what is indexed, remove papers. The
            library is persistent, so papers indexed in one session are
            still there the next time the app starts.
  Ask     - ask a question of any subset of the library, with the
            retrieval strategy selectable so the difference between
            dense, lexical, hybrid, and re-ranked retrieval is visible
            rather than buried in a config file.

The answer streams. Retrieval takes about a tenth of a second and
generation takes seconds, so the sources appear first and the answer
fills in behind them. The total wait is unchanged; what changes is that
the reader can start reading immediately and can tell a slow answer
apart from a hung process.

Scope controls are built for a library of fifty papers rather than five:
the paper list is filterable, with select-all and clear, because fifty
checkboxes with no search is not a control anyone can use.

Run with: python app.py
"""

import gradio as gr

from src.conversation import Conversation
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
        # Collapsible, with the citation always visible. Five passages of
        # academic prose is several screens of text; a reader checking one
        # citation should not have to scroll past the other four.
        blocks.append(
            f"<details open><summary><b>[Excerpt {i}]</b> "
            f"{source.chunk.citation(title)} "
            f"<i>score {source.score:.3f}</i></summary>\n\n"
            f"{source.chunk.text}\n\n</details>"
        )
    return "\n\n".join(blocks)


def filtered_choices(needle: str) -> list:
    """
    Document choices narrowed by a case-insensitive substring.

    Matched against the full title, not the shortened label shown in the
    list. Labels are truncated at 70 characters, and several papers in
    this corpus carry the word that identifies them past that point: one
    Venus paper leads with a list of molecular formulae and does not
    reach "Venus" until character 85, so filtering on the visible text
    alone would make it unfindable.
    """
    needle = (needle or "").strip().lower()
    if not needle:
        return document_choices()
    return [(d.short_title, d.doc_id) for d in pipeline.documents
            if needle in d.title.lower()]


def apply_filter(needle, selected):
    """
    Narrows the visible paper list without discarding selections that
    scroll out of view.

    Dropping hidden selections would mean typing a filter silently
    changes what gets searched, which is a surprising way to lose a
    carefully chosen scope.
    """
    choices = filtered_choices(needle)
    visible = {value for _, value in choices}
    kept = [v for v in (selected or []) if v in visible]
    return gr.update(choices=choices, value=kept)


def ask_question(question, selected_doc_ids, mode, top_k, follow_up, history):
    """
    Answers a question, streaming the result.

    Yields four times or more: once to acknowledge the question, once
    when retrieval finishes so the sources appear, then repeatedly as the
    answer arrives, then a final time with the conversation updated.

    `history` is Gradio session state, so each browser session keeps its
    own conversation rather than sharing one across users.
    """
    history = history or []

    def fail(message):
        return message, "", "", history

    if not question or not question.strip():
        yield fail("Please enter a question.")
        return
    if pipeline.corpus.is_empty():
        yield fail("The library is empty. Upload a paper on the Library tab first.")
        return
    if not selected_doc_ids:
        yield fail("Select at least one paper to search.")
        return

    # Searching every paper is the same as not filtering at all, and
    # skipping the filter lets FAISS search without an id selector.
    doc_ids = None if len(selected_doc_ids) == len(pipeline.documents) else selected_doc_ids

    conversation = None
    if follow_up and history:
        conversation = Conversation()
        for previous_question, previous_answer in history:
            conversation.add(previous_question, previous_answer)

    yield "", "", "*Searching...*", history

    try:
        retrieved, fragments = pipeline.stream_question(
            question, doc_ids=doc_ids, mode=mode, top_k=int(top_k),
            conversation=conversation,
        )
    except Exception as error:
        yield fail(f"\u274c {error}")
        return

    scope = "all papers" if doc_ids is None else f"{len(selected_doc_ids)} paper(s)"
    stats = f"*Searched {scope} \u00b7 {retrieved.retrieval_summary()}*"
    if retrieved.retrieval_question:
        stats += f"\n\n*Follow-up rewritten for search: \"{retrieved.retrieval_question}\"*"

    sources = format_sources(retrieved)
    # Sources are ready long before the answer is. Showing them now gives
    # the reader something to do during generation.
    yield "", sources, stats + "\n\n*Writing the answer...*", history

    text = ""
    try:
        for fragment in fragments:
            text += fragment
            yield text, sources, stats + "\n\n*Writing the answer...*", history
    except Exception as error:
        yield f"\u274c {error}", sources, stats, history
        return

    yield text, sources, stats, history + [(question, text)]


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

with gr.Blocks(title="Research Paper Q&A") as demo:
    # Per-session, so two browsers do not share one conversation.
    history_state = gr.State([])

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
                    with gr.Row():
                        ask_btn = gr.Button("Ask", variant="primary")
                        clear_btn = gr.Button("New conversation")

                    gr.Markdown(
                        "*The first question after startup takes longer while the local "
                        "LLM loads into GPU memory. It is then held resident for an "
                        "hour, so later questions start immediately.*"
                    )

                    answer_output = gr.Markdown(label="Answer")
                    retrieval_stats = gr.Markdown()

                with gr.Column(scale=1):
                    gr.Markdown("### Search which papers")
                    scope_filter = gr.Textbox(
                        label="Filter by title",
                        placeholder="venus, transformer, mosfet ...",
                        lines=1,
                    )
                    with gr.Row():
                        select_all_btn = gr.Button("Select all", size="sm")
                        clear_scope_btn = gr.Button("Clear", size="sm")
                    scope_count = gr.Markdown(
                        f"*{len(pipeline.documents)} of {len(pipeline.documents)} selected*"
                    )
                    scope_input = gr.CheckboxGroup(
                        label="",
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
                    follow_up_input = gr.Checkbox(
                        label="Follow-up question",
                        value=False,
                        info=("Use the conversation so far. A question like "
                              "\"and at 60 km?\" is rewritten into a standalone "
                              "one before searching."),
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
    # Indexing or removing a paper resets the filter box, because the
    # choices it was narrowing have changed underneath it.
    add_btn.click(fn=lambda: "", outputs=scope_filter)
    delete_btn.click(fn=lambda: "", outputs=scope_filter)

    add_btn.click(fn=add_document, inputs=pdf_input, outputs=library_outputs)
    delete_btn.click(fn=delete_document, inputs=delete_input, outputs=library_outputs)

    mode_input.change(
        fn=lambda mode: f"*{MODE_DESCRIPTIONS[mode]}*",
        inputs=mode_input,
        outputs=mode_help,
    )

    scope_filter.change(
        fn=apply_filter, inputs=[scope_filter, scope_input], outputs=scope_input
    )
    select_all_btn.click(
        fn=lambda needle: gr.update(
            value=[value for _, value in filtered_choices(needle)]
        ),
        inputs=scope_filter,
        outputs=scope_input,
    )
    clear_scope_btn.click(fn=lambda: gr.update(value=[]), outputs=scope_input)
    scope_input.change(
        fn=lambda selected: (
            f"*{len(selected or [])} of {len(pipeline.documents)} selected*"
        ),
        inputs=scope_input,
        outputs=scope_count,
    )

    ask_inputs = [question_input, scope_input, mode_input, top_k_input,
                  follow_up_input, history_state]
    ask_outputs = [answer_output, sources_output, retrieval_stats, history_state]
    ask_btn.click(fn=ask_question, inputs=ask_inputs, outputs=ask_outputs)
    question_input.submit(fn=ask_question, inputs=ask_inputs, outputs=ask_outputs)
    clear_btn.click(
        fn=lambda: ([], "", "", "*Conversation cleared.*"),
        outputs=[history_state, answer_output, sources_output, retrieval_stats],
    )


if __name__ == "__main__":
    demo.launch()
