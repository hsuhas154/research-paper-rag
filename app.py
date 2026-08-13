"""
app.py

Gradio UI for the Research Paper Q&A RAG system (Phase 1).

Flow: upload a PDF -> process/index it -> ask questions -> see the
answer alongside the exact source passages it was grounded in.

Run with: python app.py
"""

import gradio as gr

from src.rag_pipeline import RAGPipeline

# Loaded once at startup, not per-request. The embedding model load is
# the expensive part - reusing one instance across the whole app
# session keeps every subsequent interaction fast.
print("Starting up: loading embedding model...")
pipeline = RAGPipeline(top_k=5)
print("Ready.")


def process_pdf(pdf_file):
    """
    Called when the user uploads a PDF and clicks 'Process PDF'.
    Runs the full extract -> chunk -> embed -> index pipeline and
    reports back how many chunks were indexed.
    """
    if pdf_file is None:
        return "Please upload a PDF first."

    try:
        n_chunks = pipeline.load_document(pdf_file, document_name=pdf_file)
        return f"✅ Indexed {n_chunks} chunks. You can now ask questions below."
    except Exception as e:
        return f"❌ Error processing PDF: {e}"


def ask_question(question):
    """
    Called when the user submits a question. Returns the generated
    answer and a formatted view of the source chunks it was grounded
    in, so the person can verify the answer against the actual paper
    text rather than trusting it blindly.
    """
    if not question or not question.strip():
        return "Please enter a question.", ""

    if pipeline.store is None:
        return "Please upload and process a PDF first.", ""

    try:
        answer, retrieved = pipeline.answer_question(question)
    except Exception as e:
        return f"Error generating answer: {e}", ""

    sources_md = ""
    for i, r in enumerate(retrieved, start=1):
        sources_md += f"**[Excerpt {i}]** (similarity: {r.score:.3f})\n\n"
        sources_md += f"{r.chunk.text}\n\n---\n\n"

    return answer, sources_md


with gr.Blocks(title="Research Paper Q&A") as demo:
    gr.Markdown("# Research Paper Q&A System")
    gr.Markdown(
        "Upload a research paper (PDF), then ask questions about its content. "
        "Answers are grounded in the paper's actual text using retrieval-augmented "
        "generation - retrieved passages are shown below each answer so you can "
        "verify where the answer came from."
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            process_btn = gr.Button("Process PDF", variant="primary")
            status_output = gr.Markdown()

        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Ask a question about the paper",
                placeholder="e.g. What eddy diffusion coefficient was used in this model?",
                lines=2,
            )
            ask_btn = gr.Button("Ask", variant="primary")
            gr.Markdown(
                "*Note: the first question after starting the app or loading a new "
                "paper will take longer (~30-40s) while the local LLM loads into GPU "
                "memory. Subsequent questions are much faster.*"
            )
            answer_output = gr.Markdown(label="Answer")

    gr.Markdown("## Source Passages")
    sources_output = gr.Markdown()

    process_btn.click(fn=process_pdf, inputs=pdf_input, outputs=status_output)
    ask_btn.click(fn=ask_question, inputs=question_input, outputs=[answer_output, sources_output])
    question_input.submit(fn=ask_question, inputs=question_input, outputs=[answer_output, sources_output])


if __name__ == "__main__":
    demo.launch()