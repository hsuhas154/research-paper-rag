/*
 * Browser client for the Research Paper Q&A API.
 *
 * No build step and no framework, deliberately. The page is served from
 * the same origin as the API, so there is no CORS to configure, nothing
 * to compile, and nothing to keep in sync with a bundler. It also means
 * the container needs no Node toolchain.
 *
 * The answer arrives over Server-Sent Events. Retrieval finishes about a
 * second before generation starts, so the sources are rendered first and
 * the answer streams in behind them.
 */

const $ = (id) => document.getElementById(id);
const state = { papers: [], selected: new Set(), history: [], busy: false };

const escapeHtml = (text) =>
  String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function setStatus(message, bad = false) {
  const el = $("status");
  el.textContent = message;
  el.classList.toggle("bad", bad);
}

// ---------------------------------------------------------------- papers

async function loadLibrary() {
  try {
    const health = await (await fetch("health")).json();
    if (!health.ready) {
      setStatus("Models are still loading, retrying...");
      setTimeout(loadLibrary, 2000);
      return;
    }
    const library = await (await fetch("documents")).json();
    state.papers = library.documents;
    state.selected = new Set(state.papers.map((d) => d.doc_id));
    setStatus(`${library.n_documents} papers, ${library.n_chunks.toLocaleString()} passages`);
    renderPapers();
  } catch (error) {
    setStatus(`Cannot reach the API: ${error.message}`, true);
  }
}

function visiblePapers() {
  const needle = $("filter").value.trim().toLowerCase();
  // Matched against the full title, not the truncated label, because
  // several titles carry the word that identifies them past the cut.
  return needle
    ? state.papers.filter((d) => d.title.toLowerCase().includes(needle))
    : state.papers;
}

function renderPapers() {
  const visible = visiblePapers();
  $("papers").innerHTML = visible
    .map(
      (d) => `<label title="${escapeHtml(d.title)}">
        <input type="checkbox" value="${d.doc_id}"
          ${state.selected.has(d.doc_id) ? "checked" : ""}>
        <span>${escapeHtml(d.title.length > 80 ? d.title.slice(0, 77) + "..." : d.title)}</span>
      </label>`
    )
    .join("");

  $("papers").querySelectorAll("input").forEach((box) => {
    box.addEventListener("change", () => {
      box.checked ? state.selected.add(box.value) : state.selected.delete(box.value);
      updateCount();
    });
  });
  updateCount();
}

function updateCount() {
  const shown = visiblePapers().length;
  const total = state.papers.length;
  const scope = `${state.selected.size} of ${total} selected`;
  $("scopeCount").textContent = shown === total ? scope : `${scope} · ${shown} shown`;
}

// ---------------------------------------------------------------- asking

function renderSources(sources) {
  $("sourcesHeading").classList.toggle("hidden", sources.length === 0);
  $("sources").innerHTML = sources
    .map(
      (s) => `<details class="source" open>
        <summary><b>[Excerpt ${s.excerpt}]</b>
          <span class="cite">${escapeHtml(s.citation)} · score ${s.score.toFixed(3)}</span>
        </summary>
        <p>${escapeHtml(s.text)}</p>
      </details>`
    )
    .join("");
}

function renderHistory() {
  $("historyHeading").classList.toggle("hidden", state.history.length === 0);
  $("history").innerHTML = state.history
    .map(
      (t) => `<div class="turn">
        <div class="q">${escapeHtml(t.question)}</div>
        <div class="a">${escapeHtml(t.answer.slice(0, 280))}${t.answer.length > 280 ? "..." : ""}</div>
      </div>`
    )
    .join("");
}

/*
 * Reads a Server-Sent Events stream from a POST response.
 *
 * EventSource cannot be used here: it only issues GET requests, and the
 * question, scope and history belong in a body rather than a query
 * string. Parsing the frames by hand is a few lines and avoids inventing
 * a GET endpoint that would cap how long a conversation can get.
 */
async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let name = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) name = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (dataLines.length) onEvent(name, JSON.parse(dataLines.join("\n")));
    }
  }
}

async function ask() {
  if (state.busy) return;
  const question = $("question").value.trim();
  if (!question) return;

  if (state.selected.size === 0) {
    $("summary").textContent = "Select at least one paper to search.";
    $("summary").classList.add("error");
    return;
  }

  state.busy = true;
  $("ask").disabled = true;
  $("answer").textContent = "";
  $("answer").classList.add("waiting");
  $("sources").innerHTML = "";
  $("sourcesHeading").classList.add("hidden");
  $("rewritten").classList.add("hidden");
  $("summary").classList.remove("error");
  $("summary").textContent = "Searching...";

  // Sending every id when everything is selected is the same as sending
  // none, and omitting the filter lets the server skip it entirely.
  const everything = state.selected.size === state.papers.length;
  const body = {
    question,
    doc_ids: everything ? null : [...state.selected],
    mode: $("mode").value,
    top_k: Number($("topk").value) || 5,
    history: $("followup").checked ? state.history : null,
  };

  let answer = "";
  try {
    const response = await fetch("ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(detail.detail || `HTTP ${response.status}`);
    }

    await readEvents(response, (name, payload) => {
      if (name === "sources") {
        renderSources(payload.sources);
        $("summary").textContent = payload.retrieval_summary;
        if (payload.retrieval_question) {
          $("rewritten").textContent =
            `Searched as: "${payload.retrieval_question}"`;
          $("rewritten").classList.remove("hidden");
        }
      } else if (name === "token") {
        answer += payload.text;
        $("answer").textContent = answer;
      } else if (name === "done") {
        if (payload.error) throw new Error(payload.error);
        answer = payload.answer ?? answer;
        $("answer").textContent = answer;
      }
    });

    state.history.push({ question, answer });
    renderHistory();
    $("question").value = "";
  } catch (error) {
    $("summary").textContent = error.message;
    $("summary").classList.add("error");
  } finally {
    state.busy = false;
    $("ask").disabled = false;
    $("answer").classList.remove("waiting");
  }
}

// ---------------------------------------------------------------- upload

async function uploadPdf() {
  const file = $("pdf").files[0];
  if (!file) return;
  $("uploadNote").textContent = `Indexing ${file.name}...`;
  const form = new FormData();
  form.append("file", file);
  try {
    const response = await fetch("documents", { method: "POST", body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    $("uploadNote").textContent =
      `Indexed ${payload.title} (${payload.n_pages} pages, ${payload.n_chunks} passages)`;
    await loadLibrary();
  } catch (error) {
    $("uploadNote").textContent = error.message;
  }
}

// ---------------------------------------------------------------- wiring

$("ask").addEventListener("click", ask);
$("question").addEventListener("keydown", (event) => {
  // Enter sends, Shift+Enter makes a new line, which is what a chat box
  // is expected to do.
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ask();
  }
});
$("reset").addEventListener("click", () => {
  state.history = [];
  renderHistory();
  $("answer").textContent = "";
  $("sources").innerHTML = "";
  $("sourcesHeading").classList.add("hidden");
  $("summary").textContent = "Conversation cleared.";
});
$("filter").addEventListener("input", renderPapers);
$("selectAll").addEventListener("click", () => {
  visiblePapers().forEach((d) => state.selected.add(d.doc_id));
  renderPapers();
});
$("clearAll").addEventListener("click", () => {
  visiblePapers().forEach((d) => state.selected.delete(d.doc_id));
  renderPapers();
});
$("upload").addEventListener("click", uploadPdf);

loadLibrary();
