// Everything talks to the FastAPI backend on the same origin.

let threadId = localStorage.getItem("rag_thread_id");
if (!threadId) {
  threadId = crypto.randomUUID();
  localStorage.setItem("rag_thread_id", threadId);
}

const messagesEl = document.getElementById("messages");
const emptyState = document.getElementById("emptyState");
const composerForm = document.getElementById("composerForm");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const statusDot = document.getElementById("statusDot");
const webSearchIndicator = document.getElementById("webSearchIndicator");
const sourcesList = document.getElementById("sourcesList");
const sourceCount = document.getElementById("sourceCount");
const ingestStatus = document.getElementById("ingestStatus");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const urlForm = document.getElementById("urlForm");
const urlInput = document.getElementById("urlInput");
const crawlToggle = document.getElementById("crawlToggle");
const newChatBtn = document.getElementById("newChatBtn");

let activeEventSource = null;

init();

function init() {
  checkHealth();
  refreshSources();

  uploadBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileUpload);
  urlForm.addEventListener("submit", handleUrlIngest);

  composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = messageInput.value.trim();
    if (text) sendMessage(text);
  });

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerForm.requestSubmit();
    }
  });

  messageInput.addEventListener("input", () => {
    messageInput.style.height = "auto";
    messageInput.style.height = Math.min(messageInput.scrollHeight, 160) + "px";
  });

  newChatBtn.addEventListener("click", startNewChat);
}

// ---------------- Status ----------------

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    statusDot.classList.remove("status-ok", "status-error");
    statusDot.classList.add(data.status === "ok" ? "status-ok" : "status-error");
    statusDot.title = `Groq model: ${data.groq_model}`;

    webSearchIndicator.classList.toggle("enabled", !!data.web_search_enabled);
    webSearchIndicator.querySelector(".label").textContent = data.web_search_enabled
      ? "Web search: on"
      : "Web search: off (no TAVILY_API_KEY)";
  } catch (err) {
    statusDot.classList.add("status-error");
    statusDot.title = "Backend unreachable";
  }
}

async function refreshSources() {
  try {
    const res = await fetch("/api/sources");
    const sources = await res.json();
    sourceCount.textContent = `(${sources.length})`;
    sourcesList.innerHTML = "";

    if (sources.length === 0) {
      const li = document.createElement("li");
      li.className = "sources-empty";
      li.textContent = "Nothing ingested yet.";
      sourcesList.appendChild(li);
      return;
    }

    for (const s of sources) {
      const li = document.createElement("li");
      const name = document.createElement("span");
      name.className = "source-name";
      name.textContent = s.source;
      name.title = s.source;
      const count = document.createElement("span");
      count.className = "source-count";
      count.textContent = s.chunk_count;
      li.append(name, count);
      sourcesList.appendChild(li);
    }
  } catch (err) {
    // Sidebar list is a nice-to-have; a failed refresh isn't fatal.
  }
}

// ---------------- Ingestion ----------------

async function handleFileUpload() {
  if (!fileInput.files.length) return;
  const formData = new FormData();
  for (const file of fileInput.files) formData.append("files", file);

  setIngestStatus(`Ingesting ${fileInput.files.length} file(s)…`, "pending");
  try {
    const res = await fetch("/api/ingest/documents", { method: "POST", body: formData });
    const data = await res.json();
    if (res.ok && data.status !== "error") {
      setIngestStatus(`Added ${data.chunks_added} chunk(s) from ${fileInput.files.length} file(s).`, "ok");
    } else {
      setIngestStatus(data.details || "Ingestion failed.", "error");
    }
  } catch (err) {
    setIngestStatus("Could not reach the backend.", "error");
  }
  fileInput.value = "";
  refreshSources();
}

async function handleUrlIngest(e) {
  e.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;

  setIngestStatus(`Scraping ${url}…`, "pending");
  try {
    const res = await fetch("/api/ingest/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, crawl: crawlToggle.checked, max_pages: 5 }),
    });
    const data = await res.json();
    if (res.ok) {
      setIngestStatus(`Added ${data.chunks_added} chunk(s). ${data.details || ""}`.trim(), "ok");
      urlInput.value = "";
    } else {
      setIngestStatus(data.detail || "Could not read that page.", "error");
    }
  } catch (err) {
    setIngestStatus("Could not reach the backend.", "error");
  }
  refreshSources();
}

function setIngestStatus(text, kind) {
  ingestStatus.textContent = text;
  ingestStatus.className = `ingest-status ${kind}`;
}

// ---------------- Chat ----------------

function startNewChat() {
  if (activeEventSource) {
    activeEventSource.close();
    activeEventSource = null;
  }
  threadId = crypto.randomUUID();
  localStorage.setItem("rag_thread_id", threadId);
  messagesEl.innerHTML = "";
  messagesEl.appendChild(emptyState);
  emptyState.style.display = "block";
  sendBtn.disabled = false;
  messageInput.focus();
}

function sendMessage(text) {
  emptyState.style.display = "none";
  addMessage("user", text);
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendBtn.disabled = true;

  const assistantWrapper = addMessage("assistant", "");
  const contentEl = assistantWrapper.querySelector(".bubble-content");
  contentEl.classList.add("is-streaming");

  const url = `/api/chat/stream?message=${encodeURIComponent(text)}&thread_id=${encodeURIComponent(threadId)}`;
  const es = new EventSource(url);
  activeEventSource = es;
  let fullText = "";

  es.addEventListener("token", (e) => {
    const data = JSON.parse(e.data);
    fullText += data.content;
    contentEl.textContent = fullText;
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });

  es.addEventListener("sources", (e) => {
    const data = JSON.parse(e.data);
    renderSources(assistantWrapper, data.sources || []);
  });

  es.addEventListener("done", () => {
    finishStream(es, contentEl);
    refreshSources();
  });

  // Custom event for errors raised inside the graph (distinct from the
  // browser's built-in connection-level "error" event handled below).
  es.addEventListener("stream_error", (e) => {
    let msg = "Something went wrong while generating a response.";
    try {
      msg = JSON.parse(e.data).message || msg;
    } catch (_) {
      /* keep default message */
    }
    if (!fullText) contentEl.textContent = msg;
    contentEl.classList.add("is-error");
    finishStream(es, contentEl);
  });

  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) return;
    if (!fullText) {
      contentEl.textContent = "Could not reach the backend. Is the server running?";
      contentEl.classList.add("is-error");
    }
    finishStream(es, contentEl);
  };
}

function finishStream(es, contentEl) {
  contentEl.classList.remove("is-streaming");
  es.close();
  activeEventSource = null;
  sendBtn.disabled = false;
  messageInput.focus();
}

function addMessage(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `message message-${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const content = document.createElement("div");
  content.className = "bubble-content";
  content.textContent = text;

  bubble.appendChild(content);
  wrapper.appendChild(bubble);
  messagesEl.appendChild(wrapper);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrapper;
}

function renderSources(messageWrapper, sources) {
  if (!sources.length) return;
  const bubble = messageWrapper.querySelector(".bubble");
  const sourcesEl = document.createElement("div");
  sourcesEl.className = "message-sources";

  const seen = new Set();
  for (const s of sources) {
    const label = s.title || s.source;
    if (seen.has(label)) continue;
    seen.add(label);
    const tag = document.createElement("span");
    tag.className = "source-tag";
    tag.textContent = label;
    tag.title = s.snippet || "";
    sourcesEl.appendChild(tag);
  }
  bubble.appendChild(sourcesEl);
}
