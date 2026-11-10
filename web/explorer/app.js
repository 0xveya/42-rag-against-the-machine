/* Local repository explorer. All source text is inserted as text, never HTML. */
const $ = (id) => document.getElementById(id);
const repository = $("repository");
const question = $("question");
const status = $("status");
let repositories = [];
let sources = [];
let selectedSource = -1;
let socket;
let reconnectTimer;
let busy = false;
let indexFiles = 0;
let activeTab = "results";

function report(message, error = false) {
  status.textContent = message;
  status.classList.toggle("error", error);
}
function controls() {
  const selected = Boolean(repository.value);
  const connected = socket?.readyState === WebSocket.OPEN;
  repository.disabled = busy || !repositories.length;
  for (const id of ["reindex", "remove-index"])
    $(id).disabled = busy || !selected;
  for (const id of ["clone", "refresh", "url", "name"]) $(id).disabled = busy;
  for (const id of ["search", "generate"])
    $(id).disabled = busy || !selected || !connected || !indexFiles;
  question.disabled = busy;
  $("k").disabled = busy;
}
function setBusy(value) {
  busy = value;
  controls();
}
function tab(name) {
  activeTab = name;
  for (const button of document.querySelectorAll("[data-tab]")) {
    if (button.dataset.tab === name)
      button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
    $(button.dataset.tab).hidden = button.dataset.tab !== name;
  }
}
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) {
    const detail =
      typeof result.detail === "string"
        ? result.detail
        : JSON.stringify(result.detail);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return result;
}
function clearResults() {
  sources = [];
  selectedSource = -1;
  $("sources").replaceChildren();
  $("source-code").textContent = "";
  $("source-path").textContent = "";
  $("source-range").textContent = "";
  $("copy-source").disabled = true;
  $("result-count").textContent = "";
  $("retrieval").hidden = true;
  $("empty").hidden = false;
  $("empty").textContent = "Search this repository’s code and documentation.";
  $("messages").replaceChildren();
  $("answer-text").textContent =
    "Generate an answer to see the local model’s output.";
}
async function loadCounts() {
  const id = repository.value;
  if (!id) {
    indexFiles = 0;
    $("counts").textContent = "No repository selected";
    $("db-repository").textContent = "Repository index";
    $("db-files").textContent = "0";
    $("db-chunks").textContent = "0";
    controls();
    return;
  }
  const counts = await api(`/api/index/${encodeURIComponent(id)}`);
  if (repository.value !== id) return;
  indexFiles = counts.files;
  $("counts").textContent =
    `${counts.files.toLocaleString()} files · ${counts.chunks.toLocaleString()} chunks`;
  $("db-repository").textContent = id;
  $("db-files").textContent = counts.files.toLocaleString();
  $("db-chunks").textContent = counts.chunks.toLocaleString();
  controls();
}
async function loadRepositories(preferred = repository.value) {
  repositories = await api("/api/repositories");
  repository.replaceChildren();
  for (const item of repositories)
    repository.append(new Option(item.name, item.repository_id));
  if (!repositories.length) {
    repository.append(new Option("No repositories yet", ""));
    $("clone-details").open = true;
    report("Clone a repository to begin.");
  } else if (repositories.some((item) => item.repository_id === preferred))
    repository.value = preferred;
  await loadCounts();
  controls();
  if (!busy && repositories.length)
    report(indexFiles ? "Ready." : "Reindex this repository to search it.");
}
async function reindex() {
  report(`Indexing ${repository.value}…`);
  const result = await api(
    `/api/repositories/${encodeURIComponent(repository.value)}/reindex`,
    { method: "POST" },
  );
  clearResults();
  await loadCounts();
  report(`Indexed ${result.files_processed} files.`);
}
function shortPath(path) {
  const root = repositories.find(
    (item) => item.repository_id === repository.value,
  )?.path;
  return root && path.startsWith(`${root}/`)
    ? path.slice(root.length + 1)
    : path;
}
function selectSource(index) {
  selectedSource = index;
  const source = sources[index];
  for (const [i, button] of [
    ...$("sources").querySelectorAll("button"),
  ].entries()) {
    if (i === index) button.setAttribute("aria-current", "true");
    else button.removeAttribute("aria-current");
  }
  $("source-path").textContent = shortPath(source.file_path);
  $("source-range").textContent =
    `Characters ${source.first_character_index.toLocaleString()}–${source.last_character_index.toLocaleString()} · ${source.text.length.toLocaleString()} characters`;
  const code = $("source-code");
  code.removeAttribute("data-highlighted");
  code.className = "";
  code.textContent = source.text;
  const extension = source.file_path.split(".").pop().toLowerCase();
  const languages = {
    py: "python",
    c: "c",
    h: "c",
    cpp: "cpp",
    hpp: "cpp",
    cc: "cpp",
    go: "go",
    js: "javascript",
    ts: "typescript",
    rs: "rust",
    json: "json",
    md: "markdown",
    sh: "bash",
  };
  const language = languages[extension];
  if (language && window.hljs?.getLanguage(language)) {
    code.className = `language-${language}`;
    window.hljs.highlightElement(code);
  }
  $("copy-source").disabled = false;
}
function inspect(message) {
  sources = message.sources;
  $("fts-query").textContent = message.query || "(no searchable terms)";
  $("result-count").textContent = `(${sources.length})`;
  $("sources").replaceChildren();
  $("retrieval").hidden = !sources.length;
  $("empty").hidden = Boolean(sources.length);
  $("empty").textContent =
    "No matching chunks. Try an identifier or phrase from the repository, or reindex it.";
  for (const [index, source] of sources.entries()) {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    const rank = document.createElement("div");
    rank.className = "source-rank";
    rank.textContent = `Source ${index + 1}`;
    const path = document.createElement("div");
    path.textContent = shortPath(source.file_path);
    const score = document.createElement("div");
    score.className = "source-score";
    score.textContent = `BM25 ${source.score.toPrecision(5)}`;
    button.append(rank, path, score);
    button.addEventListener("click", () => selectSource(index));
    item.append(button);
    $("sources").append(item);
  }
  if (sources.length) selectSource(0);
  else {
    selectedSource = -1;
    $("copy-source").disabled = true;
  }
  $("messages").replaceChildren();
  for (const messagePart of message.messages) {
    const section = document.createElement("section");
    section.className = "prompt-message";
    const heading = document.createElement("h2");
    heading.textContent = messagePart.role;
    const pre = document.createElement("pre");
    pre.textContent = messagePart.content;
    section.append(heading, pre);
    $("messages").append(section);
  }
}
function connect() {
  clearTimeout(reconnectTimer);
  socket = new WebSocket(
    `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`,
  );
  socket.addEventListener("open", () => {
    if (repositories.length)
      report(indexFiles ? "Ready." : "Reindex this repository to search it.");
    controls();
  });
  socket.addEventListener("close", () => {
    if (busy)
      report(
        "Connection lost. The request was interrupted; retry when connected.",
        true,
      );
    else report("Disconnected. Reconnecting…");
    setBusy(false);
    reconnectTimer = setTimeout(connect, 1500);
  });
  socket.addEventListener("message", ({ data }) => {
    try {
      const message = JSON.parse(data);
      if (message.type === "inspection") inspect(message);
      else if (message.type === "token")
        $("answer-text").textContent += message.data;
      else if (message.type === "status")
        report(
          message.data === "retrieving"
            ? "Searching SQLite…"
            : "Generating with the local model…",
        );
      else if (message.type === "error") {
        report(message.data, true);
        setBusy(false);
      } else if (message.type === "done") {
        report(
          sources.length
            ? `Retrieved ${sources.length} sources.`
            : "No matching sources.",
        );
        if (!sources.length && activeTab === "answer")
          $("answer-text").textContent =
            "No sources matched. No answer was generated.";
        setBusy(false);
      }
    } catch (error) {
      report(error.message, true);
      setBusy(false);
    }
  });
}
function run(generate) {
  if (
    busy ||
    socket?.readyState !== WebSocket.OPEN ||
    !$("question-form").reportValidity()
  )
    return;
  if (!question.value.trim()) {
    report("Enter a question.", true);
    return;
  }
  tab(generate ? "answer" : "results");
  $("answer-text").textContent = generate
    ? ""
    : "Generate an answer to see the local model’s output.";
  setBusy(true);
  report("Searching SQLite…");
  socket.send(
    JSON.stringify({
      repository: repository.value,
      question: question.value.trim(),
      k: Number($("k").value),
      inspect: true,
      generate,
    }),
  );
}
$("question-form").addEventListener("submit", (event) => {
  event.preventDefault();
  run(false);
});
$("generate").addEventListener("click", () => run(true));
question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    run(false);
  }
});
for (const button of document.querySelectorAll("[data-tab]"))
  button.addEventListener("click", () => tab(button.dataset.tab));
repository.addEventListener("change", async () => {
  clearResults();
  indexFiles = 0;
  controls();
  report("Reading index…");
  try {
    await loadCounts();
    report(indexFiles ? "Ready." : "Reindex this repository to search it.");
  } catch (error) {
    indexFiles = 0;
    controls();
    report(error.message, true);
  }
});
$("refresh").addEventListener("click", async () => {
  setBusy(true);
  try {
    await loadRepositories();
    report(
      repositories.length
        ? "Repositories refreshed."
        : "Clone a repository to begin.",
    );
  } catch (error) {
    report(error.message, true);
  } finally {
    setBusy(false);
  }
});
$("reindex").addEventListener("click", async () => {
  setBusy(true);
  try {
    await reindex();
  } catch (error) {
    report(error.message, true);
  } finally {
    setBusy(false);
  }
});
$("clone-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(true);
  report("Cloning repository…");
  try {
    const url = $("url").value.trim();
    const name = $("name").value.trim();
    const inferred =
      name ||
      new URL(url).pathname
        .replace(/\/$/, "")
        .split("/")
        .pop()
        .replace(/\.git$/, "");
    const result = await api("/api/repositories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, name }),
    });
    if (result.exit_code !== 0)
      throw new Error(result.stderr || "Clone failed");
    await loadRepositories(inferred);
    clearResults();
    await reindex();
    $("clone-form").reset();
    $("clone-details").open = false;
  } catch (error) {
    report(error.message, true);
  } finally {
    setBusy(false);
  }
});
$("remove-index").addEventListener("click", () => {
  $("remove-name").textContent = repository.value;
  $("remove-dialog").showModal();
});
$("remove-dialog").addEventListener("close", async () => {
  if ($("remove-dialog").returnValue !== "remove") return;
  setBusy(true);
  report("Removing index…");
  try {
    await api(`/api/index/${encodeURIComponent(repository.value)}`, {
      method: "DELETE",
    });
    clearResults();
    await loadCounts();
    report("Index removed. Cloned files were kept.");
  } catch (error) {
    report(error.message, true);
  } finally {
    setBusy(false);
  }
});
$("copy-source").addEventListener("click", async () => {
  if (selectedSource < 0) return;
  try {
    await navigator.clipboard.writeText(sources[selectedSource].text);
    $("copy-source").textContent = "Copied";
    setTimeout(() => {
      $("copy-source").textContent = "Copy";
    }, 1500);
  } catch (error) {
    report(`Could not copy: ${error.message}`, true);
  }
});
loadRepositories().catch((error) => {
  report(error.message, true);
  controls();
});
connect();
