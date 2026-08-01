// Vocal Manager frontend — zero-dependency vanilla JS.
// Talks to Rust via the Tauri v2 invoke bridge (`window.__TAURI_INTERNALS__.invoke`)
// and polls worker state + log tail every ~300 ms.

const invoke = (cmd, args = {}) =>
  window.__TAURI_INTERNALS__.invoke(cmd, args);

const $ = (id) => document.getElementById(id);

// ---- tiny UI helpers ----
function fmtBytes(n) {
  if (n == null) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function phaseLabel(phase) {
  const map = {
    idle: "Idle",
    loading: "Loading…",
    ready: "Ready ✓",
    speaking: "Speaking…",
    error: "Error",
  };
  return map[phase] || phase || "—";
}

function phaseClass(phase) {
  if (phase === "ready") return "ok";
  if (phase === "error") return "err";
  if (phase === "loading" || phase === "speaking") return "warn";
  return "";
}

// ---- state ----
let cfg = null;        // latest VocalConfig from get_config
let worker = null;     // latest worker state
let status = null;     // latest StatusDto
let logOffset = 0;     // incremental log cursor

let busy = {
  start: false,
  stop: false,
  speak: false,
  save: false,
};

function renderStatus() {
  const pathEl = $("config-path");
  if (status) {
    pathEl.textContent = `vocal.config @ ${status.config_path}`;
  }
  renderCards();
  renderQwenModels();
  renderWorkerBadges();
  renderTestControls();
}

// ---- Manage tab ----
function renderCards() {
  if (!status) return;
  const container = $("backend-cards");
  container.innerHTML = "";

  for (const b of status.backends) {
    const card = document.createElement("div");
    card.className = "card" + (b.active ? " active" : "");

    const head = document.createElement("div");
    head.className = "card-head";
    const title = document.createElement("span");
    title.className = "card-title";
    title.textContent = b.label;
    const badges = document.createElement("span");
    badges.innerHTML = "";
    if (b.active) {
      const act = document.createElement("span");
      act.className = "badge active-badge";
      act.textContent = "active";
      badges.appendChild(act);
    }
    head.appendChild(title);
    head.appendChild(badges);

    const modelRow = rowEl("Model", b.model_path, b.model_exists ? "ok" : "err");
    const sizeRow = rowEl("Size", fmtBytes(b.model_bytes), "");
    const binRow = rowEl("Binary", b.binary, b.binary_exists ? "ok" : "err");
    const binStatus = rowEl("Binary ok?", b.binary_exists ? "yes" : "missing", b.binary_exists ? "ok" : "err");
    const noteRow = b.model_note ? rowEl("Note", b.model_note, "") : null;

    const actions = document.createElement("div");
    actions.className = "card-actions";

    const loadBtn = document.createElement("button");
    loadBtn.className = "primary";
    loadBtn.textContent = "Load";
    loadBtn.disabled = busy.start || (worker && worker.running);
    loadBtn.addEventListener("click", () => startWorker(b.backend));
    actions.appendChild(loadBtn);

    const unloadBtn = document.createElement("button");
    unloadBtn.textContent = "Unload";
    unloadBtn.disabled = busy.stop || !(worker && worker.running);
    unloadBtn.addEventListener("click", stopWorker);
    actions.appendChild(unloadBtn);

    card.appendChild(head);
    card.appendChild(modelRow);
    card.appendChild(sizeRow);
    card.appendChild(binRow);
    card.appendChild(binStatus);
    if (noteRow) card.appendChild(noteRow);
    card.appendChild(actions);

    container.appendChild(card);
  }
}

function rowEl(k, v, cls) {
  const row = document.createElement("div");
  row.className = "row";
  const kEl = document.createElement("span");
  kEl.className = "k";
  kEl.textContent = k;
  const vEl = document.createElement("span");
  vEl.className = "v" + (cls ? " " + cls : "");
  vEl.textContent = v;
  row.appendChild(kEl);
  row.appendChild(vEl);
  return row;
}

// ---- Qwen model discovery ----
function renderQwenModels() {
  if (!status) return;
  const container = $("qwen-models");
  container.innerHTML = "";

  for (const m of status.qwen_models || []) {
    const card = document.createElement("div");
    card.className = "card" + (m.active ? " active" : "");

    const head = document.createElement("div");
    head.className = "card-head";
    const title = document.createElement("span");
    title.className = "card-title";
    title.textContent = m.name;
    head.appendChild(title);
    if (m.active) {
      const act = document.createElement("span");
      act.className = "badge active-badge";
      act.textContent = "in use";
      head.appendChild(act);
    }

    const pathRow = rowEl("Path", m.path, m.exists ? "ok" : "err");
    const sizeRow = rowEl("Size", fmtBytes(m.bytes), "");
    const statusRow = rowEl("Status", m.exists ? "present" : "missing", m.exists ? "ok" : "err");

    const actions = document.createElement("div");
    actions.className = "card-actions";
    const useBtn = document.createElement("button");
    useBtn.className = "primary";
    useBtn.textContent = m.active ? "Active" : "Use this model";
    useBtn.disabled = m.active || (worker && worker.running);
    useBtn.addEventListener("click", () => useQwenModel(m.path));
    actions.appendChild(useBtn);

    card.appendChild(head);
    card.appendChild(pathRow);
    card.appendChild(sizeRow);
    card.appendChild(statusRow);
    card.appendChild(actions);

    container.appendChild(card);
  }
}

async function useQwenModel(path) {
  try {
    await invoke("set_qwen_model", { modelPath: path });
    cfg = await invoke("get_config", {});
    loadSettingsForm();
    refreshStatus();
  } catch (e) {
    console.error("set_qwen_model failed", e);
    alert("Failed to switch model: " + e);
  }
}

// ---- worker lifecycle ----
async function startWorker(backend) {
  // Switch backend if the user clicked a non-active card.
  if (cfg && cfg.backend !== backend) {
    cfg.backend = backend;
    try {
      await invoke("save_config", { cfg });
    } catch (e) {
      console.error("save_config failed", e);
    }
  }
  busy.start = true;
  renderCards();
  try {
    await invoke("worker_start", {});
  } catch (e) {
    console.error("worker_start failed", e);
  }
  busy.start = false;
  refreshStatus();
}

async function stopWorker() {
  busy.stop = true;
  renderCards();
  try {
    await invoke("worker_stop", {});
  } catch (e) {
    console.error("worker_stop failed", e);
  }
  busy.stop = false;
  refreshStatus();
}

async function speak() {
  const text = $("test-text").value.trim();
  if (!text) {
    $("test-status").textContent = "Type something to speak";
    return;
  }
  busy.speak = true;
  $("btn-speak").disabled = true;
  try {
    await invoke("worker_speak", { text });
    $("test-status").textContent = "Queued ✓";
    $("test-status").className = "status-note ok";
  } catch (e) {
    $("test-status").textContent = String(e);
    $("test-status").className = "status-note err";
  }
  busy.speak = false;
  $("btn-speak").disabled = false;
}

// ---- worker badges (Test tab) ----
function renderWorkerBadges() {
  const el = $("test-status");
  const btn = $("btn-speak");
  if (!worker || !worker.running) {
    el.textContent = "No model loaded — go to Manage and hit Load";
    el.className = "status-note err";
    btn.disabled = true;
    return;
  }
  const phase = phaseLabel(worker.phase);
  const loadInfo = worker.load_seconds != null ? ` · loaded in ${worker.load_seconds.toFixed(2)}s` : "";
  const pidInfo = worker.pid ? ` · pid ${worker.pid}` : "";
  el.textContent = `${phase}${loadInfo}${pidInfo}`;
  el.className = "status-note " + phaseClass(worker.phase);
  btn.disabled = !busy.speak && worker.phase !== "ready";
}

// ---- Test tab controls ----
function renderTestControls() {
  if (!cfg) return;
  const setVal = (id, v) => { if (v != null && v !== "") $(id).value = v; };
  setVal("test-speaker", cfg.speaker);
  setVal("test-language", cfg.language);
  setVal("test-instruct", cfg.instruct || "");
  setVal("test-temperature", cfg.temperature);
}

// ---- Logs ----
function renderLogs() {
  const lines = (status && status.logs) || [];
  // appended incrementally by poll; here we just mirror the current buffer
}

// ---- Settings tab ----
function loadSettingsForm() {
  if (!cfg) return;
  const setVal = (id, v) => { $(id).value = v == null ? "" : v; };
  setVal("set-backend", cfg.backend);
  setVal("set-model-path", cfg.model_path);
  setVal("set-engine-dir", cfg.engine_dir);
  setVal("set-speaker", cfg.speaker);
  setVal("set-language", cfg.language);
  setVal("set-instruct", cfg.instruct || "");
  setVal("set-temperature", cfg.temperature);
  setVal("set-python-bin", cfg.python_bin);
  setVal("set-chatterbox-worker", cfg.chatterbox_worker);
  setVal("set-chatterbox-model", cfg.chatterbox_model);
  setVal("set-ref-audio", cfg.ref_audio || "");
  setVal("set-chatterbox-model-path", cfg.chatterbox_model_path);
}

function readSettingsForm() {
  const val = (id) => $(id).value.trim();
  return {
    backend: val("set-backend"),
    model_path: val("set-model-path"),
    engine_dir: val("set-engine-dir"),
    speaker: val("set-speaker"),
    language: val("set-language"),
    instruct: val("set-instruct") || null,
    temperature: val("set-temperature"),
    python_bin: val("set-python-bin"),
    chatterbox_worker: val("set-chatterbox-worker"),
    chatterbox_model: val("set-chatterbox-model"),
    ref_audio: val("set-ref-audio") || null,
    chatterbox_model_path: val("set-chatterbox-model-path"),
  };
}

async function saveSettings() {
  const next = readSettingsForm();
  busy.save = true;
  $("btn-save-config").disabled = true;
  try {
    await invoke("save_config", { cfg: next });
    cfg = next;
    $("settings-status").textContent = "Saved ✓";
    $("settings-status").className = "status-note ok";
    refreshStatus();
  } catch (e) {
    $("settings-status").textContent = String(e);
    $("settings-status").className = "status-note err";
  }
  busy.save = false;
  $("btn-save-config").disabled = false;
}

// ---- MCP server ----
async function loadMcp() {
  try {
    const s = await invoke("mcp_status", {});
    const badges = $("mcp-badges");
    badges.innerHTML = "";
    const b = (label, ok, text) => {
      const el = document.createElement("span");
      el.className = "badge " + (ok ? "ok" : "err");
      el.textContent = `${label}: ${text}`;
      badges.appendChild(el);
    };
    b("binary", s.binary_exists, s.binary_exists ? "built" : "NOT built");
    b("config", s.config_installed, s.config_installed ? "installed" : "not installed");
    $("mcp-config").textContent = s.config_json;
  } catch (e) {
    $("mcp-config").textContent = "error: " + String(e);
  }
}

async function installMcp() {
  $("mcp-status").textContent = "writing…";
  $("mcp-status").className = "status-note";
  try {
    const p = await invoke("mcp_install_config", {});
    $("mcp-status").textContent = "wrote " + p;
    $("mcp-status").className = "status-note ok";
    loadMcp();
  } catch (e) {
    $("mcp-status").textContent = String(e);
    $("mcp-status").className = "status-note err";
  }
}

async function copyMcp() {
  try {
    await navigator.clipboard.writeText($("mcp-config").textContent);
    $("mcp-status").textContent = "copied ✓";
    $("mcp-status").className = "status-note ok";
  } catch (e) {
    $("mcp-status").textContent = "copy failed";
    $("mcp-status").className = "status-note err";
  }
}

async function testMcp() {
  $("mcp-status").textContent = "testing…";
  $("mcp-status").className = "status-note";
  $("mcp-result").textContent = "";
  try {
    const r = await invoke("mcp_test", {});
    $("mcp-status").textContent = "OK ✓";
    $("mcp-status").className = "status-note ok";
    $("mcp-result").textContent =
      "server: " + (r.server || "?") + "\ntools: " + JSON.stringify(r.tools);
  } catch (e) {
    $("mcp-status").textContent = "failed";
    $("mcp-status").className = "status-note err";
    $("mcp-result").textContent = String(e);
  }
}

// ---- polling ----
async function refreshStatus() {
  try {
    status = await invoke("check_status", {});
    worker = status.worker;
    renderStatus();
  } catch (e) {
    console.error("check_status failed", e);
  }
}

async function pollLogs() {
  try {
    const chunk = await invoke("worker_logs", { offset: logOffset });
    if (chunk && chunk.lines && chunk.lines.length) {
      logOffset = chunk.offset;
      const testLog = $("test-log");
      const fullLog = $("full-log");
      const add = chunk.lines.join("\n") + "\n";
      testLog.textContent += add;
      fullLog.textContent += add;
      testLog.scrollTop = testLog.scrollHeight;
      fullLog.scrollTop = fullLog.scrollHeight;
    }
  } catch (e) {
    // ignore — polling continues
  }
}

async function init() {
  // Tabs
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      $("tab-" + btn.dataset.tab).classList.add("active");
      if (btn.dataset.tab === "mcp") loadMcp();
    });
  });

  // Buttons
  $("btn-speak").addEventListener("click", speak);
  $("btn-save-config").addEventListener("click", saveSettings);
  $("btn-mcp-install").addEventListener("click", installMcp);
  $("btn-mcp-copy").addEventListener("click", copyMcp);
  $("btn-mcp-test").addEventListener("click", testMcp);

  // Load config once for the settings form
  try {
    cfg = await invoke("get_config", {});
    loadSettingsForm();
  } catch (e) {
    console.error("get_config failed", e);
  }

  await refreshStatus();
  loadMcp();

  // Poll loop
  setInterval(refreshStatus, 300);
  setInterval(pollLogs, 300);
}

init();
