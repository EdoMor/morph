/* Morph web client.
 * Consumes the SSE event stream from POST /api/chat and renders it.
 * No build step, no framework — it has to load fast over a phone hotspot.
 */

const $ = (sel) => document.querySelector(sel);

const log = $("#log");
const input = $("#input");
const sendBtn = $("#send");
const statusEl = $("#status");
const drawer = $("#drawer");
const scrim = $("#scrim");

let sessionId = localStorage.getItem("morph.session") || null;
let streaming = false;

/* ------------------------------------------------------------------ util */

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollToEnd() {
  requestAnimationFrame(() => log.scrollTo({ top: log.scrollHeight, behavior: "smooth" }));
}

function clearEmptyState() {
  const empty = log.querySelector(".empty");
  if (empty) empty.remove();
}

/** Render a minimal, safe subset of markdown: fenced code and inline code. */
function renderBody(node, text) {
  node.textContent = "";
  const parts = text.split(/```(?:[\w+-]*)\n?/);
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      const pre = el("pre");
      pre.appendChild(el("code", null, part.replace(/\n$/, "")));
      node.appendChild(pre);
    } else if (part) {
      part.split(/(`[^`\n]+`)/).forEach((chunk) => {
        if (chunk.startsWith("`") && chunk.endsWith("`") && chunk.length > 2) {
          node.appendChild(el("code", null, chunk.slice(1, -1)));
        } else if (chunk) {
          node.appendChild(document.createTextNode(chunk));
        }
      });
    }
  });
}

/* --------------------------------------------------------------- render */

function addMessage(role, text) {
  clearEmptyState();
  const wrap = el("div", `msg ${role}`);
  wrap.appendChild(el("div", "who", role === "user" ? "You" : "Morph"));
  const body = el("div", "body");
  renderBody(body, text);
  wrap.appendChild(body);
  log.appendChild(wrap);
  scrollToEnd();
  return body;
}

function addTyping() {
  const wrap = el("div", "msg assistant typing-wrap");
  const dots = el("div", "typing");
  dots.append(el("i"), el("i"), el("i"));
  wrap.appendChild(dots);
  log.appendChild(wrap);
  scrollToEnd();
  return wrap;
}

function addToolCall(event) {
  clearEmptyState();
  const details = el("details", "tool");
  details.id = `tool-${event.id}`;
  const summary = el("summary");
  summary.append(el("span", "dot"), el("span", "name", event.name), el("span", "took", "running…"));
  details.appendChild(summary);
  const pre = el("pre", null, JSON.stringify(event.arguments ?? {}, null, 2));
  details.appendChild(pre);
  log.appendChild(details);
  scrollToEnd();
}

function completeToolCall(event) {
  const details = document.getElementById(`tool-${event.id}`);
  if (!details) return;
  details.classList.add(event.ok ? "ok" : "bad");
  details.querySelector(".took").textContent = `${Math.round(event.duration_ms)}ms`;
  const pre = details.querySelector("pre");
  pre.textContent = event.content || "(no output)";
  if (!event.ok) details.open = true;

  const previews = event.meta && event.meta.previews;
  if (previews && previews.length) {
    const gallery = el("div", "gallery");
    previews.forEach((src, i) => {
      const img = el("img");
      img.src = src;
      img.alt = `Generated image ${i + 1}`;
      img.loading = "lazy";
      gallery.appendChild(img);
    });
    details.appendChild(gallery);
    details.open = true;
  }
  scrollToEnd();
}

function addError(message) {
  clearEmptyState();
  log.appendChild(el("div", "error", message));
  scrollToEnd();
}

/* ----------------------------------------------------------------- chat */

async function send(text) {
  if (!text.trim() || streaming) return;
  streaming = true;
  sendBtn.disabled = true;
  addMessage("user", text);
  input.value = "";
  autosize();

  let typing = addTyping();
  let assistantBody = null;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });
    if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split;
      while ((split = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const dataLine = frame.split("\n").find((l) => l.startsWith("data:"));
        if (!dataLine) continue;

        let event;
        try {
          event = JSON.parse(dataLine.slice(5).trim());
        } catch {
          continue;
        }

        if (typing) { typing.remove(); typing = null; }

        if (event.type === "text") {
          if (!assistantBody) assistantBody = addMessage("assistant", event.text);
          else renderBody(assistantBody, event.text);
          scrollToEnd();
        } else if (event.type === "tool_use") {
          addToolCall(event);
          assistantBody = null;
        } else if (event.type === "tool_result") {
          completeToolCall(event);
        } else if (event.type === "error") {
          addError(event.message || "Unknown error");
        } else if (event.type === "done") {
          const result = event.result || {};
          if (result.session_id) {
            sessionId = result.session_id;
            localStorage.setItem("morph.session", sessionId);
            loadSessions();
          }
          if (result.stop_reason === "max_steps") {
            addError("Stopped: step budget exhausted. Ask a narrower question or raise max_steps.");
          }
        }
      }
    }
  } catch (err) {
    addError(`Connection lost: ${err.message}`);
  } finally {
    if (typing) typing.remove();
    streaming = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

/* ------------------------------------------------------------- sidebars */

async function loadJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

async function loadHealth() {
  try {
    const health = await loadJSON("/api/health");
    statusEl.textContent = `${health.model} · ${health.tools} tools`;
    statusEl.className = "status ok";
  } catch {
    statusEl.textContent = "offline";
    statusEl.className = "status bad";
  }
}

async function loadTools() {
  try {
    const { tools } = await loadJSON("/api/tools");
    $("#tool-count").textContent = tools.length;
    const list = $("#tool-list");
    list.textContent = "";
    tools.forEach((tool) => {
      const li = el("li");
      li.appendChild(el("code", null, tool.name));
      list.appendChild(li);
    });
  } catch { /* offline: leave the placeholder */ }
}

async function loadSkills() {
  try {
    const { skills } = await loadJSON("/api/skills");
    $("#skill-count").textContent = skills.length;
    const list = $("#skill-list");
    list.textContent = "";
    if (!skills.length) {
      list.appendChild(el("li", "muted", "None installed"));
      return;
    }
    skills.forEach((skill) => {
      const li = el("li", "clickable");
      li.appendChild(el("code", null, skill.name));
      li.title = skill.description;
      list.appendChild(li);
    });
  } catch { /* offline */ }
}

async function loadSessions() {
  try {
    const { sessions } = await loadJSON("/api/sessions");
    const list = $("#session-list");
    list.textContent = "";
    if (!sessions.length) {
      list.appendChild(el("li", "muted", "No sessions yet"));
      return;
    }
    sessions.slice(0, 30).forEach((meta) => {
      const li = el("li", "clickable", meta.title || meta.id);
      if (meta.id === sessionId) li.classList.add("active");
      li.addEventListener("click", () => openSession(meta.id));
      list.appendChild(li);
    });
  } catch { /* offline */ }
}

async function openSession(id) {
  closeDrawer();
  try {
    const data = await loadJSON(`/api/sessions/${id}`);
    sessionId = id;
    localStorage.setItem("morph.session", id);
    log.textContent = "";
    (data.history || []).forEach((message) => {
      if (message.role === "user") addMessage("user", message.content || "");
      else if (message.role === "assistant" && message.content) addMessage("assistant", message.content);
    });
    loadSessions();
  } catch (err) {
    addError(`Could not open session: ${err.message}`);
  }
}

function newSession() {
  sessionId = null;
  localStorage.removeItem("morph.session");
  log.textContent = "";
  const empty = el("div", "empty");
  empty.appendChild(el("h1", null, "What are we building?"));
  empty.appendChild(el("p", null, "New conversation."));
  log.appendChild(empty);
  closeDrawer();
  loadSessions();
}

/* --------------------------------------------------------------- drawer */

function openDrawer() {
  drawer.classList.add("open");
  scrim.hidden = false;
  $("#menu-btn").setAttribute("aria-expanded", "true");
}

function closeDrawer() {
  drawer.classList.remove("open");
  scrim.hidden = true;
  $("#menu-btn").setAttribute("aria-expanded", "false");
}

/* ---------------------------------------------------------------- input */

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, window.innerHeight * 0.4)}px`;
}

input.addEventListener("input", autosize);
input.addEventListener("keydown", (event) => {
  // Enter sends on a physical keyboard; on touch the Enter key inserts a newline.
  const touch = window.matchMedia("(pointer: coarse)").matches;
  if (event.key === "Enter" && !event.shiftKey && !touch) {
    event.preventDefault();
    send(input.value);
  }
});

$("#composer").addEventListener("submit", (event) => {
  event.preventDefault();
  send(input.value);
});

$("#menu-btn").addEventListener("click", () =>
  drawer.classList.contains("open") ? closeDrawer() : openDrawer()
);
scrim.addEventListener("click", closeDrawer);
$("#new-btn").addEventListener("click", newSession);

log.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (chip) send(chip.dataset.prompt);
});

/* ------------------------------------------------------------------ PWA */

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch(() => {
      /* not fatal: the app works without offline caching */
    });
  });
}

loadHealth();
loadTools();
loadSkills();
loadSessions();
setInterval(loadHealth, 30000);
