/* Morph progress dashboard.
 * Reads data/dashboard.json, built by scripts/build_site.py from the history
 * and scorecard the loop commits. No framework, no build step.
 */

const $ = (sel) => document.querySelector(sel);
const SVG_NS = "http://www.w3.org/2000/svg";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

function when(ts) {
  if (!ts) return "";
  const date = new Date(ts * 1000);
  const days = (Date.now() - date) / 86400000;
  if (days < 1) return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (days < 7) return `${Math.floor(days)}d ago`;
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

/* ------------------------------------------------------------------ chart */

function drawChart(series) {
  const chart = $("#chart");
  chart.textContent = "";
  if (!series.length) return;

  const W = 760, H = 260, pad = { t: 16, r: 16, b: 32, l: 40 };
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  // Always anchor at 0–100: a self-scaling axis makes a 0.2 wobble look dramatic.
  const x = (i) => pad.l + (series.length === 1 ? innerW / 2 : (i / (series.length - 1)) * innerW);
  const y = (v) => pad.t + innerH - (Math.max(0, Math.min(100, v)) / 100) * innerH;

  for (const value of [0, 25, 50, 75, 100]) {
    chart.appendChild(svg("line", {
      class: "axis", x1: pad.l, x2: W - pad.r, y1: y(value), y2: y(value),
    }));
    const label = svg("text", { class: "axis-text", x: pad.l - 8, y: y(value) + 4, "text-anchor": "end" });
    label.textContent = value;
    chart.appendChild(label);
  }

  const points = series.map((p, i) => `${x(i)},${y(p.after)}`);
  chart.appendChild(svg("polygon", {
    class: "area",
    points: `${pad.l},${y(0)} ${points.join(" ")} ${x(series.length - 1)},${y(0)}`,
  }));
  chart.appendChild(svg("polyline", { class: "line", points: points.join(" ") }));

  series.forEach((point, i) => {
    const dot = svg("circle", {
      class: point.accepted ? "dot-accepted" : "dot-rejected",
      cx: x(i), cy: y(point.after), r: point.accepted ? 5 : 4,
    });
    const title = svg("title");
    title.textContent =
      `${point.accepted ? "accepted" : "rejected"} — ${point.before.toFixed(1)} → ` +
      `${point.after.toFixed(1)}${point.version ? ` (v${point.version})` : ""}`;
    dot.appendChild(title);
    chart.appendChild(dot);
  });

  const first = svg("text", { class: "axis-text", x: pad.l, y: H - 10 });
  first.textContent = when(series[0].ts);
  chart.appendChild(first);

  if (series.length > 1) {
    const last = svg("text", { class: "axis-text", x: W - pad.r, y: H - 10, "text-anchor": "end" });
    last.textContent = when(series[series.length - 1].ts);
    chart.appendChild(last);
  }
}

function drawRadar(data) {
  const card = $("#radar-card");
  const chart = $("#radar");
  const axes = data?.axes || [];
  if (axes.length < 3) {
    card.hidden = true;
    return;
  }

  card.hidden = false;
  chart.textContent = "";

  const W = 620, H = 460, cx = 310, cy = 214, radius = 154;
  const angle = (index) => -Math.PI / 2 + (index / axes.length) * Math.PI * 2;
  const point = (index, percent, extra = 0) => {
    const distance = radius * Math.max(0, Math.min(100, percent)) / 100 + extra;
    return [
      cx + Math.cos(angle(index)) * distance,
      cy + Math.sin(angle(index)) * distance,
    ];
  };
  const points = (field, fallback = 0) => axes
    .map((axis, index) => point(index, axis[field] ?? fallback).join(","))
    .join(" ");

  const title = svg("title", { id: "radar-title" });
  title.textContent = "Benchmark performance by category";
  chart.appendChild(title);
  const desc = svg("desc", { id: "radar-desc" });
  desc.textContent =
    `Current composite ${Number(data.current_composite ?? 0).toFixed(1)} out of 100` +
    (data.previous_composite != null
      ? `, compared with ${Number(data.previous_composite).toFixed(1)} in the previous run.`
      : ".");
  chart.appendChild(desc);

  for (const percent of [25, 50, 75, 100]) {
    chart.appendChild(svg("polygon", {
      class: "radar-grid",
      points: axes.map((_, index) => point(index, percent).join(",")).join(" "),
    }));
    const [x, y] = point(0, percent);
    const ringLabel = svg("text", {
      class: "radar-ring-label", x: x + 5, y: y + 11,
    });
    ringLabel.textContent = percent;
    chart.appendChild(ringLabel);
  }

  axes.forEach((axis, index) => {
    const [x, y] = point(index, 100);
    chart.appendChild(svg("line", {
      class: "radar-spoke", x1: cx, y1: cy, x2: x, y2: y,
    }));

    const [labelX, labelY] = point(index, 100, 42);
    const cosine = Math.cos(angle(index));
    const label = svg("text", {
      class: "radar-axis-label",
      x: labelX,
      y: labelY - 5,
      "text-anchor": cosine > 0.2 ? "start" : cosine < -0.2 ? "end" : "middle",
    });
    const name = svg("tspan", { x: labelX });
    name.textContent = axis.label;
    label.appendChild(name);
    const value = svg("tspan", { x: labelX, dy: 17, class: "radar-axis-value" });
    value.textContent =
      `${Number(axis.current).toFixed(0)}%` +
      (axis.delta != null ? ` · ${axis.delta >= 0 ? "+" : ""}${Number(axis.delta).toFixed(1)}` : "");
    label.appendChild(value);
    chart.appendChild(label);
  });

  const hasPrevious = axes.some((axis) => axis.previous != null);
  if (hasPrevious) {
    chart.appendChild(svg("polygon", {
      class: "radar-shape previous", points: points("previous"),
    }));
  }
  chart.appendChild(svg("polygon", {
    class: "radar-shape current", points: points("current"),
  }));
  axes.forEach((axis, index) => {
    const [x, y] = point(index, axis.current);
    const dot = svg("circle", { class: "radar-dot", cx: x, cy: y, r: 4 });
    const tip = svg("title");
    tip.textContent =
      `${axis.label}: ${Number(axis.current).toFixed(1)}%` +
      (axis.previous != null
        ? `, previous ${Number(axis.previous).toFixed(1)}%, change ${axis.delta >= 0 ? "+" : ""}${Number(axis.delta).toFixed(1)}`
        : "");
    dot.appendChild(tip);
    chart.appendChild(dot);
  });

  $("#radar-current-label").textContent =
    `${data.current_label || "Current run"} · ${Number(data.current_composite ?? 0).toFixed(1)}`;
  $("#radar-previous-label").textContent =
    `${data.previous_label || "Previous run"} · ${Number(data.previous_composite ?? 0).toFixed(1)}`;
  $("#radar-previous-key").hidden = !hasPrevious;

  const composite = data.composite_delta;
  const compositeHost = $("#radar-composite-delta");
  compositeHost.textContent = composite == null
    ? "First measured run"
    : `${composite >= 0 ? "+" : ""}${Number(composite).toFixed(2)} points`;
  compositeHost.className =
    composite == null ? "flat" : composite > 0.001 ? "up" : composite < -0.001 ? "down" : "flat";

  const deltas = $("#radar-deltas");
  deltas.textContent = "";
  axes.forEach((axis) => {
    const item = el("li");
    item.appendChild(el("span", "radar-delta-label", axis.label));
    const change = axis.delta;
    const klass = change == null ? "flat" : change > 0.001 ? "up" : change < -0.001 ? "down" : "flat";
    item.appendChild(el(
      "span",
      `radar-delta-value ${klass}`,
      change == null ? "—" : `${change >= 0 ? "+" : ""}${Number(change).toFixed(1)} pts`,
    ));
    deltas.appendChild(item);
  });
}

/* -------------------------------------------------------------- sections */

function drawCategories(scorecard) {
  const host = $("#categories");
  host.textContent = "";
  const categories = scorecard?.categories || {};

  if (!Object.keys(categories).length) {
    host.appendChild(el("p", "hint", "No scorecard has been committed yet."));
    return;
  }

  for (const [name, data] of Object.entries(categories)) {
    const row = el("div", "bar-row");
    row.appendChild(el("span", "bar-label", name));

    const track = el("div", "bar-track");
    const fill = el("div", `bar-fill${name === "requirements" ? " gate" : ""}`);
    const share = data.weight ? (data.points / data.weight) * 100 : 0;
    fill.style.width = `${Math.max(0, Math.min(100, share))}%`;
    track.appendChild(fill);
    row.appendChild(track);

    row.appendChild(el("span", "bar-value", `${(data.points ?? 0).toFixed(1)}/${data.weight ?? 0}`));
    host.appendChild(row);
  }
}

function drawTiers(scorecard) {
  const body = $("#tiers tbody");
  body.textContent = "";
  const diagnostics = scorecard?.diagnostics || {};

  for (const [suite, data] of Object.entries(diagnostics)) {
    const row = el("tr");
    row.appendChild(el("td", null, suite));
    const profile = data.tier_profile || {};
    for (const tier of ["1", "2", "3", "4", "5"]) {
      row.appendChild(el("td", null, profile[tier] !== undefined ? Number(profile[tier]).toFixed(2) : "—"));
    }
    row.appendChild(el("td", null, `T${data.frontier ?? 0}`));

    const state = el("td");
    state.appendChild(el("span", `pill ${data.calibration || ""}`, data.calibration || "—"));
    row.appendChild(state);
    body.appendChild(row);
  }
}

function drawWarnings(scorecard) {
  const warnings = scorecard?.instrument_warnings || [];
  $("#warnings-card").hidden = warnings.length === 0;
  const host = $("#warnings");
  host.textContent = "";
  warnings.forEach((text) => host.appendChild(el("li", null, text)));
}

function drawTargets(scorecard) {
  const targets = scorecard?.next_targets || [];
  $("#targets-card").hidden = targets.length === 0;
  const host = $("#targets");
  host.textContent = "";

  targets.forEach((target) => {
    const item = el("li");
    item.appendChild(el("span", "score", `${Math.round((target.score || 0) * 100)}%`));
    const body = el("div");
    body.appendChild(el("span", "name", target.name || ""));
    if (target.detail) body.appendChild(el("span", "detail", target.detail));
    item.appendChild(body);
    host.appendChild(item);
  });
}

function drawReleases(data) {
  const releases = data.releases || [];
  $("#releases-card").hidden = releases.length === 0;
  const host = $("#releases");
  host.textContent = "";

  releases.forEach((release) => {
    const item = el("li");
    item.appendChild(el("span", "tag", release.tag));
    if (release.date) item.appendChild(el("span", "date", release.date));
    if (data.repo) {
      const link = el("a", null, "download APK");
      link.href = `${data.repo}/releases/tag/${release.tag}`;
      link.rel = "noopener";
      item.appendChild(link);
    }
    if (release.notes) item.appendChild(el("span", "notes-text", release.notes));
    host.appendChild(item);
  });
}

function drawAttempts(history, filter) {
  const host = $("#attempts");
  host.textContent = "";

  const shown = history.filter((entry) =>
    filter === "all" ? true : filter === "accepted" ? entry.accepted : !entry.accepted
  );

  if (!shown.length) {
    host.appendChild(el("li", "hint", "Nothing matches that filter yet."));
    return;
  }

  shown.forEach((entry) => {
    const item = el("li");

    const head = el("div", "attempt-head");
    head.appendChild(el("span", `pill ${entry.accepted ? "ok" : "bad"}`, entry.accepted ? "accepted" : "rejected"));
    if (entry.version) head.appendChild(el("strong", null, `v${entry.version}`));

    const before = entry.score_before ?? 0;
    const after = entry.score_after ?? 0;
    const change = after - before;
    const klass = change > 0.001 ? "up" : change < -0.001 ? "down" : "flat";
    head.appendChild(el("span", `delta ${klass}`,
      `${before.toFixed(1)} → ${after.toFixed(1)} (${change >= 0 ? "+" : ""}${change.toFixed(2)})`));
    head.appendChild(el("span", "when", when(entry.ts)));
    item.appendChild(head);

    if (!entry.accepted && entry.rejection_reason) {
      item.appendChild(el("div", "attempt-reason", `Rejected: ${entry.rejection_reason}`));
    }
    if (entry.summary) {
      item.appendChild(el("div", "attempt-summary", entry.summary));
    }
    if (entry.files_changed?.length) {
      item.appendChild(el("div", "files", `Touched: ${entry.files_changed.join(", ")}`));
    }
    host.appendChild(item);
  });
}

function drawReasons(summary) {
  const host = $("#reasons");
  host.textContent = "";
  const reasons = summary.rejection_reasons || [];
  if (!reasons.length) return;

  host.appendChild(el("h3", null, "Why iterations were rejected"));
  reasons.forEach((reason) => {
    const row = el("div", "reason-row");
    row.appendChild(el("span", null, reason.reason));
    row.appendChild(el("span", null, String(reason.count)));
    host.appendChild(row);
  });
}

/* ----------------------------------------------------------------- boot */

function render(data) {
  const summary = data.summary || {};
  const scorecard = data.scorecard;

  $("#hd-version").textContent = `v${data.version}`;
  $("#hd-score").innerHTML = scorecard?.composite != null
    ? `${scorecard.composite.toFixed(1)}<small> / 100</small>`
    : "—";
  $("#hd-accepted").innerHTML =
    `${summary.accepted ?? 0}<small> of ${summary.attempts ?? 0}</small>`;
  $("#hd-hours").innerHTML = `${summary.model_hours ?? 0}<small> h</small>`;

  // Before the early return: a fork with no history yet still has a first run
  // to watch, and that is the most interesting moment it will ever have.
  pollLiveRun(data.repo);
  pollLiveTrace(data.repo);

  if (!summary.attempts && !scorecard) {
    $("#empty").hidden = false;
    return;
  }
  $("#content").hidden = false;

  drawChart(data.series || []);
  drawRadar(data.radar);
  drawCategories(scorecard);
  drawTiers(scorecard);
  drawWarnings(scorecard);
  drawTargets(scorecard);
  drawReleases(data);
  drawAttempts(data.history || [], "all");
  drawReasons(summary);

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach((c) => c.classList.remove("is-on"));
      chip.classList.add("is-on");
      drawAttempts(data.history || [], chip.dataset.filter);
    });
  });

  const model = scorecard?.metadata?.model;
  $("#meta").textContent =
    `Commit ${data.commit}${model ? ` · measured against ${model}` : ""} · ` +
    `updated ${new Date(data.generated_at * 1000).toLocaleString()}.`;

  if (data.repo) {
    const links = $("#links");
    [["Repository", data.repo], ["Releases", `${data.repo}/releases`], ["Runs", `${data.repo}/actions`]]
      .forEach(([label, href], index) => {
        if (index) links.appendChild(document.createTextNode(" · "));
        const link = el("a", null, label);
        link.href = href;
        link.rel = "noopener";
        links.appendChild(link);
      });
  }
}

/* ------------------------------------------------------------ live runs */

/* The committed data is a snapshot from the end of the last run, so it cannot
 * show a run that is happening now. The public GitHub API can. It needs no
 * token for a public repository, and if it is unavailable — rate limit, private
 * repo, offline — the banner simply stays hidden. */
async function pollLiveRun(repoUrl) {
  const match = /github\.com\/([^/]+)\/([^/]+)/.exec(repoUrl || "");
  if (!match) return;
  const [, owner, repo] = match;
  const banner = $("#live");

  async function check() {
    try {
      const response = await fetch(
        `https://api.github.com/repos/${owner}/${repo}/actions/runs?per_page=10`,
        { headers: { Accept: "application/vnd.github+json" }, cache: "no-store" }
      );
      if (!response.ok) return;
      const runs = (await response.json()).workflow_runs || [];
      const active = runs.find((r) => r.status === "in_progress" || r.status === "queued");

      if (!active) {
        banner.hidden = true;
        return;
      }
      const minutes = Math.floor((Date.now() - new Date(active.run_started_at)) / 60000);
      $("#live-title").textContent =
        `${active.name} #${active.run_number} is ${active.status.replace("_", " ")}`;
      $("#live-detail").textContent =
        `Started ${minutes} min ago on ${(active.head_sha || "").slice(0, 7)}. ` +
        "This page shows the last completed run until it finishes.";
      banner.href = active.html_url;
      banner.hidden = false;
    } catch {
      /* the banner is a nicety; never let it break the page */
    }
  }

  check();
  setInterval(check, 45000);
}

/* ----------------------------------------------------------- live trace */

/* The run's own event stream, pushed to the `live` branch every few seconds
 * while it happens (scripts/publish_live.py) and served from raw.github­
 * usercontent.com, which is public and sends CORS headers. Pages itself only
 * redeploys when a workflow ends, which is too late to watch anything.
 *
 * The CDN caches raw files for a few minutes; the timestamp query defeats it.
 * If it ever does not, the log lags rather than breaks. */

const TRACE_POLL_MS = 12000;
const TRACE_STALE_S = 180;   /* no heartbeat for this long: the run is over */
const TRACE_MAX_ROWS = 400;

function traceLine(record) {
  const row = el("li", `trace-row is-${record.kind}`);
  const step = record.step ? `step ${record.step}` : "";

  if (record.kind === "phase") {
    row.className = "trace-row is-phase";
    row.appendChild(el("span", "trace-what",
      record.iteration ? `iteration ${record.iteration}` : String(record.phase || "")));
    row.appendChild(el("span", "trace-body", record.text || ""));
    return row;
  }

  const marks = { text: "·", tool_use: "→", tool_result: "←", error: "!", done: "⤷" };
  row.appendChild(el("span", "trace-mark", marks[record.kind] || "·"));

  const what = el("span", "trace-what");
  if (record.kind === "tool_use") what.textContent = `${step} ${record.name}`;
  else if (record.kind === "tool_result") what.textContent = record.ok ? "ok" : "failed";
  else if (record.kind === "text") what.textContent = `${step} thinking`;
  else what.textContent = step || record.kind;
  row.appendChild(what);

  const body = el("span", "trace-body");
  if (record.kind === "tool_use") body.textContent = `(${record.text || ""})`;
  else body.textContent = record.text || "";
  row.appendChild(body);

  if (record.kind === "tool_result" && !record.ok) row.classList.add("is-fail");
  if (record.kind === "error" && record.recoverable) row.classList.add("is-retry");
  return row;
}

async function pollLiveTrace(repoUrl) {
  const match = /github\.com\/([^/]+)\/([^/]+)/.exec(repoUrl || "");
  if (!match) return;
  const [, owner, repo] = match;
  const base = `https://raw.githubusercontent.com/${owner}/${repo}/live`;
  const card = $("#trace-card");
  const list = $("#trace");
  let lastCount = -1;

  async function grab(name) {
    const response = await fetch(`${base}/${name}?ts=${Date.now()}`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.text();
  }

  async function check() {
    let body, status = {};
    try {
      body = await grab("trace.jsonl");
      try { status = JSON.parse(await grab("status.json")); } catch { /* optional */ }
    } catch {
      card.hidden = true;   /* no run has ever published, or the branch is gone */
      return;
    }

    const records = body.split("\n")
      .filter((line) => line.trim())
      .map((line) => { try { return JSON.parse(line); } catch { return null; } })
      .filter(Boolean)
      .slice(-TRACE_MAX_ROWS);
    if (!records.length) { card.hidden = true; return; }

    card.hidden = false;
    const age = (Date.now() / 1000) - (status.updated_at || records[records.length - 1].t || 0);
    const live = age < TRACE_STALE_S;
    card.classList.toggle("is-live", live);

    $("#trace-status").textContent = live
      ? `${status.phase || "running"} — ${status.activity || "working"}`
      : "The last run has finished. This is how it ended.";
    $("#trace-age").textContent = live
      ? `updated ${Math.max(0, Math.round(age))}s ago`
      : `ended ${when((status.updated_at || 0))}`;

    if (records.length !== lastCount) {
      lastCount = records.length;
      list.textContent = "";
      records.forEach((record) => list.appendChild(traceLine(record)));
      if ($("#trace-follow").checked) {
        list.scrollTop = list.scrollHeight;
      }
    }
  }

  check();
  setInterval(check, TRACE_POLL_MS);
}

fetch("data/dashboard.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  })
  .then(render)
  .catch((error) => {
    $("#empty").hidden = false;
    $("#empty").querySelector("p").textContent =
      `Could not load the dashboard data: ${error.message}`;
  });
