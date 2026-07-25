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

  if (!summary.attempts && !scorecard) {
    $("#empty").hidden = false;
    return;
  }
  $("#content").hidden = false;

  drawChart(data.series || []);
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
