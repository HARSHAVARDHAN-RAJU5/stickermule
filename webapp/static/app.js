/* Page logic. It displays what the pipeline returned and nothing more —
   no verdict is computed here. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const ROUTE = {
  APPROVED:           { cls: "ok",  title: "Approved",          say: "ready to print" },
  NEEDS_CUSTOMER_FIX: { cls: "fix", title: "Back to customer",  say: "needs a new file" },
  HUMAN_REVIEW:       { cls: "rev", title: "Human review",      say: "the agent is not sure" },
  REJECTED:           { cls: "rej", title: "Rejected",          say: "the file could not be read" },
};

/* ---------------- navigation ---------------- */

const VIEWS = ["check", "samples", "results", "how"];

function show(view, push = true) {
  if (!VIEWS.includes(view)) view = "check";
  $$(".view").forEach(v => v.classList.toggle("is-active", v.id === `view-${view}`));
  $$(".nav-item").forEach(b => b.classList.toggle("is-active", b.dataset.view === view));
  if (view === "samples") loadSamples();
  if (view === "results") loadResults();
  /* Each view gets its own address, so the back button and a shared link both
     behave the way people expect them to. */
  if (push && location.hash.slice(1) !== view) location.hash = view;
  window.scrollTo(0, 0);
}

document.addEventListener("click", e => {
  const target = e.target.closest("[data-view]");
  if (target) show(target.dataset.view);
});

window.addEventListener("hashchange", () => show(location.hash.slice(1), false));

/* ---------------- upload form ---------------- */

const file = $("#file"), drop = $("#drop"), dropText = $("#drop-text"), go = $("#go");

/* No click handler here on purpose: .drop is a <label> wrapping the input, so
   the browser opens the file dialog itself. Calling file.click() as well opens
   it twice. */
["dragenter", "dragover"].forEach(t =>
  drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(t =>
  drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", e => {
  if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; onPick(); }
});
file.addEventListener("change", onPick);

function onPick() {
  const f = file.files[0];
  go.disabled = !f;
  drop.classList.toggle("has-file", !!f);
  dropText.textContent = f ? f.name : "Drop a design here, or click to choose";
}

$("#form").addEventListener("submit", async e => {
  e.preventDefault();
  if (!file.files[0]) return;

  const body = new FormData();
  body.append("artwork", file.files[0]);
  body.append("width_in", $("#w").value);
  body.append("height_in", $("#h").value);
  body.append("shape", $("#shape").value);
  body.append("material", $("#material").value);
  body.append("writer", $("#writer").value);

  await submit("/api/check", { method: "POST", body },
               URL.createObjectURL(file.files[0]));
});

$("#reset").addEventListener("click", () => {
  file.value = ""; onPick();
  $("#result").hidden = true;
  $("#empty").hidden = false;
});

/* ---------------- running a check ---------------- */

async function submit(url, options, imageUrl) {
  go.disabled = true;
  const prev = go.textContent;
  go.textContent = $("#writer").value === "template" ? "Checking…" : "Checking and writing…";
  try {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok) { alert(data.error || "Something went wrong."); return; }
    render(data, imageUrl);
    refreshTally();
  } catch (err) {
    alert("Could not reach the server. Is it still running?");
  } finally {
    go.textContent = prev;
    onPick();
  }
}

async function runSample(name) {
  show("check");
  await submit("/api/check-sample",
    { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, writer: $("#writer").value }) },
    `/api/sample-image/${name}`);
}

/* ---------------- rendering a result ---------------- */

function render(data, imageUrl) {
  const r = ROUTE[data.route] || ROUTE.REJECTED;
  const g = data.geometry || {};
  const bits = [];

  bits.push(`<div class="verdict ${r.cls}">
      <b>${r.title}</b><span>${escapeHtml(data.reason || r.say)}</span></div>`);

  const meta = [`<span>File <b>${escapeHtml(data.filename || "upload")}</b></span>`];
  if (data.dpi) meta.push(`<span>Resolution <b>${data.dpi} DPI</b></span>`);
  if (g.px_width) meta.push(`<span>Ordered <b>${data.product.width_in}&times;${data.product.height_in} in</b></span>`);
  meta.push(`<span>Checked in <b>${Math.round(data.elapsed_ms)} ms</b></span>`);
  bits.push(`<div class="meta">${meta.join("")}</div>`);

  if (imageUrl && data.route !== "REJECTED") bits.push(preview(data, imageUrl));

  if (data.findings.length) {
    bits.push(`<div class="findings">${data.findings.map(finding).join("")}</div>`);
  }

  if (data.message) bits.push(message(data.message));

  if (data.expected_route) {
    const ok = data.expected_route === data.route;
    bits.push(`<div class="expected">
      <b>${ok ? "Matches" : "Does not match"} the known answer</b> &mdash;
      this file should be <b>${ROUTE[data.expected_route].title.toLowerCase()}</b>.
      ${escapeHtml(data.note || "")}</div>`);
  }

  $("#result").innerHTML = bits.join("");
  $("#result").hidden = false;
  $("#empty").hidden = true;
}

/* The overlay is drawn in percentages, so it scales with the image without
   anyone having to know how big it is rendered. */
function preview(data, imageUrl) {
  const g = data.geometry;
  let marks = "";

  /* The dashed guide is the safe zone, which only exists on a fixed shape.
     On a die-cut order the cut path is derived from the artwork, so drawing a
     rectangle there would show a boundary that is not real. */
  if (g.px_width && g.warn_inset_px && data.product.shape !== "die_cut") {
    const px = (g.warn_inset_px / g.px_width) * 100;
    const py = (g.warn_inset_px / g.px_height) * 100;
    marks += `<rect x="${px}%" y="${py}%" width="${100 - 2 * px}%" height="${100 - 2 * py}%"
                fill="none" stroke="#2563EB" stroke-opacity=".55"
                stroke-width="1" stroke-dasharray="5 4"/>`;
  }
  for (const f of data.findings) {
    const pt = f.overlay && f.overlay.worst_point_px;
    if (!pt || !g.px_width) continue;
    const cx = (pt[0] / g.px_width) * 100, cy = (pt[1] / g.px_height) * 100;
    const colour = f.severity === "FAIL" ? "#C0342B" : "#B45309";
    marks += `<circle cx="${cx}%" cy="${cy}%" r="7" fill="none"
                stroke="${colour}" stroke-width="2"/>`;
  }

  return `<div class="preview"><img src="${imageUrl}" alt="the uploaded artwork">
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${marks}</svg></div>`;
}

function finding(f) {
  const num = f.measured === null || f.measured === undefined
    ? ""
    : `${f.measured}${f.unit ? " " + f.unit : ""}` +
      (f.threshold !== null && f.threshold !== undefined && f.severity !== "PASS"
        ? ` / ${f.threshold}` : "");
  return `<div class="finding">
      <span class="pill ${f.severity}">${f.severity}</span>
      <span><span class="name">${label(f.code)}</span><br>
        <span class="txt">${escapeHtml(f.summary)}</span></span>
      <span class="num">${num}</span>
    </div>`;
}

/* The message and how it got there. Rejected drafts are shown, not hidden:
   watching the guard throw one out is the point of the page. */
function message(m) {
  const tries = m.attempts;
  let how;
  if (m.source === "template") how = "Fixed wording, no model used.";
  else if (m.source === "model")
    how = tries.length === 1
      ? `Written by <b>${escapeHtml(m.writer)}</b>. Passed the guard first time.`
      : `Written by <b>${escapeHtml(m.writer)}</b>. The guard refused the first draft; the retry passed.`;
  else if (tries.some(a => a.error))
    how = `<b>${escapeHtml(m.writer)}</b> could not be reached, so the fixed wording was sent.`;
  else
    how = `The guard refused every draft from <b>${escapeHtml(m.writer)}</b>, so the fixed wording was sent.`;

  const refused = tries.filter(a => !a.passed).map((a, i) => `
    <details class="refused">
      <summary>Refused draft ${i + 1}${a.error ? " (error)" : ""}</summary>
      ${a.error ? `<p class="why">${escapeHtml(a.error)}</p>` : `
        <p class="draft">${escapeHtml(a.text)}</p>
        <ul class="why">${a.problems.map(p => `<li>${escapeHtml(p)}</li>`).join("")}</ul>`}
    </details>`).join("");

  const secs = tries.length
    ? ` &middot; ${(tries.reduce((t, a) => t + a.elapsed_ms, 0) / 1000).toFixed(1)} s` : "";

  return `<div class="message">
      <h4>Message to the customer</h4>
      <p class="how">${how}${secs}</p>
      <p class="text">${escapeHtml(m.text)}</p>
      ${refused}
    </div>`;
}

const NAMES = {
  RES_DPI: "Resolution",
  ALPHA_MISSING: "Missing transparency",
  ALPHA_FAKE: "Fake transparency",
  SAFE_ZONE: "Safe zone",
  DETAIL_AT_CUT: "Detail at the cut",
  STICKER_MOCKUP: "Picture of a sticker",
  CHECK_ERROR: "Check error",
};
const label = code => NAMES[code] || code.replace(/_/g, " ").toLowerCase();

/* ---------------- session tally ---------------- */

async function refreshTally() {
  const t = await (await fetch("/api/tally")).json();
  $("#s-total").textContent = t.checked;
  $("#s-ok").textContent = t.by_route.APPROVED || 0;
  $("#s-fix").textContent = t.by_route.NEEDS_CUSTOMER_FIX || 0;
  $("#s-rev").textContent = t.by_route.HUMAN_REVIEW || 0;
}

/* ---------------- samples ---------------- */

let samplesLoaded = false;
async function loadSamples() {
  if (samplesLoaded) return;
  const rows = await (await fetch("/api/samples")).json();
  $("#sample-count").textContent = `${rows.length} files`;
  $("#sample-rows").innerHTML = rows.map(s => {
    const r = ROUTE[s.expect_route];
    const size = `${s.product.width_in}&times;${s.product.height_in} in, ` +
                 s.product.shape.replace(/_/g, " ");
    return `<tr data-sample="${s.name}">
      <td class="file">${s.name}</td>
      <td>${size}</td>
      <td><span class="tag ${r.cls}">${r.title}</span></td>
      <td class="note">${escapeHtml(s.note)}</td>
      <td class="action"><span class="cap">Check &rsaquo;</span></td>
    </tr>`;
  }).join("");
  $$("#sample-rows tr").forEach(tr =>
    tr.addEventListener("click", () => runSample(tr.dataset.sample)));
  samplesLoaded = true;
}

/* ---------------- results ---------------- */

let resultsLoaded = false;
async function loadResults() {
  if (resultsLoaded) return;
  const { markdown } = await (await fetch("/api/results")).json();
  $("#results-body").innerHTML = markdown
    ? miniMarkdown(markdown)
    : `<p class="cap">No results yet. Run <code>python -m evalkit run --write docs/results.md</code>.</p>`;
  resultsLoaded = true;
}

/* Just enough markdown for the generated report: headings, tables, lists,
   bold, italic and code. Not a general parser, and not pretending to be one.
   The one rule that matters: a block continues across wrapped lines until a
   blank line or the start of a new block, otherwise every wrapped sentence
   becomes its own paragraph and bold spanning two lines never closes. */
function miniMarkdown(md) {
  const out = [];
  let table = null;   // rows of the table being collected
  let items = null;   // list items being collected
  let buf = [];       // lines of the current paragraph or list item

  const text = () => buf.splice(0).join(" ").trim();
  const flush = () => {
    const t = text();
    if (!t) return;
    if (items) items.push(`<li>${inline(t)}</li>`);
    else out.push(`<p>${inline(t)}</p>`);
  };
  const closeList = () => {
    flush();
    if (items) { out.push(`<ul>${items.join("")}</ul>`); items = null; }
  };
  const closeTable = () => {
    if (table) { out.push(renderTable(table)); table = null; }
  };

  for (const raw of md.split("\n")) {
    const line = raw.trim();

    if (line.startsWith("|")) {
      closeList(); 
      const cells = line.split("|").slice(1, -1).map(c => c.trim());
      if (!cells.every(c => /^:?-+:?$/.test(c))) (table ??= []).push(cells);
      continue;
    }
    closeTable();

    if (!line) { closeList(); continue; }

    const heading = line.match(/^(#{2,4})\s+(.*)$/);
    if (heading) {
      closeList();
      out.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`);
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.*)$/);
    if (bullet) {
      flush();
      items ??= [];
      buf.push(bullet[1]);
      continue;
    }

    buf.push(line);   // a wrapped continuation of whatever is open
  }
  closeList(); closeTable();
  return out.join("");
}

function renderTable(rows) {
  const [head, ...body] = rows;
  return `<table><thead><tr>${head.map(c => `<th>${inline(c)}</th>`).join("")}</tr></thead>
    <tbody>${body.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

function inline(s) {
  return escapeHtml(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>")
    .replace(/(^|[\s(])\*([^*\s][^*]*)\*/g, "$1<i>$2</i>");
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

/* ---------------- message writers ---------------- */

/* A writer that cannot run right now (no key, Ollama not started) is shown
   but disabled, so nobody picks it and silently gets the fallback. */
async function loadWriters() {
  try {
    const w = await (await fetch("/api/writers")).json();
    for (const opt of $$("#writer option")) {
      const info = w[opt.value];
      if (!info) continue;
      opt.textContent = info.label + (info.ok ? "" : " (not set up)");
      opt.disabled = !info.ok;
    }
  } catch { /* the template still works */ }
  try {
    const saved = localStorage.getItem("writer");
    const opt = saved && $(`#writer option[value="${saved}"]`);
    if (opt && !opt.disabled) $("#writer").value = saved;
  } catch { /* storage blocked */ }
}
$("#writer").addEventListener("change", () => {
  try { localStorage.setItem("writer", $("#writer").value); } catch {}
});

/* ---------------- start ---------------- */
show(location.hash.slice(1) || "check", false);
refreshTally();
loadWriters();
