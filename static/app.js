/**
 * app.js — FakeNewsDB Frontend Logic
 * ====================================
 * All Oracle SQL queries and ML predictions go through the Flask backend.
 * No mock data or local JS engine — every result is real.
 *
 * Public functions called from index.html:
 *   runQuery()           — POST to /query, render result table
 *   predictNews()        — POST to /predict, render verdict
 *   compareAllModels()   — Run all 4 models and render comparison table
 *   clearQuery()         — Reset SQL section
 *   clearPredict()       — Reset prediction section
 *   setQuery(sql)        — Populate SQL textarea (quick buttons)
 *   setSample(type)      — Populate news textarea with sample text
 *   loadSchema(table)    — GET /schema/<table>, render column pills
 *   exportCSV()          — Download current result table as CSV
 *   loadPerformance()    — GET model_results from Oracle, render cards
 *   selectModel(el)      — Toggle active model pill
 *   updateCharCount()    — Update news textarea char counter
 *   updateSqlCount()     — Update SQL textarea char counter
 */

"use strict";

/* ── Module state ────────────────────────────────────────────── */
let _lastData    = null;   // { columns, data } used for CSV export
let _activeModel = "NaiveBayes";

/* ── Sample news texts ──────────────────────────────────────── */
const SAMPLES = {
  real: `The Federal Reserve raised its benchmark interest rate by 0.25 percentage points on Wednesday, bringing it to the highest level in 16 years. Fed Chair Jerome Powell said the central bank remains committed to returning inflation to its 2% target while carefully monitoring economic data.`,

  fake: `BREAKING: Scientists have discovered that drinking bleach cures COVID-19 and the mainstream media is suppressing the information. A whistleblower from the CDC claims the government has been hiding this cure for years to protect pharmaceutical companies.`,

  political: `The Senate passed a bipartisan infrastructure bill 69-30 on Tuesday. The $1.2 trillion package includes funding for roads, bridges, broadband internet, and public transit. President Biden signed the legislation at the White House the following week.`,

  health: `A new study published in The Lancet found that regular moderate exercise — around 150 minutes per week — significantly reduces the risk of cardiovascular disease, type 2 diabetes, and certain cancers. Researchers followed 90,000 participants over 10 years.`,
};

/* ── API helpers ─────────────────────────────────────────────── */
async function apiPost(endpoint, body) {
  try {
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (err) {
    return { error: "Network error: " + err.message };
  }
}

async function apiGet(endpoint) {
  try {
    const res = await fetch(endpoint);
    return await res.json();
  } catch (err) {
    return { error: "Network error: " + err.message };
  }
}

async function apiPut(endpoint, body) {
  try {
    const res = await fetch(endpoint, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return await res.json();
  } catch (err) {
    return { error: "Network error: " + err.message };
  }
}

async function apiDelete(endpoint) {
  try {
    const res = await fetch(endpoint, { method: "DELETE" });
    return await res.json();
  } catch (err) {
    return { error: "Network error: " + err.message };
  }
}

/* ── Status helpers ─────────────────────────────────────────── */
function showStatus(id, msg, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = msg;
  el.className = "status-bar " + type;
}

function hideStatus(id) {
  const el = document.getElementById(id);
  if (el) el.className = "status-bar hidden";
}

/* ── Initialisation on page load ────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadTableList();
  loadModels();
  loadPerformance();
  fetchCustomNews();
});

/* ── Health check (Oracle + ML models) ──────────────────────── */
async function checkHealth() {
  const dot    = document.getElementById("badge-dot");
  const txt    = document.getElementById("db-status-text");
  const badge  = document.getElementById("db-badge");

  const data = await apiGet("/health");

  if (data.error) {
    dot.className  = "badge-dot error";
    txt.textContent = "Connection error";
    badge.title    = data.error;
    return;
  }

  if (data.database && data.database.connected) {
    dot.className  = "badge-dot connected";
    txt.textContent = "Oracle connected";
  } else {
    dot.className  = "badge-dot error";
    txt.textContent = "Oracle offline";
    badge.title    = (data.database || {}).message || "DB unavailable";
  }

  const modelsEl = document.getElementById("stat-models");
  if (modelsEl && data.ml_models) {
    modelsEl.textContent = data.ml_models.count + "/4";
  }

  // Mark unavailable model pills
  if (data.ml_models && data.ml_models.loaded) {
    const loaded = new Set(data.ml_models.loaded);
    document.querySelectorAll(".mpill").forEach(btn => {
      const m = btn.dataset.model;
      if (m && !loaded.has(m)) {
        btn.classList.add("unavailable");
        btn.title = m + " not loaded — run train_models.py";
      }
    });
  }
}

/* ── Load table list and stats ───────────────────────────────── */
async function loadTableList() {
  const data = await apiGet("/tables");
  if (data.error) return;

  const tables = data.tables || [];
  const statT  = document.getElementById("stat-tables");
  const statR  = document.getElementById("stat-records");
  if (statT) statT.textContent = tables.length;

  let total = 0;
  tables.forEach(t => { if (t.rows > 0) total += t.rows; });
  if (statR) statR.textContent = total > 0 ? total.toLocaleString() : "—";
}

/* ── Load model list ─────────────────────────────────────────── */
async function loadModels() {
  const data = await apiGet("/models");
  if (!data.models) return;
  // Update model pills to show loaded/unloaded state
  const loadedSet = new Set(data.models.filter(m => m.loaded).map(m => m.name));
  const statM = document.getElementById("stat-models");
  if (statM) statM.textContent = loadedSet.size + "/4";
}

/* ── SQL QUERY SECTION ───────────────────────────────────────── */

/** Populate the SQL textarea from a quick button */
function setQuery(sql) {
  const ta = document.getElementById("sql-input");
  if (!ta) return;
  ta.value = sql;
  updateSqlCount();
  ta.focus();
}

/** Update SQL char counter */
function updateSqlCount() {
  const ta = document.getElementById("sql-input");
  const cc = document.getElementById("sql-char-count");
  if (ta && cc) cc.textContent = ta.value.length + " chars";
}

/** Clear SQL section */
function clearQuery() {
  const ta = document.getElementById("sql-input");
  if (ta) { ta.value = ""; updateSqlCount(); }
  document.getElementById("query-results").classList.add("hidden");
  hideStatus("query-status");
  document.getElementById("export-btn").disabled = true;
  document.getElementById("stat-rows").textContent = "—";
  _lastData = null;
}

/** Run the SQL query against Oracle via /query */
async function runQuery() {
  const sql   = (document.getElementById("sql-input").value || "").trim();
  const limit = parseInt(document.getElementById("row-limit").value) || 100;

  if (!sql) {
    showStatus("query-status", "Please enter a SQL query.", "error");
    return;
  }

  showStatus("query-status", "Executing on Oracle…", "loading");
  document.getElementById("query-results").classList.add("hidden");
  document.getElementById("export-btn").disabled = true;

  const data = await apiPost("/query", { sql, limit });

  if (data.error) {
    showStatus("query-status", "Error: " + data.error, "error");
    return;
  }

  hideStatus("query-status");
  renderTable(data.columns, data.data);
  document.getElementById("stat-rows").textContent = data.row_count.toLocaleString();
  _lastData = { columns: data.columns, data: data.data };
  document.getElementById("export-btn").disabled = false;
}

/** Render query results into the table */
function renderTable(columns, rows) {
  const head = document.getElementById("result-head");
  const body = document.getElementById("result-body");
  const wrapper = document.getElementById("query-results");
  const countEl  = document.getElementById("results-count");

  // Header
  head.innerHTML = "<tr>" + columns.map(c =>
    "<th title='" + c + "'>" + c + "</th>"
  ).join("") + "</tr>";

  // Body
  body.innerHTML = rows.map(row =>
    "<tr>" + row.map((cell, i) => {
      const col = (columns[i] || "").toLowerCase();
      let cls = "";
      const cellStr = String(cell);
      if (col === "label" || col === "binary_label") {
        cls = cellStr.toUpperCase() === "FAKE" ? "cell-fake" :
              cellStr.toUpperCase() === "REAL" ? "cell-real" : "";
      } else if (!isNaN(cell) && cell !== "") {
        cls = "cell-num";
      }
      return "<td class='" + cls + "' title='" + escHtml(cellStr) + "'>"
             + escHtml(truncate(cellStr, 80)) + "</td>";
    }).join("") + "</tr>"
  ).join("");

  countEl.textContent = rows.length.toLocaleString() + " rows";
  wrapper.classList.remove("hidden");
}

/* ── Schema browser ─────────────────────────────────────────── */
async function loadSchema(table) {
  const display = document.getElementById("schema-display");
  if (!table || !display) return;
  display.textContent = "Loading…";

  const data = await apiGet("/schema/" + table);
  if (data.error) {
    display.innerHTML = "<span style='color:var(--red);font-size:.78rem'>" + data.error + "</span>";
    return;
  }

  display.innerHTML = (data.columns || []).map(col =>
    "<span class='schema-pill'>" + col.name.toLowerCase() +
    "<span class='pill-type'>" + col.type + "</span></span>"
  ).join("");
}

/* ── NEWS CLASSIFIER SECTION ─────────────────────────────────── */

/** Select active model pill */
function selectModel(btn) {
  document.querySelectorAll(".mpill").forEach(p => p.classList.remove("active"));
  btn.classList.add("active");
  _activeModel = btn.dataset.model;
}

/** Populate news textarea with sample text */
function setSample(type) {
  const ta = document.getElementById("news-input");
  if (ta) {
    ta.value = SAMPLES[type] || "";
    updateCharCount();
  }
}

/** Update news textarea char counter */
function updateCharCount() {
  const ta = document.getElementById("news-input");
  const cc = document.getElementById("news-char-count");
  if (ta && cc) {
    const len = ta.value.length;
    cc.textContent = len.toLocaleString() + " / 5000 chars";
    cc.style.color = len > 4800 ? "var(--red)" : "";
  }
}

/** Clear prediction section */
function clearPredict() {
  const ta = document.getElementById("news-input");
  if (ta) { ta.value = ""; updateCharCount(); }
  document.getElementById("predict-result").classList.add("hidden");
  document.getElementById("predict-placeholder").classList.remove("hidden");
  document.getElementById("single-result").classList.remove("hidden");
  document.getElementById("compare-result").classList.add("hidden");
  hideStatus("predict-status");
}

/** Classify news text with selected model */
async function predictNews() {
  const text  = (document.getElementById("news-input").value || "").trim();
  const model = _activeModel;

  if (!text) {
    showStatus("predict-status", "Please enter some news text first.", "error");
    return;
  }

  showStatus("predict-status", "Running " + model + " model…", "loading");
  document.getElementById("predict-placeholder").classList.add("hidden");
  document.getElementById("predict-result").classList.add("hidden");
  document.getElementById("single-result").classList.remove("hidden");
  document.getElementById("compare-result").classList.add("hidden");

  const data = await apiPost("/predict", { text, model });

  if (data.error) {
    showStatus("predict-status", "Error: " + data.error + (data.hint ? " — " + data.hint : ""), "error");
    return;
  }

  hideStatus("predict-status");
  renderVerdict(data);
}

/** Compare all 4 models in parallel */
async function compareAllModels() {
  const text = (document.getElementById("news-input").value || "").trim();
  if (!text) {
    showStatus("predict-status", "Please enter some news text first.", "error");
    return;
  }

  showStatus("predict-status", "Running all 4 models in parallel…", "loading");
  document.getElementById("predict-placeholder").classList.add("hidden");
  document.getElementById("predict-result").classList.add("hidden");
  document.getElementById("single-result").classList.add("hidden");
  document.getElementById("compare-result").classList.remove("hidden");

  const models = ["NaiveBayes", "LogReg", "BiLSTM", "BERT"];
  const promises = models.map(m => apiPost("/predict", { text, model: m }));
  const results  = await Promise.all(promises);

  const tbody = document.getElementById("compare-body");
  tbody.innerHTML = results.map((r, i) => {
    if (r.error) {
      return "<tr><td>" + models[i] + "</td><td colspan='2' style='color:var(--text2)'>"
             + r.error + "</td></tr>";
    }
    const cls  = r.prediction === "FAKE" ? "cmp-fake" : "cmp-real";
    const bar  = buildMiniBar(r.confidence);
    return "<tr><td>" + models[i] + "</td>"
         + "<td class='" + cls + "'>" + r.prediction + "</td>"
         + "<td>" + bar + " " + r.confidence + "%</td></tr>";
  }).join("");

  document.getElementById("predict-result").classList.remove("hidden");
  hideStatus("predict-status");
}

/** Render the verdict badge + confidence bar */
function renderVerdict(data) {
  const isFake = data.prediction === "FAKE";

  const badge = document.getElementById("verdict-badge");
  badge.className = "verdict-badge " + (isFake ? "fake" : "real");
  document.getElementById("verdict-label").textContent = data.prediction;

  const conf = data.confidence || 0;
  document.getElementById("conf-value").textContent = conf.toFixed(1) + "%";

  const bar = document.getElementById("conf-bar");
  bar.style.width = conf + "%";
  bar.className = "conf-bar" +
    (conf >= 85 ? " high" : conf >= 65 ? " medium" : " low");

  document.getElementById("result-model").textContent = data.model || "—";

  document.getElementById("predict-result").classList.remove("hidden");
}

/** Build a tiny inline progress bar (for compare table) */
function buildMiniBar(pct) {
  const w = Math.min(100, pct);
  const color = w >= 85 ? "var(--green)" : w >= 65 ? "var(--yellow)" : "var(--red)";
  return "<span style='display:inline-block;width:60px;height:7px;background:#21262d;"
       + "border-radius:4px;vertical-align:middle;margin-right:4px'>"
       + "<span style='display:block;width:" + w + "%;height:100%;background:" + color
       + ";border-radius:4px'></span></span>";
}

/* ── PERFORMANCE DASHBOARD ───────────────────────────────────── */
async function loadPerformance() {
  const placeholder = document.getElementById("perf-placeholder");
  const content     = document.getElementById("perf-content");
  const grid        = document.getElementById("perf-grid");

  if (placeholder) placeholder.classList.add("hidden");
  if (content)     content.classList.add("hidden");

  const data = await apiPost("/query", {
    sql: "SELECT model_name, dataset_name, accuracy, f1_score, precision_score, recall_score " +
         "FROM model_results ORDER BY accuracy DESC",
    limit: 50,
  });

  if (data.error || !data.data || data.data.length === 0) {
    if (placeholder) {
      placeholder.innerHTML =
        "<span class='placeholder-icon'>&#128202;</span>" +
        "<p style='color:var(--text2)'>" +
        (data.error ? data.error : "No performance data found. Run load_data.py first.") +
        "</p>";
      placeholder.classList.remove("hidden");
    }
    return;
  }

  const cols = data.columns;
  grid.innerHTML = data.data.map(row => {
    const obj = {};
    cols.forEach((c, i) => { obj[c] = row[i]; });

    const acc = parseFloat(obj.accuracy  || obj.ACCURACY  || 0);
    const f1  = parseFloat(obj.f1_score  || obj.F1_SCORE  || 0);
    const pre = parseFloat(obj.precision_score || obj.PRECISION_SCORE || 0);
    const rec = parseFloat(obj.recall_score    || obj.RECALL_SCORE    || 0);
    const model   = obj.model_name   || obj.MODEL_NAME   || "?";
    const dataset = obj.dataset_name || obj.DATASET_NAME || "?";

    return "<div class='perf-card'>"
      + "<div class='perf-model-name'>" + model + "</div>"
      + "<div class='perf-dataset'>Dataset: " + dataset + "</div>"
      + "<div class='perf-metric'><span class='perf-metric-label'>Accuracy</span>"
        + "<span class='perf-metric-val'>" + (acc * 100).toFixed(2) + "%</span></div>"
      + "<div class='perf-metric'><span class='perf-metric-label'>F1 Score</span>"
        + "<span class='perf-metric-val'>" + (f1 * 100).toFixed(2) + "%</span></div>"
      + "<div class='perf-metric'><span class='perf-metric-label'>Precision</span>"
        + "<span class='perf-metric-val'>" + (pre * 100).toFixed(2) + "%</span></div>"
      + "<div class='perf-metric'><span class='perf-metric-label'>Recall</span>"
        + "<span class='perf-metric-val'>" + (rec * 100).toFixed(2) + "%</span></div>"
      + "</div>";
  }).join("");

  if (content) content.classList.remove("hidden");
}

/* ── CSV Export ──────────────────────────────────────────────── */
function exportCSV() {
  if (!_lastData) return;
  const { columns, data } = _lastData;

  const rows = [columns.join(",")].concat(
    data.map(row => row.map(cell => {
      const s = String(cell).replace(/"/g, '""');
      return s.includes(",") || s.includes("\n") || s.includes('"') ? '"' + s + '"' : s;
    }).join(","))
  );

  const blob = new Blob([rows.join("\n")], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href     = url;
  a.download = "fakenewsdb_query.csv";
  a.click();
  URL.revokeObjectURL(url);
}

/* ── Utilities ───────────────────────────────────────────────── */
function escHtml(str) {
  return String(str)
    .replace(/&/g,  "&amp;")
    .replace(/</g,  "&lt;")
    .replace(/>/g,  "&gt;")
    .replace(/"/g,  "&quot;")
    .replace(/'/g,  "&#39;");
}

function truncate(str, maxLen) {
  return str.length > maxLen ? str.slice(0, maxLen) + "…" : str;
}

/* ── CUSTOM DATASET MANAGER (CRUD) ───────────────────────────── */

async function fetchCustomNews() {
  const tbody = document.getElementById("crud-table-body");
  if (!tbody) return;
  tbody.innerHTML = "<tr><td colspan='6' style='text-align:center'>Loading...</td></tr>";
  
  const data = await apiGet("/api/news");
  if (data.error || !data.data) {
    tbody.innerHTML = "<tr><td colspan='6' style='text-align:center; color:var(--red)'>" + escHtml(data.error || "Failed to load") + "</td></tr>";
    return;
  }

  if (data.data.length === 0) {
    tbody.innerHTML = "<tr><td colspan='6' style='text-align:center; color:var(--text2)'>No custom records found. Create one above!</td></tr>";
    return;
  }

  tbody.innerHTML = data.data.map(item => {
    // Escape string keys for proper parsing in onclick handlers
    const objStr = JSON.stringify(item)
                    .replace(/\\/g, "\\\\")
                    .replace(/'/g, "\\'")
                    .replace(/"/g, "&quot;");
                    
    return "<tr>"
      + "<td>" + item.id + "</td>"
      + "<td title='" + escHtml(item.title) + "'>" + escHtml(truncate(item.title, 40)) + "</td>"
      + "<td>" + escHtml(item.subject) + "</td>"
      + "<td><span class='verdict-badge " + (item.label === 'FAKE' ? "fake" : "real") + "' style='font-size:0.6rem; padding: 2px 6px; position:static'>" + item.label + "</span></td>"
      + "<td>" + item.created_at + "</td>"
      + "<td style='text-align:center; min-width: 120px'>"
        + "<button class='btn btn-ghost btn-sm' style='padding: 2px 8px; font-size: 0.75rem' onclick='editCustomNews(\"" + objStr + "\")'>Edit</button> "
        + "<button class='btn btn-ghost btn-sm' style='padding: 2px 8px; font-size: 0.75rem; color: var(--red)' onclick='deleteCustomNews(" + item.id + ")'>Delete</button>"
      + "</td>"
    + "</tr>";
  }).join("");
}

function clearCrudForm() {
  document.getElementById("crud-id").value = "";
  document.getElementById("crud-title").value = "";
  document.getElementById("crud-subject").value = "";
  document.getElementById("crud-text").value = "";
  document.getElementById("crud-label").value = "FAKE";
  hideStatus("crud-status");
}

function editCustomNews(itemStr) {
  try {
    const item = JSON.parse(itemStr.replace(/&quot;/g, '"').replace(/\\'/g, "'").replace(/\\\\/g, "\\"));
    clearCrudForm();
    document.getElementById("crud-id").value = item.id;
    document.getElementById("crud-title").value = item.title || "";
    document.getElementById("crud-subject").value = item.subject || "";
    document.getElementById("crud-text").value = item.text || "";
    document.getElementById("crud-label").value = item.label || "FAKE";
    
    // Smooth scroll back to form
    document.getElementById("crud-section").scrollIntoView({ behavior: 'smooth' });
  } catch (e) {
    console.error("Failed to parse row data: ", e);
  }
}

async function saveCustomNews() {
  const id = document.getElementById("crud-id").value;
  const title = document.getElementById("crud-title").value.trim();
  const subject = document.getElementById("crud-subject").value.trim();
  const text = document.getElementById("crud-text").value.trim();
  const label = document.getElementById("crud-label").value;

  if (!title || !text) {
    showStatus("crud-status", "Title and Content are required.", "error");
    return;
  }

  showStatus("crud-status", "Saving record...", "loading");
  
  const payload = { title, text, subject, label };
  let res;
  
  if (id) {
    res = await apiPut("/api/news/" + id, payload);
  } else {
    res = await apiPost("/api/news", payload);
  }

  if (res.error) {
    showStatus("crud-status", "Error: " + res.error, "error");
  } else {
    showStatus("crud-status", "Record saved successfully!", "connected");
    clearCrudForm();
    await fetchCustomNews();
    setTimeout(() => hideStatus("crud-status"), 3000);
  }
}

async function deleteCustomNews(id) {
  if (!confirm("Are you sure you want to delete record #" + id + "?")) return;
  
  showStatus("crud-status", "Deleting record...", "loading");
  const res = await apiDelete("/api/news/" + id);
  if (res.error) {
    showStatus("crud-status", "Error: " + res.error, "error");
  } else {
    showStatus("crud-status", "Record deleted.", "connected");
    await fetchCustomNews();
    setTimeout(() => hideStatus("crud-status"), 3000);
  }
}
