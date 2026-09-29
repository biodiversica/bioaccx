/* The results explorer: what a run left behind, made navigable.
 *
 * Everything here reads artifacts already on disk — metadata, the evaluation
 * table, the UMAP projection, the dataset list — so it works on models trained
 * long before this page existed. Nothing is recomputed.
 */

import { t } from "/static/i18n.js";

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) if (child != null) node.append(child);
  return node;
};

/* Colours come with the projection: the server generates them for the number
 * of classes actually present (see bioaccx/umap.py), which is also what the
 * PNG plots use — so a class is the same colour here and in the figure on
 * disk, and there is no second palette here to run out or drift. */
const colourOf = (colours, index) => colours[index] ?? "#888888";

const fixed = (value, places = 4) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(places);

export function initExplorer({ api, status }) {
  const ui = {
    list: $("model-list"), dir: $("models-dir"), browse: $("models-browse"),
    title: $("detail-title"),
    metrics: $("panel-metrics"), map: $("panel-map"), compare: $("panel-compare"),
    viewMetrics: $("view-metrics"), viewMap: $("view-map"), viewCompare: $("view-compare"),
    canvas: $("map-canvas"), legend: $("map-legend"), note: $("map-note"),
    colour: $("map-colour"), split: $("map-split"),
    sample: $("map-sample"), sampleLabel: $("sample-label"),
    sampleFile: $("sample-file"), audio: $("sample-audio"), spec: $("sample-spec"),
    compareWith: $("compare-with"), compareBody: $("compare-body"),
  };

  const state = {
    models: [], stem: null, detail: null, umap: null,
    view: "metrics", sort: { column: "f1", ascending: true }, drawn: [],
  };

  /* The directory being browsed travels with every request rather than being
   * held by the server, so two tabs can look at two collections and a reloaded
   * page comes back where it was. Empty means the one the editor was started
   * with. */
  const dirParam = () => {
    const chosen = ui.dir.value.trim();
    return chosen ? `dir=${encodeURIComponent(chosen)}` : "";
  };
  const withDir = (url) => {
    const param = dirParam();
    if (!param) return url;
    return url + (url.includes("?") ? "&" : "?") + param;
  };

  /* ── model list ────────────────────────────────────────────────────── */

  async function loadModels() {
    try {
      const body = await api(withDir("/api/models"));
      state.models = body.models;
      ui.dir.title = body.models_dir;          // the resolved absolute path
      renderList();
      if (!state.models.length) {
        ui.metrics.replaceChildren(
          el("p", { className: "empty",
                    textContent: t("models.empty", { dir: body.models_dir }) }));
      }
    } catch (error) {
      status(error.message, "bad");
      ui.list.replaceChildren(el("li", {}, el("p", { className: "empty", textContent: error.message })));
    }
  }

  function renderList() {
    ui.list.replaceChildren();
    for (const model of state.models) {
      const card = el("button", { type: "button", className: "model-card" });
      if (model.stem === state.stem) card.classList.add("is-active");
      card.append(el("span", { className: "name", textContent: model.stem }));

      const meta = el("div", { className: "meta" });
      if (model.kind !== "model") {
        // An embeddings or dataset run has no classifier; saying so up front
        // explains the missing F1 rather than leaving it looking broken.
        meta.append(el("span", { className: `kind kind-${model.kind}`,
                                 textContent: model.kind }));
      }
      meta.append(el("span", { textContent: model.backbone }));
      if (model.n_classes) {
        meta.append(el("span", { textContent: t("models.classes", { count: model.n_classes }) }));
      }
      if (model.macro.f1 !== null) {
        meta.append(el("span", { className: "f1", textContent: `F1 ${fixed(model.macro.f1)}` }));
      }
      if (model.has_umap) meta.append(el("span", { textContent: t("models.has_map") }));
      if (model.created_at) meta.append(el("span", { textContent: model.created_at.slice(0, 10) }));
      card.append(meta);

      card.addEventListener("click", () => selectModel(model.stem));
      ui.list.append(el("li", {}, card));
    }
  }

  async function selectModel(stem) {
    state.stem = stem;
    renderList();
    ui.title.textContent = stem;
    try {
      state.detail = await api(withDir(`/api/models/${encodeURIComponent(stem)}`));
      renderMetrics();
      fillCompareOptions();
      state.umap = null;
      // A run with no evaluation has nothing on the metrics tab, so open the
      // projection it does have instead of an empty panel.
      if (state.view === "metrics" && !state.detail.has_evaluation
          && state.detail.has_umap) {
        setView("map");
        return;
      }
      if (state.view === "map") await showMap();
      if (state.view === "compare") await showCompare();
    } catch (error) {
      status(error.message, "bad");
    }
  }

  /* ── metrics ───────────────────────────────────────────────────────── */

  const COLUMNS = ["label", "precision", "recall", "f1", "f1_opt",
                   "auprc", "auroc", "threshold", "samples"];
  const columnTitle = (key) => t(`metrics.col_${key}`);

  function renderMetrics() {
    const model = state.detail;
    ui.metrics.replaceChildren();
    if (!model) return;

    const grid = el("div", { className: "summary-grid" });
    const facts = {
      backbone: `${model.backbone} (${model.backbone_format || "?"})`,
      classifier: model.classifier,
      classes: model.n_classes,
      train: model.n_train, test: model.n_test,
      created: model.created_at,
      excluded: (model.excluded_labels || []).join(", ") || "—",
      directory: model.dir,
    };
    for (const [key, value] of Object.entries(facts)) {
      if (value === undefined || value === null) continue;
      grid.append(el("b", { textContent: t(`metrics.fact_${key}`) }));
      grid.append(el("span", { textContent: String(value) }));
    }
    ui.metrics.append(grid);

    if (!model.evaluation.length) {
      ui.metrics.append(el("p", {
        className: "empty",
        textContent: model.kind === "embeddings" ? t("metrics.empty_embeddings")
          : model.kind === "dataset" ? t("metrics.empty_dataset")
            : t("metrics.empty_none"),
      }));
      if (model.report) ui.metrics.append(el("pre", { className: "run-log", textContent: model.report }));
      return;
    }

    const table = el("table", { className: "metrics-table" });
    const head = el("tr");
    for (const key of COLUMNS) {
      const title = columnTitle(key);
      const th = el("th", { textContent: title, title: t("metrics.sort", { column: title }) });
      th.addEventListener("click", () => {
        state.sort = {
          column: key,
          ascending: state.sort.column === key ? !state.sort.ascending : true,
        };
        renderMetrics();
      });
      head.append(th);
    }
    table.append(el("thead", {}, head));

    const overall = model.evaluation.filter((row) => row.overall);
    const classes = model.evaluation.filter((row) => !row.overall);
    const { column, ascending } = state.sort;
    classes.sort((a, b) => {
      const x = a[column], y = b[column];
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      const order = typeof x === "string" ? x.localeCompare(y) : x - y;
      return ascending ? order : -order;
    });

    const body = el("tbody");
    for (const row of [...overall, ...classes]) {
      const tr = el("tr");
      if (row.overall) tr.className = "overall";
      for (const key of COLUMNS) {
        const value = row[key];
        const text = key === "label" ? value
          : key === "samples" ? (value === null ? "—" : String(value))
          : fixed(value);
        const td = el("td", { textContent: text });
        // Anything under half is worth the reader's eye before anything else.
        if (key === "f1" && value !== null && value < 0.5 && !row.overall) td.className = "weak";
        tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);
    ui.metrics.append(table);
  }

  /* ── embedding map ─────────────────────────────────────────────────── */

  async function showMap() {
    if (!state.stem) return;
    if (!state.umap) {
      try {
        state.umap = await api(withDir(`/api/models/${encodeURIComponent(state.stem)}/umap`));
      } catch (error) {
        ui.legend.replaceChildren();
        ui.note.textContent = error.message;
        clearCanvas();
        return;
      }
    }
    ui.note.textContent = state.umap.key_source === "row-order" ? t("map.note_row_order")
      : state.umap.key_source === "none" ? t("map.note_none")
        : "";
    drawMap();
  }

  function clearCanvas() {
    const ctx = ui.canvas.getContext("2d");
    ctx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
  }

  function groupsOf(umap) {
    return ui.colour.value === "cluster"
      ? umap.clusters.map(String)
      : umap.labels;
  }

  const coloursOf = (umap) =>
    (ui.colour.value === "cluster" ? umap.cluster_colors : umap.label_colors) ?? [];

  function groupOf(point) {
    return ui.colour.value === "cluster"
      ? String(point.cluster === null ? "" : Math.trunc(point.cluster))
      : point.label;
  }

  function drawMap() {
    const umap = state.umap;
    if (!umap) return;
    const canvas = ui.canvas;
    const ctx = canvas.getContext("2d");
    const wanted = ui.split.value;
    const points = umap.points.filter((p) => !wanted || p.split === wanted);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!points.length) {
      state.drawn = [];
      return;
    }

    const xs = points.map((p) => p.x), ys = points.map((p) => p.y);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = 18;
    const scaleX = (canvas.width - pad * 2) / (maxX - minX || 1);
    const scaleY = (canvas.height - pad * 2) / (maxY - minY || 1);

    const groups = groupsOf(umap);
    const colours = coloursOf(umap);
    const index = new Map(groups.map((name, i) => [name, i]));

    state.drawn = points.map((point) => {
      const x = pad + (point.x - minX) * scaleX;
      // Canvas y grows downward; flip so the projection reads the usual way up.
      const y = canvas.height - pad - (point.y - minY) * scaleY;
      ctx.fillStyle = colourOf(colours, index.get(groupOf(point)) ?? 0);
      ctx.globalAlpha = 0.75;
      ctx.beginPath();
      ctx.arc(x, y, 2.6, 0, Math.PI * 2);
      ctx.fill();
      return { x, y, point };
    });
    ctx.globalAlpha = 1;

    ui.legend.replaceChildren();
    for (const [i, name] of groups.entries()) {
      ui.legend.append(el("span", {}, [
        el("i", { style: `background:${colourOf(colours, i)}` }),
        document.createTextNode(name || "—"),
      ]));
    }
  }

  ui.canvas.addEventListener("click", async (event) => {
    if (!state.drawn.length) return;
    const box = ui.canvas.getBoundingClientRect();
    const scale = ui.canvas.width / box.width;
    const cx = (event.clientX - box.left) * scale;
    const cy = (event.clientY - box.top) * scale;

    let best = null, bestDistance = Infinity;
    for (const item of state.drawn) {
      const distance = (item.x - cx) ** 2 + (item.y - cy) ** 2;
      if (distance < bestDistance) { bestDistance = distance; best = item; }
    }
    if (!best || bestDistance > 400) return;      // 20px, in canvas units
    await showSample(best.point);
  });

  async function showSample(point) {
    const stem = encodeURIComponent(state.stem);
    if (!point.key) {
      status(t("status.no_key"), "bad");
      return;
    }
    const query = `key=${encodeURIComponent(point.key)}`;
    try {
      const info = await api(withDir(`/api/models/${stem}/sample?${query}`));
      ui.sample.hidden = false;
      ui.sampleLabel.textContent =
        `${point.label}${point.split ? ` · ${point.split}` : ""}`;
      ui.sampleFile.textContent = info.exists
        ? t("map.clip_range", { filename: info.filename,
                                start: fixed(info.start_time, 2),
                                end: fixed(info.end_time, 2) })
        : t("map.clip_missing", { path: info.filepath });
      ui.sampleFile.title = info.filepath;

      if (!info.exists) {
        // The dataset may live on another machine, or have moved since training.
        ui.audio.removeAttribute("src");
        ui.spec.removeAttribute("src");
        return;
      }
      ui.audio.src = withDir(`/api/models/${stem}/clip?${query}`);
      ui.spec.src = withDir(`/api/models/${stem}/spectrogram?${query}`);
    } catch (error) {
      status(error.message, "bad");
    }
  }

  ui.colour.addEventListener("change", drawMap);
  ui.split.addEventListener("change", drawMap);

  /* ── compare ───────────────────────────────────────────────────────── */

  function fillCompareOptions() {
    ui.compareWith.replaceChildren(el("option", { value: "", textContent: t("compare.pick") }));
    for (const model of state.models) {
      if (model.stem === state.stem) continue;
      ui.compareWith.append(el("option", { value: model.stem, textContent: model.stem }));
    }
  }

  async function showCompare() {
    const other = ui.compareWith.value;
    ui.compareBody.replaceChildren();
    if (!state.stem || !other) {
      ui.compareBody.append(el("p", {
        className: "empty", textContent: t("compare.empty"),
      }));
      return;
    }
    try {
      const body = await api(withDir(
        `/api/models/compare?left=${encodeURIComponent(state.stem)}` +
        `&right=${encodeURIComponent(other)}`));
      renderCompare(body);
    } catch (error) {
      status(error.message, "bad");
    }
  }

  function renderCompare(body) {
    ui.compareBody.replaceChildren();

    const table = el("table", { className: "metrics-table" });
    table.append(el("thead", {}, el("tr", {}, [
      el("th", { textContent: t("compare.col_class") }),
      el("th", { textContent: body.left.stem }),
      el("th", { textContent: body.right.stem }),
      el("th", { textContent: t("compare.col_delta") }),
    ])));
    const rows = el("tbody");
    for (const row of body.metrics) {
      const tr = el("tr");
      if (row.overall) tr.className = "overall";
      tr.append(el("td", { textContent: row.label }));
      tr.append(el("td", { textContent: fixed(row.left_f1) }));
      tr.append(el("td", { textContent: fixed(row.right_f1) }));
      const delta = el("td", {
        textContent: row.delta === null
          ? (row.only_in ? t("compare.only_in", { stem: row.only_in }) : "—")
          : `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(4)}`,
      });
      if (row.delta !== null) delta.className = row.delta >= 0 ? "delta-up" : "delta-down";
      tr.append(delta);
      rows.append(tr);
    }
    table.append(rows);
    ui.compareBody.append(table);

    ui.compareBody.append(el("h3", {
      textContent: t("compare.settings", { count: body.config.length }),
      style: "font-size:0.85rem;margin:1.2rem 0 0.4rem",
    }));
    if (!body.config.length) {
      ui.compareBody.append(el("p", { className: "empty", textContent: t("compare.identical") }));
      return;
    }
    const diff = el("table", { className: "metrics-table" });
    diff.append(el("thead", {}, el("tr", {}, [
      el("th", { textContent: t("compare.col_setting") }),
      el("th", { textContent: body.left.stem }),
      el("th", { textContent: body.right.stem }),
    ])));
    const diffRows = el("tbody");
    for (const row of body.config) {
      diffRows.append(el("tr", {}, [
        el("td", { textContent: row.path }),
        el("td", { textContent: row.left === undefined ? "—" : JSON.stringify(row.left) }),
        el("td", { textContent: row.right === undefined ? "—" : JSON.stringify(row.right) }),
      ]));
    }
    diff.append(diffRows);
    ui.compareBody.append(diff);
  }

  ui.compareWith.addEventListener("change", showCompare);

  /* ── which directory ───────────────────────────────────────────────── */

  async function useDirectory(path) {
    if (path !== undefined) ui.dir.value = path;
    try {
      localStorage.setItem("bioaccx.modelsDir", ui.dir.value.trim());
    } catch { /* storage unavailable; the choice just will not be remembered */ }
    // The previous selection belongs to the old directory.
    state.stem = null;
    state.detail = null;
    state.umap = null;
    ui.title.textContent = t("models.select");
    ui.metrics.replaceChildren();
    await loadModels();
  }

  ui.dir.addEventListener("change", () => useDirectory());
  ui.browse.addEventListener("click", async () => {
    const picked = await window.bioaccxPickFolder();
    if (picked) await useDirectory(picked);
  });

  /* ── view switching ────────────────────────────────────────────────── */

  function setView(view) {
    state.view = view;
    for (const [name, button, panel] of [
      ["metrics", ui.viewMetrics, ui.metrics],
      ["map", ui.viewMap, ui.map],
      ["compare", ui.viewCompare, ui.compare],
    ]) {
      button.classList.toggle("is-active", name === view);
      panel.hidden = name !== view;
    }
    if (view === "map") showMap();
    if (view === "compare") showCompare();
  }

  ui.viewMetrics.addEventListener("click", () => setView("metrics"));
  ui.viewMap.addEventListener("click", () => setView("map"));
  ui.viewCompare.addEventListener("click", () => setView("compare"));

  function restoreDirectory() {
    try {
      ui.dir.value = localStorage.getItem("bioaccx.modelsDir") ?? "";
    } catch { /* storage unavailable; start from the server's default */ }
  }

  /* Re-stamp everything this view builds in JavaScript. Called after a
   * language change; the data itself is already in hand, so nothing is
   * re-fetched and the selected model, sort order and view all survive. */
  function retranslate() {
    const against = ui.compareWith.value;
    if (!state.stem) ui.title.textContent = t("models.select");
    renderList();
    if (state.detail) renderMetrics();
    fillCompareOptions();
    ui.compareWith.value = against;
    if (state.view === "map") showMap();
    if (state.view === "compare") showCompare();
  }

  return {
    loadModels: async () => { restoreDirectory(); await loadModels(); },
    // After a run: pick up the directory it wrote, and re-read the selected
    // model in case that is the one the run just rewrote.
    refresh: async () => {
      await loadModels();
      if (state.stem && state.models.some((m) => m.stem === state.stem)) {
        await selectModel(state.stem);
      }
    },
    hasModels: () => state.models.length > 0,
    retranslate,
  };
}
