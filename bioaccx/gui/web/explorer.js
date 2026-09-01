/* The results explorer: what a run left behind, made navigable.
 *
 * Everything here reads artifacts already on disk — metadata, the evaluation
 * table, the UMAP projection, the dataset list — so it works on models trained
 * long before this page existed. Nothing is recomputed.
 */

const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) if (child != null) node.append(child);
  return node;
};

/* Categorical colours, ordered so neighbouring classes stay distinguishable.
 * Deliberately not a gradient: these encode identity, not magnitude. */
const PALETTE = [
  "#1f6b54", "#c2571f", "#3b6ea5", "#a03d63", "#6b8f1f", "#8a5cb0",
  "#b8912a", "#417d7a", "#a44a3f", "#5a6b8c", "#7a9c3d", "#96566f",
];
const colourFor = (index) => PALETTE[index % PALETTE.length];

const fixed = (value, places = 4) =>
  value === null || value === undefined ? "—" : Number(value).toFixed(places);

export function initExplorer({ api, status }) {
  const ui = {
    list: $("model-list"), dir: $("models-dir"), title: $("detail-title"),
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

  /* ── model list ────────────────────────────────────────────────────── */

  async function loadModels() {
    try {
      const body = await api("/api/models");
      state.models = body.models;
      ui.dir.textContent = body.models_dir;
      ui.dir.title = body.models_dir;
      renderList();
      if (!state.models.length) {
        ui.metrics.replaceChildren(
          el("p", { className: "empty", textContent: `Nothing to show in ${body.models_dir}.` }));
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
      if (model.n_classes) meta.append(el("span", { textContent: `${model.n_classes} classes` }));
      if (model.macro.f1 !== null) {
        meta.append(el("span", { className: "f1", textContent: `F1 ${fixed(model.macro.f1)}` }));
      }
      if (model.has_umap) meta.append(el("span", { textContent: "map" }));
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
      state.detail = await api(`/api/models/${encodeURIComponent(stem)}`);
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

  const COLUMNS = [
    ["label", "class"], ["precision", "prec"], ["recall", "recall"], ["f1", "F1"],
    ["f1_opt", "F1 opt"], ["auprc", "AUPRC"], ["auroc", "AUROC"],
    ["threshold", "thr"], ["samples", "n"],
  ];

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
      grid.append(el("b", { textContent: key }));
      grid.append(el("span", { textContent: String(value) }));
    }
    ui.metrics.append(grid);

    if (!model.evaluation.length) {
      ui.metrics.append(el("p", {
        className: "empty",
        textContent: model.kind === "embeddings"
          ? "An embeddings run — no classifier was trained, so there are no "
            + "metrics. Its projection is on the map tab."
          : model.kind === "dataset"
            ? "A dataset export — no embeddings or classifier were produced."
            : "No evaluation table — this run wrote no per-class metrics.",
      }));
      if (model.report) ui.metrics.append(el("pre", { className: "run-log", textContent: model.report }));
      return;
    }

    const table = el("table", { className: "metrics-table" });
    const head = el("tr");
    for (const [key, title] of COLUMNS) {
      const th = el("th", { textContent: title, title: `sort by ${title}` });
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
      for (const [key] of COLUMNS) {
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
        state.umap = await api(`/api/models/${encodeURIComponent(state.stem)}/umap`);
      } catch (error) {
        ui.legend.replaceChildren();
        ui.note.textContent = error.message;
        clearCanvas();
        return;
      }
    }
    ui.note.textContent = state.umap.key_source === "row-order"
      ? "keys inferred from row order — this projection predates the key column"
      : state.umap.key_source === "none"
        ? "no sample keys: points cannot be traced back to audio"
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
    const index = new Map(groups.map((name, i) => [name, i]));

    state.drawn = points.map((point) => {
      const x = pad + (point.x - minX) * scaleX;
      // Canvas y grows downward; flip so the projection reads the usual way up.
      const y = canvas.height - pad - (point.y - minY) * scaleY;
      ctx.fillStyle = colourFor(index.get(groupOf(point)) ?? 0);
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
        el("i", { style: `background:${colourFor(i)}` }),
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
      status("this point has no key — cannot find its audio", "bad");
      return;
    }
    const query = `key=${encodeURIComponent(point.key)}`;
    try {
      const info = await api(`/api/models/${stem}/sample?${query}`);
      ui.sample.hidden = false;
      ui.sampleLabel.textContent =
        `${point.label}${point.split ? ` · ${point.split}` : ""}`;
      ui.sampleFile.textContent = info.exists
        ? `${info.filename} [${fixed(info.start_time, 2)}–${fixed(info.end_time, 2)}s]`
        : `missing: ${info.filepath}`;
      ui.sampleFile.title = info.filepath;

      if (!info.exists) {
        // The dataset may live on another machine, or have moved since training.
        ui.audio.removeAttribute("src");
        ui.spec.removeAttribute("src");
        return;
      }
      ui.audio.src = `/api/models/${stem}/clip?${query}`;
      ui.spec.src = `/api/models/${stem}/spectrogram?${query}`;
    } catch (error) {
      status(error.message, "bad");
    }
  }

  ui.colour.addEventListener("change", drawMap);
  ui.split.addEventListener("change", drawMap);

  /* ── compare ───────────────────────────────────────────────────────── */

  function fillCompareOptions() {
    ui.compareWith.replaceChildren(el("option", { value: "", textContent: "— pick a model —" }));
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
        className: "empty", textContent: "Pick a model to compare against.",
      }));
      return;
    }
    try {
      const body = await api(
        `/api/models/compare?left=${encodeURIComponent(state.stem)}` +
        `&right=${encodeURIComponent(other)}`);
      renderCompare(body);
    } catch (error) {
      status(error.message, "bad");
    }
  }

  function renderCompare(body) {
    ui.compareBody.replaceChildren();

    const table = el("table", { className: "metrics-table" });
    table.append(el("thead", {}, el("tr", {}, [
      el("th", { textContent: "class" }),
      el("th", { textContent: body.left.stem }),
      el("th", { textContent: body.right.stem }),
      el("th", { textContent: "Δ F1" }),
    ])));
    const rows = el("tbody");
    for (const row of body.metrics) {
      const tr = el("tr");
      if (row.overall) tr.className = "overall";
      tr.append(el("td", { textContent: row.label }));
      tr.append(el("td", { textContent: fixed(row.left_f1) }));
      tr.append(el("td", { textContent: fixed(row.right_f1) }));
      const delta = el("td", {
        textContent: row.delta === null ? (row.only_in ? `only in ${row.only_in}` : "—")
                                        : `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(4)}`,
      });
      if (row.delta !== null) delta.className = row.delta >= 0 ? "delta-up" : "delta-down";
      tr.append(delta);
      rows.append(tr);
    }
    table.append(rows);
    ui.compareBody.append(table);

    ui.compareBody.append(el("h3", {
      textContent: `settings that differ (${body.config.length})`,
      style: "font-size:0.85rem;margin:1.2rem 0 0.4rem",
    }));
    if (!body.config.length) {
      ui.compareBody.append(el("p", { className: "empty", textContent: "Identical settings." }));
      return;
    }
    const diff = el("table", { className: "metrics-table" });
    diff.append(el("thead", {}, el("tr", {}, [
      el("th", { textContent: "setting" }),
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

  return { loadModels, hasModels: () => state.models.length > 0 };
}
