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
    colour: $("map-colour"), split: $("map-split"), zoomReset: $("map-zoom-reset"),
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
      if (model.macro_included?.f1 != null) {
        meta.append(el("span", {
          className: "f1",
          textContent: t("models.f1_included", { value: fixed(model.macro_included.f1) }),
          title: t("models.f1_included_title"),
        }));
      }
      if (model.mixup?.n_mixes) {
        // Trained partly on synthetic audio: worth knowing before reading its F1.
        meta.append(el("span", {
          className: "mixup",
          textContent: t("models.mixup", { count: model.mixup.n_mixes }),
          title: t("models.mixup_title", { mixes: model.mixup.n_mixes,
                                            real: model.mixup.n_real ?? "?" }),
        }));
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
      resetView();
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
  /* Added when the run trained on audio mixes: the make-up of each class's
   * training signal, next to the scores it produced. */
  const MIX_COLUMNS = ["real_train", "in_mixes", "mix_share"];
  const columnTitle = (key) => t(`metrics.col_${key}`);
  /* The macro-average rows keep the names the evaluation CSV gives them
   * (results.OVERALL_ROW, results.INCLUDED_ROW); they are named for reading here. */
  const OVERALL_NAMES = {
    "OVERALL (Macro-avg)": "metrics.overall",
    "OVERALL (Macro-avg, included)": "metrics.overall_included",
  };
  const rowName = (label) => (OVERALL_NAMES[label] ? t(OVERALL_NAMES[label]) : label);

  const percent = (value) =>
    value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;

  function mixupFact(model) {
    const s = model.mixup_settings || {};
    if (!model.mixup?.n_mixes) return undefined;
    return t("metrics.mixup_summary", {
      mixes: model.mixup.n_mixes, real: model.mixup.n_real ?? "?",
      snr: JSON.stringify(s.snr_db ?? []), pairing: s.pairing ?? "?",
      background: (s.background_labels || []).join(", ") || "—",
      target: s.background_target ?? "?",
    });
  }

  function cellText(key, value, row) {
    if (key === "label") return rowName(value);
    // A background label dropped from mix targets is still summed into mixes:
    // say so, or "0" reads as "never mixed".
    if (key === "in_mixes" && row.as_source > (value ?? 0)) {
      return t("metrics.in_mixes_as_source", { count: value ?? 0, source: row.as_source });
    }
    if (key === "mix_share") return percent(value);
    if (["samples", "real_train", "in_mixes"].includes(key)) {
      return value === null || value === undefined ? "—" : String(value);
    }
    return fixed(value);
  }

  /* One evaluation table. `sortable` wires the headers to the shared sort;
   * `absentTitle(row)` explains a class with no positive sample (a mixed test
   * set has none for a dropped background label). */
  function metricsTable(rows, columns, { sortable = false, excluded = new Set(),
                                         absentTitle = null, partners = {} } = {}) {
    const table = el("table", { className: "metrics-table" });
    const head = el("tr");
    for (const key of columns) {
      const title = columnTitle(key);
      const th = el("th", { textContent: title });
      if (sortable) {
        th.title = t("metrics.sort", { column: title });
        th.addEventListener("click", () => {
          state.sort = {
            column: key,
            ascending: state.sort.column === key ? !state.sort.ascending : true,
          };
          renderMetrics();
        });
      }
      head.append(th);
    }
    table.append(el("thead", {}, head));

    const overall = rows.filter((row) => row.overall);
    const classes = rows.filter((row) => !row.overall);
    if (sortable) {
      const { column, ascending } = state.sort;
      classes.sort((a, b) => {
        const x = a[column], y = b[column];
        if (x === null || x === undefined) return 1;
        if (y === null || y === undefined) return -1;
        const order = typeof x === "string" ? x.localeCompare(y) : x - y;
        return ascending ? order : -order;
      });
    }

    const body = el("tbody");
    for (const row of [...overall, ...classes]) {
      const tr = el("tr");
      if (row.overall) tr.className = "overall";
      if (excluded.has(row.label)) {
        tr.className = "excluded";
        tr.title = t("metrics.excluded_row");
      } else if (absentTitle && !row.overall && row.samples === 0) {
        tr.className = "excluded";
        tr.title = absentTitle(row);
      }
      for (const key of columns) {
        const value = row[key];
        const td = el("td", { textContent: cellText(key, value, row) });
        // Anything under half is worth the reader's eye before anything else.
        if (key === "f1" && value !== null && value < 0.5 && !row.overall) td.className = "weak";
        if (key === "in_mixes" && partners[row.label]) {
          const top = Object.entries(partners[row.label]).slice(0, 5)
            .map(([name, n]) => `${name} ${n}`).join(", ");
          if (top) td.title = t("metrics.partners", { partners: top });
        }
        tr.append(td);
      }
      body.append(tr);
    }
    table.append(body);
    return table;
  }

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
      mixup: mixupFact(model),
      test_mixes: model.mixup?.n_test_mixes || undefined,
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

    // The mix make-up joins the scores by class, so it sorts like any column.
    const composition = model.composition || {};
    const hasMix = Object.keys(composition).length > 0;
    const rows = model.evaluation.map((row) => ({ ...row, ...(composition[row.label] || {}) }));
    const partners = Object.fromEntries(
      Object.entries(composition).map(([label, c]) => [label, c.partners || {}]));
    ui.metrics.append(metricsTable(rows, hasMix ? [...COLUMNS, ...MIX_COLUMNS] : COLUMNS, {
      sortable: true, excluded: new Set(model.excluded_labels || []), partners,
    }));
    if (hasMix) {
      ui.metrics.append(el("p", { className: "table-note", textContent: t("metrics.mix_note") }));
    }

    if (model.evaluation_mixed?.length) {
      ui.metrics.append(el("h3", {
        className: "table-title",
        textContent: t("metrics.mixed_title", { count: model.mixup?.n_test_mixes ?? "?" }),
      }));
      ui.metrics.append(el("p", { className: "table-note", textContent: t("metrics.mixed_note") }));
      ui.metrics.append(metricsTable(model.evaluation_mixed, COLUMNS, {
        absentTitle: (row) => t("metrics.mixed_absent_row", { fp: row.false_positives ?? "?" }),
      }));
    }
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

  /* Zoom and pan: a scale `k` and an offset applied, in canvas pixels, on top
   * of the fit that puts the whole projection in the frame. Points keep their
   * size on screen, so zooming in separates them instead of enlarging them. */
  const view = { k: 1, tx: 0, ty: 0 };
  const MAX_ZOOM = 40;

  function resetView() {
    view.k = 1;
    view.tx = 0;
    view.ty = 0;
    ui.zoomReset.hidden = true;
  }

  // One redraw per frame however many wheel or drag events arrive in it.
  let frame = 0;
  const redraw = () => {
    if (!frame) frame = requestAnimationFrame(() => { frame = 0; drawPoints(); });
  };

  /* The bounds of every point, whichever split is shown, so that filtering
   * keeps the frame — and any zoom — where it was. */
  function boundsOf(umap) {
    if (!umap.bounds) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const p of umap.points) {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
      }
      umap.bounds = { minX, maxX, minY, maxY };
    }
    return umap.bounds;
  }

  function drawMap() {
    if (!state.umap) return;
    drawPoints();
    drawLegend();
  }

  /* Match the drawing buffer to the size the canvas is shown at, in device
   * pixels, so the map fills its box and stays sharp. A zoomed view keeps its
   * place when the box changes size. Returns the device pixel ratio. */
  function fitCanvas() {
    const canvas = ui.canvas;
    const dpr = window.devicePixelRatio || 1;
    const width = Math.round(canvas.clientWidth * dpr);
    const height = Math.round(canvas.clientHeight * dpr);
    if (width && height && (canvas.width !== width || canvas.height !== height)) {
      view.tx *= width / canvas.width;
      view.ty *= height / canvas.height;
      canvas.width = width;
      canvas.height = height;
    }
    return dpr;
  }

  // Redraw when the window, the panel or the pixel density changes.
  new ResizeObserver(() => { if (state.umap) redraw(); }).observe(ui.canvas);

  function drawPoints() {
    const umap = state.umap;
    if (!umap) return;
    const canvas = ui.canvas;
    const dpr = fitCanvas();
    const ctx = canvas.getContext("2d");
    const wanted = ui.split.value;
    const points = umap.points.filter((p) => !wanted || p.split === wanted);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ui.zoomReset.hidden = view.k === 1 && view.tx === 0 && view.ty === 0;
    if (!points.length) {
      state.drawn = [];
      return;
    }

    const { minX, maxX, minY, maxY } = boundsOf(umap);
    const pad = 18 * dpr, radius = 2.6 * dpr;
    // One scale for both axes, so distances read the same in every direction;
    // the projection is centred and the rest of the box is margin, which zoom
    // and drag can still bring points into.
    const scale = Math.min((canvas.width - pad * 2) / (maxX - minX || 1),
                           (canvas.height - pad * 2) / (maxY - minY || 1));
    const offsetX = (canvas.width - (maxX - minX) * scale) / 2;
    const offsetY = (canvas.height - (maxY - minY) * scale) / 2;

    const groups = groupsOf(umap);
    const colours = coloursOf(umap);
    const index = new Map(groups.map((name, i) => [name, i]));

    ctx.globalAlpha = 0.75;
    state.drawn = [];
    for (const point of points) {
      const x = view.k * (offsetX + (point.x - minX) * scale) + view.tx;
      // Canvas y grows downward; flip so the projection reads the usual way up.
      const y = view.k * (canvas.height - offsetY - (point.y - minY) * scale) + view.ty;
      // Off-screen points cost nothing, and cannot be clicked either.
      if (x < -radius || y < -radius || x > canvas.width + radius || y > canvas.height + radius) {
        continue;
      }
      ctx.fillStyle = colourOf(colours, index.get(groupOf(point)) ?? 0);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      state.drawn.push({ x, y, point });
    }
    ctx.globalAlpha = 1;
  }

  function drawLegend() {
    const umap = state.umap;
    const groups = groupsOf(umap);
    const colours = coloursOf(umap);
    ui.legend.replaceChildren();
    for (const [i, name] of groups.entries()) {
      ui.legend.append(el("span", {}, [
        el("i", { style: `background:${colourOf(colours, i)}` }),
        document.createTextNode(name || "—"),
      ]));
    }
  }

  /* A pointer position in canvas pixels; the canvas is drawn smaller than its
   * pixel size, so screen distances are scaled up to it. */
  function canvasPoint(event) {
    const box = ui.canvas.getBoundingClientRect();
    return [(event.clientX - box.left) * (ui.canvas.width / box.width),
            (event.clientY - box.top) * (ui.canvas.height / box.height)];
  }

  // The wheel zooms around the pointer; zooming all the way out shows the whole map again.
  ui.canvas.addEventListener("wheel", (event) => {
    if (!state.umap) return;
    event.preventDefault();
    const [mx, my] = canvasPoint(event);
    const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;   // lines → pixels
    const k = Math.min(MAX_ZOOM, Math.max(1, view.k * Math.exp(-delta * 0.0015)));
    const f = k / view.k;
    view.tx = mx - (mx - view.tx) * f;
    view.ty = my - (my - view.ty) * f;
    view.k = k;
    if (k === 1) { view.tx = 0; view.ty = 0; }
    redraw();
  }, { passive: false });

  // Dragging moves a zoomed map. A drag is not a click, so it never opens a sample.
  let drag = null;
  let dragged = false;
  ui.canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !state.umap) return;
    const [x, y] = canvasPoint(event);
    drag = { x, y, tx: view.tx, ty: view.ty, sx: event.clientX, sy: event.clientY };
    dragged = false;
    ui.canvas.setPointerCapture(event.pointerId);
  });
  ui.canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    if (!dragged && Math.abs(event.clientX - drag.sx) + Math.abs(event.clientY - drag.sy) < 5) return;
    dragged = true;
    if (view.k === 1) return;
    const [x, y] = canvasPoint(event);
    view.tx = drag.tx + (x - drag.x);
    view.ty = drag.ty + (y - drag.y);
    ui.canvas.classList.add("is-panning");
    redraw();
  });
  const endDrag = () => {
    drag = null;
    ui.canvas.classList.remove("is-panning");
  };
  ui.canvas.addEventListener("pointerup", endDrag);
  ui.canvas.addEventListener("pointercancel", endDrag);

  ui.zoomReset.addEventListener("click", () => {
    resetView();
    drawPoints();
  });

  ui.canvas.addEventListener("click", async (event) => {
    if (dragged) {
      dragged = false;
      return;
    }
    if (!state.drawn.length) return;
    const [cx, cy] = canvasPoint(event);

    let best = null, bestDistance = Infinity;
    for (const item of state.drawn) {
      const distance = (item.x - cx) ** 2 + (item.y - cy) ** 2;
      if (distance < bestDistance) { bestDistance = distance; best = item; }
    }
    const reach = 20 * (window.devicePixelRatio || 1);   // 20 screen pixels, in canvas units
    if (!best || bestDistance > reach * reach) return;
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
      let text = info.exists
        ? t("map.clip_range", { filename: info.filename,
                                start: fixed(info.start_time, 2),
                                end: fixed(info.end_time, 2) })
        : t("map.clip_missing", { path: info.filepath });
      if (info.exists && info.noise_file) {
        // An augmented window: played mixed with its noise, as it was embedded
        const mixed = info.exported || info.noise_found;
        text += " " + t(mixed ? "map.clip_noise" : "map.clip_noise_missing",
                        { noise: info.noise_file, snr: info.snr_db });
      }
      if (info.exported) text += " " + t("map.clip_exported");
      ui.sampleFile.textContent = text;
      ui.sampleFile.title = info.exported || info.filepath;

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

    // Mixing changes what each class trained on; with it on either side the
    // F1 change is shown next to each side's synthetic share.
    const mixed = Boolean(body.left.mixup?.n_mixes || body.right.mixup?.n_mixes);
    if (mixed) {
      const side = (m) => m.mixup?.n_mixes
        ? t("compare.mixup_side", { stem: m.stem, mixes: m.mixup.n_mixes, real: m.mixup.n_real ?? "?" })
        : t("compare.no_mixup_side", { stem: m.stem });
      ui.compareBody.append(el("p", { className: "table-note",
                                      textContent: `${side(body.left)} · ${side(body.right)}` }));
    }

    const table = el("table", { className: "metrics-table" });
    table.append(el("thead", {}, el("tr", {}, [
      el("th", { textContent: t("compare.col_class") }),
      el("th", { textContent: body.left.stem }),
      el("th", { textContent: body.right.stem }),
      el("th", { textContent: t("compare.col_delta") }),
      ...(mixed ? [
        el("th", { textContent: t("compare.col_mix_share", { stem: body.left.stem }) }),
        el("th", { textContent: t("compare.col_mix_share", { stem: body.right.stem }) }),
      ] : []),
    ])));
    const rows = el("tbody");
    for (const row of body.metrics) {
      const tr = el("tr");
      if (row.overall) tr.className = "overall";
      tr.append(el("td", { textContent: rowName(row.label) }));
      tr.append(el("td", { textContent: fixed(row.left_f1) }));
      tr.append(el("td", { textContent: fixed(row.right_f1) }));
      const delta = el("td", {
        textContent: row.delta === null
          ? (row.only_in ? t("compare.only_in", { stem: row.only_in }) : "—")
          : `${row.delta >= 0 ? "+" : ""}${row.delta.toFixed(4)}`,
      });
      if (row.delta !== null) delta.className = row.delta >= 0 ? "delta-up" : "delta-down";
      tr.append(delta);
      if (mixed) {
        tr.append(el("td", { textContent: row.overall ? "" : percent(row.left_mix_share ?? 0) }));
        tr.append(el("td", { textContent: row.overall ? "" : percent(row.right_mix_share ?? 0) }));
      }
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
    resetView();
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
