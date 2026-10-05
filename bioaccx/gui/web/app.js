/* Wires the form, the file preview and the API together.
 *
 * The YAML text is the single source of truth. A form edit is not applied in
 * the browser: it is sent to the server with the current text, and the server
 * hands back the serialised document. What the preview shows is therefore
 * byte-for-byte what Save would write, and the form can never drift from it.
 */

import { renderForm } from "/static/form.js";
import { initExplorer } from "/static/explorer.js";
import { initTheme } from "/static/theme.js";
import { applyStatic, currentLang, setBundle, t, translateSchema } from "/static/i18n.js";

const $ = (id) => document.getElementById(id);

const ui = {
  form: $("form"), yaml: $("yaml"), validation: $("validation"),
  path: $("path"), status: $("status"),
  save: $("save"), open: $("open"), neu: $("new"), browse: $("browse"),
  editable: $("editable"), revert: $("revert"), toggleYaml: $("toggle-yaml"),
  picker: $("picker"), pickerList: $("picker-list"),
  pickerPath: $("picker-path"), pickerTitle: $("picker-title"),
  pickerChoose: $("picker-choose"),
  runCommand: $("run-command"), runStart: $("run-start"), runCancel: $("run-cancel"),
  runCmd: $("run-cmd"), runCopy: $("run-copy"), runLog: $("run-log"),
  runProgress: $("run-progress"), runBar: $("run-bar-fill"), runStep: $("run-step"),
  theme: $("theme"), lang: $("lang"),
};

// Before anything else, so the control works even if the backend never answers.
const repaintTheme = initTheme(ui.theme);

/* `schema` is the English schema the server sent; `translated` is it under the
 * language on screen. Both are kept so switching language costs no request for
 * the schema and cannot lose the original text. */
const state = {
  schema: null, translated: null, secretPaths: [], text: "", values: {}, saved: "",
  painted: null, version: "",
};

/* ── helpers ─────────────────────────────────────────────────────────── */

function status(message, kind = "") {
  ui.status.textContent = message;
  ui.status.className = `status ${kind}`;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch { /* response had no JSON body */ }
    throw new Error(detail);
  }
  return response.json();
}

/* ── rendering ───────────────────────────────────────────────────────── */

function paint(document_) {
  state.painted = document_;             // kept so a language change can repaint
  const { text, values, validation } = document_;
  state.text = text;
  state.values = values;
  ui.yaml.value = text;
  paintValidation(validation);

  const focused = document.activeElement?.closest?.(".field")?.dataset?.path;
  const scroll = ui.form.parentElement.scrollTop;
  renderForm(ui.form, state.translated, values, emit, state.secretPaths);
  ui.form.parentElement.scrollTop = scroll;
  if (focused) {
    ui.form
      .querySelector(`.field[data-path="${CSS.escape(focused)}"] input, ` +
                     `.field[data-path="${CSS.escape(focused)}"] select, ` +
                     `.field[data-path="${CSS.escape(focused)}"] textarea`)
      ?.focus();
  }
  ui.revert.hidden = state.text === state.saved;
  paintCommandLine();
}

function paintValidation(validation) {
  ui.validation.replaceChildren();
  if (!validation) return;

  if (validation.valid) {
    ui.validation.append(
      Object.assign(document.createElement("div"),
                    { className: "ok", textContent: t("validation.ok") }));
    const s = validation.summary || {};
    const grid = document.createElement("div");
    grid.className = "summary";
    for (const [key, value] of Object.entries({
      backbone: s.foundation_model, label_mode: s.label_mode,
      classifier: s.classifier, sources: s.sources,
      output_dir: s.output_dir,
    })) {
      if (value === undefined) continue;
      grid.append(Object.assign(document.createElement("b"),
                                { textContent: t(`validation.${key}`) }));
      grid.append(Object.assign(document.createElement("span"),
                                { textContent: String(value) }));
    }
    ui.validation.append(grid);
    return;
  }

  for (const error of validation.errors || []) {
    ui.validation.append(Object.assign(document.createElement("div"), {
      className: "err",
      textContent: error.path ? `${error.path}: ${error.message}` : error.message,
    }));
  }
}

/* ── edits ───────────────────────────────────────────────────────────── */

let pending = Promise.resolve();

function emit(path, value) {
  status(t("status.applying"), "busy");
  pending = pending.then(async () => {
    try {
      paint(await api("/api/document", {
        method: "POST",
        body: JSON.stringify({ text: state.text, edits: { [path]: value } }),
      }));
      status(t("status.updated", { path }), "ok");
    } catch (error) {
      status(error.message, "bad");
    }
  });
  return pending;
}

async function reparse(text) {
  status(t("status.parsing"), "busy");
  try {
    paint(await api("/api/document", {
      method: "POST", body: JSON.stringify({ text, edits: {} }),
    }));
    status(t("status.parsed"), "ok");
  } catch (error) {
    status(error.message, "bad");
  }
}

/* ── files ───────────────────────────────────────────────────────────── */

async function openPath(path) {
  status(t("status.opening"), "busy");
  try {
    const result = await api(`/api/config?path=${encodeURIComponent(path)}`);
    ui.path.value = result.path;
    state.saved = result.text;
    paint(result);
    status(t("status.opened", { name: result.path.split("/").pop() }), "ok");
  } catch (error) {
    status(error.message, "bad");
  }
}

async function saveCurrent() {
  const path = ui.path.value.trim();
  if (!path) {
    status(t("status.need_path"), "bad");
    ui.path.focus();
    return;
  }
  status(t("status.saving"), "busy");
  try {
    const result = await api("/api/save", {
      method: "POST", body: JSON.stringify({ path, text: ui.yaml.value }),
    });
    ui.path.value = result.path;
    state.saved = ui.yaml.value;
    ui.revert.hidden = true;
    status(t("status.saved", { name: result.path.split("/").pop() }), "ok");
  } catch (error) {
    status(error.message, "bad");
  }
}

/* ── folder / file picker ────────────────────────────────────────────── */

let pickerResolve = null;
let pickerDir = "";

async function showPicker({ title, files }) {
  ui.pickerTitle.textContent = title;
  ui.pickerChoose.hidden = files;
  await loadPicker("", files);
  ui.picker.showModal();
  return new Promise((resolve) => { pickerResolve = resolve; });
}

async function loadPicker(path, files) {
  try {
    const listing = await api(`/api/browse?path=${encodeURIComponent(path)}`);
    pickerDir = listing.path;
    ui.pickerPath.textContent = listing.path;
    ui.pickerList.replaceChildren();

    const row = (label, target, isFile) => {
      const button = Object.assign(document.createElement("button"),
                                   { type: "button", textContent: label });
      button.addEventListener("click", () => {
        if (isFile) finishPicker(target);
        else loadPicker(target, files);
      });
      const li = document.createElement("li");
      li.append(button);
      return li;
    };

    if (listing.parent) ui.pickerList.append(row(t("picker.up"), listing.parent, false));
    for (const dir of listing.dirs) ui.pickerList.append(row(`${dir.name}/`, dir.path, false));
    if (files) {
      for (const file of listing.files) ui.pickerList.append(row(file.name, file.path, true));
    }
  } catch (error) {
    status(error.message, "bad");
  }
}

function finishPicker(value) {
  ui.picker.close();
  const resolve = pickerResolve;
  pickerResolve = null;
  if (resolve) resolve(value);
}

ui.pickerChoose.addEventListener("click", () => finishPicker(pickerDir));
ui.picker.addEventListener("close", () => finishPicker(null));

/** Used by the paths widget in form.js. */
window.bioaccxPickFolder = () =>
  showPicker({ title: t("picker.folder"), files: false });

/* ── events ──────────────────────────────────────────────────────────── */

ui.save.addEventListener("click", saveCurrent);

ui.open.addEventListener("click", async () => {
  const path = ui.path.value.trim();
  if (path) return openPath(path);
  const picked = await showPicker({ title: t("picker.file"), files: true });
  if (picked) openPath(picked);
});

ui.browse.addEventListener("click", async () => {
  const picked = await showPicker({ title: t("picker.file"), files: true });
  if (picked) openPath(picked);
});

ui.neu.addEventListener("click", async () => {
  status(t("status.new_config"), "busy");
  try {
    const result = await api("/api/template");
    state.saved = "";
    ui.path.value = "";
    paint(result);
    status(t("status.template"), "ok");
  } catch (error) {
    status(error.message, "bad");
  }
});

/* The file preview is off by default for most work — the form and the
 * validation summary say what matters — but it is one click away, and the
 * choice is remembered so it does not have to be made every session. The
 * column keeps its width either way, so hiding the file does not reflow the
 * page around it. */
function showPreview(visible) {
  ui.yaml.hidden = !visible;
  paintPreviewToggle();
  try {
    localStorage.setItem("bioaccx.preview", visible ? "1" : "0");
  } catch {
    // Private browsing, or storage disabled — the toggle still works, it just
    // will not be remembered.
  }
}

function paintPreviewToggle() {
  const visible = !ui.yaml.hidden;
  ui.toggleYaml.textContent = visible ? t("preview.hide") : t("preview.show");
  ui.toggleYaml.setAttribute("aria-expanded", String(visible));
}

ui.toggleYaml.addEventListener("click", () => showPreview(ui.yaml.hidden));

ui.editable.addEventListener("change", () => {
  ui.yaml.readOnly = !ui.editable.checked;
  // Editing the file by hand is impossible while it is hidden.
  if (ui.editable.checked) {
    showPreview(true);
    ui.yaml.focus();
  }
});

ui.yaml.addEventListener("change", () => {
  if (!ui.yaml.readOnly) reparse(ui.yaml.value);
});

ui.revert.addEventListener("click", () => {
  if (state.saved) reparse(state.saved);
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "s") {
    event.preventDefault();
    saveCurrent();
  }
});

/* ── language ────────────────────────────────────────────────────────── */

/* Bundles are fetched rather than shipped inside the page, so a language is
 * added by dropping a JSON file next to the others (see gui/i18n.py). The
 * schema is not re-fetched: it is served in English once and translated here,
 * so switching language costs one small request and loses no state. */
async function setLanguage(code, { persist = true } = {}) {
  const data = await api(`/api/i18n/${encodeURIComponent(code)}`);
  setBundle(data.lang, data.bundle);
  if (persist) {
    try {
      localStorage.setItem("bioaccx.lang", data.lang);
    } catch { /* storage unavailable; the choice just is not remembered */ }
  }
  ui.lang.value = data.lang;

  applyStatic();
  document.title = `bioaccx ${state.version} — ${t("app.title")}`;
  repaintTheme();
  // This reads a state rather than a fixed string, so applyStatic cannot
  // reach it.
  paintPreviewToggle();
  if (state.schema) {
    state.translated = translateSchema(state.schema);
    if (state.painted) paint(state.painted);
  }
  explorer.retranslate();
}

ui.lang.addEventListener("change", async () => {
  try {
    await setLanguage(ui.lang.value);
  } catch (error) {
    status(error.message, "bad");
  }
});

/* ── views ───────────────────────────────────────────────────────────── */

const explorer = initExplorer({ api, status });
let modelsLoaded = false;

/* Each tab and the view it shows. Config and run share the config file: the
 * run executes the file named in the path box, so the file controls stay on
 * screen for both. */
const views = {
  config: { tab: $("tab-config"), view: $("editor"), fileControls: true },
  run: { tab: $("tab-run"), view: $("runner"), fileControls: true },
  analysis: { tab: $("tab-analysis"), view: $("explorer"), fileControls: false },
};

async function showView(name) {
  for (const [key, { tab, view }] of Object.entries(views)) {
    const active = key === name;
    view.hidden = !active;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  }

  // The path box and file buttons do nothing to the analysis view; hide them
  // rather than leave dead controls on screen.
  const fileControls = views[name].fileControls;
  for (const node of [ui.path, ui.browse, ui.neu, ui.open, ui.save]) node.hidden = !fileControls;

  if (name === "analysis" && !modelsLoaded) {
    modelsLoaded = true;
    await explorer.loadModels();
  }
}

for (const [key, { tab }] of Object.entries(views)) {
  tab.addEventListener("click", () => showView(key));
}

/* ── runs ────────────────────────────────────────────────────────────── */

/* A run reads the file on disk, never the draft in the browser, so whatever
 * ran can always be repeated from a terminal. That is also why the equivalent
 * command line is shown next to the button rather than hidden. */

let stream = null;

function currentCommandLine() {
  const path = ui.path.value.trim();
  return `bioaccx ${ui.runCommand.value} ${path || "<config>"}`;
}

function paintCommandLine() {
  ui.runCmd.textContent = currentCommandLine();
}

function paintJob(job) {
  const running = job && job.status === "running";
  ui.runStart.hidden = !!running;
  ui.runCancel.hidden = !running;
  views.run.tab.classList.toggle("is-running", !!running);

  if (!job) {
    ui.runProgress.hidden = true;
    return;
  }
  const known = job.step && job.step_total;
  ui.runProgress.hidden = !known && !running;
  if (known) {
    // A step is complete only once the next begins, so the last one ([4/4],
    // often the longest) must not fill the bar while it is still running.
    const completed = running ? job.step - 1 : job.step;
    ui.runBar.style.width = `${Math.round((completed / job.step_total) * 100)}%`;
    ui.runStep.textContent = t("run.progress", {
      step: job.step, total: job.step_total, name: job.step_name, elapsed: job.elapsed,
    });
  } else if (running) {
    ui.runBar.style.width = "0%";
    ui.runStep.textContent = t("run.starting", { elapsed: job.elapsed });
  }
  if (!running) {
    const done = job.status === "done";
    status(t("status.run_finished",
                 { state: t(`run_state.${job.status}`), code: job.returncode }),
           done ? "ok" : "bad");
    if (done) ui.runBar.style.width = "100%";
  }
}

/* Embedding runs print a line per sample, so a run can emit tens of thousands.
 * Lines are queued and painted once per frame, and only the tail is kept (the
 * server buffers the same amount): one element and one layout per line made
 * the tab freeze on large runs while the run itself carried on. */
const LOG_LINES = 4000;
let pendingLines = [];

function appendLog(text) {
  if (!pendingLines.length) requestAnimationFrame(flushLog);
  pendingLines.push(text);
}

function flushLog() {
  const texts = pendingLines.slice(-LOG_LINES);
  pendingLines = [];
  if (!texts.length) return;
  const atBottom =
    ui.runLog.scrollHeight - ui.runLog.scrollTop - ui.runLog.clientHeight < 40;
  const batch = document.createDocumentFragment();
  for (const text of texts) {
    const line = document.createElement("div");
    if (/^\[(failed|cancelled)\]/.test(text) || /error|traceback/i.test(text)) {
      line.className = "line-bad";
    } else if (/^\[done\]/.test(text) || text.startsWith("$ ")) {
      line.className = "line-ok";
    }
    line.textContent = text;
    batch.append(line);
  }
  ui.runLog.append(batch);
  for (let extra = ui.runLog.childElementCount - LOG_LINES; extra > 0; extra--) {
    ui.runLog.firstElementChild.remove();
  }
  ui.runLog.hidden = false;
  if (atBottom) ui.runLog.scrollTop = ui.runLog.scrollHeight;
}

function clearLog() {
  pendingLines = [];
  ui.runLog.replaceChildren();
}

function listen() {
  stream?.close();
  // The stream replays the server's buffered tail from the start, so a
  // reconnect starts from an empty log rather than duplicating it.
  clearLog();
  stream = new EventSource("/api/run/stream");
  stream.addEventListener("line", (event) => appendLog(JSON.parse(event.data).text));
  stream.addEventListener("status", (event) => {
    const job = JSON.parse(event.data);
    paintJob(job && job.id ? job : null);
    if (job && job.id && job.status !== "running") {
      stream.close();
      stream = null;
      // The models list is fetched once per page; without this a finished
      // run's outputs only showed up after a reload.
      if (job.status === "done" && modelsLoaded) explorer.refresh();
    }
  });
  // A dropped connection used to leave the page frozen on the last update
  // while the run went on; pick the run back up if it is still going.
  stream.onerror = () => {
    stream?.close();
    stream = null;
    setTimeout(async () => {
      if (stream) return;
      try {
        const state = await api("/api/run");
        if (state.running) listen();
        else paintJob(state.job);
      } catch {
        setTimeout(() => stream || listen(), 3000);
      }
    }, 1000);
  };
}

ui.runStart.addEventListener("click", async () => {
  const path = ui.path.value.trim();
  if (!path) {
    status(t("status.run_needs_save"), "bad");
    ui.path.focus();
    return;
  }
  if (ui.yaml.value !== state.saved) {
    status(t("status.run_unsaved"), "bad");
    return;
  }
  ui.runLog.replaceChildren();
  status(t("status.starting"), "busy");
  try {
    const job = await api("/api/run", {
      method: "POST",
      body: JSON.stringify({ command: ui.runCommand.value, path }),
    });
    paintJob(job);
    listen();
    status(t("status.running", { command: ui.runCommand.value }), "busy");
  } catch (error) {
    status(error.message, "bad");
  }
});

ui.runCancel.addEventListener("click", async () => {
  status(t("status.cancelling"), "busy");
  try {
    await api("/api/run/cancel", { method: "POST" });
  } catch (error) {
    status(error.message, "bad");
  }
});

ui.runCopy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(currentCommandLine());
    status(t("status.copied"), "ok");
  } catch {
    status(t("status.copy_failed"), "bad");
  }
});

ui.runCommand.addEventListener("change", paintCommandLine);
ui.path.addEventListener("input", paintCommandLine);

/* ── boot ────────────────────────────────────────────────────────────── */

(async function boot() {
  // The bundle comes first, before anything is labelled: every string this
  // file writes goes through t(), and t() has nothing to say until it lands.
  let chosen = null;
  try {
    chosen = localStorage.getItem("bioaccx.lang");
  } catch { /* storage unavailable; the server's language decides */ }
  try {
    await setLanguage(chosen || "en", { persist: false });
  } catch { /* offline or unauthorized — the bootstrap below reports it */ }

  status(t("status.loading"), "busy");
  // Hidden unless this browser has been told otherwise: the form and the
  // validation summary carry the day-to-day work, and the file is one click
  // away when it is wanted.
  let remembered = null;
  try {
    remembered = localStorage.getItem("bioaccx.preview");
  } catch { /* storage unavailable; fall through to the default */ }
  showPreview(remembered === "1");

  try {
    const bootstrap = await api("/api/bootstrap");
    state.schema = bootstrap.schema;
    state.translated = bootstrap.schema;
    state.secretPaths = bootstrap.secret_paths;
    state.version = bootstrap.version;

    for (const language of bootstrap.languages ?? []) {
      ui.lang.append(Object.assign(document.createElement("option"),
                                   { value: language.code, textContent: language.name }));
    }
    ui.lang.value = currentLang();
    // --lang is what a browser that has never chosen sees; a choice made in
    // the picker outlives it, which is why that one is the first asked for.
    if (!chosen && bootstrap.lang !== currentLang()) {
      await setLanguage(bootstrap.lang, { persist: false });
    } else {
      state.translated = translateSchema(state.schema);
    }

    for (const command of bootstrap.runnable ?? []) {
      ui.runCommand.append(
        Object.assign(document.createElement("option"),
                      { value: command, textContent: command }));
    }
    paintCommandLine();
    // A run started before this page loaded (or before a reload) is picked up
    // rather than orphaned: the subprocess outlives the browser tab.
    if (bootstrap.job) {
      paintJob(bootstrap.job);
      if (bootstrap.job.status === "running") listen();
    }

    if (bootstrap.open_path) {
      await openPath(bootstrap.open_path);
    } else {
      const result = await api("/api/template");
      paint(result);
      status(t("status.ready"), "ok");
    }
  } catch (error) {
    status(error.message, "bad");
  }
})();
