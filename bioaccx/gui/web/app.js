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

const $ = (id) => document.getElementById(id);

const ui = {
  form: $("form"), yaml: $("yaml"), validation: $("validation"),
  path: $("path"), status: $("status"),
  save: $("save"), open: $("open"), neu: $("new"), browse: $("browse"),
  editable: $("editable"), revert: $("revert"), toggleYaml: $("toggle-yaml"),
  picker: $("picker"), pickerList: $("picker-list"),
  pickerPath: $("picker-path"), pickerTitle: $("picker-title"),
  pickerChoose: $("picker-choose"),
  run: document.querySelector(".run"),
  runCommand: $("run-command"), runStart: $("run-start"), runCancel: $("run-cancel"),
  runCmd: $("run-cmd"), runCopy: $("run-copy"), runLog: $("run-log"),
  runProgress: $("run-progress"), runBar: $("run-bar-fill"), runStep: $("run-step"),
  theme: $("theme"),
};

// Before anything else, so the control works even if the backend never answers.
initTheme(ui.theme);

const state = { schema: null, secretPaths: [], text: "", values: {}, saved: "" };

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

function paint({ text, values, validation }) {
  state.text = text;
  state.values = values;
  ui.yaml.value = text;
  paintValidation(validation);

  const focused = document.activeElement?.closest?.(".field")?.dataset?.path;
  const scroll = ui.form.parentElement.scrollTop;
  renderForm(ui.form, state.schema, values, emit, state.secretPaths);
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
                    { className: "ok", textContent: "✓ config is valid" }));
    const s = validation.summary || {};
    const grid = document.createElement("div");
    grid.className = "summary";
    for (const [key, value] of Object.entries({
      backbone: s.foundation_model, "label mode": s.label_mode,
      classifier: s.classifier, sources: s.sources,
      "output dir": s.output_dir,
    })) {
      if (value === undefined) continue;
      grid.append(Object.assign(document.createElement("b"), { textContent: key }));
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
  status("applying…", "busy");
  pending = pending.then(async () => {
    try {
      paint(await api("/api/document", {
        method: "POST",
        body: JSON.stringify({ text: state.text, edits: { [path]: value } }),
      }));
      status(`${path} updated`, "ok");
    } catch (error) {
      status(error.message, "bad");
    }
  });
  return pending;
}

async function reparse(text) {
  status("parsing…", "busy");
  try {
    paint(await api("/api/document", {
      method: "POST", body: JSON.stringify({ text, edits: {} }),
    }));
    status("parsed", "ok");
  } catch (error) {
    status(error.message, "bad");
  }
}

/* ── files ───────────────────────────────────────────────────────────── */

async function openPath(path) {
  status("opening…", "busy");
  try {
    const result = await api(`/api/config?path=${encodeURIComponent(path)}`);
    ui.path.value = result.path;
    state.saved = result.text;
    paint(result);
    status(`opened ${result.path.split("/").pop()}`, "ok");
  } catch (error) {
    status(error.message, "bad");
  }
}

async function saveCurrent() {
  const path = ui.path.value.trim();
  if (!path) {
    status("give the file a path first", "bad");
    ui.path.focus();
    return;
  }
  status("saving…", "busy");
  try {
    const result = await api("/api/save", {
      method: "POST", body: JSON.stringify({ path, text: ui.yaml.value }),
    });
    ui.path.value = result.path;
    state.saved = ui.yaml.value;
    ui.revert.hidden = true;
    status(`saved ${result.path.split("/").pop()}`, "ok");
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

    if (listing.parent) ui.pickerList.append(row("../", listing.parent, false));
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
  showPicker({ title: "Choose a folder", files: false });

/* ── events ──────────────────────────────────────────────────────────── */

ui.save.addEventListener("click", saveCurrent);

ui.open.addEventListener("click", async () => {
  const path = ui.path.value.trim();
  if (path) return openPath(path);
  const picked = await showPicker({ title: "Open a config file", files: true });
  if (picked) openPath(picked);
});

ui.browse.addEventListener("click", async () => {
  const picked = await showPicker({ title: "Open a config file", files: true });
  if (picked) openPath(picked);
});

ui.neu.addEventListener("click", async () => {
  status("new config", "busy");
  try {
    const result = await api("/api/template");
    state.saved = "";
    ui.path.value = "";
    paint(result);
    status("started from a template", "ok");
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
  ui.toggleYaml.textContent = visible ? "Hide" : "Show file";
  ui.toggleYaml.setAttribute("aria-expanded", String(visible));
  try {
    localStorage.setItem("bioaccx.preview", visible ? "1" : "0");
  } catch {
    // Private browsing, or storage disabled — the toggle still works, it just
    // will not be remembered.
  }
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

/* ── views ───────────────────────────────────────────────────────────── */

const explorer = initExplorer({ api, status });
let modelsLoaded = false;

const tabs = {
  config: $("tab-config"), models: $("tab-models"),
  editor: $("editor"), explorer: $("explorer"),
};

async function showView(view) {
  const editing = view === "config";
  tabs.editor.hidden = !editing;
  tabs.explorer.hidden = editing;
  tabs.config.classList.toggle("is-active", editing);
  tabs.models.classList.toggle("is-active", !editing);
  tabs.config.setAttribute("aria-selected", String(editing));
  tabs.models.setAttribute("aria-selected", String(!editing));

  // The path box and file buttons belong to the editor; hide them rather than
  // leaving controls that do nothing to the view on screen.
  for (const node of [ui.path, ui.browse, ui.neu, ui.open, ui.save]) node.hidden = !editing;

  if (!editing && !modelsLoaded) {
    modelsLoaded = true;
    await explorer.loadModels();
  }
}

tabs.config.addEventListener("click", () => showView("config"));
tabs.models.addEventListener("click", () => showView("models"));

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
  ui.run.classList.toggle("is-running", !!running);

  if (!job) {
    ui.runProgress.hidden = true;
    return;
  }
  const known = job.step && job.step_total;
  ui.runProgress.hidden = !known && !running;
  if (known) {
    ui.runBar.style.width = `${Math.round((job.step / job.step_total) * 100)}%`;
    ui.runStep.textContent =
      `${job.step}/${job.step_total} ${job.step_name} · ${job.elapsed}s`;
  } else if (running) {
    ui.runBar.style.width = "0%";
    ui.runStep.textContent = `starting · ${job.elapsed}s`;
  }
  if (!running) {
    const done = job.status === "done";
    status(`run ${job.status} (exit ${job.returncode})`, done ? "ok" : "bad");
    if (done) ui.runBar.style.width = "100%";
  }
}

function appendLog(text) {
  const atBottom =
    ui.runLog.scrollHeight - ui.runLog.scrollTop - ui.runLog.clientHeight < 40;
  const line = document.createElement("div");
  if (/^\[(failed|cancelled)\]/.test(text) || /error|traceback/i.test(text)) {
    line.className = "line-bad";
  } else if (/^\[done\]/.test(text) || text.startsWith("$ ")) {
    line.className = "line-ok";
  }
  line.textContent = text;
  ui.runLog.append(line);
  ui.runLog.hidden = false;
  ui.run.classList.add("has-log");
  if (atBottom) ui.runLog.scrollTop = ui.runLog.scrollHeight;
}

function listen() {
  stream?.close();
  stream = new EventSource("/api/run/stream");
  stream.addEventListener("line", (event) => appendLog(JSON.parse(event.data).text));
  stream.addEventListener("status", (event) => {
    const job = JSON.parse(event.data);
    paintJob(job && job.id ? job : null);
    if (job && job.id && job.status !== "running") {
      stream.close();
      stream = null;
    }
  });
  stream.onerror = () => { stream?.close(); stream = null; };
}

ui.runStart.addEventListener("click", async () => {
  const path = ui.path.value.trim();
  if (!path) {
    status("save the config before running it", "bad");
    ui.path.focus();
    return;
  }
  if (ui.yaml.value !== state.saved) {
    status("unsaved changes — a run reads the file on disk", "bad");
    return;
  }
  ui.runLog.replaceChildren();
  status("starting…", "busy");
  try {
    const job = await api("/api/run", {
      method: "POST",
      body: JSON.stringify({ command: ui.runCommand.value, path }),
    });
    paintJob(job);
    listen();
    status(`running ${ui.runCommand.value}`, "busy");
  } catch (error) {
    status(error.message, "bad");
  }
});

ui.runCancel.addEventListener("click", async () => {
  status("cancelling…", "busy");
  try {
    await api("/api/run/cancel", { method: "POST" });
  } catch (error) {
    status(error.message, "bad");
  }
});

ui.runCopy.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(currentCommandLine());
    status("command copied", "ok");
  } catch {
    status("could not copy — select the command instead", "bad");
  }
});

ui.runCommand.addEventListener("change", paintCommandLine);
ui.path.addEventListener("input", paintCommandLine);

/* ── boot ────────────────────────────────────────────────────────────── */

(async function boot() {
  status("loading…", "busy");
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
    state.secretPaths = bootstrap.secret_paths;
    document.title = `bioaccx ${bootstrap.version} — config editor`;

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
      status("ready", "ok");
    }
  } catch (error) {
    status(error.message, "bad");
  }
})();
