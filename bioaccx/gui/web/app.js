/* Wires the form, the file preview and the API together.
 *
 * The YAML text is the single source of truth. A form edit is not applied in
 * the browser: it is sent to the server with the current text, and the server
 * hands back the serialised document. What the preview shows is therefore
 * byte-for-byte what Save would write, and the form can never drift from it.
 */

import { renderForm } from "/static/form.js";

const $ = (id) => document.getElementById(id);

const ui = {
  form: $("form"), yaml: $("yaml"), validation: $("validation"),
  path: $("path"), status: $("status"),
  save: $("save"), open: $("open"), neu: $("new"), browse: $("browse"),
  editable: $("editable"), revert: $("revert"),
  picker: $("picker"), pickerList: $("picker-list"),
  pickerPath: $("picker-path"), pickerTitle: $("picker-title"),
  pickerChoose: $("picker-choose"),
};

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

ui.editable.addEventListener("change", () => {
  ui.yaml.readOnly = !ui.editable.checked;
  if (ui.editable.checked) ui.yaml.focus();
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

/* ── boot ────────────────────────────────────────────────────────────── */

(async function boot() {
  status("loading…", "busy");
  try {
    const bootstrap = await api("/api/bootstrap");
    state.schema = bootstrap.schema;
    state.secretPaths = bootstrap.secret_paths;
    document.title = `bioaccx ${bootstrap.version} — config editor`;

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
