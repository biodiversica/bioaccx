/* Renders the config form from the schema the server generates.
 *
 * Nothing here knows the names of bioaccx's config fields. Each widget is
 * chosen by the descriptor's `widget`, so a field added to config.py shows up
 * with the right control and help text without this file changing.
 *
 * Every control reports edits as {path, value} where path is dotted
 * ("training.keras.epochs") and a null value means "remove this key", which is
 * how a cleared optional field avoids writing an explicit null.
 */

const el = (tag, props = {}, children = []) => {
  const node = Object.assign(document.createElement(tag), props);
  for (const child of [].concat(children)) {
    if (child != null) node.append(child);
  }
  return node;
};

/* ── value coercion ──────────────────────────────────────────────────── */

const parseTags = (text, itemType) => {
  const parts = text.split(",").map((s) => s.trim()).filter(Boolean);
  if (!parts.length) return null;
  return itemType === "float" || itemType === "int"
    ? parts.map(Number).filter((n) => !Number.isNaN(n))
    : parts;
};

const parsePaths = (text) => {
  const lines = text.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!lines.length) return null;
  return lines.length === 1 ? lines[0] : lines;
};

const parseFreq = (text) => {
  const parts = text.split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  if (!parts.length || parts.some(Number.isNaN)) return null;
  return parts.length === 1 ? parts[0] : parts;
};

const parseGroups = (text) => {
  const out = {};
  for (const line of text.split("\n")) {
    const [name, members] = line.split(":");
    if (!name || !members) continue;
    const list = members.split(",").map((s) => s.trim()).filter(Boolean);
    if (name.trim() && list.length) out[name.trim()] = list;
  }
  return Object.keys(out).length ? out : null;
};

const showTags = (value) => (Array.isArray(value) ? value.join(", ") : value ?? "");
const showPaths = (value) =>
  Array.isArray(value) ? value.join("\n") : value ?? "";
const showGroups = (value) =>
  value && typeof value === "object"
    ? Object.entries(value).map(([k, v]) => `${k}: ${[].concat(v).join(", ")}`).join("\n")
    : "";

/* ── widgets ─────────────────────────────────────────────────────────── */

function buildControl(field, value, emit) {
  const set = (v) => emit(field.path, v);
  const has = value !== undefined && value !== null;

  switch (field.widget) {
    case "checkbox": {
      const input = el("input", { type: "checkbox", checked: has ? !!value : !!field.default });
      input.addEventListener("change", () => set(input.checked));
      return el("div", { className: "row" }, [input]);
    }

    case "number": {
      const input = el("input", {
        type: "number",
        step: field.step || "any",
        value: has ? value : "",
        placeholder: field.default ?? "",
      });
      input.addEventListener("change", () =>
        set(input.value === "" ? null : Number(input.value)));
      return input;
    }

    case "select": {
      const select = el("select");
      if (field.optional || field.default === null) {
        select.append(el("option", { value: "", textContent: "— unset —" }));
      }
      for (const choice of field.choices || []) {
        select.append(el("option", { value: choice, textContent: choice }));
      }
      select.value = has ? String(value) : (field.optional ? "" : String(field.default ?? ""));
      select.addEventListener("change", () => set(select.value === "" ? null : select.value));
      return select;
    }

    case "tags": {
      const input = el("input", {
        type: "text",
        value: showTags(value),
        placeholder: showTags(field.default) || "comma, separated",
      });
      input.addEventListener("change", () => set(parseTags(input.value, field.item)));
      return input;
    }

    case "paths": {
      const area = el("textarea", {
        rows: 2,
        value: showPaths(value),
        placeholder: "one folder per line",
      });
      area.addEventListener("change", () => set(parsePaths(area.value)));
      const browse = el("button", {
        type: "button", className: "ghost small", textContent: "Browse…",
      });
      browse.addEventListener("click", async () => {
        const picked = await window.bioaccxPickFolder();
        if (!picked) return;
        area.value = area.value.trim() ? `${area.value.trim()}\n${picked}` : picked;
        set(parsePaths(area.value));
      });
      return el("div", {}, [area, el("div", { className: "row" }, [browse])]);
    }

    case "freq": {
      const input = el("input", {
        type: "text",
        value: Array.isArray(value) ? value.join(", ") : value ?? "",
        placeholder: "1000  (or  500, 8000  for a band-pass)",
      });
      input.addEventListener("change", () => set(parseFreq(input.value)));
      return input;
    }

    case "groups": {
      const area = el("textarea", {
        rows: 3,
        value: showGroups(value),
        placeholder: "SPECIES_A: SPECIES_A1, SPECIES_A2\nSPECIES_B: SPECIES_B1, SPECIES_B2",
      });
      area.addEventListener("change", () => set(parseGroups(area.value)));
      return area;
    }

    case "registry": {
      const select = el("select");
      select.append(el("option", { value: "", textContent: "— none (fields set by hand) —" }));
      for (const choice of field.choices || []) {
        select.append(el("option", { value: choice.id, textContent: choice.label }));
      }
      select.value = has ? String(value) : "";
      const desc = el("div", { className: "registry-desc" });
      const describe = () => {
        const found = (field.choices || []).find((c) => c.id === select.value);
        desc.textContent = found ? found.description : "";
        desc.hidden = !found;
      };
      describe();
      select.addEventListener("change", () => { describe(); set(select.value || null); });
      return el("div", {}, [select, desc]);
    }

    case "secret-set": {
      const input = el("input", {
        type: "password", value: "", placeholder: "•••••••• (set — leave blank to keep)",
      });
      input.addEventListener("change", () => set(input.value === "" ? "__SET__" : input.value));
      return input;
    }

    case "raw": {
      const area = el("textarea", { rows: 2, value: value == null ? "" : JSON.stringify(value) });
      area.addEventListener("change", () => {
        try {
          set(area.value.trim() === "" ? null : JSON.parse(area.value));
        } catch {
          set(area.value.trim() === "" ? null : area.value);
        }
      });
      return area;
    }

    default: {
      const input = el("input", {
        type: "text",
        value: has ? value : "",
        placeholder: field.default ?? "",
      });
      input.addEventListener("change", () => set(input.value === "" ? null : input.value));
      return input;
    }
  }
}

/* ── sources: repeating dataset blocks ───────────────────────────────── */

/* A source's own augmentation block, edited in place.
 *
 * The fields are the same AugmentationConfig descriptors the top-level block
 * uses, so this cannot drift from config.py either. They are bound straight to
 * the source object rather than to a dotted path, because `sources` is an
 * array written back whole. */
function buildAugmentation(block, spec, push) {
  const box = el("div", { className: "source-aug" });
  const settings = block.augmentation;

  for (const descriptor of spec.fields) {
    const bound = { ...descriptor, path: descriptor.name };
    const control = buildControl(bound, settings[descriptor.name], (name, next) => {
      if (next === null) delete settings[name];
      else settings[name] = next;
      push();
    });
    const label = el("label", { textContent: descriptor.label });
    if (descriptor.required) label.append(el("span", { className: "req", textContent: "*" }));
    const cell = el("div", {}, [control]);
    if (descriptor.help) {
      cell.append(el("div", { className: "help", textContent: descriptor.help }));
    }
    box.append(el("div", { className: "source-field" }, [label, cell]));
  }
  return box;
}

function buildSources(field, value, emit, values, nested) {
  const blocks = Array.isArray(value) ? value.map((b) => ({ ...b })) : [];
  const wrap = el("div", { className: "sources-list" });

  const push = () => emit(field.path, blocks.length ? blocks : null);
  const spec = nested?.AugmentationConfig;
  // Own blocks survive a trip through "inherit" or "no augmentation" and back.
  const remembered = new Map();

  const inherited = (key) => {
    const v = values[`dataset.${key}`];
    return v === undefined ? "" : String(Array.isArray(v) ? v.join(", ") : v);
  };

  const render = () => {
    wrap.replaceChildren();
    blocks.forEach((block, i) => {
      const head = el("div", { className: "source-head" }, [
        el("span", { textContent: `source ${i + 1}` }),
      ]);
      const remove = el("button", {
        type: "button", className: "ghost small", textContent: "Remove",
      });
      remove.addEventListener("click", () => { blocks.splice(i, 1); render(); push(); });
      head.append(remove);

      const row = (label, control) =>
        el("div", { className: "source-field" }, [el("label", { textContent: label }), control]);

      const dir = el("input", {
        type: "text",
        value: Array.isArray(block.data_dir) ? block.data_dir.join(", ") : block.data_dir ?? "",
        placeholder: inherited("data_dir") || "inherited",
      });
      dir.addEventListener("change", () => {
        const parsed = parseTags(dir.value, "str");
        if (parsed === null) delete block.data_dir;
        else block.data_dir = parsed.length === 1 ? parsed[0] : parsed;
        push();
      });

      const mode = el("select");
      mode.append(el("option", { value: "", textContent: `inherited (${inherited("label_mode") || "subfolders"})` }));
      for (const m of ["subfolders", "table", "file_per_label"]) {
        mode.append(el("option", { value: m, textContent: m }));
      }
      mode.value = block.label_mode ?? "";
      mode.addEventListener("change", () => {
        if (mode.value) block.label_mode = mode.value; else delete block.label_mode;
        push();
      });

      const table = el("input", {
        type: "text", value: block.table_file ?? "", placeholder: inherited("table_file") || "inherited",
      });
      table.addEventListener("change", () => {
        if (table.value.trim()) block.table_file = table.value.trim();
        else delete block.table_file;
        push();
      });

      // Augmentation is one of three states, not a checkbox: a source can
      // inherit the shared block, replace it with its own, or opt out. The
      // previous own block is remembered, so flipping through the states does
      // not discard settings.
      const own = block.augmentation;
      const state = own === null ? "none" : own === undefined ? "inherit" : "own";
      if (state === "own") remembered.set(i, own);

      const augMode = el("select");
      const inheritedNote = values["dataset.augmentation"]
        ? "inherit shared block" : "inherit (none is set)";
      for (const [key, text] of [["inherit", inheritedNote],
                                 ["own", "own block"],
                                 ["none", "no augmentation"]]) {
        augMode.append(el("option", { value: key, textContent: text }));
      }
      augMode.value = state;
      augMode.addEventListener("change", () => {
        if (augMode.value === "none") block.augmentation = null;
        else if (augMode.value === "inherit") delete block.augmentation;
        else block.augmentation = remembered.get(i) ?? { snr_levels: [10] };
        render();
        push();
      });

      const parts = [head, row("data dir", dir), row("label mode", mode),
                     row("table file", table), row("augmentation", augMode)];
      if (state === "own" && spec) {
        parts.push(buildAugmentation(block, spec, push));
      }
      wrap.append(el("div", { className: "source-block" }, parts));
    });

    const add = el("button", { type: "button", className: "ghost small", textContent: "Add source" });
    add.addEventListener("click", () => { blocks.push({}); render(); push(); });
    wrap.append(add);
  };

  render();
  return wrap;
}

/* ── field + section assembly ────────────────────────────────────────── */

function buildField(field, values, emit, secretPaths, nested) {
  const value = values[field.path];
  const isSet = value !== undefined;

  const label = el("label", { textContent: field.label });
  if (field.required) label.append(el("span", { className: "req", textContent: "*" }));

  const control = el("div", { className: "control" });

  let widget;
  if (field.widget === "sources") {
    widget = buildSources(field, value, emit, values, nested);
  } else if (secretPaths.includes(field.path) && value === "__SET__") {
    widget = buildControl({ ...field, widget: "secret-set" }, value, emit);
  } else if (secretPaths.includes(field.path)) {
    widget = buildControl({ ...field, widget: "text" }, value, emit);
  } else {
    widget = buildControl(field, value, emit);
  }
  control.append(widget);

  const badges = el("div", { className: "row" });
  if (field.run_level) {
    badges.append(el("span", {
      className: "badge run-level", textContent: "run level",
      title: "Taken from the top-level dataset block only; per-source overrides are ignored.",
    }));
  }
  if (secretPaths.includes(field.path)) {
    badges.append(el("span", {
      className: "badge secret", textContent: "secret",
      title: "Never sent back to the browser once set.",
    }));
  }
  if (badges.childElementCount) control.append(badges);

  if (field.help) control.append(el("div", { className: "help", textContent: field.help }));

  const node = el("div", { className: `field${isSet ? " is-set" : ""}` }, [label, control]);
  // The controller restores focus by path after re-rendering, so tabbing
  // through the form doesn't drop the caret when an edit round-trips.
  node.dataset.path = field.path;
  return node;
}

function buildFieldList(fields, values, emit, secretPaths, nested) {
  const frag = document.createDocumentFragment();
  const common = fields.filter((f) => !f.advanced);
  const advanced = fields.filter((f) => f.advanced);

  const addAll = (target, list) => {
    for (const field of list) {
      if (field.widget === "nested") {
        const spec = nested[field.dataclass];
        if (!spec) continue;   // rendered as its own group (keras, sklearn)
        target.append(buildNested(field, spec, values, emit, secretPaths));
        continue;
      }
      target.append(buildField(field, values, emit, secretPaths, nested));
    }
  };

  addAll(frag, common);
  if (advanced.length) {
    const details = el("details", { className: "advanced" });
    details.append(el("summary", { textContent: `advanced (${advanced.length})` }));
    const body = el("div");
    addAll(body, advanced);
    details.append(body);
    frag.append(details);
  }
  return frag;
}

function buildNested(field, spec, values, emit, secretPaths) {
  const group = el("div", { className: "group" });
  const enabled = Object.keys(values).some((k) => k.startsWith(`${field.path}.`));

  const toggle = el("input", { type: "checkbox", checked: enabled });
  const head = el("h3", {}, []);
  head.append(el("label", { className: "inline" }, [toggle, ` ${spec.title.toLowerCase()}`]));
  group.append(head);

  const body = el("div");
  body.hidden = !enabled;
  body.append(buildFieldList(
    spec.fields.map((f) => ({ ...f, path: `${field.path}.${f.name}` })),
    values, emit, secretPaths, {},
  ));
  group.append(body);

  toggle.addEventListener("change", () => {
    body.hidden = !toggle.checked;
    if (!toggle.checked) emit(field.path, null);
  });
  return group;
}

export function renderForm(root, schema, values, emit, secretPaths) {
  root.replaceChildren();
  for (const section of schema.sections) {
    const node = el("section", { className: "section" }, [
      el("h2", { textContent: section.title }),
      section.help ? el("p", { className: "section-help", textContent: section.help }) : null,
    ]);
    node.append(buildFieldList(section.fields, values, emit, secretPaths, schema.nested));

    for (const group of section.groups || []) {
      const box = el("div", { className: "group" }, [el("h3", { textContent: group.title })]);
      box.append(buildFieldList(group.fields, values, emit, secretPaths, schema.nested));
      node.append(box);
    }
    root.append(node);
  }
}
