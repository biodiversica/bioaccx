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

function buildSources(field, value, emit, values) {
  const blocks = Array.isArray(value) ? value.map((b) => ({ ...b })) : [];
  const wrap = el("div", { className: "sources-list" });

  const push = () => emit(field.path, blocks.length ? blocks : null);

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

      // A source can carry its own augmentation block, which the form does not
      // edit — it is written in the YAML pane. Hold on to it so that toggling
      // "no augmentation" on and off again restores it rather than deleting
      // hand-written settings, and say it is there so it is not invisible.
      const own = block.augmentation;
      const hasOwn = own !== null && own !== undefined;

      const noAug = el("input", { type: "checkbox", checked: own === null });
      noAug.addEventListener("change", () => {
        if (noAug.checked) block.augmentation = null;
        else if (hasOwn) block.augmentation = own;
        else delete block.augmentation;
        push();
      });

      const augNote = el("span", { className: "source-note" });
      if (hasOwn) {
        const levels = [].concat(own.snr_levels ?? []).join(", ");
        augNote.textContent =
          `own block${own.augmentation_dir ? ` · ${own.augmentation_dir}` : ""}` +
          `${levels ? ` · SNR ${levels}` : ""} — edit it in the YAML pane`;
      } else if (own === null) {
        augNote.textContent = "opted out of the shared block";
      } else {
        augNote.textContent = "inherits the shared block, if there is one";
      }

      wrap.append(el("div", { className: "source-block" }, [
        head,
        row("data dir", dir),
        row("label mode", mode),
        row("table file", table),
        row("no augmentation", el("div", { className: "row" }, [noAug, augNote])),
      ]));
    });

    const add = el("button", { type: "button", className: "ghost small", textContent: "Add source" });
    add.addEventListener("click", () => { blocks.push({}); render(); push(); });
    wrap.append(add);
  };

  render();
  return wrap;
}

/* ── field + section assembly ────────────────────────────────────────── */

function buildField(field, values, emit, secretPaths) {
  const value = values[field.path];
  const isSet = value !== undefined;

  const label = el("label", { textContent: field.label });
  if (field.required) label.append(el("span", { className: "req", textContent: "*" }));

  const control = el("div", { className: "control" });

  let widget;
  if (field.widget === "sources") {
    widget = buildSources(field, value, emit, values);
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
      target.append(buildField(field, values, emit, secretPaths));
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
