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

import { t } from "/static/i18n.js";

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
        select.append(el("option", { value: "", textContent: t("form.unset") }));
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
        placeholder: showTags(field.default) || t("form.tags_ph"),
      });
      input.addEventListener("change", () => set(parseTags(input.value, field.item)));
      return input;
    }

    case "paths": {
      const area = el("textarea", {
        rows: 2,
        value: showPaths(value),
        placeholder: t("form.paths_ph"),
      });
      area.addEventListener("change", () => set(parsePaths(area.value)));
      const browse = el("button", {
        type: "button", className: "ghost small", textContent: t("form.browse"),
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
        placeholder: t("form.freq_ph"),
      });
      input.addEventListener("change", () => set(parseFreq(input.value)));
      return input;
    }

    case "groups": {
      const area = el("textarea", {
        rows: 3,
        value: showGroups(value),
        placeholder: t("form.groups_ph"),
      });
      area.addEventListener("change", () => set(parseGroups(area.value)));
      return area;
    }

    case "registry": {
      const select = el("select");
      select.append(el("option", { value: "", textContent: t("form.registry_none") }));
      for (const choice of field.choices || []) {
        select.append(el("option", { value: choice.id, textContent: choice.label }));
      }
      select.value = has ? String(value) : "";

      const desc = el("div", { className: "registry-desc" });
      const describe = () => {
        const found = (field.choices || []).find((c) => c.id === select.value);
        desc.replaceChildren();
        desc.hidden = !found;
        if (!found) return;
        desc.append(el("div", { textContent: found.description }));
        // The audio contract is the thing people come here to check, so state
        // it rather than making them read it off the fields below.
        const d = found.defaults || {};
        const window =
          d.window_seconds !== undefined ? t("form.fact_window_seconds", { value: d.window_seconds })
          : d.window_samples !== undefined ? t("form.fact_window_samples", { value: d.window_samples })
          : null;
        const facts = [
          d.sample_rate !== undefined ? t("form.fact_rate", { value: d.sample_rate }) : null,
          window,
          d.embedding_size !== undefined
            ? t("form.fact_embedding", { value: d.embedding_size }) : null,
        ].filter(Boolean);
        if (facts.length) {
          desc.append(el("div", { className: "registry-facts", textContent: facts.join(" · ") }));
        }
        // Where the weights come from. The fields that hold it — hf_repo and
        // hf_filename, or the kaggle pair, or path for a local file — are
        // greyed placeholders further down the form, because the entry
        // supplies them rather than the file; saying it here means the choice
        // can be checked where it is made.
        const where = [d.source, d.hf_repo ?? d.kaggle_handle ?? d.path,
                       d.hf_filename ?? d.kaggle_filename].filter(Boolean);
        if (where.length > 1) {
          desc.append(el("div", { className: "registry-where", textContent: where.join(" · ") }));
        }
      };
      describe();
      select.addEventListener("change", () => { describe(); set(select.value || null); });
      return el("div", {}, [select, desc]);
    }

    case "secret-set": {
      const input = el("input", {
        type: "password", value: "", placeholder: t("form.secret_ph"),
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
    // Inside a source's own block the key is written under that source, so the
    // tooltip names the key within the block rather than the shared path.
    const label = el("label", { textContent: descriptor.label, title: descriptor.name });
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
        el("span", { textContent: t("form.source", { n: i + 1 }) }),
      ]);
      const remove = el("button", {
        type: "button", className: "ghost small", textContent: t("form.remove"),
      });
      remove.addEventListener("click", () => { blocks.splice(i, 1); render(); push(); });
      head.append(remove);

      const row = (label, control) =>
        el("div", { className: "source-field" }, [el("label", { textContent: label }), control]);

      const dir = el("input", {
        type: "text",
        value: Array.isArray(block.data_dir) ? block.data_dir.join(", ") : block.data_dir ?? "",
        placeholder: inherited("data_dir") || t("form.inherited"),
      });
      dir.addEventListener("change", () => {
        const parsed = parseTags(dir.value, "str");
        if (parsed === null) delete block.data_dir;
        else block.data_dir = parsed.length === 1 ? parsed[0] : parsed;
        push();
      });

      const mode = el("select");
      mode.append(el("option", { value: "", textContent:
        t("form.inherited_value", { value: inherited("label_mode") || "subfolders" }) }));
      for (const m of ["subfolders", "table", "file_per_label"]) {
        mode.append(el("option", { value: m, textContent: m }));
      }
      mode.value = block.label_mode ?? "";
      mode.addEventListener("change", () => {
        if (mode.value) block.label_mode = mode.value; else delete block.label_mode;
        push();
      });

      const table = el("input", {
        type: "text", value: block.table_file ?? "",
        placeholder: inherited("table_file") || t("form.inherited"),
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
        ? t("form.aug_inherit") : t("form.aug_inherit_none");
      for (const [key, text] of [["inherit", inheritedNote],
                                 ["own", t("form.aug_own")],
                                 ["none", t("form.aug_none")]]) {
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

      const parts = [head,
                     row(t("form.source_data_dir"), dir),
                     row(t("form.source_label_mode"), mode),
                     row(t("form.source_table_file"), table),
                     row(t("form.source_augmentation"), augMode)];
      if (state === "own" && spec) {
        parts.push(buildAugmentation(block, spec, push));
      }
      wrap.append(el("div", { className: "source-block" }, parts));
    });

    const add = el("button", {
      type: "button", className: "ghost small", textContent: t("form.add_source"),
    });
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

  // The label reads in the interface language, which in every language but
  // English is no longer the key itself — so the key it writes is on the
  // tooltip, and the form still names the file it produces.
  const label = el("label", { textContent: field.label, title: field.path });
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
      className: "badge run-level", textContent: t("form.badge_run_level"),
      title: t("form.badge_run_level_title"),
    }));
  }
  if (secretPaths.includes(field.path)) {
    badges.append(el("span", {
      className: "badge secret", textContent: t("form.badge_secret"),
      title: t("form.badge_secret_title"),
    }));
  }
  if (field.fromRegistry && !isSet) {
    // Shown, not written: the value comes from the registry entry at load time,
    // and typing over it here is what makes it an override in the file.
    badges.append(el("span", {
      className: "badge registry", textContent: t("form.badge_registry", { id: field.fromRegistry }),
      title: t("form.badge_registry_title"),
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
    details.append(el("summary", {
      textContent: t("form.advanced", { count: advanced.length }),
    }));
    const body = el("div");
    addAll(body, advanced);
    details.append(body);
    frag.append(details);
  }
  return frag;
}

function buildNested(field, spec, values, emit, secretPaths) {
  const group = el("div", { className: "group" });
  // The block is on when the file has it — even empty, since `audio_mixup: {}`
  // means "on, with the defaults" — not only when one of its fields is set.
  const present = values[field.path] !== undefined && values[field.path] !== null;
  const enabled = present || Object.keys(values).some((k) => k.startsWith(`${field.path}.`));

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

  // Switching on writes the block (empty: every field at its default) so the
  // redraw after the next edit still finds it; switching off removes it.
  toggle.addEventListener("change", () => {
    body.hidden = !toggle.checked;
    emit(field.path, toggle.checked ? {} : null);
  });
  return group;
}

/* A chosen registry_id supplies the rest of the foundation model — the loader
 * merges those defaults at read time, so the file keeps its one line. The form
 * would otherwise show the dataclass defaults, which are wrong for any backbone
 * that is not BirdNET: Perch's 32 kHz / 5 s window would read as 48 kHz and
 * blank. Substituting them here makes the fields show what the run will
 * actually use, still overridable by typing over them. */
function withRegistryDefaults(section, values) {
  const picker = section.fields.find((f) => f.widget === "registry");
  const chosen = picker && values[picker.path];
  if (!chosen) return section.fields;

  const entry = (picker.choices || []).find((c) => c.id === chosen);
  if (!entry || !entry.defaults) return section.fields;

  return section.fields.map((field) => {
    const supplied = entry.defaults[field.name];
    if (supplied === undefined || field.widget === "registry") return field;
    return { ...field, default: supplied, fromRegistry: chosen };
  });
}

/* The value a path will actually have at load time: what the file says, or the
 * field's default when the file is silent about it. */
function effectiveValue(schema, values, path) {
  if (values[path] !== undefined) return values[path];
  for (const section of schema.sections) {
    for (const field of [...section.fields,
                         ...(section.groups || []).flatMap((g) => g.fields)]) {
      if (field.path === path) return field.default;
    }
  }
  return undefined;
}

/* A group can declare that it only applies for certain values of another field
 * — `{path, in: [...]}` from the schema. Which fields those are is the
 * server's business; this only evaluates the condition. */
function groupApplies(group, schema, values) {
  const when = group.visible_when;
  if (!when) return true;
  return when.in.includes(effectiveValue(schema, values, when.path));
}

/* ── section folding ─────────────────────────────────────────────────── */

/* Which sections are folded away. Held here rather than in the DOM because the
 * form is rebuilt from scratch after every edit, and remembered per browser so
 * a form opens the way it was left. */
const COLLAPSED_KEY = "bioaccx.collapsed";

const stored = (() => {
  try {
    return JSON.parse(localStorage.getItem(COLLAPSED_KEY) ?? "null");
  } catch {
    return null;   // storage disabled, or a stale value
  }
})();

const collapsed = new Set(Array.isArray(stored) ? stored : []);
// Nothing remembered for this browser: every section starts folded, so the
// form opens as a five-line table of contents instead of every field at once.
let seeded = Array.isArray(stored);

const rememberCollapsed = () => {
  try {
    localStorage.setItem(COLLAPSED_KEY, JSON.stringify([...collapsed]));
  } catch {
    // Private browsing, or storage disabled — folding still works, it just
    // will not be remembered.
  }
};

/* The title doubles as the section's show/hide control: five sections of
 * fields is more than any one run cares about, so folding the rest is the
 * difference between scrolling and reading. */
function buildSectionHead(section, body) {
  body.id = `section-${section.name}`;

  const chevron = el("span", { className: "chev", textContent: "▾" });
  chevron.setAttribute("aria-hidden", "true");
  const toggle = el("button", { type: "button", className: "section-toggle" },
                    [chevron, el("span", { textContent: section.title })]);
  toggle.setAttribute("aria-controls", body.id);

  const apply = (open) => {
    body.hidden = !open;
    toggle.classList.toggle("is-closed", !open);
    toggle.setAttribute("aria-expanded", String(open));
  };
  apply(!collapsed.has(section.name));

  toggle.addEventListener("click", () => {
    const open = collapsed.has(section.name);
    if (open) collapsed.delete(section.name);
    else collapsed.add(section.name);
    apply(open);
    rememberCollapsed();
  });
  return el("h2", {}, [toggle]);
}

export function renderForm(root, schema, values, emit, secretPaths) {
  if (!seeded) {
    for (const section of schema.sections) collapsed.add(section.name);
    seeded = true;
  }
  root.replaceChildren();
  for (const section of schema.sections) {
    const body = el("div", { className: "section-body" }, [
      section.help ? el("p", { className: "section-help", textContent: section.help }) : null,
    ]);
    body.append(buildFieldList(withRegistryDefaults(section, values), values, emit,
                               secretPaths, schema.nested));

    for (const group of section.groups || []) {
      if (!groupApplies(group, schema, values)) continue;
      const box = el("div", { className: "group" }, [el("h3", { textContent: group.title })]);
      box.append(buildFieldList(group.fields, values, emit, secretPaths, schema.nested));
      body.append(box);
    }
    root.append(el("section", { className: "section" },
                   [buildSectionHead(section, body), body]));
  }
}
