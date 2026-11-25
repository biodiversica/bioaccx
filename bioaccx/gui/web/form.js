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
        select.append(el("option", { value: "", textContent: field.unsetLabel ?? t("form.unset") }));
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
      return el("div", { className: "paths" }, [area, el("div", { className: "row" }, [browse])]);
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

/* Settings picked from "Add a setting…" that have no value yet, as
 * "index:name". Kept outside the form because it is rebuilt after every edit,
 * and an empty row would otherwise vanish before anything is typed into it. */
const pendingOverrides = new Set();

const showValue = (v) => (Array.isArray(v) ? v.join(", ") : String(v));

/* The source blocks. `perSource` holds the dataset fields a source may set
 * (every field but the run-level ones); `runLevel` those it may not, so a
 * block that sets one anyway can say that it is ignored. */
function buildSources(field, layout, value, emit, values, nested, perSource, runLevel) {
  const blocks = Array.isArray(value) ? value.map((b) => ({ ...b })) : [];
  const wrap = el("div", { className: "sources-list" });

  const push = () => emit(field.path, blocks.length ? blocks : null);
  const byName = new Map(perSource.map((f) => [f.name, f]));
  const runNames = new Set(runLevel.map((f) => f.name));
  const nestedField = perSource.find((f) => f.widget === "nested");
  const spec = nestedField && nested?.[nestedField.dataclass];
  // Own blocks survive a trip through "inherit" or "no augmentation" and back.
  const remembered = new Map();

  /* One setting of one source, bound to the block: a value it leaves unset
   * shows what it inherits — the shared value, or the default. */
  const settingRow = (block, i, descriptor, removable) => {
    const shared = values[descriptor.path];
    const inherited = shared !== undefined ? shared : descriptor.default;
    const bound = {
      ...descriptor, path: descriptor.name, default: inherited, optional: true,
      unsetLabel: inherited == null ? t("form.inherited")
        : t("form.inherited_value", { value: showValue(inherited) }),
    };
    const control = buildControl(bound, block[descriptor.name], (name, next) => {
      if (next === null) delete block[name];
      else block[name] = next;
      pendingOverrides.delete(`${i}:${name}`);
      push();
    });
    const cell = el("div", { className: "cell" }, [control]);
    if (descriptor.help) cell.append(el("div", { className: "help", textContent: descriptor.help }));
    return sourceRow(block, i, descriptor.name, descriptor.label, cell, removable);
  };

  const sourceRow = (block, i, name, labelText, cell, removable) => {
    const label = el("label", { textContent: labelText, title: name });
    let tail = el("span");
    if (removable) {
      tail = el("button", {
        type: "button", className: "ghost small source-unset", textContent: "✕",
        title: t("form.remove_setting"),
      });
      tail.setAttribute("aria-label", t("form.remove_setting"));
      tail.addEventListener("click", () => {
        pendingOverrides.delete(`${i}:${name}`);
        const written = name in block;
        delete block[name];
        render();
        if (written) push();
      });
    }
    return el("div", { className: "source-field" }, [label, cell, tail]);
  };

  /* Augmentation is one of three states, not a checkbox: a source can inherit
   * the shared block, replace it with its own, or opt out. The previous own
   * block is remembered, so flipping through the states does not discard
   * settings. */
  const augmentationRows = (block, i) => {
    const name = nestedField.name;
    const own = block[name];
    const state = own === null ? "none" : own === undefined ? "inherit" : "own";
    if (state === "own") remembered.set(i, own);

    const augMode = el("select");
    const inheritedNote = values[nestedField.path]
      ? t("form.aug_inherit") : t("form.aug_inherit_none");
    for (const [key, text] of [["inherit", inheritedNote],
                               ["own", t("form.aug_own")],
                               ["none", t("form.aug_none")]]) {
      augMode.append(el("option", { value: key, textContent: text }));
    }
    augMode.value = state;
    augMode.addEventListener("change", () => {
      if (augMode.value === "none") block[name] = null;
      else if (augMode.value === "inherit") delete block[name];
      else block[name] = remembered.get(i) ?? { snr_levels: [10] };
      render();
      push();
    });
    const rows = [sourceRow(block, i, name, nestedField.label,
                            el("div", { className: "cell" }, [augMode]), false)];
    if (state === "own" && spec) rows.push(buildAugmentation(block, spec, push));
    return rows;
  };

  const render = () => {
    wrap.replaceChildren();
    blocks.forEach((block, i) => {
      const head = el("div", { className: "source-head" }, [
        el("span", { textContent: t("form.source", { n: i + 1 }) }),
      ]);
      // The last source goes by switching to one source, which keeps its
      // settings, rather than by removing it, which would not.
      if (blocks.length > 1) {
        const remove = el("button", {
          type: "button", className: "ghost small", textContent: t("form.remove"),
        });
        remove.addEventListener("click", () => {
          blocks.splice(i, 1);
          pendingOverrides.clear();
          render();
          push();
        });
        head.append(remove);
      }

      const parts = [head];
      const shown = new Set();
      for (const name of layout.primary) {
        const descriptor = byName.get(name);
        if (!descriptor) continue;
        parts.push(settingRow(block, i, descriptor, false));
        shown.add(name);
      }
      if (nestedField) {
        parts.push(...augmentationRows(block, i));
        shown.add(nestedField.name);
      }

      // Everything else the block sets, then what was just added, in form order.
      const extra = [...Object.keys(block)];
      for (const key of pendingOverrides) {
        const [index, name] = key.split(":");
        if (Number(index) === i && !extra.includes(name)) extra.push(name);
      }
      for (const name of extra) {
        if (shown.has(name)) continue;
        shown.add(name);
        const descriptor = byName.get(name);
        if (descriptor) {
          parts.push(settingRow(block, i, descriptor, true));
        } else {
          // A run-level key is ignored here, and an unknown one is not ours to
          // interpret; both stay visible, so the block hides nothing it holds.
          const cell = el("div", { className: "cell" }, [
            buildControl({ widget: "raw", path: name }, block[name], (key, next) => {
              if (next === null) delete block[key]; else block[key] = next;
              push();
            }),
          ]);
          if (runNames.has(name)) {
            cell.append(el("div", { className: "help warn", textContent: t("form.ignored_run_level") }));
          }
          const label = runLevel.find((f) => f.name === name)?.label ?? name;
          parts.push(sourceRow(block, i, name, label, cell, true));
        }
      }

      const available = perSource.filter((f) => !shown.has(f.name) && f.widget !== "nested");
      if (available.length) {
        const add = el("select", { className: "source-add" });
        add.setAttribute("aria-label", t("form.add_setting"));
        add.append(el("option", { value: "", textContent: t("form.add_setting") }));
        for (const f of available) add.append(el("option", { value: f.name, textContent: f.label }));
        add.addEventListener("change", () => {
          if (!add.value) return;
          pendingOverrides.add(`${i}:${add.value}`);
          render();
        });
        parts.push(el("div", { className: "row source-add-row" }, [add]));
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

/* The one-or-several switch at the top of the dataset section.
 *
 * Going to several sources moves the fields that locate the audio into
 * Source 1, so the run reads the same data and a second source does not
 * inherit them. Going back folds Source 1 into the top-level block: its own
 * values win, and the other sources are removed once that is confirmed. */
function buildSourceMode(layout, values, emit, several, runNames) {
  const blocks = values[layout.field];
  const name = `source-mode-${layout.field}`;
  const radio = (checked, text) => {
    const input = el("input", { type: "radio", name, checked });
    return [input, el("label", { className: "inline" }, [input, ` ${text}`])];
  };
  const [one, oneLabel] = radio(!several, t("form.mode_single"));
  const [many, manyLabel] = radio(several, t("form.mode_several"));

  many.addEventListener("change", () => {
    const section = layout.field.split(".").slice(0, -1).join(".");
    const first = {};
    const edits = {};
    for (const key of layout.moved) {
      const path = `${section}.${key}`;
      if (values[path] === undefined) continue;
      first[key] = values[path];
      edits[path] = null;
    }
    edits[layout.field] = [first];
    emit(edits);
  });

  one.addEventListener("change", () => {
    const others = Array.isArray(blocks) ? blocks.length - 1 : 0;
    if (others > 0 && !window.confirm(t("form.mode_single_confirm", { count: others }))) {
      many.checked = true;
      return;
    }
    const section = layout.field.split(".").slice(0, -1).join(".");
    const edits = {};
    for (const [key, value] of Object.entries((blocks && blocks[0]) || {})) {
      if (runNames.has(key)) continue;   // ignored in a source; not promoted either
      edits[`${section}.${key}`] = value;   // null removes, as "no augmentation" should
    }
    edits[layout.field] = null;
    emit(edits);
  });

  const box = el("div", { className: "source-mode" });
  box.setAttribute("role", "radiogroup");
  box.setAttribute("aria-label", t("form.mode_aria"));
  box.append(el("div", { className: "row" }, [oneLabel, manyLabel]));
  box.append(el("div", {
    className: "help",
    textContent: several ? t("form.mode_several_help") : t("form.mode_single_help"),
  }));
  return box;
}

/* A titled box with a one-line explanation under its title. */
function fieldGroup(title, help, content) {
  const box = el("div", { className: "group" }, [el("h3", { textContent: title })]);
  if (help) box.append(el("p", { className: "group-help", textContent: help }));
  box.append(content);
  return box;
}

/* The dataset section, laid out by how many sources it has.
 *
 * One source: its fields, then the run settings. Several: the source blocks
 * first (they are the data), then the settings they share, then the run
 * settings, which no source can override. */
function buildSourcesSection(section, fields, values, emit, secretPaths, nested) {
  const layout = section.sources;
  const frag = document.createDocumentFragment();
  const sourcesField = fields.find((f) => f.path === layout.field);
  const blocks = values[layout.field];
  const several = Array.isArray(blocks) && blocks.length > 0;
  const runLevel = fields.filter((f) => f.run_level);
  const perSource = fields.filter((f) => !f.run_level && f !== sourcesField);
  const runNames = new Set(runLevel.map((f) => f.name));

  frag.append(buildSourceMode(layout, values, emit, several, runNames));

  if (several) {
    frag.append(fieldGroup(t("form.group_sources"), sourcesField?.help,
      buildSources(sourcesField, layout, blocks, emit, values, nested, perSource, runLevel)));
    // A field that locates audio is per source; it is listed as shared only
    // when the top-level block still sets it, so nothing in the file is hidden.
    const shared = perSource.filter((f) =>
      !layout.moved.includes(f.name) || values[f.path] !== undefined);
    frag.append(fieldGroup(t("form.group_shared"), t("form.group_shared_help"),
      buildFieldList(shared, values, emit, secretPaths, nested)));
  } else {
    pendingOverrides.clear();
    frag.append(buildFieldList(perSource, values, emit, secretPaths, nested));
  }

  frag.append(fieldGroup(t("form.group_run"), t("form.group_run_help"),
    buildFieldList(runLevel, values, emit, secretPaths, nested)));
  return frag;
}

/* ── field + section assembly ────────────────────────────────────────── */

function buildField(field, values, emit, secretPaths) {
  const value = values[field.path];
  const isSet = value !== undefined;

  // The label reads in the interface language, which in every language but
  // English is no longer the key itself — so the key it writes is on the
  // tooltip, and the form still names the file it produces.
  const label = el("label", { textContent: field.label, title: field.path });
  if (field.required) label.append(el("span", { className: "req", textContent: "*" }));

  const control = el("div", { className: "control" });

  let widget;
  if (secretPaths.includes(field.path) && value === "__SET__") {
    widget = buildControl({ ...field, widget: "secret-set" }, value, emit);
  } else if (secretPaths.includes(field.path)) {
    widget = buildControl({ ...field, widget: "text" }, value, emit);
  } else {
    widget = buildControl(field, value, emit);
  }
  control.append(widget);

  const badges = el("div", { className: "row" });
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
      target.append(buildField(field, values, emit, secretPaths));
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
  head.append(el("label", { className: "inline" }, [toggle, ` ${spec.title}`]));
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
    const fields = withRegistryDefaults(section, values);
    body.append(section.sources
      ? buildSourcesSection(section, fields, values, emit, secretPaths, schema.nested)
      : buildFieldList(fields, values, emit, secretPaths, schema.nested));

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
