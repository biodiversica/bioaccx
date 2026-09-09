/* Interface language for the editor.
 *
 * One bundle is held at a time; the server has already merged it over English
 * (see gui/i18n.py), so a key a locale has not translated still reads in
 * English rather than showing the key.
 *
 * Static markup carries `data-i18n` attributes and is re-stamped by
 * applyStatic(); anything built in JavaScript calls t(). The config schema is
 * a third case, handled by translateSchema() below.
 */

let bundle = {};
let language = "en";

/** The string at a dotted key, or null when the bundle has no such key. */
const raw = (key) => {
  const found = key.split(".").reduce(
    (node, part) => (node && node[part] !== undefined ? node[part] : null), bundle);
  return typeof found === "string" ? found : null;
};

/** Translate `key`, substituting `{name}` placeholders from `vars`. */
export function t(key, vars) {
  let out = raw(key);
  if (out === null) return key;
  if (vars) {
    for (const [name, value] of Object.entries(vars)) {
      out = out.split(`{${name}}`).join(value);
    }
  }
  return out;
}

export const currentLang = () => language;

export function setBundle(code, data) {
  language = code;
  bundle = data || {};
  document.documentElement.lang = code;
}

/** Re-stamp every translatable attribute in the static markup. */
export function applyStatic(root = document) {
  const stamp = (attribute, apply) => {
    for (const node of root.querySelectorAll(`[${attribute}]`)) {
      apply(node, t(node.getAttribute(attribute)));
    }
  };
  stamp("data-i18n", (node, text) => { node.textContent = text; });
  stamp("data-i18n-title", (node, text) => { node.title = text; });
  stamp("data-i18n-ph", (node, text) => { node.placeholder = text; });
  stamp("data-i18n-aria", (node, text) => node.setAttribute("aria-label", text));
  stamp("data-i18n-alt", (node, text) => { node.alt = text; });
}

/* The schema is served in English, because that is where its text comes from:
 * the config dataclasses themselves. A locale may translate a section's title
 * and help and any field's label and help, and whatever it leaves out keeps
 * what the dataclass said — so a new config key shows up in every language the
 * day it is added, in English until someone translates it.
 *
 * An English label is the YAML key spelled out ("sample rate" for
 * `sample_rate`); a translated one is not, so form.js puts the dotted path on
 * the label's tooltip and the file stays one hover away. */
export function translateSchema(schema) {
  // Field paths are dotted and are the keys of `fields` verbatim, so these
  // lookups are direct rather than walking the bundle a dot at a time.
  const field = (f) => {
    const translated = bundle.fields?.[f.path] || {};
    const out = { ...f };
    for (const key of ["label", "help"]) {
      if (typeof translated[key] === "string") out[key] = translated[key];
    }
    return out;
  };

  const sections = schema.sections.map((section) => ({
    ...section,
    title: raw(`sections.${section.name}.title`) ?? section.title,
    help: raw(`sections.${section.name}.help`) ?? section.help,
    fields: section.fields.map(field),
    groups: (section.groups || []).map((group) => ({
      ...group,
      title: raw(`sections.${section.name}.groups.${group.name}`) ?? group.title,
      fields: group.fields.map(field),
    })),
  }));

  const nested = Object.fromEntries(
    Object.entries(schema.nested || {}).map(([name, spec]) => [name, {
      ...spec,
      title: raw(`nested.${name}.title`) ?? spec.title,
      fields: spec.fields.map(field),
    }]));

  return { ...schema, sections, nested };
}
