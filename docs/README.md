# bioaccx documentation

- **English** — [`docs/en/`](en/), indexed from the [README](../README.md)
- **Português (Brasil)** — [`docs/pt-BR/`](pt-BR/), indexado pelo
  [README.pt-BR](../README.pt-BR.md)

A translated page has the same file name in both languages, and each copy links
to the other on the line under its title, which reads "English · Português
(Brasil)" with the other language as the link. Pages that exist only in
English have no such line. `README.md` is also the package description on PyPI,
where relative links do not resolve, so its links are absolute GitHub URLs.

Two pages are generated from the code by `tools/gen_docs.py` and should not be
edited by hand between their `<!-- generated: -->` markers:

| Page | Generated from |
|---|---|
| [`en/config-reference.md`](en/config-reference.md), [`pt-BR/config-reference.md`](pt-BR/config-reference.md) | the dataclasses in `bioaccx/config.py`, with the Portuguese text from `bioaccx/gui/locales/pt-BR.json` |
| [`en/cli.md`](en/cli.md) | the Typer app in `bioaccx/cli.py` |

```bash
python tools/gen_docs.py            # rebuild the generated blocks
python tools/gen_docs.py --stamp    # record what a translation was written from
python tools/gen_docs.py --check    # what CI (tests/unit/test_docs.py) asserts
```

Each translated page carries a `translated-from` stamp naming the English page
and its content hash. When the English page changes, `--check` reports which
translations are out of date, so that an outdated translation is detected
rather than diverging unnoticed.
