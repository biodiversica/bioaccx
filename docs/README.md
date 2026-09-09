# bioaccx documentation

- **English** — [`docs/en/`](en/), indexed from the [README](../README.md)
- **Português (Brasil)** — [`docs/pt-BR/`](pt-BR/), indexado pelo
  [README.pt-BR](../README.pt-BR.md)

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
and its content hash. When the English page changes, `--check` says which
translations are behind — a page that is out of date says so instead of quietly
diverging.
