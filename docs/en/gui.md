# The browser GUI

`bioaccx gui` opens a config editor in a browser. It writes the same YAML files
every other command reads — the terminal stays the way runs are started.

```bash
uv tool install "bioaccx[cpu,gui]"      # the GUI needs the [gui] extra

bioaccx gui                             # start from a template
bioaccx gui my_config.yaml              # open an existing config
bioaccx gui --port 9000 --no-open       # pick a port, don't launch a browser
bioaccx gui --lang pt-BR                # open it in Portuguese
```

The form is **generated from the config dataclasses**, so it always offers
exactly the keys this version of bioaccx understands, with their real defaults
and the documentation written beside them in the source. There is no second
copy of the schema to drift.

What it gives you over a text editor:

- **The registry picker.** Choosing a backbone shows what it actually uses —
  sample rate, window, embedding size, tensor names — read from the same
  registry `bioaccx registry` prints, and marked `from 0xbb10`. Under those, a
  second line says where the weights come from — `huggingface ·
  biodiversica/BirdNET-onnx-backbone · model_backbone.onnx`, or the Kaggle
  handle, or a local path — so the download an ID implies is visible at the
  drop-down rather than only in the greyed `hf_repo` / `hf_filename` fields
  further down. The values are shown rather than written: the file keeps its one
  `registry_id:` line, the loader merges the rest at read time, and typing over
  a field is what makes it a real override.
- **Only what applies.** The Keras and sklearn boxes appear according to
  `training.classifier`, so you are not scrolling settings for a head that will
  not be trained. Switching between them leaves the other one's settings in the
  file.
- **Validation before the run.** The draft is checked with the same
  `load_config` the CLI uses, so a bad `registry_id`, an augmentation block with
  no noise source, or a grouped softmax without `label_groups` is caught while
  you edit rather than thirty seconds into a three-hour run.
- **Discoverability.** Around 120 config keys, tiered into common and advanced,
  each with its help text — rather than scrolling the
  [config reference](config-reference.md).
- **Your comments survive.** Files are read and written through a
  comment-preserving YAML round-trip: opening `example_config.yaml` and saving
  it back is a byte-for-byte no-op, and editing two fields changes only those
  two fields.
- **It stays out of the way.** The five sections open folded — a short table of
  contents rather than 120 fields — and unfold with a click on their title,
  remembered per browser. The button at the right of the toolbar cycles the
  theme: ◐ follows your system, ☀ light, ☾ dark.
- **English or Portuguese.** The picker beside the theme button switches the
  interface; `--lang pt-BR` decides what a browser that has never chosen sees.
  Field labels, help text, sections, buttons and messages are all translated —
  and because a translated label no longer spells out the YAML key, hovering it
  shows the key it writes. See [Interface languages](#interface-languages).

## Browsing what a run produced

The **models** tab reads `custom_models/` (or `--models-dir`) and turns the
files every run already writes into something you can interrogate. Nothing is
recomputed, so it works on models trained long before this existed.

- **Model list** — every run as a card: backbone, class count, macro F1 pulled
  from the evaluation table (plus *F1 incl.*, the macro F1 without the
  `exclude_labels` classes, when the run excluded any), and whether it has an
  embedding map. The directory
  is editable at the top of the list, so results can be read from anywhere
  without restarting; `--models-dir` only sets where it starts.
- **Metrics** — the per-class evaluation table, sortable. Click *F1* to bring
  the weakest classes to the top; anything under 0.5 is marked. Classes in
  `exclude_labels` are greyed out and left out of the
  `OVERALL (Macro-avg, included)` row. For runs trained before that row existed,
  it is computed from the per-class rows; *Compare* lines it up too.
- **Map** — the UMAP projection on a canvas, coloured by label or by KMeans
  cluster and filterable by split. The colours are generated for the number of
  classes actually present — never two classes the same colour — and are the
  ones the `_umap.png` figure uses, so the map and the figure read alike. **Click a point to hear the clip it came
  from**, with a spectrogram beside it. That is the thing the CLI cannot do:
  seeing *which recordings* sit in a confused region.
- **Compare** — two runs side by side, with per-class F1 deltas and the exact
  settings that differ between them, read from each run's metadata. It answers
  "what did changing `hidden_units` actually do" without diffing two reports by
  eye.

Clicking a point needs to know which sample it is. Runs now write a `key`
column in `<model>_umap.csv` for exactly that. Projections written before that
column existed still work: their keys are inferred from row order, but only
when the projection and the dataset list are the same length *and* agree label
for label — otherwise the map says the points cannot be traced back rather than
guessing. The map notes which of the two it used.

Source audio is read from the paths recorded in `<model>_dataset_list.csv`. If
a dataset has moved or lives on another machine, the point still shows its
metadata and says the file is missing instead of failing.

## Interface languages

The editor ships in English and Brazilian Portuguese. The picker in the toolbar
switches it without a reload and remembers the choice in that browser;
`bioaccx gui --lang pt-BR` sets what a browser that has never chosen sees.

```bash
bioaccx gui --lang pt-BR        # opens in Portuguese
bioaccx gui -l pt               # any tag that resolves; unknown ones warn and use en
```

Adding a language is one file: drop `<code>.json` into `bioaccx/gui/locales/`
next to `en.json`, and it appears in the picker. A locale only has to carry
what it translates — the server merges it over English, so a half-finished
translation reads English for the rest instead of showing raw keys.

Two things are deliberate:

- **The key is always one hover away.** In English a field's label *is* its key
  spelled out (`sample_rate` reads as *sample rate*); translated, it is not — so
  every label carries its dotted config path as a tooltip, and the form still
  names the file it writes.
- **English lives in the source, not in `en.json`.** The field labels and help
  text come from the config dataclasses, so `en.json` holds only the strings the
  browser itself invents. A config key added to `config.py` therefore appears in
  every language the day it is added — in English until someone translates it,
  never as a blank or a missing key.

Runs are not translated: the log streamed into the page is the CLI's own
output, byte for byte what the terminal would show.

## Starting a run from the editor

Below the file preview is a run panel: pick `train`, `dataset`, `embeddings` or
`validate`, press **Run**, and the output streams into the page as it happens,
with a progress bar driven by the `[3/5]` step markers the pipeline already
prints. **Cancel** sends the same interrupt Ctrl+C would, so the run stops the
way it always has.

Three things are deliberate:

- **The run reads the file on disk, not the draft in the browser.** Unsaved
  changes are refused with a message rather than silently run, so whatever ran
  can always be reproduced from a terminal.
- **The equivalent command is shown next to the button** (with a copy button).
  Nothing the editor does is unavailable from the shell.
- **The run is a separate process in its own session.** It does not share
  memory with the editor, a crash cannot take the editor down, and closing the
  browser — or restarting the server — does not stop training. Reopening the
  page reattaches to a run already in progress.

Only those four commands can be launched. The model-surgery commands
(`merge`, `extract-head`, `convert-head`) take model paths rather than a
config, so they stay in the terminal. One run at a time: a second is refused
rather than queued, because two runs on one machine compete for the same CPU,
GPU and output directory.

### Per-source augmentation

Each source picks one of three states — **inherit** the shared block, **own
block**, or **no augmentation** — and choosing *own block* opens the full set of
augmentation settings for that source alone. A source's own block replaces the
inherited one entirely:

```yaml
dataset:
  label_mode: subfolders
  augmentation:                     # shared by sources that say nothing
    augmentation_dir: /noise/general
    snr_levels: [20, 10]

  sources:
    - data_dir: /audio/quiet_site
      augmentation:                 # its own block, replaces the shared one
        augmentation_dir: /noise/rain
        snr_levels: [3]

    - data_dir: /audio/clean_recordings
      augmentation: null            # opts out entirely

    - data_dir: /audio/normal_site  # says nothing, inherits the shared block
```

Switching a source between the three states keeps its own block, so flipping to
*no augmentation* and back does not discard settings. Run-level fields
(`test_ratio`, `random_seed`, `audio_extensions`, credentials) are never
overridable per source and are marked `run level` in the form.

The file being written is shown beside the form as you edit, so what you see is
exactly what lands on disk. Tick **edit directly** to type YAML into that pane
instead — anything the form does not cover can be written by hand.

That preview is hidden by default and toggled with **Show file** / **Hide** in
its header; the validation summary and the run panel stay visible either way,
and the choice is remembered.

Secrets (`xc_api_key`) are never sent back to the browser: an existing key shows
as set, and leaving the field blank keeps it unchanged in the file.

By default the server binds `127.0.0.1` and is reachable only from this machine.
Binding anything else generates an access token, included in the printed link
and then stored as a cookie; `--no-auth` disables that.

To reach the editor from another machine, prefer binding that one interface
rather than everything:

```bash
bioaccx gui --host "$(tailscale ip -4)"   # tailnet only — not your LAN
bioaccx gui --host 192.168.0.30           # this LAN only
bioaccx gui --host 0.0.0.0                # every interface; prints each URL
```

Over Tailscale the traffic is already WireGuard-encrypted end to end, so the
token travels safely on plain HTTP within the tailnet. Do not put this behind
`tailscale funnel` — that publishes it to the open internet, and the editor can
read and write files anywhere the user running it can.
