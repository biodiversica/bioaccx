# The browser GUI

`bioaccx gui` opens a configuration editor in the browser. It writes the same
YAML files that every other command reads, and runs can still be started from
the terminal.

```bash
uv tool install "bioaccx[cpu,gui]"      # the GUI requires the [gui] extra

bioaccx gui                             # start from a template
bioaccx gui my_config.yaml              # open an existing configuration
bioaccx gui --port 9000 --no-open       # choose a port without opening a browser
bioaccx gui --lang pt-BR                # open the interface in Portuguese
```

The form is **generated from the configuration dataclasses**, so it always offers
exactly the keys supported by this version of bioaccx, with their actual defaults
and the documentation written next to them in the source code. There is no
second copy of the schema that could become outdated.

Compared with a text editor, it offers the following:

- **The registry picker.** Choosing a backbone shows the values it uses (sample
  rate, window, embedding size and tensor names), read from the same registry
  that `bioaccx registry` prints and marked `From 0xbb10`. Below them, a second
  line states where the weights are obtained from (`huggingface ·
  biodiversica/BirdNET-onnx-backbone · model_backbone.onnx`, the Kaggle handle,
  or a local path), so the download implied by an ID is visible in the
  drop-down list and not only in the greyed-out `hf_repo` and `hf_filename`
  fields further down. These values are displayed rather than written: the file
  keeps its single `registry_id:` line, the loader merges the remaining values
  when the file is read, and only a value entered in a field becomes an actual
  override.
- **Only applicable settings.** The Keras and scikit-learn sections are shown
  according to `training.classifier`, so settings for a head that will not be
  trained are hidden. Switching between them leaves the settings of the other
  head in the file.
- **Validation before the run.** The draft is checked with the same
  `load_config` function that the CLI uses, so an invalid `registry_id`, an
  augmentation block without a noise source, or a grouped softmax without
  `label_groups` is detected during editing rather than shortly after a long run
  has started.
- **Discoverability.** Around 120 configuration keys, divided into common and
  advanced settings, are presented with their help text, so the
  [config reference](config-reference.md) does not need to be consulted.
- **Comments are preserved.** Files are read and written with a
  comment-preserving YAML round trip: opening `example_config.yaml` and saving it
  again leaves it byte-for-byte unchanged, and editing two fields changes only
  those two fields.
- **A compact layout.** The five sections are initially folded, presenting a
  short table of contents rather than 120 fields, and are unfolded by clicking
  their title; this state is remembered per browser. The button at the right of
  the toolbar cycles through the themes: ◐ follows the system setting, ☀ light,
  ☾ dark.
- **English or Portuguese.** The selector next to the theme button switches the
  interface language; `--lang pt-BR` sets the language for a browser in which no
  language has been chosen. Field names, help text, sections, buttons and
  messages are all translated, and hovering over a field name shows the YAML key
  it writes. See [Interface languages](#interface-languages).

## Browsing what a run produced

The **Analysis** tab reads `custom_models/` (or `--models-dir`) and presents the
files that every run already writes in a form that can be explored. Nothing is
recomputed, so it also works with models trained before this feature existed.

- **Model list:** every run is shown as a card with the backbone, the number of
  classes, the macro F1 taken from the evaluation table (and *Included F1*, the
  macro F1 without the `exclude_labels` classes, when the run excluded any),
  whether it has an embedding map and, for a run trained with audio mixup, the
  number of synthetic mixes it was trained on. The directory can be edited at
  the top of the list, so results can be read from any location without
  restarting; `--models-dir` only sets the initial directory.
- **Metrics:** the per-class evaluation table, which can be sorted. Clicking
  *F1* brings the weakest classes to the top, and values below 0.5 are
  highlighted. Classes in `exclude_labels` are greyed out and excluded from the
  *Macro average (included labels)* row. For runs trained before that row
  existed, it is computed from the per-class rows, and the *Comparison* view
  also includes it. For a run trained with audio mixup, each class additionally
  shows its real training windows, the mixes whose target contains it (hovering
  shows which classes it was mixed with) and its mix share, that is, the
  synthetic proportion of its training signal, next to the scores, which are
  always computed on real test windows. Its mixed test set, when built, has a
  separate table below.
- **Map:** the UMAP projection on a canvas, coloured by label or by KMeans
  cluster and filterable by split. The colours are generated for the number of
  classes actually present, so no two classes share a colour, and they are the
  same as in the `_umap.png` figure, so the map and the figure can be compared
  directly. **Clicking a point plays the clip it was computed from**, with a
  spectrogram next to it. This is not possible from the CLI and shows *which
  recordings* lie in a region where classes are confused.
- **Comparison:** two runs side by side, with the per-class F1 differences and
  the exact settings that differ between them, read from the metadata of each
  run. It shows the effect of a change, such as a different `hidden_units`
  value, without comparing two reports manually. When either run used audio
  mixup, the mix share of each class in both runs is shown next to its F1
  change, and the mixup settings are included in the comparison.

To play a point, the GUI must identify its sample. Runs therefore write a `key`
column in `<model>_umap.csv`. Projections written before this column existed
are still supported: their keys are inferred from the row order, but only when
the projection and the dataset list have the same length *and* agree label by
label. Otherwise, the map states that the points cannot be traced to their audio
rather than guessing. The map indicates which of the two methods was used.

A point plays the window that the model embedded. When the run exported its
dataset (`export_dataset`), this is the exported WAV file in `<model>/dataset/`.
Otherwise, the window is cut from the source recording listed in
`<model>_dataset_list.csv` and rebuilt as it was embedded: a noisy copy is mixed
with its noise file at its SNR, and a short signal is placed at its random
offset. Dataset lists written before the noise path was recorded contain only
the name of the noise file, which is then looked up in the `augmentation_dir`
of the run. These lists also lack the signal duration, so their shifted windows
are played without the shift. If the audio has been moved or is stored on
another machine, the point still shows its metadata and reports that the file
is missing instead of failing.

## Interface languages

The editor is available in English and Brazilian Portuguese. The selector in the
toolbar switches the language without reloading the page and remembers the
choice in that browser; `bioaccx gui --lang pt-BR` sets the language for a
browser in which no language has been chosen.

```bash
bioaccx gui --lang pt-BR        # opens in Portuguese
bioaccx gui -l pt               # any tag that can be resolved; unknown tags produce a warning and use en
```

Adding a language requires a single file: placing `<code>.json` in
`bioaccx/gui/locales/` next to `en.json` adds it to the selector. A locale file
needs to contain only the strings it translates. The server merges it over the
English strings, so an incomplete translation shows English for the remaining
text instead of raw keys.

Two aspects are intentional:

- **The key is always shown on hover.** Field names are written out in full in
  every language (`hf_repo` is shown as *Hugging Face repository* in English
  and *Repositório do Hugging Face* in Portuguese), so every field name shows
  its dotted configuration path as a tooltip, and the form still identifies the
  file it writes.
- **English text is defined in the source code, not in `en.json`.** The field
  help text comes from the configuration dataclasses, and English field names
  are derived from the keys, so `en.json` contains only the strings created by
  the browser interface itself. A configuration key added to `config.py`
  therefore appears in every language as soon as it is added: in English until
  it is translated, and never as an empty value or a missing key.

Runs are not translated: the log streamed to the page is the output of the CLI
itself, identical to what the terminal would show.

## Starting a run from the editor

The **Run** tab runs the configuration named in the path field. After selecting
`train`, `dataset`, `embeddings` or `validate` and pressing **Run**, the output
is streamed to the page as it is produced, with a progress bar driven by the
`[3/5]` step markers that the pipeline already prints. **Cancel** sends the same
interrupt as Ctrl+C, so the run stops in the usual way. While a run is in
progress, a dot on the **Run** tab indicates this from the other tabs.

Three aspects are intentional:

- **The run reads the file on disk, not the draft in the browser.** A run with
  unsaved changes is refused with a message rather than started, so every run
  can always be reproduced from a terminal.
- **The equivalent command is shown next to the button**, together with a copy
  button. Everything the editor does is also available from the shell.
- **The run is a separate process in its own session.** It does not share
  memory with the editor, a crash cannot stop the editor, and closing the
  browser, or restarting the server, does not stop training. Reopening the page
  reconnects to a run that is already in progress.

Only these four commands can be started. The model-surgery commands (`merge`,
`extract-head`, `convert-head`) take model paths rather than a configuration
file, so they remain terminal commands. Only one run is allowed at a time: a
second run is refused rather than queued, because two runs on one machine
compete for the same CPU, GPU and output directory.

### Per-source augmentation

Each source has one of three states: **inherit** the shared block, **own
block**, or **no augmentation**. Choosing *own block* shows the full set of
augmentation settings for that source alone. The own block of a source replaces
the inherited one entirely:

```yaml
dataset:
  label_mode: subfolders
  augmentation:                     # shared by sources that do not define their own
    augmentation_dir: /noise/general
    snr_levels: [20, 10]

  sources:
    - data_dir: /audio/quiet_site
      augmentation:                 # its own block, which replaces the shared one
        augmentation_dir: /noise/rain
        snr_levels: [3]

    - data_dir: /audio/clean_recordings
      augmentation: null            # no augmentation

    - data_dir: /audio/normal_site  # no setting: inherits the shared block
```

Switching a source between the three states keeps its own block, so switching
to *no augmentation* and back does not discard settings. Run-level fields
(`test_ratio`, `random_seed`, `audio_extensions` and credentials) can never be
overridden per source and are marked **Run-level** in the form.

The file being written is shown next to the form during editing, so the content
displayed is exactly what is written to disk. Selecting **Edit YAML directly**
allows YAML to be entered in that pane instead, so that settings not covered by
the form can be written by hand.

The preview is hidden by default and is shown or hidden with **Show file** /
**Hide** in its header. The validation summary remains visible in both cases,
and the choice is remembered.

Secrets (`xc_api_key`) are never sent back to the browser: an existing key is
shown as set, and leaving the field empty keeps it unchanged in the file.

By default, the server binds to `127.0.0.1` and can be reached only from the
local machine. Binding to any other address generates an access token, which is
included in the printed link and then stored as a cookie; `--no-auth` disables
it.

To access the editor from another machine, it is preferable to bind only the
required interface rather than all of them:

```bash
bioaccx gui --host "$(tailscale ip -4)"   # tailnet only, not the local network
bioaccx gui --host 192.168.0.30           # this local network only
bioaccx gui --host 0.0.0.0                # every interface; prints each URL
```

Over Tailscale, the traffic is already encrypted end to end with WireGuard, so
the token can be transmitted safely over plain HTTP within the tailnet. The
editor must not be exposed with `tailscale funnel`, which publishes it to the
public internet, because the editor can read and write files in any location
accessible to the user running it.
