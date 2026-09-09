# Remote sound sources

Use `ext_table_file` to include audio from [iNaturalist](https://www.inaturalist.org/), [Xeno-canto](https://xeno-canto.org/), and/or [Arbimon](https://arbimon.org/) alongside (or instead of) local files.

```yaml
dataset:
  data_dir: /path/to/local_recordings   # optional; omit for remote-only
  ext_table_file: /path/to/observations.csv
```

**Row dispatch** — each row is identified by the first non-empty ID column:

| Priority | Column | Source |
|---|---|---|
| 1 | `observation_id` | iNaturalist observation |
| 2 | `xc_id` | Xeno-canto recording (`12345` or `XC12345`) |
| 3 | `stream_id` | Arbimon 1-minute recording (also requires `date`, `time`, `utc_offset`) |
| 4 | `filename` | Local audio file |

**All columns:**

| Column | Notes |
|---|---|
| `observation_id` | iNaturalist observation ID |
| `sound_index` | 0-based sound index within the observation; defaults to `0` (iNaturalist only) |
| `xc_id` | Xeno-canto recording ID — numeric or with `XC` prefix |
| `stream_id` | Arbimon stream/site ID (requires `date`, `time`, `utc_offset` columns) |
| `date` | Local recording date for Arbimon rows: `YYYY-MM-DD` |
| `time` | Local recording start time for Arbimon rows: `HH:MM` or `HH:MM:SS` |
| `utc_offset` | UTC offset in hours for Arbimon rows (e.g. `-3`, `+5.5`, `UTC-3`, `UTC+5:30`) |
| `filename` | Path to a local audio file (absolute or relative to `data_dir`) |
| `label` | Falls back to the taxon name / stream ID for remote rows if empty |
| `start_time` / `end_time` | Seconds; same partial-time rules as local table mode |
| `split` | `train` or `test`; auto-split if empty |

**Mixed table** — all source types can coexist in one file:

```csv
filename,observation_id,sound_index,xc_id,stream_id,date,time,utc_offset,label,start_time,end_time
/data/rec.wav,,,,,,,,cicada,0.0,3.0
,12345678,0,,,,,,,1.0,6.0
,,,98765,,,,, Turdus merula,,,train
,,,,abc123,2023-07-14,06:00,-3,Guira guira,10.0,30.0
```

**Caching** — audio files and metadata are cached locally on first download; subsequent runs skip the network entirely.

```yaml
dataset:
  ext_cache_dir: /path/to/cache   # default: ~/.cache/bioaccx/ext
```

## iNaturalist and Xeno-canto

Rows are identified by `observation_id` (iNaturalist) or `xc_id` (Xeno-canto). Labels fall back to the taxon scientific name fetched from the respective API when the `label` column is empty.

**Xeno-canto API key** — Xeno-canto uses API v3, which requires a personal key for metadata queries (scientific name lookup). Without a key, audio is still downloaded directly but the label falls back to `"xc_<id>"` unless you set it explicitly in the table. Register at [xeno-canto.org/explore/api](https://xeno-canto.org/explore/api).

```yaml
dataset:
  xc_api_key: YOUR_KEY_HERE   # optional; enables scientific name lookup for XC rows
```

## Arbimon

Each Arbimon row identifies a specific 1-minute recording by its stream (site) ID and local timestamp. The label falls back to `stream_id` when the `label` column is empty.

**Required columns:**

| Column | Default name | Description |
|---|---|---|
| `stream_id` | `stream_id` | Arbimon recording site / stream ID |
| `date` | `date` | Local recording date, ISO format: `YYYY-MM-DD` |
| `time` | `time` | Local recording start time: `HH:MM` or `HH:MM:SS` |
| `utc_offset` | `utc_offset` | UTC offset of the local time in hours (e.g. `-3`, `+5.5`, `UTC-3`, `UTC+5:30`) |

**Config:**

```yaml
dataset:
  ext_table_file: /path/to/observations.csv
  arbimon_credentials_path: /path/to/.rfcx_credentials   # required
  # arbimon_stream_id_col: stream_id   # column name overrides (optional)
  # arbimon_date_col: date
  # arbimon_time_col: time
  # arbimon_utc_offset_col: utc_offset
```

**Authentication** — Arbimon uses the rfcx SDK for downloads. Authenticate once to create a credentials file:

```python
import rfcx
client = rfcx.Client()
client.authenticate(persisted_credentials_path="/path/to/.rfcx_credentials")
```

This opens a browser URL for device authorisation and saves a token to disk. All subsequent runs load the token from that file without any user interaction.

**Installing the rfcx SDK** — the SDK is not on PyPI; install it directly from the GitHub release:

```bash
pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl
```

**Caching** — downloaded audio is stored under `<ext_cache_dir>/arbimon/<stream_id>/`. A lightweight sentinel file is written for each downloaded minute so that repeated runs skip the network entirely and locate the audio file without re-scanning the directory.
