# Remote sound sources

`ext_table_file` includes audio from [iNaturalist](https://www.inaturalist.org/), [Xeno-canto](https://xeno-canto.org/) and [Arbimon](https://arbimon.org/), in addition to or instead of local files.

```yaml
dataset:
  data_dir: /path/to/local_recordings   # optional; omit when all audio is remote
  ext_table_file: /path/to/observations.csv
```

**Row dispatch:** each row is identified by its first non-empty ID column.

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
| `xc_id` | Xeno-canto recording ID, numeric or with the `XC` prefix |
| `stream_id` | Arbimon stream/site ID (requires `date`, `time`, `utc_offset` columns) |
| `date` | Local recording date for Arbimon rows: `YYYY-MM-DD` |
| `time` | Local recording start time for Arbimon rows: `HH:MM` or `HH:MM:SS` |
| `utc_offset` | UTC offset in hours for Arbimon rows (for example `-3`, `+5.5`, `UTC-3`, `UTC+5:30`) |
| `filename` | Path to a local audio file (absolute or relative to `data_dir`) |
| `label` | When empty, the taxon name or stream ID is used for remote rows |
| `start_time` / `end_time` | In seconds; the same partial-time rules apply as in the local table mode |
| `split` | `train` or `test`; split automatically when empty |

**Mixed table:** all source types can be combined in one file.

```csv
filename,observation_id,sound_index,xc_id,stream_id,date,time,utc_offset,label,start_time,end_time
/data/rec.wav,,,,,,,,cicada,0.0,3.0
,12345678,0,,,,,,,1.0,6.0
,,,98765,,,,, Turdus merula,,,train
,,,,abc123,2023-07-14,06:00,-3,Guira guira,10.0,30.0
```

**Caching:** audio files and metadata are cached locally on the first download, so subsequent runs do not access the network.

```yaml
dataset:
  ext_cache_dir: /path/to/cache   # default: ~/.cache/bioaccx/ext
```

## iNaturalist and Xeno-canto

Rows are identified by `observation_id` (iNaturalist) or `xc_id` (Xeno-canto). When the `label` column is empty, the scientific name of the taxon, fetched from the respective API, is used as the label.

**Xeno-canto API key:** Xeno-canto uses API v3, which requires a personal key for metadata queries (scientific name lookup). Without a key, the audio is still downloaded directly, but the label defaults to `"xc_<id>"` unless it is set explicitly in the table. Keys are issued at [xeno-canto.org/explore/api](https://xeno-canto.org/explore/api).

```yaml
dataset:
  xc_api_key: YOUR_KEY_HERE   # optional; enables scientific name lookup for XC rows
```

## Arbimon

Each Arbimon row identifies a specific one-minute recording by its stream (site) ID and local timestamp. When the `label` column is empty, `stream_id` is used as the label.

**Required columns:**

| Column | Default name | Description |
|---|---|---|
| `stream_id` | `stream_id` | Arbimon recording site (stream) ID |
| `date` | `date` | Local recording date, ISO format: `YYYY-MM-DD` |
| `time` | `time` | Local recording start time: `HH:MM` or `HH:MM:SS` |
| `utc_offset` | `utc_offset` | UTC offset of the local time in hours (for example `-3`, `+5.5`, `UTC-3`, `UTC+5:30`) |

**Configuration:**

```yaml
dataset:
  ext_table_file: /path/to/observations.csv
  arbimon_credentials_path: /path/to/.rfcx_credentials   # required
  # arbimon_stream_id_col: stream_id   # column name overrides (optional)
  # arbimon_date_col: date
  # arbimon_time_col: time
  # arbimon_utc_offset_col: utc_offset
```

**Authentication:** Arbimon downloads use the rfcx SDK. A credentials file is created by authenticating once:

```python
import rfcx
client = rfcx.Client()
client.authenticate(persisted_credentials_path="/path/to/.rfcx_credentials")
```

This opens a browser URL for device authorisation and saves a token to disk. Subsequent runs load the token from that file without user interaction.

**Installing the rfcx SDK:** the SDK is not published on PyPI and must be installed from the GitHub release:

```bash
pip install https://github.com/rfcx/rfcx-sdk-python/releases/download/0.3.1/rfcx-0.3.1-py3-none-any.whl
```

**Caching:** downloaded audio is stored under `<ext_cache_dir>/arbimon/<stream_id>/`. A small sentinel file is written for each downloaded minute, so that repeated runs do not access the network and locate the audio file without scanning the directory again.
