# Image Dataset Generator

A cross-platform Python CLI that converts scanned comics and illustrated pages into **ordered, redraw-ready play scripts** with OpenAI vision. Designed for a folder containing roughly 2,000 images, without requiring you to manage 2,000 API calls manually.

It writes **two synchronized datasets**: a comprehensive, panel-by-panel description and a concise script. Dialogue appears verbatim in both. The detailed version adds spatial blocking, poses, camera, background, prop interactions, design notes, bubble placement, stylistic details and uncertainties. A deterministic abridger makes the short version from the detailed result, avoiding a second model call for every image.

## Install (Windows, macOS, Linux)

Requires Python 3.10+ and internet access on the machine running the tool.

```bash
git clone https://github.com/goosnav/image-dataset-generator.git
cd image-dataset-generator
python -m pip install .
image-dataset-generator configure
```

`configure` requests your OpenAI API key without echoing it. You may instead set `OPENAI_API_KEY` as an environment variable. No key is committed, printed or exported. The `configure` command saves a local user-profile config with file mode `0600` on Unix; for stricter credential management, use the environment variable or your OS's secret manager.

Also works without installing the console command: `python image_dataset_generator.py --help` from the checkout.

## Quick start: 2,000 comics, $49 target, $54 maximum

First do a **free dry run**:

```bash
image-dataset-generator plan "/path/to/comics" --recursive
```

Then test 10 images (only a pilot; it will not expand silently):

```bash
image-dataset-generator run "/path/to/comics" --recursive --limit 10
```

Finally, start a fresh full run:

```bash
image-dataset-generator run "/path/to/comics" --recursive \
  --transport batch --model gpt-6.1-sol \
  --target-usd 49 --hard-cap-usd 54
```

In Windows PowerShell, put those options on one line or use PowerShell's backtick continuation. **By default, the program waits and collects batch results**. Batch requests may take up to 24 hours. Press Ctrl+C safely and resume later:

```bash
image-dataset-generator resume "comic-dataset-export-2026-11-18_09-30-00"
```

If you want to submit and exit immediately, use `--detach` and later run `resume`. Nothing is silently processed by ChatGPT; **you** execute the CLI and the OpenAI API handles any submitted batch requests.

The export directory is created in the **current working directory**, not automatically inside your source-image folder. You can override this using `--output-base`. Source files are never modified. The folder name is local date/time stamped, with a suffix if a collision occurs.

## Files written to each export

```text
comic-dataset-export-YYYY-MM-DD_HH-MM-SS/
  manifest.json                 # configuration, pricing snapshot, coverage, spend
  dataset.sqlite3              # indexed, queryable master archive
  comprehensive.jsonl.gz       # one machine-readable detailed object per source
  abridged.jsonl.gz            # one compact structured script per source
  comprehensive_scripts.md     # readable stage directions + exact dialogue
  abridged_scripts.md          # readable short scripts
  review_queue.csv             # identified uncertainties, ambiguous text
  status.csv                   # one row per source, status and token/cost totals
```

SQLite is the **working source of truth**. The compressed JSONL files are the most convenient streaming input for downstream machine-learning and retrieval workflows. Markdown is convenient for people. The `export` command rebuilds the portable outputs from SQLite without spending API tokens.

The original images **are not embedded in the exported datasets**. Each record contains a stable source relative path, file SHA-256, and image ID so it can be traced to the exact original file. Keep original images elsewhere. Data is UTF-8 and losslessly compressed.

## Why structured records instead of plain CSV?

CSV poorly represents multi-panel images, nested actors, speech balloons and props. Each comprehensive JSON object includes:

- page description, layout/gutters, style and stable character-design IDs;
- `panels[]` strictly in Western reading order, with normalized page-relative `xywh` bounding boxes;
- shot, composition, foreground/background, per-character position, facing, gaze, pose, expression and physical interactions;
- props and all visible text, with order, exact wording, speaker, lettering, balloon tail/location and legibility;
- drawable reconstruction directions, panel continuity, final beat, ambiguities and redraw checklist.

The abridged version contains only summary, style, per-panel action, minimal blocking, shot and exact dialogue, plus ending. The long Markdown script renders the nested structure as a screenplay-like series of panels.

**Image fidelity:** The tool sends a lossless PNG, after EXIF rotation and optional resize (default max side 2048 px), with `detail: "original"`. It never generates images. Very small text and ambiguous lettering can still be misread: review `review_queue.csv` and spot-check the corpus against original scans. No vision model can guarantee enough information to reproduce an image pixel-for-pixel from text alone.

## Cost model and ceilings

The default uses **GPT-6.1 Sol via OpenAI Batch** (50% lower than standard). This is the high-accuracy / high-volume compromise. The strongest available model may be preferable for particularly complex samples; `refine` or `--model gpt-6-astra` permits that at a higher unit cost. The tool **does not intentionally spend extra money just to meet the $49 target**; spend is driven by the number of images and actual token usage.

The CLI stores a November 2026 standard price card per million tokens:

| Model | Input | Output | Batch multiplier |
|---|---:|---:|---:|
| GPT-6.1 Sol | $2 | $10 | 0.5x |
| GPT-6.1 Terra | $1 | $5 | 0.5x |
| GPT-6 Astra | $5 | $25 | 0.5x |
| GPT-5.4 mini | $0.75 | $4.50 | 0.5x |

Before each submission the CLI reserves a conservative **upper-bound estimate** based on prepared image dimensions, approximate vision tokens, prompt length and `max_output_tokens` (default 3600). It will stop submitting when **tracked spent + outstanding reservations** approaches `$54`, leaving a small safety margin. When actual usage returns, unused reservations are released and more batches can be submitted. Request and model token caps are enforced by the API, but actual billed input can vary; therefore **this is not a mathematically guaranteed or account-level OpenAI spending limit**. Costs from other applications, price changes, regional processing surcharges, caching adjustments, failed ambiguous requests, and API-side billing are outside its scope. Set an organization/project budget alert or limit in the OpenAI dashboard as an additional safeguard.

**Important:** `$49` is a cost target, not a promise that all 2,000 images fit. Image density, resolution, output verbosity and model choice change actual spend. The program never intentionally exceeds its configured `$54` hard cap, but can stop with unfinished images. Do **not** treat `plan` as a provider quote. Current documentation: [API pricing](https://developers.openai.com/api/docs/pricing), [vision tokenization](https://developers.openai.com/api/docs/guides/images-vision), [Batch API](https://developers.openai.com/api/docs/guides/batch).

## Quality and speed controls

```bash
# Fast interactive API (normal rates), for a handful of image tests
image-dataset-generator run ./comics --transport live --limit 5

# More detail at higher cost: may need multiple resumptions / a larger budget
image-dataset-generator run ./comics -r --max-side 2400 --max-output-tokens 5000

# Pick GPT-6 Astra for a small subset (premium cost)
image-dataset-generator run ./comics --model gpt-6-astra --limit 10

# After a successful base run, upgrade up to 100 most uncertain/complex pages
image-dataset-generator refine "./comic-dataset-export-..." --model gpt-6-astra --limit 100

# Inspect export and regenerate results locally
image-dataset-generator status "./comic-dataset-export-..."
image-dataset-generator export "./comic-dataset-export-..."
```

`refine` **re-analyzes the selected images** with a stronger model, ranked by reported uncertainty and panel/text density. It replaces a successful detailed result and regenerates the short one, while retaining cumulative API spend. It only runs after the base workload is complete (no pending/submitted items), and remains subject to the original $54 cap. The tool does not change output if a review pass fails.

## Batch recovery and safety

- Batch input files are broken into small chunks (default at most 45 images and 24 MiB each), so base64 images do not exceed the platform's Batch upload limit. Results are associated by unique `custom_id` even when returned out of order.
- `resume` collects completed batches, updates actual/estimated cost, and submits more pending work when budget permits; repeated collects are idempotent.
- A source file that has changed since its initial SHA-256 scan is **not** processed silently. Start a new timestamped export to prevent identity drift.
- If the upload succeeds but batch creation is interrupted, requests remain in `sending` and **are not replayed automatically**. Verify the OpenAI Batches dashboard, then run `image-dataset-generator attach-batch EXPORT BATCH_ID` to associate the remote batch. If you have positively confirmed no batch was ever created, run `image-dataset-generator release-unconfirmed EXPORT --i-confirm-no-batch` to release the reservation and resubmit.
- API or malformed model failures are recorded in `status.csv`. Run `resume EXPORT --retry-failed` only after inspecting the failures; retrying may incur extra cost. Live-mode requests are deliberately not automatically retried after ambiguous network failures.
- If you move the original folder, keep the relative file structure and edit `source_folder` in the run's `manifest.json` before resuming. No API key is written there.

## Test locally without spending money

```bash
python -m unittest discover -s tests -v
python image_dataset_generator.py plan ./example_images --recursive
```

The unit tests exercise a mocked Batch API and do not touch the network. To verify real model output quality, you must run a small real pilot and inspect scripts against original comic scans.

## Known limits

1. `detail: original` availability and model pricing can change. If the API rejects an unsupported model/detail combination, choose a currently supported model and rerun the failed images; update the local price card only after verifying current prices.
2. A single very dense comic can exceed the default 3,600 output tokens and yield an incomplete JSON response. Increase `--max-output-tokens` on a fresh run (or run a separately configured higher-budget pilot).
3. Reading order, speaker attribution and handwritten OCR are model judgments and can be wrong. Spatial coordinate estimates are approximate, not vector tracing.
4. The first frame only is taken from animated GIFs; this is an illustration/comic-page pipeline, not a video frame extractor.
5. Region-specific API surcharges, transient provider behavior and changes to model pricing can make tracked amounts differ from final billing. Keep a project-level financial ceiling on the OpenAI platform.

No source drawings are altered or generated by the tool. The script's output is text-only derivative documentation to support archiving, search, creative planning and manual redrawing.
