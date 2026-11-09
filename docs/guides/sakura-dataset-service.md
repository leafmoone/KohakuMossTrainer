# Sakuradatasetservice input

This input is step-controlled. The separate service samples up to `per_folder`
distinct TARs per parent folder in each allocation round and shuffles their order.
Configure the count in `[service]` in `sakura_service.toml`; it defaults to 5. If a
folder contains fewer TARs, each available TAR is selected once. A round is not a
trainer epoch: use `STEPS`, `SAVE_EVERY` and `SAMPLE_EVERY`, and set `EPOCHS = 0`.
The service and trainer must run on the same Linux host. One service instance is used
by one training consumer; do not share its worker 0 with another process.

Clone the service branch with configurable TAR sampling and install it into the trainer's
environment (Python 3.11+ and Pydantic 2.7+ are required):

```bash
git clone --branch codex/balanced-multiple-tars https://github.com/leafmoone/Sakuradatasetservice.git
uv pip install --python .venv/bin/python -e ./Sakuradatasetservice
```

Edit [sakura_service.toml](../../configs/data/sakura_service.toml), including the
local dataset path or ModelScope `owner/name`, revision, cache and state paths, then
start the service in a separate terminal:

```bash
.venv/bin/python scripts/data/sakura_service.py --config configs/data/sakura_service.toml
```

The launcher creates the manifest if missing, then starts a single-worker, repeating
balance service. It rejects an existing manifest for a different configured repo or
revision. Alternatively, for local shards, invoke the service CLI directly:

```bash
.venv/bin/sakura-dataset-service manifest --local-root /path/to/tars --repo-id local/my-data --revision v1 --output /path/to/manifest.json
.venv/bin/sakura-dataset-service serve --local-root /path/to/tars --manifest /path/to/manifest.json --cache-root /path/to/dedicated-cache --state-dir /path/to/service-state --cache-high-bytes 107374182400 --sampling balance --per-folder 5 --seed 0 --workers 1 --repeat
```

For ModelScope replace `--local-root /path/to/tars` with `--modelscope` on both
commands, use the dataset's actual `owner/name` and revision when creating the
manifest, and set `MODELSCOPE_API_TOKEN` in the environment. Never put a token in a
configuration file. The manifest is fixed; inventory changes require a new manifest,
state directory and dedicated cache directory. The service reserves up to twice a
shard's size for an in-progress transfer; set the quota and disk headroom accordingly.

In a config under `configs/user/`, use
`use_config("../moss/data/data_sakura_service.py")` and set `DATASETS` to the
`data.sock` path under the service's `state_dir`. Configure weights, optimizer, adapter,
`STEPS`, sample prompts and logging as usual. TARs must be uncompressed WebDataset
archives containing same-key image and JSON or TXT files. The `multichoice` caption
template reads the structured JSON described in [data.md](../concepts/data.md).

The current integration requires one data rank, `NUM_WORKERS = 0`, no separate
`ENCODE_DEVICES`, and `LATENT_CACHE = TEXT_CACHE = "off"`. Thus the VAE and text
encoder run during training; this may be slower than a fully cached run. Each TAR is
planned separately, including its final partial batch. Samples are shuffled within a
TAR; they are not globally shuffled across all TARs. `MAX_TEXT_TOKENS` is the
prewarm bound for later TARs, so choose a value that accommodates their captions.

Model, optimizer and scheduler checkpoints work as usual. The service queue advances
independently of those checkpoints and is not rewound when one is restored: after an
interruption, data between the checkpoint and the service's current position can be
skipped, and the current TAR can be repeated. This includes batches fetched ahead by
the encoders. Restart the service with the same manifest, seed, sampling settings,
state and cache to retain its queue. An unfinished TAR remains leased and will be
retried by worker 0, or requeued when the service restarts. A checkpoint therefore does
not guarantee exact sample replay. Use fixed `SAMPLE_PROMPTS` and provide any reference
images separately; this streaming source does not automatically select reference images.
