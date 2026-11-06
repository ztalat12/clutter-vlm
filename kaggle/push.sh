#!/usr/bin/env bash
# pushes the notebook to kaggle and starts a gpu run (GPU T4 x2, internet on)
# usage: kaggle/push.sh trial      the 20-images-per-version trial
#        kaggle/push.sh full all   the main run in one version (or: full 1, then full 2)
# needs the kaggle cli logged in; the HF_TOKEN secret is attached once in the kaggle notebook page
# (or set TOKEN_DATASET=<user>/<slug> to attach your own private dataset holding hf_token.txt)
# --no-run as third argument only saves the notebook without running it
set -euo pipefail
STAGE=${1:-trial}
PART=${2:-all}
here=$(cd "$(dirname "$0")" && pwd)
staging=$(mktemp -d)
python3 - "$here/../study3_qwen3vl8b_kaggle.ipynb" "$staging/study3_qwen3vl8b_kaggle.ipynb" "$STAGE" "$PART" <<'PY'
import json, sys
source, target, stage, part = sys.argv[1:]
notebook = json.load(open(source))
cell = notebook["cells"][1]
text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
assert 'STAGE = "trial"' in text and 'PART = "all"' in text, "settings cell not found"
cell["source"] = text.replace('STAGE = "trial"', f'STAGE = "{stage}"').replace('PART = "all"', f'PART = "{part}"')
json.dump(notebook, open(target, "w"), indent=1)
print(f"notebook set to STAGE={stage} PART={part}")
PY
python3 - "$here/kernel-metadata.json" "$staging/kernel-metadata.json" "${TOKEN_DATASET:-}" <<'PY'
import json, sys
source, target, dataset = sys.argv[1:]
meta = json.load(open(source))
if dataset:
    meta["dataset_sources"] = [dataset]
json.dump(meta, open(target, "w"), indent=1)
PY
kaggle kernels push -p "$staging" --accelerator NvidiaTeslaT4 ${3:-}
