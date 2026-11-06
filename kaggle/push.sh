#!/usr/bin/env bash
# pushes the notebook to kaggle as its own notebook per model and starts a gpu run (GPU T4 x2, internet on)
# usage: kaggle/push.sh MODEL STAGE [PART] [--no-run]
#   MODEL   qwen3vl8b or gemma4e4b
#   STAGE   trial (20 images per version) or full
#   PART    all (default); qwen3vl8b: 1 (shapes + hard) then 2 (objects + lines + naming);
#           gemma4e4b: answers (all four datasets) then naming (the naming step), as in zoya's runner's guide
#   --no-run only saves the notebook on kaggle without running it
# examples: kaggle/push.sh qwen3vl8b trial
#           kaggle/push.sh gemma4e4b full answers
# needs the kaggle cli logged in. kaggle gives notebook secrets only to runs started in its web page,
# so for an api run set TOKEN_DATASET=<user>/<slug>: your own private dataset holding hf_token.txt
set -euo pipefail
MODEL=${1:?give the model: qwen3vl8b or gemma4e4b}
STAGE=${2:?give the stage: trial or full}
PART=${3:-all}
RUN=${4:-}
case "$MODEL" in qwen3vl8b|gemma4e4b) ;; *) echo "unknown model $MODEL" >&2; exit 1;; esac
case "$STAGE" in trial|full) ;; *) echo "unknown stage $STAGE" >&2; exit 1;; esac
case "$PART" in all|1|2|answers|naming) ;; *) echo "unknown part $PART" >&2; exit 1;; esac
case "$RUN" in ""|--no-run) ;; *) echo "unknown option $RUN" >&2; exit 1;; esac
here=$(cd "$(dirname "$0")" && pwd)
staging=$(mktemp -d)
python3 - "$here/../study3_kaggle.ipynb" "$staging/study3_kaggle.ipynb" "$MODEL" "$STAGE" "$PART" <<'PY'
import json, sys
source, target, model, stage, part = sys.argv[1:]
notebook = json.load(open(source))
cell = notebook["cells"][1]
text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
for setting in ('STAGE = "trial"', 'PART = "all"', 'MODEL = "qwen3vl8b"'):
    assert setting in text, f"settings cell has no {setting}"
text = text.replace('STAGE = "trial"', f'STAGE = "{stage}"').replace('PART = "all"', f'PART = "{part}"')
cell["source"] = text.replace('MODEL = "qwen3vl8b"', f'MODEL = "{model}"')
json.dump(notebook, open(target, "w"), indent=1)
print(f"notebook set to MODEL={model} STAGE={stage} PART={part}")
PY
python3 - "$here/kernel-metadata.json" "$staging/kernel-metadata.json" "$MODEL" "${KAGGLE_USERNAME:-}" "${TOKEN_DATASET:-}" <<'PY'
import json, sys
source, target, model, user, dataset = sys.argv[1:]
meta = json.load(open(source))
owner = user or meta["id"].split("/")[0]
# one kaggle notebook per model, so their outputs and versions stay apart
meta["id"] = f"{owner}/clutter-vlm-study3-{model}"
meta["title"] = f"clutter-vlm study3 {model}"
meta["code_file"] = "study3_kaggle.ipynb"
if dataset:
    meta["dataset_sources"] = [dataset]
json.dump(meta, open(target, "w"), indent=1)
print(f"kaggle notebook {meta['id']}")
PY
kaggle kernels push -p "$staging" --accelerator NvidiaTeslaT4 $RUN
