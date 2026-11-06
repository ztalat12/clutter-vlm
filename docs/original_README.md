# clutter-vlm

ATML project, study 3: does visual clutter make a VLM worse at counting shapes and following lines, and does clutter the model can name hurt more than clutter it can't name?

The datasets are the four Shapes/Objects/Lines VQA sets in the `atml-shapes-vqa` Hugging Face organisation. The predictions and methods are fixed in `docs/study3_methodology.pdf` before any model runs.

## Running on Kaggle

GPU T4, internet on, secret `HF_TOKEN` (a read token of an account in the `atml-shapes-vqa` organisation). One notebook cell:

```
!git clone -q https://github.com/zoyakhan51/clutter-vlm /tmp/code && cd /tmp/code && git checkout -q COMMIT
import os; from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!cd /tmp/code && python run_predictions.py --model qwen3vl4b --split all
!cd /tmp/code && python run_nameability.py --model qwen3vl4b
!cd /tmp/code && python analysis.py --model qwen3vl4b
```

Replace `COMMIT` with the commit being run. Add `--per-version 20` to the first two scripts for a trial run.

- `run_predictions.py` answers every image of the four datasets and runs each dataset's official `vqa_scoring.py`. The model's reply is started with the first words of each answer format (`circle:`, `apple:`, `box:`), so it answers instead of explaining.
- `run_nameability.py` cuts every squiggle and every `other_shapes` shape of Shapes Hard out of its image and records what the model calls it (H2 nameability score, Stage 0 naming check).
- `analysis.py` computes every H1-H8 result with 95% bootstrap intervals over scenes, exactly as fixed in `docs/study3_methodology.pdf`.

Outputs: `/kaggle/working/preds` (answers, scorer output, nameability answers, `environment_<model>.json`) and `/kaggle/working/results` (`results_<model>.json` and a plain-text `summary_<model>.txt`).

## More models (study 3b)

The same study is repeated on more models, each run on its own Kaggle account. Everyone runs the same commit, so every model gets exactly the same questions, answer start and scoring. Only the model name changes.

| Model name | Model | Size |
|---|---|---|
| `gemma4e4b` | google/gemma-4-E4B-it | about 4B (8B with embeddings) |
| `internvl35_4b` | OpenGVLab/InternVL3_5-4B-HF | 4B |
| `molmo2_4b` | allenai/Molmo2-4B | about 5B |
| `qwen3vl8b` | Qwen/Qwen3-VL-8B-Instruct | 8B, optional size check |

Rule fixed before any of these runs: a finding counts as general if it holds (same verdict) in at least 3 of the 4 models (Qwen3-VL-4B plus the three new ones). Otherwise it is reported as model-specific. The floor rule of the plan applies to each model on each dataset.

Notebook settings: GPU T4 x2, internet on, phone-verified Kaggle account, secret `HF_TOKEN` (a read token of a Hugging Face account that can see the atml-shapes-vqa datasets). One cell, with `MODEL` replaced by the model name and `COMMIT` by the commit you were given:

```
!pip install -q transformers==5.22.1
!git clone -q https://github.com/zoyakhan51/clutter-vlm /tmp/code && cd /tmp/code && git checkout -q COMMIT
import os; from kaggle_secrets import UserSecretsClient; os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
!cd /tmp/code && python run_predictions.py --model MODEL --split all --per-version 20
!cd /tmp/code && python run_nameability.py --model MODEL --per-version 20
```

1. Trial first (the cell above, about 20 minutes). Save Version, then Save & Run All. Send the log back so the answers can be checked. The model weights are already pinned in `models.py`.
2. Main run: the same cell without `--per-version 20` (and with a new commit only if the trial needed a fix) (about 5 hours for a 4B model, longer for 8B). Save & Run All, then the tab can be closed.
3. Results: share the notebook with the study owner's Kaggle account, or download its output (the `preds` folder) and send it.
