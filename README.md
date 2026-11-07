# clutter-vlm: study 3 with Qwen3-VL-8B and Gemma 4 E4B

Study 3 of [zoyakhan51/clutter-vlm](https://github.com/zoyakhan51/clutter-vlm), run on Kaggle with two models, each in its own Kaggle notebook:

| Model name | Model | Weights revision |
|---|---|---|
| `qwen3vl8b` | Qwen/Qwen3-VL-8B-Instruct | `0c351dd0` |
| `gemma4e4b` | google/gemma-4-E4B-it | `4ecc2ba6` |

## Files

| File | What it is |
|---|---|
| `models.py`, `run_predictions.py`, `run_nameability.py`, `analysis.py` | Zoya's study code, byte-identical to her commit `cadcd07` |
| `docs/study3_methodology.pdf` | The predictions and methods fixed before any model ran |
| `docs/original_README.md` | Zoya's README |
| `study3_kaggle.ipynb` | The Kaggle notebook that runs the study for one model (`MODEL` setting) |
| `kaggle/kernel-metadata.json`, `kaggle/push.sh` | Push the notebook to Kaggle (one notebook per model) and start it from the command line |

The notebook clones this repo at commit `a2a127a` and stops unless the four scripts are byte-identical to Zoya's `cadcd07`. Every run gets the same questions, answer start, scoring and analysis as the other study 3b models. Only the model name differs.

## Running

Kaggle settings: GPU T4 x2, internet on, phone-verified account, secret `HF_TOKEN` that can read the `atml-shapes-vqa` datasets, switched on for the notebook. Kaggle gives secrets only to runs started from its web page (Save Version → Save & Run All). Runs started through the API need `TOKEN_DATASET=<user>/<slug>`: your own private Kaggle dataset holding `hf_token.txt`.

The plan follows Zoya's runner's guide (`docs/original_README.md`, her commit `e77972b`). Each part is its own Kaggle version, and Kaggle runs at most 2 GPU notebooks at once per account.

| Run | Command | Time |
|---|---|---|
| Gemma 4 E4B part 1: answers for all four datasets | `kaggle/push.sh gemma4e4b full answers` | about 8 h |
| Gemma 4 E4B part 2: the naming step | `kaggle/push.sh gemma4e4b full naming` | about 3 h |
| Qwen3-VL-8B trial | `kaggle/push.sh qwen3vl8b trial` | about 30 min |
| Qwen3-VL-8B part A: shapes and hard | `kaggle/push.sh qwen3vl8b full 1` | 4.5-7 h |
| Qwen3-VL-8B part B: objects, lines and naming | `kaggle/push.sh qwen3vl8b full 2` | 4.3-7 h |
| H1-H8 analysis, once a model's parts are all done (CPU, no GPU) | `INPUT_DATASET=<user>/<slug> kaggle/push.sh <model> full analysis` | a few minutes |

Gemma runs in two parts because in one go it would take 11-12 hours, too close to Kaggle's 12-hour limit. Its trial already passed on `cadcd07`.

A model's second part can run beside its first. If the first part's output is attached, the second part copies those files in and runs the analysis over everything. Otherwise both parts' `preds` files go into one folder afterwards.

For the analysis, `INPUT_DATASET` is a private Kaggle dataset of yours holding the model's `preds_*`, `nameability_*` and `environment_*` files from every part. The notebook copies them in and runs only `analysis.py`, which still needs the `HF_TOKEN` secret because it reads the dataset metadata from Hugging Face.

After the trial, check the log the way Zoya asks: "Successfully ran", no `Traceback`, `100.0` in every `parsed` column, and the "s per image" speed lines. Send her a screenshot of the end of the log.

Results: `kaggle kernels output <user>/clutter-vlm-study3-<model> -p out`, or the notebook's Output tab: `preds/` (send this folder to Zoya), `results/` and `log_<model>_*.txt`.

If the run stops with "cannot read atml-shapes-vqa/...", the lines above that error say why: the token is not accepted, the account is not in `atml-shapes-vqa`, or the token is fine-grained instead of type Read.
