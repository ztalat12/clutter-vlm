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

For each model:

1. Trial: `kaggle/push.sh <model> trial`. It checks the answers and estimates the full run time.
2. Full run: `kaggle/push.sh <model> full all` if the trial says it fits in 12 hours. Otherwise run `full 1`, then `full 2` with the part 1 output attached.
3. Results: `kaggle kernels output <user>/clutter-vlm-study3-<model> -p out`, or the notebook's Output tab: `preds/`, `results/` and `log_<model>_*.txt`.
