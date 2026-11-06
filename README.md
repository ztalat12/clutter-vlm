# clutter-vlm: study 3 with Qwen3-VL-8B

Study 3 of [zoyakhan51/clutter-vlm](https://github.com/zoyakhan51/clutter-vlm), run with the `qwen3vl8b` model (`Qwen/Qwen3-VL-8B-Instruct`, weights revision `0c351dd0`) on Kaggle.

## Files

| File | What it is |
|---|---|
| `models.py`, `run_predictions.py`, `run_nameability.py`, `analysis.py` | Zoya's study code, byte-identical to her commit `cadcd07` |
| `docs/study3_methodology.pdf` | The predictions and methods fixed before any model ran |
| `docs/original_README.md` | Zoya's README |
| `study3_qwen3vl8b_kaggle.ipynb` | The Kaggle notebook that runs the study |
| `kaggle/kernel-metadata.json`, `kaggle/push.sh` | Push the notebook to Kaggle and start it from the command line |

The notebook clones this repo at commit `a2a127a` and stops unless the four scripts are byte-identical to Zoya's `cadcd07`. The run gets the same questions, answer start, scoring and analysis as the other study 3b models; only the model name differs.

## Running

Kaggle settings: GPU T4 x2, internet on, phone-verified account, secret `HF_TOKEN` that can read the `atml-shapes-vqa` datasets, switched on for the notebook.

1. Trial: `kaggle/push.sh trial` (or import the notebook and Save & Run All). It checks the answers and estimates the full run time.
2. Full run: `kaggle/push.sh full all` if the trial says it fits in 12 hours, otherwise `full 1` and then `full 2` with the part 1 output attached.
3. Results: `kaggle kernels output talhaimran27/clutter-vlm-study3-qwen3vl8b -p out`, or the notebook's Output tab: `preds/`, `results/` and `log_qwen3vl8b_*.txt`.
