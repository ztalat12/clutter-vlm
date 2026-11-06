# clutter-vlm: study 3 with Qwen3-VL-8B

Runs the study 3 experiment of [zoyakhan51/clutter-vlm](https://github.com/zoyakhan51/clutter-vlm) unchanged, with the `qwen3vl8b` model (`Qwen/Qwen3-VL-8B-Instruct`, weights revision `0c351dd0`), on Kaggle.

Import `study3_qwen3vl8b_kaggle.ipynb` on Kaggle (Create → New Notebook → File → Import Notebook). Settings: GPU T4 x2, internet on, phone-verified account, secret `HF_TOKEN` that can read the `atml-shapes-vqa` datasets.

1. Trial: `STAGE = "trial"`, Save & Run All. It checks the answers and estimates the full run time.
2. Full run: `STAGE = "full"` with `PART = "all"` if the trial says it fits in 12 hours. Otherwise run `PART = "1"`, then `PART = "2"` with the part 1 output attached as input.
3. Results: `/kaggle/working/preds`, `/kaggle/working/results` and the log file `log_qwen3vl8b_*.txt`.

The notebook clones Zoya's code at commit `cadcd07`, so the run gets the same questions, answer start, scoring and analysis as the other study 3b models. Only the model name differs.
