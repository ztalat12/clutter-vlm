# clutter-vlm: study 3 with Qwen3-VL-8B

Runs the study 3 experiment from [zoyakhan51/clutter-vlm](https://github.com/zoyakhan51/clutter-vlm) unchanged, with the `qwen3vl8b` model (`Qwen/Qwen3-VL-8B-Instruct`, revision `0c351dd0`), on Kaggle.

Upload `study3_qwen3vl8b_kaggle.ipynb` to Kaggle (Create → New Notebook → File → Import Notebook) and follow the steps in its first cell. You need GPU T4 x2, internet on, and an `HF_TOKEN` secret that can read the `atml-shapes-vqa` datasets.

The notebook clones Zoya's code at commit `cadcd07`, so you get the same questions, answer start, scoring and analysis as the other study 3b models.
