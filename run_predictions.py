# runs one vlm on the four clutter datasets and saves its raw answers as one csv per dataset
# meant for a kaggle gpu notebook with the HF_TOKEN secret set as an environment variable
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
from datasets import concatenate_datasets
from datasets import load_dataset
from huggingface_hub import HfApi
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

from models import MODELS

DATASETS = {
    "shapes":"atml-shapes-vqa/shapes-vqa-distractors",
    "hard":"atml-shapes-vqa/shapes-vqa-hard",
    "objects":"atml-shapes-vqa/objects-vqa",
    "lines":"atml-shapes-vqa/lines-vqa",
}
# the reply is started with the first words of the required answer format so the model answers instead of explaining
# the same start is used for every version of a dataset so versions stay comparable
PREFILL = {
    "shapes":"circle:",
    "hard":"circle:",
    "objects":"apple:",
    "lines":"box:",
}
MAX_NEW_TOKENS = 64

parser = argparse.ArgumentParser(description="save the answers of one vlm on the clutter datasets")
parser.add_argument("--model",default="qwen3vl4b",choices=list(MODELS))
parser.add_argument("--split",default="all",help="train, validation, test or all")
parser.add_argument("--per-version",type=int,default=0,help="keep only the first n images of each version, 0 keeps all")
parser.add_argument("--datasets",default=",".join(DATASETS))
parser.add_argument("--out",default="/kaggle/working/preds")
args = parser.parse_args()

out = Path(args.out)
out.mkdir(parents=True,exist_ok=True)
repo,revision = MODELS[args.model]
# a model without a pinned revision uses the newest weights, whose exact revision is printed and saved below
if revision is None:
    revision = HfApi().model_info(repo).sha
print(f"{args.model}: {repo} at revision {revision}")
# the two models already run keep the single gpu setup of the first main run, the others are spread over every gpu kaggle gives
first_setup = args.model in ("qwen3vl2b","qwen3vl4b")
# the new models pick their own attention code because some of their parts (like gemma's audio part) can't use sdpa
if first_setup:
    model = AutoModelForImageTextToText.from_pretrained(repo,revision=revision,dtype=torch.float16,device_map="cuda:0",attn_implementation="sdpa")
else:
    model = AutoModelForImageTextToText.from_pretrained(repo,revision=revision,dtype=torch.float16,device_map="auto")
processor = AutoProcessor.from_pretrained(repo,revision=revision)
model.eval()

environment = {
    "model":repo,"revision":revision,"dtype":"float16","split":args.split,"per_version":args.per_version,
    "max_new_tokens":MAX_NEW_TOKENS,"prefill":PREFILL,"decoding":"greedy","python":sys.version,
    "torch":torch.__version__,"transformers":__import__("transformers").__version__,
    "gpu":torch.cuda.get_device_name(0),"gpus":torch.cuda.device_count(),"started":time.strftime("%Y-%m-%d %H:%M:%S"),
}
(out/f"environment_{args.model}.json").write_text(json.dumps(environment,indent=1),encoding="utf-8")
print(f"{repo} at {revision[:8]} on {environment['gpu']} with transformers {environment['transformers']}")


def ask_model(image,question,prefill):
    """Return the greedy text answer of the model with its reply started by the prefill"""
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)+prefill
    # the chat template already holds every special token, so the processor adds none of its own
    inputs = processor(text=[prompt],images=[image],return_tensors="pt",add_special_tokens=False).to(model.device)
    # the other models need the image numbers in the same 16-bit format as their weights
    if not first_setup:
        inputs = inputs.to(dtype=model.dtype)
    with torch.no_grad():
        output = model.generate(**inputs,max_new_tokens=MAX_NEW_TOKENS,do_sample=False)
    # putting the prefill back in front so the scorer sees the whole answer
    return prefill+processor.batch_decode(output[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0].rstrip()


for name in args.datasets.split(","):
    data = load_dataset(DATASETS[name])
    # all joins the three splits since nobody trains on these datasets in this study
    data = concatenate_datasets([data[split] for split in data]) if args.split=="all" else data[args.split]
    path = out/f"preds_{name}_{args.model}.csv"
    started = time.time()
    seen = {}
    kept = 0
    with open(path,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id","prediction"])
        for example in data:
            version = example["version"]
            seen[version] = seen.get(version,0)+1
            # skipping the rest of a version once the trial limit is reached
            if args.per_version and seen[version]>args.per_version:
                continue
            answer = ask_model(example["image"],example["question"],PREFILL[name])
            writer.writerow([example["image_id"],answer])
            kept += 1
            # printing one object list of the versions whose distractors the analysis needs to read
            if version in ("other_shapes","lookalikes") and seen[version]==1:
                print(f"--- {name} {version} objects ---\n{str(example['objects'])[:1500]}")
            # printing the first question and answer so the exact wording can be checked
            if kept==1:
                print(f"--- {name} first question ---\n{example['question']}\n--- answer ---\n{answer}\n--- truth ---\n{example['answer']}")
    minutes = (time.time()-started)/60
    print(f"{name}: {kept} images in {minutes:.1f} min ({60*minutes/max(kept,1):.2f} s per image) saved to {path}")
    # the scorer skips images without a prediction so trial runs are scored on the images they kept
    scorer = hf_hub_download(DATASETS[name],"vqa_scoring.py",repo_type="dataset",local_dir=str(out/f"scorer_{name}"))
    # a failing scorer is reported but does not stop the remaining datasets from running
    code = subprocess.run([sys.executable,scorer,"--predictions",str(path),"--split",args.split,"--out",str(out/f"scores_{name}_{args.model}.csv")]).returncode
    print(f"{name}: scorer exit code {code}")
