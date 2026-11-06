# shows every squiggle and every other_shapes shape of shapes hard on its own and records what the model calls it
# feeds the nameability score of H2 and the naming check of the new shapes in stage 0
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from datasets import concatenate_datasets
from datasets import load_dataset
from huggingface_hub import HfApi
from PIL import Image
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText

from models import MODELS

REPO = "atml-shapes-vqa/shapes-vqa-hard"
VERSIONS = ["squiggles_2","squiggles_4","squiggles_8","squiggles_16","other_shapes"]
LOOKS_LIKE = "What does this shape look like? Answer in 1 word."
NAME_IT = "What shape is this? Answer in one word."
SAMPLES = 5
# sampling settings fixed before the run so the five answers can differ
SAMPLING = {"do_sample":True,"temperature":0.7,"top_p":1.0,"top_k":50}
SEED = 0

parser = argparse.ArgumentParser(description="record what the model calls each squiggle and each other_shapes shape")
parser.add_argument("--model",default="qwen3vl4b",choices=list(MODELS))
parser.add_argument("--per-version",type=int,default=0,help="keep only the first n images of each version, 0 keeps all")
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
torch.manual_seed(SEED)


def answers_for(image,question,sample):
    """Return five sampled answers or one greedy answer of the model for one cropped shape"""
    messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":question}]}]
    prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    # the chat template already holds every special token, so the processor adds none of its own
    inputs = processor(text=[prompt],images=[image],return_tensors="pt",add_special_tokens=False).to(model.device)
    # the other models need the image numbers in the same 16-bit format as their weights
    if not first_setup:
        inputs = inputs.to(dtype=model.dtype)
    settings = dict(SAMPLING,num_return_sequences=SAMPLES) if sample else {"do_sample":False}
    with torch.no_grad():
        output = model.generate(**inputs,max_new_tokens=8,**settings)
    texts = processor.batch_decode(output[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
    return [t.strip().replace("|","/") for t in texts]


data = load_dataset(REPO)
data = concatenate_datasets([data[split] for split in data])
path = out/f"nameability_{args.model}.csv"
started = time.time()
seen = {}
shapes = 0
with open(path,"w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id","scene_id","version","object_id","shape","answers","greedy_name"])
    for example in data:
        version = example["version"]
        if version not in VERSIONS:
            continue
        seen[version] = seen.get(version,0)+1
        if args.per_version and seen[version]>args.per_version:
            continue
        image = example["image"].convert("RGB")
        objects = example["objects"]
        if isinstance(objects,str):
            objects = json.loads(objects)
        for o in objects:
            if o["role"]!="distractor":
                continue
            # cutting a square around the shape with some margin and padding with white where it leaves the picture
            x0,y0,x1,y1 = o["bbox"]
            side = int(max(x1-x0,y1-y0)*1.4)+2
            left = int((x0+x1)/2-side/2)
            top = int((y0+y1)/2-side/2)
            box = (max(left,0),max(top,0),min(left+side,image.width),min(top+side,image.height))
            crop = Image.new("RGB",(side,side),"white")
            crop.paste(image.crop(box),(box[0]-left,box[1]-top))
            crop = crop.resize((256,256))
            answers = answers_for(crop,LOOKS_LIKE,True)
            greedy = answers_for(crop,NAME_IT,False)[0] if version=="other_shapes" else ""
            writer.writerow([example["image_id"],example["scene_id"],version,o["id"],o["shape"],"|".join(answers),greedy])
            shapes += 1
            if shapes==1:
                print(f"first shape {o['shape']} in {example['image_id']}: {answers} {greedy}")
minutes = (time.time()-started)/60
print(f"nameability: {shapes} shapes in {minutes:.1f} min ({60*minutes/max(shapes,1):.2f} s per shape) saved to {path}")
