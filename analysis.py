# computes every study 3 result (H1 to H8) from the saved answers and the dataset metadata
# all intervals are 95% bootstrap intervals over whole scenes as fixed in docs/study3_methodology.pdf
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {
    "shapes":"atml-shapes-vqa/shapes-vqa-distractors",
    "hard":"atml-shapes-vqa/shapes-vqa-hard",
    "objects":"atml-shapes-vqa/objects-vqa",
    "lines":"atml-shapes-vqa/lines-vqa",
}
CLASSES = {
    "shapes":["circle","triangle","square","star"],
    "hard":["circle","triangle","square","star"],
    "objects":["apple","cat","dog","car"],
}
SQUIGGLES = ["squiggles_2","squiggles_4","squiggles_8","squiggles_16"]
# versions that add one kind of clutter so the everything version is left out of H4
SINGLE_TRAPS = {
    "shapes":["squiggles","blobs","lookalikes"],
    "hard":["wrong_color","lookalikes","text_traps"]+SQUIGGLES+["other_shapes"],
    "objects":["random_objects","lookalikes","text_traps"],
}
NUMBER_WORDS = {"zero":0,"no":0,"none":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12}
# answers that describe a line or a mark rather than name a thing, fixed before the run
NON_NAMES = {"squiggle","scribble","line","curve","wave","wavy","zigzag","doodle","loop","swirl","scrawl","stroke","mark","pattern","shape","abstract","random","design","drawing","sketch"}
# word forms of the same descriptions added after the first sampled answers of the main run showed them (curved, curl, curvy)
NON_NAMES |= {"curved","curvy","curly","curl","curling","squiggly","wiggly","wiggle","scribbled","scribbly","swirly","swirled","swirling","loopy","looped","looping","lined","linear","wavey","waving","zigzagged","zigzaggy","doodled","scrawled","sketched","stroked","marked","shaped","designed","drawn"}
RESAMPLES = 10000
NUMBER = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"


def to_int(token):
    """Turn a digit string or a number word into an integer"""
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def parse_counts(text,classes):
    """Return the count the answer gives for every class or None when any class is missing"""
    text = str(text).lower().replace("*","")
    counts = {}
    for c in classes:
        found = re.search(r"\b"+c+r"(?:e?s)?\b\s*[:=\-]?\s*"+NUMBER+r"\b",text)
        if found is None:
            return None
        counts[c] = to_int(found.group(1))
    return counts


def parse_box(text):
    """Return the box number the answer gives or None when there is none"""
    found = re.search(r"\bbox\b\s*[:#]?\s*"+NUMBER+r"\b",str(text).lower())
    return to_int(found.group(1)) if found else None


def scene_groups(scenes):
    """Return the row positions of every scene so whole scenes can be resampled"""
    codes,unique = pd.factorize(pd.Series(scenes))
    return [np.flatnonzero(codes==k) for k in range(len(unique))]


def interval(point,draws):
    """Point value with the 2.5 and 97.5 percent bootstrap limits"""
    low,high = np.percentile(draws,[2.5,97.5])
    return {"value":round(float(point),4),"low":round(float(low),4),"high":round(float(high),4)}


def boot_mean(values,seed):
    """Mean of one value per scene with its bootstrap interval, skipping scenes without a value"""
    values = np.asarray(values,dtype=float)
    values = values[~np.isnan(values)]
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0,len(values),(RESAMPLES,len(values)))].mean(1)
    out = interval(values.mean(),draws)
    out["scenes"] = int(len(values))
    return out


def boot_ratio_diff(num_a,den_a,num_b,den_b,seed):
    """Difference of two ratios of sums over scenes with a bootstrap that resamples the same scenes for both"""
    keep = ~(np.isnan(num_a)|np.isnan(den_a)|np.isnan(num_b)|np.isnan(den_b))
    num_a,den_a,num_b,den_b = num_a[keep],den_a[keep],num_b[keep],den_b[keep]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0,len(num_a),(RESAMPLES,len(num_a)))
    ratio_a = num_a[pick].sum(1)/np.maximum(den_a[pick].sum(1),1e-9)
    ratio_b = num_b[pick].sum(1)/np.maximum(den_b[pick].sum(1),1e-9)
    a = interval(num_a.sum()/den_a.sum(),ratio_a)
    b = interval(num_b.sum()/den_b.sum(),ratio_b)
    diff = interval(num_a.sum()/den_a.sum()-num_b.sum()/den_b.sum(),ratio_a-ratio_b)
    return a,b,diff,int(keep.sum())


def boot_slope(frame,columns,seed):
    """Least squares coefficients of correct on the given columns with scenes resampled as whole groups"""
    x = np.column_stack([np.ones(len(frame))]+[frame[c].to_numpy(float) for c in columns])
    y = frame["correct"].to_numpy(float)
    groups = scene_groups(frame["scene_id"])
    point = np.linalg.lstsq(x,y,rcond=None)[0]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(RESAMPLES):
        rows = np.concatenate([groups[k] for k in rng.integers(0,len(groups),len(groups))])
        draws.append(np.linalg.lstsq(x[rows],y[rows],rcond=None)[0])
    draws = np.array(draws)
    return {c:interval(point[j+1],draws[:,j+1]) for j,c in enumerate(columns)}


def mcnemar(clean,other):
    """Exact two sided McNemar p value for paired right or wrong answers"""
    broke = int(((clean==1)&(other==0)).sum())
    fixed = int(((clean==0)&(other==1)).sum())
    n = broke+fixed
    if n==0:
        return 1.0
    k = min(broke,fixed)
    tail = sum(math.comb(n,i) for i in range(k+1))*0.5**n
    return float(min(1.0,2*tail))


def verdict(result,want="above"):
    """Say whether a 95% interval lies wholly above or below zero"""
    if result["low"]>0:
        return "above 0" if want=="above" else "above 0 (opposite of prediction)"
    if result["high"]<0:
        return "below 0" if want=="below" else "below 0 (opposite of prediction)"
    return "includes 0"


def rank_state(values):
    """Turn the rank_kept column into kept changed or empty whatever way it was stored"""
    text = pd.Series(values).astype(str).str.strip().str.lower()
    return text.map(lambda t:"kept" if t in ("yes","true","1","1.0") else ("changed" if t in ("no","false","0","0.0") else ""))


def objects_of(value):
    """Return the object list of one image whether it was stored as text or as a list"""
    return json.loads(value) if isinstance(value,str) else list(value)


def load_frame(name,meta,preds):
    """Join the metadata and the answers of one dataset and add the parsed answers"""
    frame = meta.merge(preds,on="image_id",how="inner")
    if name=="lines":
        frame["pred_box"] = [parse_box(t) for t in frame["prediction"]]
        frame["parsed"] = frame["pred_box"].notna()
        frame["correct"] = (frame["pred_box"]==frame["target_box"].astype(int)).astype(float)
        return frame
    classes = CLASSES[name]
    # the dataset stores the true counts as text so they are turned into numbers first
    for c in classes:
        frame["count_"+c] = pd.to_numeric(frame["count_"+c])
    parsed = [parse_counts(t,classes) for t in frame["prediction"]]
    frame["parsed"] = [p is not None for p in parsed]
    for c in classes:
        frame["pred_"+c] = [np.nan if p is None else p[c] for p in parsed]
    frame["pred_total"] = frame[["pred_"+c for c in classes]].sum(1,min_count=len(classes))
    truth = np.column_stack([frame["count_"+c].to_numpy(float) for c in classes])
    guess = np.column_stack([frame["pred_"+c].to_numpy(float) for c in classes])
    frame["correct"] = np.all(truth==guess,axis=1).astype(float)
    return frame


def wide(frame,column):
    """Put one column into a table with one row per scene and one column per version"""
    return frame.pivot_table(index="scene_id",columns="version",values=column,aggfunc="first")


def analyse(frames,nameability):
    """Compute every table and verdict of the plan from the joined frames"""
    out = {"accuracy":{},"H1":{},"H2":{},"H3":{},"H4":{},"H5":{},"H6":{},"H7":{},"H8":{}}
    correct = {name:wide(frame,"correct") for name,frame in frames.items()}
    drop = {name:table.rsub(table["clean"],axis=0) for name,table in correct.items()}
    # accuracy and parse rate of every version, always reported
    for name,frame in frames.items():
        rows = {}
        for version,part in frame.groupby("version"):
            row = {"images":int(len(part)),"accuracy":round(float(part["correct"].mean()),4),"parsed":round(float(part["parsed"].mean()),4)}
            if name!="lines":
                truth_total = part[["count_"+c for c in CLASSES[name]]].sum(1)
                row["abs_err"] = round(float((part[["pred_"+c for c in CLASSES[name]]].to_numpy(float)-part[["count_"+c for c in CLASSES[name]]].to_numpy(float)).__abs__().sum(1)[part["parsed"].to_numpy()].mean()),4)
                row["total_bias"] = round(float((part["pred_total"]-truth_total).mean()),4)
            rows[version] = row
        out["accuracy"][name] = rows
    # H1 names test and similarity test
    if "hard" in drop:
        d = drop["hard"]
        out["H1"]["names_hard"] = boot_mean(d["other_shapes"]-d[SQUIGGLES].mean(1),11)
        out["H1"]["similarity_hard"] = boot_mean(d["lookalikes"]-d["other_shapes"],12)
    if "objects" in drop:
        d = drop["objects"]
        out["H1"]["similarity_objects"] = boot_mean(d["lookalikes"]-d["random_objects"],13)
    if "shapes" in drop:
        d = drop["shapes"]
        out["H1"]["combined_shapes_no_verdict"] = boot_mean(d["lookalikes"]-d[["squiggles","blobs"]].mean(1),14)
    if "names_hard" in out["H1"]:
        names = verdict(out["H1"]["names_hard"])=="above 0"
        similar = all(verdict(out["H1"][k])=="above 0" for k in ("similarity_hard","similarity_objects") if k in out["H1"])
        out["H1"]["conclusion"] = {(True,True):"both names and similarity matter",(True,False):"names matter, similarity doesn't",(False,True):"similarity matters, names don't",(False,False):"neither explains the drops"}[(names,similar)]
    # H2 squiggle complexity and nameability
    if "hard" in frames:
        part = frames["hard"][frames["hard"]["version"].isin(SQUIGGLES)].copy()
        part["doublings"] = np.log2(part["version"].str.replace("squiggles_","").astype(int))-1
        out["H2"]["accuracy_by_curves"] = {v:round(float(part[part["version"]==v]["correct"].mean()),4) for v in SQUIGGLES}
        out["H2"]["main_slope_per_doubling"] = boot_slope(part,["doublings"],21)["doublings"]
        if nameability is not None:
            nameability = nameability.copy()
            nameability["score"] = [name_share(a) for a in nameability["answers"]]
            out["H2"]["mean_score"] = {v:round(float(s),4) for v,s in nameability.groupby("version")["score"].mean().items()}
            others = nameability[nameability["version"]=="other_shapes"]
            out["H2"]["naming_check_other_shapes"] = {shape:round(float(np.mean([named_right(shape,g) for g in grp["greedy_name"]])),4) for shape,grp in others.groupby("shape")}
            per_image = nameability[nameability["version"].isin(SQUIGGLES)].groupby("image_id")["score"].mean()
            part = part.merge(per_image.rename("nameability"),left_on="image_id",right_index=True,how="inner")
            out["H2"]["untangling"] = boot_slope(part,["doublings","nameability"],22)
    # H3 every version against clean with McNemar
    for name,table in correct.items():
        rows = {}
        for version in table.columns:
            if version=="clean":
                continue
            pair = table[["clean",version]].dropna()
            rows[version] = dict(boot_mean(pair["clean"]-pair[version],31+len(rows)),mcnemar_p=mcnemar(pair["clean"].to_numpy(),pair[version].to_numpy()))
        out["H3"][name] = rows
    # H4 excess count over the same scene's clean answer
    for name in ("shapes","hard","objects"):
        if name not in frames:
            continue
        total = wide(frames[name],"pred_total")
        excess = total.sub(total["clean"],axis=0)
        means = {v:boot_mean(excess[v],41+k) for k,v in enumerate(SINGLE_TRAPS[name]) if v in excess}
        others = [v for v in means if v!="lookalikes"]
        runner = max(others,key=lambda v:means[v]["value"])
        out["H4"][name] = {"excess_count":means,"next_highest":runner,"lookalikes_minus_next":boot_mean(excess["lookalikes"]-excess[runner],49)}
        if "everything" in excess:
            out["H4"][name]["everything_reported_only"] = boot_mean(excess["everything"],48)
    # H5 absorption of incomplete shapes against other clutter
    for name in ("shapes","hard"):
        if name in frames:
            out["H5"][name] = absorption(frames[name],name)
    # H6 text against the same amount of other clutter and the copy check
    if "hard" in drop:
        d = drop["hard"]
        out["H6"]["text_vs_squiggles_hard"] = boot_mean(d["text_traps"]-d[SQUIGGLES].mean(1),61)
    if "objects" in drop:
        d = drop["objects"]
        out["H6"]["text_vs_random_objects"] = boot_mean(d["text_traps"]-d["random_objects"],62)
    for name in ("hard","objects"):
        if name in frames:
            out["H6"]["copy_"+name] = copying(frames[name],name)
    # H7 and H8 on the lines
    if "lines" in frames:
        lines = frames["lines"]
        d = drop["lines"]
        out["H7"]["plain_clutter_drop"] = boot_mean(d["crossings_0"],71)
        changed = lines[lines["version"].isin(["crossings_2","crossings_4","crossings_8"])&(rank_state(lines["rank_kept"]).to_numpy()=="changed")].copy()
        changed["crossings"] = changed["num_crossings"].astype(float)
        out["H7"]["rank_changed_images"] = {v:int((changed["version"]==v).sum()) for v in ["crossings_2","crossings_4","crossings_8"]}
        out["H7"]["slope_per_crossing_rank_changed"] = boot_slope(changed,["crossings"],72)["crossings"]
        out["H7"]["by_version"] = line_table(lines)
        c = correct["lines"]
        out["H8"]["colored_minus_plain"] = boot_mean(c["crossings_4_colored"]-c["crossings_4"],81)
    return out


def name_share(answers):
    """Share of the sampled answers that name a concrete thing rather than describe a line"""
    words = []
    for answer in str(answers).split("|"):
        word = re.sub(r"[^a-z ]","",answer.lower()).strip().split(" ")[0] if answer.strip() else ""
        words.append(word)
    named = [w!="" and w not in NON_NAMES and w.rstrip("s") not in NON_NAMES for w in words]
    return float(np.mean(named))


def named_right(shape,answer):
    """Say whether the one word greedy answer names the shape that was drawn"""
    answer = str(answer).lower()
    # the dataset calls the lightning shape bolt so both spellings point to the same synonyms
    synonyms = {"heart":["heart"],"moon":["moon","crescent"],"cross":["cross","plus"],"bolt":["lightning","bolt","thunder","zigzag"],"lightning":["lightning","bolt","thunder","zigzag"]}
    key = next((k for k in synonyms if k in str(shape).lower()),str(shape).lower())
    return any(s in answer for s in synonyms.get(key,[key]))


def absorption(frame,name):
    """Share of incomplete shapes the model counts as complete against extra counts per item of other clutter"""
    classes = CLASSES[name]
    by_image = frame.set_index("image_id")
    clean = frame[frame["version"]=="clean"].set_index("scene_id")
    look = frame[frame["version"]=="lookalikes"].set_index("scene_id")
    scenes = sorted(set(clean.index)&set(look.index))
    num,den = [],[]
    by_subtype = {"gap":[[],[]],"wavy":[[],[]]}
    by_class = {c:[[],[]] for c in classes}
    for s in scenes:
        objects = objects_of(look.loc[s,"objects"])
        n = {c:sum(1 for o in objects if o.get("lookalike_of")==c) for c in classes}
        if not (look.loc[s,"parsed"] and clean.loc[s,"parsed"]):
            num.append(np.nan)
            den.append(np.nan)
            continue
        total = 0.0
        for c in classes:
            extra = float(np.clip(look.loc[s,"pred_"+c]-clean.loc[s,"pred_"+c],0,n[c]))
            total += extra
            by_class[c][0].append(extra)
            by_class[c][1].append(n[c])
            kinds = {o.get("subtype") for o in objects if o.get("lookalike_of")==c}
            # only classes whose incomplete shapes all share one defect can be split by defect
            if n[c] and len(kinds)==1 and list(kinds)[0] in by_subtype:
                by_subtype[list(kinds)[0]][0].append(extra)
                by_subtype[list(kinds)[0]][1].append(n[c])
        num.append(total)
        den.append(sum(n.values()))
    num,den = np.array(num),np.array(den,dtype=float)
    comparisons = {"shapes":{"squiggles_and_blobs":["squiggles","blobs"]},"hard":{"squiggles":SQUIGGLES,"other_shapes":["other_shapes"]}}[name]
    out = {"comparisons":{}}
    for label,versions in comparisons.items():
        cnum,cden = [],[]
        for s in scenes:
            extra,items = 0.0,0
            ok = bool(clean.loc[s,"parsed"])
            for v in versions:
                row = frame[(frame["scene_id"]==s)&(frame["version"]==v)]
                if len(row)==0 or not bool(row["parsed"].iloc[0]):
                    ok = False
                    break
                extra += max(0.0,float(row["pred_total"].iloc[0]-clean.loc[s,"pred_total"]))
                items += sum(1 for o in objects_of(row["objects"].iloc[0]) if o["role"]=="distractor")
            cnum.append(extra if ok else np.nan)
            cden.append(items if ok else np.nan)
        rate,other,diff,used = boot_ratio_diff(num,den,np.array(cnum),np.array(cden,dtype=float),51)
        out["absorption_rate"] = rate
        out["comparisons"][label] = {"extra_per_item":other,"difference":diff,"scenes":used}
    out["by_defect"] = {k:round(float(np.sum(v[0])/max(np.sum(v[1]),1)),4) for k,v in by_subtype.items()}
    out["by_class"] = {k:round(float(np.sum(v[0])/max(np.sum(v[1]),1)),4) for k,v in by_class.items()}
    rate = out["absorption_rate"]
    diffs = [c["difference"] for c in out["comparisons"].values()]
    if all(d["low"]>0 for d in diffs) and rate["low"]>0.5:
        out["verdict"] = "supported"
    elif all(d["low"]>0 for d in diffs):
        out["verdict"] = "partly supported"
    else:
        out["verdict"] = "not supported"
    return out


def copying(frame,name):
    """How often the model gives exactly the count a written claim makes, against the same scene's clean answer"""
    classes = CLASSES[name]
    clean = frame[frame["version"]=="clean"].set_index("scene_id")
    text = frame[frame["version"]=="text_traps"].set_index("scene_id")
    per_scene = []
    examples = 0
    for s in sorted(set(clean.index)&set(text.index)):
        if not (text.loc[s,"parsed"] and clean.loc[s,"parsed"]):
            continue
        hits,base,claims = 0,0,0
        for o in objects_of(text.loc[s,"objects"]):
            words = str(o.get("text") or "").lower()
            for c in classes:
                found = re.search(NUMBER+r"\s+"+c+r"(?:e?s)?\b",words) or re.search(r"\b"+c+r"(?:e?s)?\s*:\s*"+NUMBER+r"\b",words)
                if found:
                    claimed = to_int(found.group(1))
                    hits += int(text.loc[s,"pred_"+c]==claimed)
                    base += int(clean.loc[s,"pred_"+c]==claimed)
                    claims += 1
        if claims:
            per_scene.append((hits,base,claims))
            examples += claims
    if not per_scene:
        return {"claims":0}
    values = np.array(per_scene,dtype=float)
    copy,baseline,diff,used = boot_ratio_diff(values[:,0],values[:,2],values[:,1],values[:,2],63)
    return {"claims":examples,"copy_rate":copy,"clean_baseline":baseline,"difference":diff,"scenes":used}


def line_table(lines):
    """Accuracy jumped other and no line shares per version and accuracy for kept and changed rank"""
    rows = {}
    for version,part in lines.groupby("version"):
        crossed = [set(map(int,re.findall(r"\d+",str(b)))) for b in part["crossed_boxes"]]
        others = [set(map(int,re.findall(r"\d+",str(b)))) for b in part["distractor_boxes"]]
        guess = part["pred_box"].tolist()
        target = part["target_box"].astype(int).tolist()
        jumped = [g is not None and not np.isnan(g) and g!=t and int(g) in cr for g,t,cr in zip(guess,target,crossed)]
        other = [g is not None and not np.isnan(g) and g!=t and int(g) in ot and int(g) not in cr for g,t,cr,ot in zip(guess,target,crossed,others)]
        kept = rank_state(part["rank_kept"]).to_numpy()
        rows[version] = {
            "accuracy":round(float(part["correct"].mean()),4),
            "jumped":round(float(np.mean(jumped)),4),
            "other_line":round(float(np.mean(other)),4),
            "accuracy_rank_kept":round(float(part[kept=="kept"]["correct"].mean()),4) if (kept=="kept").any() else None,
            "accuracy_rank_changed":round(float(part[kept=="changed"]["correct"].mean()),4) if (kept=="changed").any() else None,
        }
    return rows


def summary(out):
    """Write the main results as plain lines for the log"""
    lines = []
    for key in ("names_hard","similarity_hard","similarity_objects","combined_shapes_no_verdict"):
        if key in out["H1"]:
            r = out["H1"][key]
            lines.append(f"H1 {key}: {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r)}")
    if "conclusion" in out["H1"]:
        lines.append(f"H1 conclusion: {out['H1']['conclusion']}")
    if "main_slope_per_doubling" in out["H2"]:
        r = out["H2"]["main_slope_per_doubling"]
        # h2 has two competing ideas so the label names the idea that wins instead of one prediction
        idea = {"below 0":"busy clutter idea wins","above 0":"names idea wins"}.get(verdict(r,"below").split(" (")[0],"unclear")
        lines.append(f"H2 accuracy per doubling of curves: {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {idea}")
    for k,r in out["H2"].get("untangling",{}).items():
        lines.append(f"H2 untangling {k}: {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r,'below').split(' (')[0]}")
    for name,rows in out["H4"].items():
        r = rows["lookalikes_minus_next"]
        lines.append(f"H4 {name} lookalikes minus {rows['next_highest']}: {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r)}")
    for name,rows in out["H5"].items():
        r = rows["absorption_rate"]
        lines.append(f"H5 {name} absorption {r['value']:.3f} [{r['low']:.3f},{r['high']:.3f}] -> {rows['verdict']}")
    for key in ("text_vs_squiggles_hard","text_vs_random_objects"):
        if key in out["H6"]:
            r = out["H6"][key]
            lines.append(f"H6 {key}: {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r)}")
    for key in ("copy_hard","copy_objects"):
        if key in out["H6"] and out["H6"][key].get("claims"):
            r = out["H6"][key]["difference"]
            lines.append(f"H6 {key} ({out['H6'][key]['claims']} claims): {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r)}")
    if "slope_per_crossing_rank_changed" in out["H7"]:
        r = out["H7"]["slope_per_crossing_rank_changed"]
        lines.append(f"H7 accuracy per crossing (rank changed): {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r,'below')}")
    if "colored_minus_plain" in out["H8"]:
        r = out["H8"]["colored_minus_plain"]
        lines.append(f"H8 colored minus plain: {r['value']:+.3f} [{r['low']:+.3f},{r['high']:+.3f}] -> {verdict(r)}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute the study 3 results")
    parser.add_argument("--model",default="qwen3vl4b")
    parser.add_argument("--preds",default="/kaggle/working/preds")
    parser.add_argument("--out",default="/kaggle/working/results")
    parser.add_argument("--datasets",default=",".join(DATASETS))
    args = parser.parse_args()
    from datasets import concatenate_datasets
    from datasets import load_dataset
    # finding the answers of an earlier run attached as input when the given folder does not exist
    if not Path(args.preds).exists():
        found = sorted(Path("/kaggle/input").rglob(f"preds_shapes_{args.model}.csv"))
        assert found,f"no answers of {args.model} found in {args.preds} or under /kaggle/input"
        args.preds = str(found[0].parent)
    print(f"reading answers from {args.preds}")
    frames = {}
    for name in args.datasets.split(","):
        path = Path(args.preds)/f"preds_{name}_{args.model}.csv"
        if not path.exists():
            print(f"{name}: no answers found at {path}, skipped")
            continue
        data = load_dataset(DATASETS[name])
        data = concatenate_datasets([data[split] for split in data]).remove_columns(["image"])
        meta = data.to_pandas()
        frames[name] = load_frame(name,meta,pd.read_csv(path))
        print(f"{name}: {len(frames[name])} images joined, {frames[name]['parsed'].mean():.3f} parsed")
    names_path = Path(args.preds)/f"nameability_{args.model}.csv"
    nameability = pd.read_csv(names_path) if names_path.exists() else None
    out = analyse(frames,nameability)
    folder = Path(args.out)
    folder.mkdir(parents=True,exist_ok=True)
    (folder/f"results_{args.model}.json").write_text(json.dumps(out,indent=1),encoding="utf-8")
    text = summary(out)
    (folder/f"summary_{args.model}.txt").write_text(text,encoding="utf-8")
    print(text)
    for name,rows in out["accuracy"].items():
        print(name+": "+", ".join(f"{v} {r['accuracy']:.3f}" for v,r in rows.items()))
    print(f"saved {folder}")
