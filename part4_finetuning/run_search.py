#!/usr/bin/env python
# coding: utf-8

import os
import itertools
import json
import numpy as np

# רק הקונפיגורציה של run 4
EPOCHS = [2,3,4]
LRS = [2e-4]
LORA_R = [32]
BATCH = [2,4,8]

run_id = 10
results = []   # ← כאן נאסוף את כל הנתונים להשוואה

def load_eval_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def score_hebrew_quality(outputs):
    """מדד איכות עברית בסיסי: יחס אותיות עבריות, אורך, ניקיון."""
    if not outputs:
        return 0
    scores = []
    for ex in outputs:
        txt = ex["output"]
        heb = sum(1 for c in txt if "א" <= c <= "ת")
        ratio = heb / max(len(txt), 1)
        weird = sum(1 for c in txt if ord(c) > 126 and not ("א" <= c <= "ת"))
        weird_ratio = weird / max(len(txt), 1)
        weird_score = 1 - weird_ratio
        length_score = min(len(txt) / 40, 1)
        scores.append(0.6 * ratio + 0.2 * weird_score + 0.2 * length_score)
    return float(np.mean(scores))

import re

def extract_train_metrics(run_dir):
    metrics = {"train_loss": None, "eval_loss": None, "eval_acc": None}

    # חיפוש קובץ לוג
    log_files = [f for f in os.listdir(run_dir) if f.endswith(".out") or f.endswith(".log")]
    if not log_files:
        return metrics

    with open(os.path.join(run_dir, log_files[0]), "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Regex גמיש שתופס כל פורמט אפשרי
    def extract_last(pattern):
        matches = re.findall(pattern, text)
        return float(matches[-1]) if matches else None

    metrics["train_loss"] = extract_last(r"train_loss['\"]?\s*[:=]\s*([0-9]+\.[0-9]+)")
    metrics["eval_loss"] = extract_last(r"eval_loss['\"]?\s*[:=]\s*([0-9]+\.[0-9]+)")
    metrics["eval_acc"]  = extract_last(r"eval_mean_token_accuracy['\"]?\s*[:=]\s*([0-9]+\.[0-9]+)")

    return metrics



# ───────────────────────────────────────────────
# אימון כל המודלים
# ───────────────────────────────────────────────

for epochs, lr, r, bs in itertools.product(EPOCHS, LRS, LORA_R, BATCH):

    output_dir = f"runs/run_{run_id}_e{epochs}_lr{lr}_r{r}_bs{bs}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n========================================")
    print(f"Running model #{run_id}")
    print(f"Epochs: {epochs} | LR: {lr} | r: {r} | BS: {bs}")
    print(f"Saving to: {output_dir}")
    print("========================================\n")

    os.system(
        f"python train_qwen.py "
        f"--epochs {epochs} "
        f"--lr {lr} "
        f"--lora_r {r} "
        f"--batch_size {bs} "
        f"--output_dir {output_dir}"
    )

    # ───────────────────────────────────────────────
    # ניתוח המודל אחרי האימון
    # ───────────────────────────────────────────────

    train_metrics = extract_train_metrics(output_dir)

    eval_no = load_eval_jsonl(f"{output_dir}/eval_no_prompt.jsonl")
    eval_yes = load_eval_jsonl(f"{output_dir}/eval_with_prompt.jsonl")

    hebrew_score = score_hebrew_quality(eval_yes)

    results.append({
        "model": output_dir,
        "epochs": epochs,
        "lr": lr,
        "r": r,
        "bs": bs,
        "train_loss": train_metrics["train_loss"],
        "eval_loss": train_metrics["eval_loss"],
        "eval_acc": train_metrics["eval_acc"],
        "hebrew_score": hebrew_score
    })

    run_id += 1


# ───────────────────────────────────────────────
# סיכום כל המודלים
# ───────────────────────────────────────────────

print("\n\n===== MODEL COMPARISON =====\n")

# מיון לפי איכות עברית (הכי חשוב למטלה)
results_sorted = sorted(results, key=lambda x: x["hebrew_score"], reverse=True)

for r in results_sorted:
    print(f"Model: {r['model']}")
    print(f"  Epochs: {r['epochs']}")
    print(f"  Train Loss: {r['train_loss']}")
    print(f"  Eval Loss: {r['eval_loss']}")
    print(f"  Eval Accuracy: {r['eval_acc']}")
    print(f"  Hebrew Quality Score: {r['hebrew_score']:.3f}")
    print("-" * 60)

print("\n🏆 BEST MODEL:", results_sorted[0]["model"])
print("========================================")
