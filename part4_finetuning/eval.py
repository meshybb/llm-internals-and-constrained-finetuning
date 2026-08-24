import torch
import json
import string
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import os

# ─────────────────────────────────────────────────────────
# פונקציות עזר להערכת שפה (Heuristics)
# ─────────────────────────────────────────────────────────

def clean_text(text):
    text = re.sub(r'[\d\s]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def calculate_metrics(text):
    cleaned = clean_text(text)
    if len(cleaned) == 0:
        return {"heb_ratio": 0.0, "leak_ratio": 0.0, "lexical_diversity": 0.0}

    heb_chars = sum(1 for c in cleaned if '\u0590' <= c <= '\u05FF')
    eng_chars = sum(1 for c in cleaned if re.match(r'[a-zA-Z]', c))
    leak_chars = len(cleaned) - heb_chars
    
    words = [w for w in text.split() if w.strip(string.punctuation)]
    lexical_diversity = len(set(words)) / len(words) if words else 0.0

    return {
        "heb_ratio": heb_chars / max(len(cleaned), 1),
        "leak_ratio": leak_chars / max(len(cleaned), 1),
        "lexical_diversity": lexical_diversity
    }

# ─────────────────────────────────────────────────────────
# טעינת מודל ופקודת מערכת (System Prompt)
# ─────────────────────────────────────────────────────────

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_dir = "./qwen_hebrew_lora_best"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

print("Loading tokenizer and base model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

# Build list of token IDs to suppress during generation.
# Blocks CJK, Korean, Arabic, and Japanese tokens that leak into Hebrew output.
#
# Qwen uses a tiktoken-style byte-to-unicode encoding in its vocab, so token
# strings like 'è·¯æ®µ' are actually Chinese bytes in disguise.  We must
# reverse that mapping before checking for unwanted scripts.

def _build_byte_decoder():
    """Return the tiktoken unicode→byte mapping used by Qwen's tokenizer."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}  # unicode_char → byte value

_uni2byte = _build_byte_decoder()

def _decode_qwen_token(tok_str):
    """Decode a Qwen vocab token string to actual UTF-8 text."""
    try:
        raw = bytes([_uni2byte[c] for c in tok_str])
        return raw.decode("utf-8", errors="ignore")
    except KeyError:
        return ""

def _has_unwanted_script(text):
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or   # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or   # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or # CJK Extension B
                0xAC00 <= cp <= 0xD7AF or   # Korean Hangul Syllables
                0x1100 <= cp <= 0x11FF or   # Korean Hangul Jamo
                0x0600 <= cp <= 0x06FF or   # Arabic
                0x3040 <= cp <= 0x30FF):    # Japanese Hiragana/Katakana
            return True
    return False

suppress_tokens = [
    tid for tok, tid in tokenizer.get_vocab().items()
    if _has_unwanted_script(_decode_qwen_token(tok))
]
print(f"Suppressing {len(suppress_tokens)} non-Hebrew/non-Latin tokens during generation.")

print("Loading LoRA adapters...")
ft_model = PeftModel.from_pretrained(base_model, adapter_dir)
ft_model.eval()

SYSTEM_PROMPT = "You are a helpful assistant. Always respond in Hebrew only, regardless of the language of the question."

def generate_eval(model, messages, max_new_tokens=200):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.3,
            suppress_tokens=suppress_tokens,
        )
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# ─────────────────────────────────────────────────────────
# הערכה השוואתית ושמירה ל-JSONL
# ─────────────────────────────────────────────────────────

TEST_FILE_NAME = 'test_data.json'
OUTPUT_FILE_NAME = f'{adapter_dir}/eval_outputs.jsonl'
ROOT_OUTPUT_FILE = 'eval_outputs.jsonl'  # submitted file — always overwritten with latest run

print(f"\n=== Starting Evaluation on {TEST_FILE_NAME} ===")

try:
    with open(TEST_FILE_NAME, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: Could not find {TEST_FILE_NAME}")
    exit(1)

# Track metrics for both base and finetuned for comparison
base_heb_scores, base_leak_scores = [], []
ft_heb_scores, ft_leak_scores, ft_lex_scores = [], [], []

records = []

print(f"Processing {len(test_data)} examples...\n")

for idx, ex in enumerate(test_data, 1):
    user_prompt = next(m["content"] for m in ex["messages"] if m["role"] == "user")

    print(f"[{idx}/{len(test_data)}] Evaluating: {user_prompt[:50]}...")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # 1. Base model (adapters disabled)
    with ft_model.disable_adapter():
        base_output = generate_eval(ft_model, messages)

    # 2. Fine-tuned model (adapters on)
    finetuned_output = generate_eval(ft_model, messages)

    # 3. Compute metrics for BOTH outputs so we can compare architectures
    base_m = calculate_metrics(base_output)
    ft_m   = calculate_metrics(finetuned_output)

    base_heb_scores.append(base_m["heb_ratio"])
    base_leak_scores.append(base_m["leak_ratio"])
    ft_heb_scores.append(ft_m["heb_ratio"])
    ft_leak_scores.append(ft_m["leak_ratio"])
    ft_lex_scores.append(ft_m["lexical_diversity"])

    delta_heb = (ft_m["heb_ratio"] - base_m["heb_ratio"]) * 100
    notes_str = (
        f"Base Hebrew: {base_m['heb_ratio']*100:.1f}% → FT Hebrew: {ft_m['heb_ratio']*100:.1f}% "
        f"(delta {delta_heb:+.1f}pp) | FT Leakage: {ft_m['leak_ratio']*100:.1f}%, "
        f"Lexical Div: {ft_m['lexical_diversity']:.2f}"
    )

    records.append({
        "prompt": user_prompt,
        "base_output": base_output,
        "finetuned_output": finetuned_output,
        "base_heb_ratio": round(base_m["heb_ratio"], 3),
        "ft_heb_ratio": round(ft_m["heb_ratio"], 3),
        "notes": notes_str,
    })

# Write to both the adapter dir copy and the root submission file
for path in [OUTPUT_FILE_NAME, ROOT_OUTPUT_FILE]:
    with open(path, "w", encoding="utf-8") as out_file:
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")

print("\n=========================================")
print("📊 OVERALL EVALUATION SUMMARY")
print("=========================================")
print(f"📉 Base  Avg Hebrew Purity:  {np.mean(base_heb_scores)*100:.2f}%")
print(f"✅ FT    Avg Hebrew Purity:  {np.mean(ft_heb_scores)*100:.2f}%")
print(f"📈 Avg Improvement (delta):  {(np.mean(ft_heb_scores)-np.mean(base_heb_scores))*100:+.2f}pp")
print(f"⚠️  FT   Avg Foreign Leakage: {np.mean(ft_leak_scores)*100:.2f}%")
print(f"🧠 FT   Avg Lexical Diversity:{np.mean(ft_lex_scores):.3f}")
print("=========================================")
print(f"Saved to: {OUTPUT_FILE_NAME}")
print(f"Saved to: {ROOT_OUTPUT_FILE}  ← submission copy")