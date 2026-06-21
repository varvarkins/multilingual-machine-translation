import json
import os
import pickle

import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
SRC_LANG = "rus_Cyrl"
TGT_LANG = "abk_Cyrl"
MAX_LEN = 128
NUM_BEAMS = 5
BATCH = 32


def write_output(results):
    payload = json.dumps(results, ensure_ascii=False)
    for path in ("output.json", os.path.join("out", "output.json")):
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                continue
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)
        except OSError:
            pass


def main():
    with open("input.pickle", "rb") as f:
        rows = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = NllbTokenizer.from_pretrained(MODEL_DIR)
    tokenizer.src_lang = SRC_LANG
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    forced_bos = tokenizer.convert_tokens_to_ids(TGT_LANG)

    srcs = [str(r["src"]) for r in rows]
    rids = [r["rid"] for r in rows]
    translations = [""] * len(rows)

    with torch.inference_mode():
        for i in range(0, len(srcs), BATCH):
            chunk = srcs[i:i + BATCH]
            enc = tokenizer(
                chunk, return_tensors="pt", padding=True,
                truncation=True, max_length=MAX_LEN,
            ).to(device)
            out = model.generate(
                **enc,
                forced_bos_token_id=forced_bos,
                max_length=MAX_LEN,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
            )
            dec = tokenizer.batch_decode(out, skip_special_tokens=True)
            for j, t in enumerate(dec):
                translations[i + j] = t.strip()

    results = [{"rid": rid, "translation": tr}
               for rid, tr in zip(rids, translations)]
    write_output(results)


if __name__ == "__main__":
    main()
