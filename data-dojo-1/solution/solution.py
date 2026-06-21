import json
import os
import re
import pickle

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


MODEL_DIR = os.environ.get("MODEL_DIR", "./weights")
INPUT_PATH = os.environ.get("INPUT_PATH", "input.pickle")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "output.json")

NUM_BEAMS = int(os.environ.get("GEN_NUM_BEAMS", "4"))
LENGTH_PENALTY = float(os.environ.get("GEN_LENGTH_PENALTY", "1.0"))
REPETITION_PENALTY = float(os.environ.get("GEN_REPETITION_PENALTY", "1.0"))
MAX_NEW_CAP = int(os.environ.get("GEN_MAX_NEW_CAP", "1536"))
LEN_FACTOR = float(os.environ.get("GEN_LEN_FACTOR", "2.0"))
MIN_NEW_TOKENS = 8


PROMPT = """Ты — профессиональный переводчик с английского языка на русский.
Переведи приведённый ниже текст целиком, как единый связный фрагмент.

Требования:
- Переводи весь абзац связно: сохраняй согласование рода, числа, лица и времён между всеми предложениями.
- Сохраняй стиль, тон и регистр оригинала (формальный/разговорный), а также пунктуацию и абзацы.
- Используй единообразную и корректную терминологию во всём тексте. Названия кнопок, элементов интерфейса, команд, продуктов и брендов переводи устоявшимися русскими терминами, а при их отсутствии оставляй на английском.
- Ничего не добавляй, не убирай и не комментируй; переводи смысл, а не пословно.
- Выведи ТОЛЬКО перевод на русском языке, без пояснений, предисловий, заголовков и кавычек.

Текст:
"""


_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"конечно[!,.\s]*|разумеется[!,.\s]*|хорошо[!,.\s]*|"
    r"вот\s+(?:ваш\s+)?перевод[^:\n]*:?\s*|перевод[^:\n]*:\s*|"
    r"sure[!,.\s]*|here(?:'s| is)[^:\n]*:?\s*|translation[^:\n]*:\s*"
    r")",
    flags=re.IGNORECASE,
)

_SPECIAL_TOKENS = ("<end_of_turn>", "<start_of_turn>", "<eos>", "<bos>", "<pad>")


def clean(text):
    if text is None:
        return ""
    t = text
    for tok in _SPECIAL_TOKENS:
        t = t.replace(tok, "")
    t = t.strip()
    for _ in range(3):
        new = _PREAMBLE_RE.sub("", t, count=1).strip()
        if new == t:
            break
        t = new
    pairs = [("«", "»"), ('"', '"'), ("'", "'"), ("“", "”"), ("```", "```"), ("`", "`")]
    changed = True
    while changed:
        changed = False
        t = t.strip()
        for lq, rq in pairs:
            if len(t) > len(lq) + len(rq) and t.startswith(lq) and t.endswith(rq):
                inner = t[len(lq):len(t) - len(rq)]
                if lq not in inner and rq not in inner:
                    t = inner
                    changed = True
                    break
    return t.strip()


def extract(processor, response_text):
    content = None
    try:
        parsed = processor.parse_response(response_text)
        content = parsed.get("content") if isinstance(parsed, dict) else None
    except Exception:
        content = None
    if not content:
        content = response_text
    return clean(content)


def main():
    with open(INPUT_PATH, "rb") as f:
        rows = pickle.load(f)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    gen_kwargs = dict(
        do_sample=False,
        num_beams=NUM_BEAMS,
        length_penalty=LENGTH_PENALTY,
        repetition_penalty=REPETITION_PENALTY,
        early_stopping=(NUM_BEAMS > 1),
    )

    results = []
    for row in rows:
        text = processor.apply_chat_template(
            [{"role": "user", "content": PROMPT + row["src"]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        max_new = max(MIN_NEW_TOKENS, min(MAX_NEW_CAP, int(input_len * LEN_FACTOR) + 64))

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                min_new_tokens=MIN_NEW_TOKENS,
                **gen_kwargs,
            )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        translation = extract(processor, response)
        if not translation:
            translation = clean(processor.decode(outputs[0][input_len:], skip_special_tokens=True))

        results.append({"rid": row["rid"], "translation": translation})
        inputs.to("cpu")

    payload = json.dumps(results, ensure_ascii=False)
    with open(OUTPUT_PATH, "w") as f:
        f.write(payload)
    out_dir = os.path.dirname(OUTPUT_PATH) or "."
    alt_dir = os.path.join(out_dir, "out")
    try:
        os.makedirs(alt_dir, exist_ok=True)
        with open(os.path.join(alt_dir, "output.json"), "w") as f:
            f.write(payload)
    except Exception:
        pass


if __name__ == "__main__":
    main()
