import json
import sys

import sacrebleu


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("usage: python3 score.py <output.json> [references.json]")
        sys.exit(1)
    out_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else "references.json"

    outputs = {r["rid"]: r["translation"] for r in load_json(out_path)}
    refs = {r["rid"]: r["reference"] for r in load_json(ref_path)}

    scores = []
    print(f"{'rid':>4} | {'BLEU':>6} | hypothesis")
    print("-" * 70)
    for rid in sorted(refs):
        hyp = outputs.get(rid, "")
        ref = refs[rid]
        bleu = sacrebleu.sentence_bleu(hyp, [ref]).score
        scores.append(bleu)
        snippet = hyp.replace("\n", " ")[:48]
        print(f"{rid:>4} | {bleu:6.2f} | {snippet}")

    mean = sum(scores) / len(scores) if scores else 0.0
    print("-" * 70)
    print(f"MEAN sentence-BLEU over {len(scores)} examples: {mean:.2f}")

    hyps = [outputs.get(rid, "") for rid in sorted(refs)]
    rlist = [[refs[rid] for rid in sorted(refs)]]
    corpus = sacrebleu.corpus_bleu(hyps, rlist).score
    print(f"corpus BLEU: {corpus:.2f}")


if __name__ == "__main__":
    main()
