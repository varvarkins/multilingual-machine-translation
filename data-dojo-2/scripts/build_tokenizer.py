import os
import re
from transformers import NllbTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

BASE = "facebook/nllb-200-distilled-600M"
AB_SPM = "data/ab_spm.model"
OUT = "data/tokenizer_extended"
NEW_LANG = "abk_Cyrl"


def main():
    tok = NllbTokenizer.from_pretrained(BASE)
    base_spm_path = tok.vocab_file

    base = sp_pb2.ModelProto()
    base.ParseFromString(open(base_spm_path, "rb").read())
    existing = {p.piece for p in base.pieces}
    default_score = min(p.score for p in base.pieces) - 1.0

    ab = sp_pb2.ModelProto()
    ab.ParseFromString(open(AB_SPM, "rb").read())

    added = 0
    for p in ab.pieces:
        if p.piece in existing:
            continue
        if p.type not in (sp_pb2.ModelProto.SentencePiece.NORMAL,
                          sp_pb2.ModelProto.SentencePiece.USER_DEFINED):
            continue
        np = base.pieces.add()
        np.piece = p.piece
        np.score = default_score
        existing.add(p.piece)
        added += 1

    code_re = re.compile(r"^[a-z]{3}_[A-Z][a-z]{3}$")
    lang_codes = [t.content for t in tok.added_tokens_decoder.values()
                  if code_re.match(t.content)]
    lang_codes = sorted(set(lang_codes))

    os.makedirs(OUT, exist_ok=True)
    merged_path = os.path.join(OUT, "sentencepiece.bpe.model")
    with open(merged_path, "wb") as f:
        f.write(base.SerializeToString())

    new_tok = NllbTokenizer(
        vocab_file=merged_path,
        additional_special_tokens=lang_codes + [NEW_LANG],
    )
    new_tok.save_pretrained(OUT)

    print(f"base pieces:     {len(base.pieces) - added}")
    print(f"added Abkhaz:    {added}")
    print(f"merged pieces:   {len(base.pieces)}")
    print(f"new vocab size:  {new_tok.vocab_size}")
    print(f"abk_Cyrl id:     {new_tok.convert_tokens_to_ids(NEW_LANG)}")
    print(f"saved to:        {OUT}")


if __name__ == "__main__":
    main()
