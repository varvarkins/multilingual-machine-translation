import os
import numpy as np
import pandas as pd
import torch
import sacrebleu
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset


def find_input_dir():
    for root, dirs, files in os.walk("/kaggle/input"):
        if "tokenizer_extended" in dirs and "train.tsv" in files:
            return root
    raise FileNotFoundError(
        "dataset not found under /kaggle/input — add it via Add Input "
        "and check the files include train.tsv, dev.tsv, tokenizer_extended/"
    )


INPUT_DIR = find_input_dir()
BASE_MODEL = "facebook/nllb-200-distilled-600M"
OUT_DIR = "/kaggle/working/nllb-ru-ab"

SRC_LANG = "rus_Cyrl"
TGT_LANG = "abk_Cyrl"
MAX_LEN = 128

EPOCHS = 4
BATCH = 16
GRAD_ACCUM = 2
LR = 3e-4
WARMUP_RATIO = 0.05
LABEL_SMOOTHING = 0.0
EVAL_BEAMS = 4
EVAL_SUBSET = 600


def load_data(tokenizer):
    train = pd.read_csv(f"{INPUT_DIR}/train.tsv", sep="\t").dropna()
    dev = pd.read_csv(f"{INPUT_DIR}/dev.tsv", sep="\t").dropna()
    uid = tokenizer.unk_token_id

    def clean(df):
        keep = [uid not in tokenizer.encode(str(s), add_special_tokens=False)
                for s in df["ab"]]
        return df[keep].reset_index(drop=True)

    train, dev = clean(train), clean(dev)

    def to_ds(df):
        return Dataset.from_dict({"ru": df["ru"].tolist(), "ab": df["ab"].tolist()})

    return to_ds(train), to_ds(dev)


def tokenize_fn(batch, tokenizer):
    tokenizer.src_lang = SRC_LANG
    tokenizer.tgt_lang = TGT_LANG
    return tokenizer(
        batch["ru"], text_target=batch["ab"],
        max_length=MAX_LEN, truncation=True,
    )


def reinit_embeddings(model, new_tok, old_tok):
    old_emb = model.get_input_embeddings().weight.data.clone()
    old_vocab = old_tok.get_vocab()
    model.resize_token_embeddings(len(new_tok))
    emb = model.get_input_embeddings().weight.data

    mean_vec = old_emb.mean(0)
    for tok_str, new_id in new_tok.get_vocab().items():
        if tok_str in old_vocab:
            emb[new_id] = old_emb[old_vocab[tok_str]]
        else:
            surface = new_tok.convert_tokens_to_string([tok_str]).strip()
            sub_ids = old_tok.encode(surface, add_special_tokens=False) if surface else []
            sub_ids = [i for i in sub_ids if i < old_emb.shape[0]
                       and i != old_tok.unk_token_id]
            emb[new_id] = old_emb[sub_ids].mean(0) if sub_ids else mean_vec

    abk_id = new_tok.convert_tokens_to_ids(TGT_LANG)
    rus_id = new_tok.convert_tokens_to_ids(SRC_LANG)
    emb[abk_id] = emb[rus_id].clone()


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = [sacrebleu.sentence_bleu(p.strip(), [r.strip()]).score
                  for p, r in zip(dec_preds, dec_labels)]
        return {"bleu": float(np.mean(scores))}
    return compute_metrics


def main():
    tokenizer = NllbTokenizer.from_pretrained(f"{INPUT_DIR}/tokenizer_extended")
    old_tok = NllbTokenizer.from_pretrained(BASE_MODEL)

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    reinit_embeddings(model, tokenizer, old_tok)

    abk_id = tokenizer.convert_tokens_to_ids(TGT_LANG)
    model.config.forced_bos_token_id = abk_id
    model.generation_config.forced_bos_token_id = abk_id
    model.config.max_length = MAX_LEN

    train_ds, dev_ds = load_data(tokenizer)
    eval_ds = dev_ds.select(range(min(EVAL_SUBSET, len(dev_ds))))

    tok_train = train_ds.map(lambda b: tokenize_fn(b, tokenizer),
                             batched=True, remove_columns=train_ds.column_names)
    tok_eval = eval_ds.map(lambda b: tokenize_fn(b, tokenizer),
                           batched=True, remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=OUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        label_smoothing_factor=LABEL_SMOOTHING,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=MAX_LEN,
        generation_num_beams=EVAL_BEAMS,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        report_to="none",
        dataloader_num_workers=2,
        optim="adafactor",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()

    full_eval = dev_ds.map(lambda b: tokenize_fn(b, tokenizer),
                           batched=True, remove_columns=dev_ds.column_names)
    metrics = trainer.evaluate(full_eval, num_beams=EVAL_BEAMS, max_length=MAX_LEN)
    print("FULL DEV:", metrics)

    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
