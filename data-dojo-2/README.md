# Ru → Ab перевод (NLLB-200-distilled-600M)

Цель: максимальный sentence-level BLEU (sacrebleu) на закрытом тесте.
Бейзлайн Gemma — нижняя планка; цель — побить 17.6 BLEU.

## Подход (этап 1: быстрый первый сабмит)

Дообучаем `facebook/nllb-200-distilled-600M` для пары `rus_Cyrl → abk_Cyrl`.
NLLB — encoder-decoder, заточенный под малоресурсный перевод; абхазского в
нём нет, но он добавляется через расширение токенизатора (проверенный рецепт,
которым добавляли чеченский/башкирский/казахский).

**Главное открытие:** базовый токенизатор NLLB отправляет 8 абхазских букв
(ӷ ӡ ҟ ԥ ҭ ҵ ҿ ҩ ҕ ҧ) в `<unk>` — 93% абхазских предложений ломаются. Поэтому
обязательно расширяем словарь: обучаем SentencePiece на абхазском моно,
вливаем 13 979 новых токенов в NLLB, добавляем `abk_Cyrl`. После этого UNK
исчезают (0/1500 на dev), фертильность падает с 0.565 до 0.27 токена/символ.

## Структура

```
scripts/clean_data.py       # чистка параллельного корпуса -> data/{train,dev}.tsv
scripts/build_tokenizer.py  # обучение+слияние SP -> data/tokenizer_extended/
kaggle/train_nllb.py        # обучение на Kaggle (NLLB-600M, reinit эмбеддингов, sacrebleu)
submission/                 # docker-сабмит
  ├── Dockerfile            # transformers 4.44.2 (нужен SP-токенизатор)
  ├── solution.py           # инференс ru->ab, beam=5, формат output.json
  └── weights/              # сюда кладёшь обученную модель из Kaggle
kaggle_dataset/             # ЭТО загружаешь как Kaggle dataset (train/dev/tokenizer/скрипт)
```

## Шаги

### 0. Локальная подготовка (уже сделано, CPU)
```bash
python scripts/clean_data.py        # -> data/train.tsv (182 988), data/dev.tsv (1500)
python scripts/build_tokenizer.py   # -> data/tokenizer_extended/
```
Результаты уже собраны в `kaggle_dataset/`.

### 1. Обучение на Kaggle (GPU)
1. Создай Kaggle Dataset из папки `kaggle_dataset/` (train.tsv, dev.tsv,
   tokenizer_extended/, train_nllb.py). Назови его так, чтобы он смонтировался
   в `/kaggle/input/ru-ab-nllb` — иначе поправь `INPUT_DIR` в train_nllb.py.
2. Новый Notebook → Add Data → твой датасет. Accelerator: **GPU P100**.
   Internet: ON (только для скачивания базовой модели и pip).
3. В ячейке:
   ```python
   !pip -q install "transformers==4.44.2" "tokenizers<0.20" sentencepiece \
       sacrebleu sacremoses accelerate datasets
   !python /kaggle/input/ru-ab-nllb/train_nllb.py
   ```
4. Смотри `FULL DEV: {'eval_bleu': ...}` в логах. Модель сохранится в
   `/kaggle/working/nllb-ru-ab`. Скачай эту папку.

   Время: ~4 эпохи на 183k пар на P100 ≈ 2–4 часа. Если упираешься в лимит,
   уменьши `EPOCHS` до 2–3 в train_nllb.py.

### 2. Сборка сабмита
1. Положи содержимое скачанной `nllb-ru-ab/` в `submission/weights/`
   (config.json, model.safetensors, sentencepiece.bpe.model, tokenizer*, ...).
2. Проверь локально (без GPU — убери `--gpus all`):
   ```bash
   cd submission
   docker build -t ru-ab .
   docker run --rm \
     -v "$PWD/../baseline/gemma-4/example-input.pickle":/workspace/input.pickle \
     -v "$PWD/out":/workspace/out  ru-ab
   cat out/output.json
   ```
3. Упакуй СОДЕРЖИМОЕ `submission/` в zip (Dockerfile, solution.py, weights/ в
   корне архива, без обёрточной папки) и загрузи в контест.

## Что дальше (этап 2 — чтобы уверенно побить 17.6)
- **Back-translation:** обучить Ab→Ru на тех же парах, перевести часть из
  3.3M абхазских моно → синтетический русский, дообучить Ru→Ab на
  (реальные + синтетические) пары. Это главный рычаг для low-resource.
- Перейти на NLLB-1.3B (тоже влезает в L4 на инференсе).
- Подбор beam size / length_penalty под метрику.
