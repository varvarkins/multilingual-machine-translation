# Multilingual Machine Translation — Data Dojo

Условия (правила тренировок) и решения двух соревнований Data Dojo по
машинному переводу.

Большие артефакты (веса моделей `*.safetensors`, обучающие корпуса,
`.venv`, zip-сабмиты, бинарные токенизаторы) намеренно **не** хранятся в
репозитории — здесь только постановки задач, исходный код решений и
небольшие конфиги/примеры. Веса воссоздаются обучением (см. инструкции)
или скачиванием базовых моделей через `download_weights.py`.

## [`data-dojo-1/`](data-dojo-1) — En → Ru (Gemma)

Перевод абзацев с английского на русский. Бейзлайн и решение на базе
мультимодальной модели Gemma (`AutoModelForCausalLM` + `AutoProcessor`),
промпт-инжиниринг + beam search.

- `Правила_Тренировки_Data Dojo.pdf` — условие.
- `solution/` — решение: `solution.py` (инференс), `Dockerfile`,
  `download_weights.py`, `eval/` (скрипты и референсы для оценки),
  `weights/` (только конфиги; сами веса не коммитятся).
- `baseline/gemma-4/` — исходный бейзлайн.

## [`data-dojo-2/`](data-dojo-2) — Ru → Ab (NLLB-200)

Перевод с русского на абхазский. Дообучение
`facebook/nllb-200-distilled-600M` с расширением токенизатора (добавление
абхазских токенов через SentencePiece). Цель — побить 17.6 BLEU. Подробности
в [`data-dojo-2/README.md`](data-dojo-2/README.md).

- `Правила_Тренировки_Data Dojo.pdf` — условие.
- `scripts/` — `clean_data.py`, `build_tokenizer.py` (подготовка данных и
  расширение токенизатора).
- `kaggle/`, `kaggle_dataset/` — `train_nllb.py` и материалы для обучения на
  Kaggle GPU.
- `submission/` — docker-сабмит: `solution.py`, `Dockerfile`, `weights/`
  (только конфиги).
- `baseline/gemma-4/` — бейзлайн Gemma (нижняя планка).
