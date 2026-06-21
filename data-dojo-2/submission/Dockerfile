FROM cr.yandex/crp2q2b12lka2f8enigt/pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    "transformers==4.44.2" \
    "tokenizers<0.20" \
    "sentencepiece" \
    "protobuf" \
    "sacremoses" \
    "accelerate"

WORKDIR /workspace
COPY . .

ENTRYPOINT ["python3", "solution.py"]
