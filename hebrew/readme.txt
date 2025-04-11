cloud.vastai.com

vastai/pytorch:2.5.1-cuda-12.1.1

curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd piper/src/python
uv venv
uv pip install -e .
./build_monotonic_align.sh

sudo apt-get install espeak-ng


mkdir data
metadata.csv

python3 -m piper_train.preprocess \
  --language he \
  --input-dir /path/to/dataset_dir/ \
  --output-dir /path/to/training_dir/ \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050

https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/es/es_ES/davefx/medium

uv pip install pytorch-lightning


uv run python -m piper_train \
    --dataset-dir /path/to/training_dir/ \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 32 \
    --validation-split 0.0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint /path/to/lessac/epoch=2164-step=1355540.ckpt \
    --checkpoint-epochs 1 \
    --precision 32


cat test_en-us.jsonl | \
    python3 -m piper_train.infer \
        --sample-rate 22050 \
        --checkpoint /path/to/training_dir/lightning_logs/version_0/checkpoints/*.ckpt \
        --output-dir /path/to/training_dir/output"


python3 -m piper_train.export_onnx \
    /path/to/model.ckpt \
    /path/to/model.onnx
    
cp /path/to/training_dir/config.json \
   /path/to/model.onnx.json


english, bryce