cloud.vastai.com

vastai/pytorch:2.5.1-cuda-12.1.1

sudo apt-get install espeak-ng -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

cd piper/src/python
uv venv
uv pip install -e .
./build_monotonic_align.sh


uv pip install "numpy<2"
uv run python -m piper_train.preprocess \
  --language he \
  --input-dir ../../hebrew/dummy_dataset \
  --output-dir ./train \
  --dataset-format ljspeech \
  --single-speaker \
  --sample-rate 22050 \
  --raw-phonemes


uv pip install pytorch-lightning


wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/epoch=4641-step=3104302.ckpt
uv pip install torchmetrics==0.11.4
uv run python -m piper_train \
    --dataset-dir "./train" \
    --accelerator 'gpu' \
    --devices 1 \
    --batch-size 32 \
    --validation-split 0 \
    --num-test-examples 0 \
    --max_epochs 10000 \
    --resume_from_checkpoint ./epoch=4641-step=3104302.ckpt \
    --checkpoint-epochs 1 \
    --precision 32


cat ./train/dataset.jsonl | \
    uv run python -m piper_train.infer \
        --sample-rate 22050 \
        --checkpoint bryce-3499.ckpt \
        --output-dir ./output

ls ./output


uv pip install torchmetrics==0.11.4
uv run python -m piper_train.export_onnx \
    bryce-3499.ckpt \
    bryce-3499.onnx
    
cp /path/to/training_dir/config.json \
   /path/to/model.onnx.json


english, bryce
https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/bryce/medium
https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/bryce/medium/bryce-3499.ckpt


wget.exe https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx
wget.exe https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/bryce/medium/en_US-bryce-medium.onnx.json
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json
&./piper.exe --model en_US-bryce-medium.onnx --config en_US-bryce-medium.onnx.json --text "Hello, world!" --output_file output.wav


cmake -B build .
cmake --build build

https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/ryan/medium
https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main/en/en_US/ryan/medium
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/epoch=4641-step=3104302.ckpt
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/config.json
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/dataset.jsonl.gz
wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/train.sh