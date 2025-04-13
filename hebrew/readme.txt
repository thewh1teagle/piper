Cloud: cloud.vastai.com
Image: vastai/pytorch:2.5.1-cuda-12.1.1

1.
    sudo apt-get install espeak-ng -y
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
2.
    cd src/python
    uv venv
    uv pip install -e .
    ./build_monotonic_align.sh
3.
    uv run python -m piper_train.preprocess \
    --language he \
    --input-dir ../../hebrew/dummy_dataset \
    --output-dir ./train \
    --dataset-format ljspeech \
    --single-speaker \
    --sample-rate 22050 \
    --raw-phonemes

4. Prepare checkpoint
    wget https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/ryan/medium/epoch=4641-step=3104302.ckpt
5. Train
    uv run python -m piper_train \
        --dataset-dir "./train" \
        --accelerator 'gpu' \
        --devices 1 \
        --batch-size 16 \
        --validation-split 0 \
        --num-test-examples 0 \
        --max_epochs 10000 \
        --resume_from_checkpoint ./epoch=4641-step=3104302.ckpt \
        --checkpoint-epochs 1 \
        --precision 32

6. Check while train
    cat ../../etc/test_sentences/test_he.jsonl  | \
        python3 -m piper_train.infer \
            --sample-rate 22050 \
            --checkpoint ./train/lightning_logs/version_0/checkpoints/*.ckpt \
            --output-dir ./output

7. check loss_disc_all graph and ensure it keep decreasing
    tensorboard --logdir ./train/lightning_logs/
    
8. Export onnx
    uv run python -m piper_train.export_onnx file.ckpt model.onnx
    cp ./train/config.json config.json

Depcrecated
    uv pip install torchmetrics==0.11.4
    uv pip install "numpy<2"
    uv pip install pytorch-lightning
    cmake -B build .
    cmake --build build
    &./piper.exe --model en_US-bryce-medium.onnx --config config.json --text "Hello, world!" --output_file output.wav
    english, bryce medium