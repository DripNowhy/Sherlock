# Sherlock Evaluation Guide

This guide provides detailed instructions for the evaluation of Sherlock model.

## Installation
```bash
cd inference/VLMEvalKit
pip install -e .
pip install transformers==4.45.2
```

## Inference

Here is two way to conduct inference of our Sherlock model.

### Quick Use

Runing the following command to quick use our Sherlock model and self-correction inference:
```bash
python ./inference/demo/inference.py
```

### Benchmark Evaluation

Our evaluation using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) package.

1. Create `.env` file in `./inference/VLMEvalKit/`.

2. Prepare your OPENAI_API_KEY in `.env` file.

3. Using the following command to evaluate sherlock model (8 GPU evaluation):
    ```bash
    bash ./inference/VLMEvalKit/scripts/eval_sherlock.sh
    ```