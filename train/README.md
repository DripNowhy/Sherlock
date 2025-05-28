# Sherlock Training Guide

This guide provides detailed instructions for training the Sherlock model.

## Prerequisites

- Python 3.10
- 8 x A100 80G GPU

## Installation

1. Install LLaMA-Factory:
```bash
cd train/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

2. Install specific versions of required packages:
```bash
pip install transformers==4.45.2
pip install trl==0.9.6
```

3. Install [vLLM](https://github.com/vllm-project/vllm) for more efficient data construction:
```bash
pip install vllm
```

4. Replace the DPO trainer implementation:
   - Locate the `dpo_trainer.py` file in your Python environment: `YOUR_ENV/lib/python3.10/site-packages/trl/trainer/dpo_trainer.py`
   - Replace it with the custom implementation from [dpo_trainer.py](https://github.com/DripNowhy/Sherlock/blob/main/train/dpo_trainer.py)

## Training Pipeline

<div align="center">
    <img src="../assets/sherlock_pipeline.png" alt="Sherlock Training Pipeline">
</div>

### 1. Supervised Fine-Tuning (SFT)

#### Data Preparation
1. Format your LLaVA-CoT data in ShareGPT format:
```json
{
    "conversations": [
        {
            "from": "human",
            "value": "<image>{question}"
        },
        {
            "from": "gpt",
            "value": "{response}"
        }
    ],
    "images": [
        "image_path"
    ]
}
```

2. Create training datasets:
   - Randomly sample 10k examples for $\mathcal{D}_A$
   - Randomly sample 10k examples for $\mathcal{D}_B$
   - Save the ShareGPT format data as JSON files in `./train/LLaMA-Factory/data/`

#### Training Steps
1. Train the initial R0 VLM model:
```bash
bash ./train/LLaMA-Factory/bash_train/train_sft_ro_vlm.sh
```

2. Generate Sherlock-SFT dataset:
   - Use the trained R0 VLM model
   - Run the data generation script: `./train/data_construction/sft/gen_data.py`

3. Train the Sherlock SFT model:
```bash
bash ./train/LLaMA-Factory/bash_train/train_sft.sh
```

### 2. Offline

1. Generate offline preference data:
   - Use the script to obtain preference data: `./train/data_construction/offline/gen_data.py`
   - The format of offline data should be:
        ```json
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>{question}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": "{chosen_response}"
            },
            "rejected": {
                "from": "gpt",
                "value": "{rejected_response}"
            },
            "prefix": "{chosen_prefix}",
            "prefix_l": "{rejected_prefix}",
            "weights": "{int}"
            "images": [
                "image_path"
            ]
        }
        ```

2. Train the Sherlock Offline model:
    ```bash
    bash ./train/LLaMA-Factory/bash_train/train_offline.sh
    ```

### Online
1. Sample only question and image from LLaVA-CoT dataset.
2. First generate candidate response with three turn of self-correction using `./train/data_construction/online/gen_candidate.py` file.
3. Running 
`./train/data_construction/online/gen_chosen_data.py` to filter and obtain 5k chosen reasoning trajectory based on self-consistency.
4. Construct online preference data
    -  Running `./train/data_construction/online/gen_rejected_data.py` file.
    - The format of online data should be:
        ```json
        {
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>{question}"
                }
            ],
            "chosen": {
                "from": "gpt",
                "value": "{chosen_response}"
            },
            "rejected": {
                "from": "gpt",
                "value": "{rejected_response}"
            },
            "prefix": "{chosen_prefix}",
            "prefix_l": "{rejected_prefix}",
            "weights": "{int}"
            "images": [
                "image_path"
            ]
        }
        ```

5. Start Online training to obtain **Sherlock Iter1** and **Sherlock Iter2** model.
    ```bash
    bash .train/LLaMA-Factory/bash_train/train_online.sh
    ```