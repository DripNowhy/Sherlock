import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
import json
import os
from tqdm import tqdm
import random
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate self-correction data for training')
    parser.add_argument('--model_path', type=str, default=True,
                      help='Path to the model directory')
    parser.add_argument('--input_file', type=str, default=True,
                      help='Path to input JSON file')
    parser.add_argument('--output_file', type=str, default=True,
                      help='Path to output JSON file')
    parser.add_argument('--max_model_len', type=int, default=4096,
                      help='Maximum model length')
    parser.add_argument('--max_num_seqs', type=int, default=32,
                      help='Maximum number of sequences')
    return parser.parse_args()

def add_noise_to_image(img, noise_level):
    """
    Add Gaussian noise to an image.
    
    Args:
        img: PIL Image object
        noise_level: Float between 0 and 1 indicating noise intensity
    
    Returns:
        PIL Image with added noise
    """
    img_array = np.array(img)
    noise = np.random.normal(0, noise_level * 255, img_array.shape).astype(np.uint8)
    noisy_img_array = np.clip(img_array + noise, 0, 255)
    return Image.fromarray(noisy_img_array.astype(np.uint8))

def generate_response(text, image, prefix, llm, tokenizer):
    """
    Generate response using the LLM model.
    
    Args:
        text: Input text prompt
        image: Input image
        prefix: Prefix for the response
        llm: LLM model instance
        tokenizer: Tokenizer instance
    
    Returns:
        Generated response string
    """
    messages = [{
        "role": "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": text.replace('<image>', '')
        }]
    }]

    prompt = tokenizer.apply_chat_template(messages,
                                        add_generation_prompt=True,
                                        tokenize=False)
    prompt = prompt + prefix

    sampling_params = SamplingParams(temperature=0.7,
                                    max_tokens=2048,
                                    stop_token_ids=None)

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": image
        },
    }

    output = llm.generate(inputs, sampling_params)
    response = output[0].outputs[0].text.split('</CONCLUSION>')[0] + '</CONCLUSION>'
    print(prefix+response)
    return prefix+response

def main():
    args = parse_args()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True
    )

    # Load input data
    with open(args.input_file, "r") as f:
        datas = json.load(f)

    save_data = []

    for data in tqdm(datas):
        text = data["conversations"][0]["value"]
        response = data["rejected"]["value"]
        image = data["images"]
        
        # Process image
        img = Image.open(image).convert("RGB")
        noise_level = random.uniform(0.15, 0.25)
        random_number = random.randint(0, 2)

        chosen_response = data["chosen"]["value"]
        
        # Generate prefix based on random number
        if random_number == 0:
            prefix = ""
            prefix_chosen = ""
        elif random_number == 1:
            prefix = response.split('<CAPTION>')[0]
            prefix_chosen = chosen_response.split('<CAPTION>')[0]
        elif random_number == 2:
            noise_level = 1
            prefix = response.split('<REASONING>')[0]
            prefix_chosen = chosen_response.split('<REASONING>')[0]

        # Generate noisy image
        if noise_level == 0:
            noisy_image = img
        elif noise_level == 1:
            noisy_image = Image.fromarray(np.random.randint(0, 256, img.size[::-1] + (3,), dtype=np.uint8))
        else:
            noisy_image = add_noise_to_image(img, noise_level)

        # Generate response and create corrected data
        reject_response = generate_response(text, noisy_image, prefix, llm, tokenizer)

        i = data["truncation"]
        e = data["noise_level"]

        self_correction_data = {
            "conversations": data["conversations"],
            "chosen": data["chosen"],
            "rejected": {
                "from": "gpt",
                "value": reject_response
            },
            "prefix": prefix_chosen,
            "prefix_l": prefix,
            "images": data["images"],
            "weights": 1 / (0.5 + (i/4) ** (0.5 + (e/2)))
        }
        save_data.append(self_correction_data)

    # Save processed data
    with open(args.output_file, 'w') as f:
        json.dump(save_data, f, indent=4)

if __name__ == "__main__":
    main()