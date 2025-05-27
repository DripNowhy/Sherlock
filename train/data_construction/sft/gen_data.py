from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
import json
import os
from tqdm import tqdm
import random
import argparse

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate and self-correction responses using LLM')
    
    parser.add_argument('--model_path', type=str, 
                      default="R0-VLM-DIR",
                      help='Path to the model')
    
    parser.add_argument('--input_file', type=str,
                      default="10k-randomly-sampled-data-dir",
                      help='Path to input JSON file')
    
    parser.add_argument('--output_file', type=str,
                      default='./train/LLaMA-Factory/data/sherlock_sft.json',
                      help='Path to output JSON file')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature for generation')
    
    parser.add_argument('--max_tokens', type=int, default=2048,
                      help='Maximum number of tokens to generate')
    
    return parser.parse_args()

# Initialize tokenizer for the model
args = parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

# Template for refining responses
CORRECTION_TEMPLATE = """<image>Below is a QUESTION from a user and an EXAMPLE RESPONSE.
Please provide a more helpful RESPONSE, improving the EXAMPLE RESPONSE by making the content even clearer, more accurate, and with a reasonable logic. 
Focus on addressing the human's QUESTION step by step based on the image without including irrelevant content.

QUESTION: 
{Question}  

EXAMPLE RESPONSE: 
{Example_Response}

Now, refine and improve the RESPONSE further. You can consider two approaches: 
1. REFINEMENT: If the SUMMARY section in the response is closely related to the question, the CAPTION section accurately describes the image, the REASONING section is logically clear and correct without any contradictions, and the CONCLUSION provides an accurate answer based on the previous steps, enhance clarity, accuracy, or reasoning logic as needed.
2. NEW RESPONSE: If the SUMMARY section incorrectly summarizes the intent of the issue, the CAPTION contains content unrelated to or incorrect about the image, there are logical errors or contradictions in the REASONING, or the CONCLUSION incorrectly states the findings, please enhance the accuracy and quality of each step, and craft a more effective RESPONSE that thoroughly resolves the QUESTION. 

RESPONSE:
"""

# Initialize LLM model with specific parameters
llm = LLM(
    model=args.model_path,
    max_model_len=4096,
    max_num_seqs=32,
    enforce_eager=True
)

def generate(text, image):
    """
    Generate a response using the LLM model based on the input text and image.
    
    Args:
        text (str): Input text prompt
        image (str): Path to the input image
        
    Returns:
        str: Generated response with conclusion
    """
    # Prepare messages for the model
    messages = [{
        "role": "user",
        "content": [{
            "type": "image"
        }, {
            "type": "text",
            "text": text.replace('<image>', '')
        }]
    }]

    # Apply chat template and set sampling parameters
    prompt = tokenizer.apply_chat_template(messages,
                                         add_generation_prompt=True,
                                         tokenize=False)

    sampling_params = SamplingParams(temperature=args.temperature,
                                   max_tokens=args.max_tokens,
                                   stop_token_ids=None)

    # Process image
    img = Image.open(image).convert("RGB")

    # Prepare input for model
    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": img
        },
    }

    # Generate response
    output = llm.generate(inputs, sampling_params)
    response = output[0].outputs[0].text
    print(response)
    return response

def main():
    """
    Main function to process data and generate self-correction responses
    """
    # Load input data
    with open(args.input_file, "r") as f:
        datas = json.load(f)

    save_data = []

    random.shuffle(datas)

    # Process each data entry
    for data in tqdm(datas):
        text = data["conversations"][-2]["value"] # only use last turn to conduct self-correction if multi-turn prompt
        image = data["images"][0]
        save_data.append(data)
        
        # Generate and self-correction response
        reject_response = generate(text, image)
        correction_prompt = CORRECTION_TEMPLATE.format(
            Question=text.replace('<image>', ''),
            Example_Response=reject_response
        )
        
        # Create refined SFT data
        self_correction_data = {
            "conversations": [
                {
                    "from": "human",
                    "value": correction_prompt
                },
                {
                    "from": "gpt",
                    "value": data["conversations"][-1]["value"]
                }
            ],
            "images": data["images"]
        }
        save_data.append(self_correction_data)

    # Save processed data
    with open(args.output_file, 'w') as f:
        json.dump(save_data, f, indent=4)

if __name__ == "__main__":
    main()