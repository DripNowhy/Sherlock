import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
import json
import os
from tqdm import tqdm
import random

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate refined responses for image-based questions')
    parser.add_argument('--model_path', type=str, default='SHERLOCK-SFT-DIR',
                      help='Path to the model checkpoint')
    parser.add_argument('--input_file', type=str, default='INPUT-DIR',
                      help='Path to input JSON file containing questions and images')
    parser.add_argument('--output_file', type=str, default='./train/LLaMA-Factory/data/sherlock_online_candidate.json',
                      help='Path to save the output JSON file')
    parser.add_argument('--max_model_len', type=int, default=4096,
                      help='Maximum model length')
    parser.add_argument('--max_num_seqs', type=int, default=32,
                      help='Maximum number of sequences')
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Sampling temperature')
    parser.add_argument('--max_tokens', type=int, default=2048,
                      help='Maximum number of tokens to generate')
    return parser.parse_args()

# Template for refining responses
REFINE_TEMPLATE = """<image>Below is a QUESTION from a user and an EXAMPLE RESPONSE.
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

def initialize_model(args):
    """Initialize the tokenizer and LLM model."""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    llm = LLM(
        model=args.model_path,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True
    )
    return tokenizer, llm

def generate_response(text, image, tokenizer, llm, args):
    """Generate a response for the given text and image."""
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

    sampling_params = SamplingParams(temperature=args.temperature,
                                    max_tokens=args.max_tokens,
                                    stop_token_ids=None)

    img = Image.open(image).convert("RGB")

    inputs = {
        "prompt": prompt,
        "multi_modal_data": {
            "image": img
        },
    }

    output = llm.generate(inputs, sampling_params)
    return output[0].outputs[0].text.split('</CONCLUSION>')[0] + '</CONCLUSION>'

def main():
    """Main function to process data and generate refined responses."""
    args = parse_args()
    tokenizer, llm = initialize_model(args)

    # Load input data
    with open(args.input_file, "r") as f:
        datas = json.load(f)

    save_data = []

    # Process each data point
    for data in tqdm(datas):
        text = data["question"]
        image = data["images"]
        
        # Generate initial response
        response = generate_response(text, image, tokenizer, llm, args)
        
        # Create data structure for refined responses
        correction_data = {
            "conversations": [
                {
                    "from": "human",
                    "value": text
                },
                {
                    "correction_turn": 0,
                    "from": "gpt",
                    "value": response
                }
            ]
        }

        # Generate multiple refinements
        for correction_turn in range(1, 4):
            refine_prompt = REFINE_TEMPLATE.format(Question=text, Example_Response=response)
            response = generate_response(refine_prompt, image, tokenizer, llm, args)
            correction_data["conversations"].append({
                "correction_turn": correction_turn,
                "from": "gpt",
                "value": response
            })

        # Add image and ground truth
        correction_data["images"] = image
        save_data.append(correction_data)

    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(save_data, f, indent=4)

if __name__ == "__main__":
    main()