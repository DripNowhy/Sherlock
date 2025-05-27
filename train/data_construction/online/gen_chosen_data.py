import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from PIL import Image
import json
import os
from tqdm import tqdm
import random

JUDGE_TEMPLATE = """I will provide you a question and three responses. Your task is to determine whether these three responses are semantically equivalent to each other.

For multiple-choice questions, responses are considered equivalent if they express the same answer choice.
For subjective questions, responses are considered equivalent if they convey the same meaning or conclusion.

[QUESTION]:
{question}

[ANSWER1]:
{answer1}
[ANSWER2]:
{answer2}
[ANSWER3]:
{answer3}

Please analyze if these three responses are semantically equivalent to each other.
You should only output Yes or No.

Yes means all three responses are semantically equivalent to each other.
No means at least one response differs in meaning from the others.
"""

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate and judge responses using Qwen model')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                      help='Name of the model to use')
    parser.add_argument('--input_file', type=str, default='./train/LLaMA-Factory/data/sherlock_online_candidate.json',
                      help='Path to input JSON file containing questions and responses')
    parser.add_argument('--output_file', type=str, default='./train/LLaMA-Factory/data/sherlock_online_chosen.json',
                      help='Path to save the output JSON file')
    parser.add_argument('--max_model_len', type=int, default=4096,
                      help='Maximum model length')
    parser.add_argument('--max_num_seqs', type=int, default=32,
                      help='Maximum number of sequences')
    return parser.parse_args()

def initialize_model(model_name, max_model_len, max_num_seqs):
    """Initialize the LLM model and tokenizer."""
    llm = LLM(
        model=model_name,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return llm, tokenizer

def load_data(input_file):
    """Load data from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)

def create_judge_prompt(question, answers):
    """Create the prompt for judging responses."""
    
    return JUDGE_TEMPLATE.format(
        question=question,
        answer1=answers[0],
        answer2=answers[1],
        answer3=answers[2]
    )

def process_item(item, llm, tokenizer):
    """Process a single item and get model's judgment."""
    try:
        # Extract question and answers
        question = item['conversations'][0]['value']
        answers = []
        initial_response = item['conversations'][1]['value']
        final_response = item['conversations'][4]['value']
        for conv in item['conversations'][2:5]:
            ans = conv['value'].split('<CONCLUSION>')[1].split('</CONCLUSION>')[0]
            answers.append(ans)
        
        # Create and format prompt
        prompt = create_judge_prompt(question, answers)
        messages = [{
            "role": "system",
            "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        }, {
            "role": "user",
            "content": prompt
        }]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Generate response
        sampling_params = SamplingParams(temperature=0.0, max_tokens=64, stop_token_ids=None)
        output = llm.generate(prompt, sampling_params)
        
        if output and output[0].outputs and output[0].outputs[0].text:
            response_text = output[0].outputs[0].text.lower()
            if 'yes' in response_text:
                new_item = {}
                new_item["conversations"] = [item['conversations'][0]]
                new_item["chosen"] = {
                    'from': 'gpt',
                    'value': final_response
                }
                new_item["rejected"] = {
                    'from': 'gpt',
                    'value': initial_response
                }
                new_item["images"] = item["images"]
                return new_item
            else:
                return None
        else:
            raise ValueError("Invalid model output")
            
    except Exception as e:
        print(f"Error processing item: {e}")
        return None

def main():
    """Main function to process data and generate judgments."""
    args = parse_args()
    
    # Initialize model and tokenizer
    llm, tokenizer = initialize_model(args.model_name, args.max_model_len, args.max_num_seqs)
    
    # Load data
    data = load_data(args.input_file)
    
    # Process each item
    save_list = []
    for item in tqdm(data):
        processed_item = process_item(item, llm, tokenizer)
        if processed_item is not None:
            save_list.append(processed_item)
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(save_list, f, indent=4)

if __name__ == "__main__":
    main()