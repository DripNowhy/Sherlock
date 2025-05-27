from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

CORRECTION_TEMPLATE = """Below is a QUESTION from a user and an EXAMPLE RESPONSE.
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

question = """Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: What is the age gap between these two people in image? (Unit: years)"""

image_path = "./demo/case.jpg"

model_path = "Tuwhy/Llama-3.2V-11B-Sherlock-iter2"

model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map='cpu',
            ).cuda().eval()

device = 'cuda'
processor = AutoProcessor.from_pretrained(model_path)

# kwargs_default = dict(do_sample=False, max_new_tokens=2048, temperature=0.0, top_p=None, num_beams=1)
kwargs_default = dict(do_sample=True, max_new_tokens=2048, temperature=0.6, top_p=0.7, num_beams=1)

image = Image.open(image_path)
messages = [
    {'role': 'user', 'content': [
        {'type': 'image'},
        {'type': 'text', 'text': question}
    ]}
]

input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, return_tensors='pt').to(device)
output = model.generate(**inputs, **kwargs_default)
response = processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')

print(f"INTIAL RESPONSE: {response}")

for i in range(3):
    prompt = CORRECTION_TEMPLATE.format(
        Question=question,
        Example_Response=response
    )
    messages = [
        {'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(image, input_text, return_tensors='pt').to(device)
    output = model.generate(**inputs, **kwargs_default)
    response = processor.decode(output[0][inputs['input_ids'].shape[1]:]).replace('<|eot_id|>', '')

    print(f"REFINED RESPONSE {i+1}: {response}")