from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

llama_series={
    'Llama-3.2-11B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-11B-Vision-Instruct'),
    'LLaVA-CoT': partial(llama_vision, model_path='Xkev/Llama-3.2V-11B-cot'),
    'Sherlock-SFT': partial(llama_vision, model_path='Tuwhy/Llama-3.2V-11B-Sherlock-SFT'),
    'Sherlock-Offline': partial(llama_vision, model_path='Tuwhy/Llama-3.2V-11B-Sherlock-Offline'),
    'Sherlock-iter1': partial(llama_vision, model_path='Tuwhy/Llama-3.2V-11B-Sherlock-iter1'),
    'Sherlock-iter2': partial(llama_vision, model_path='Tuwhy/Llama-3.2V-11B-Sherlock-iter2'),
    'Llama-3.2-90B-Vision-Instruct': partial(llama_vision, model_path='meta-llama/Llama-3.2-90B-Vision-Instruct'),
    
}

supported_VLM = {}

model_groups = [
    ungrouped, api_models,
    xtuner_series, qwen_series, llava_series, internvl_series, yivl_series,
    xcomposer_series, minigpt4_series, idefics_series, instructblip_series,
    deepseekvl_series, deepseekvl2_series, janus_series, minicpm_series, cogvlm_series, wemm_series,
    cambrian_series, chameleon_series, video_models, ovis_series, vila_series,
    mantis_series, mmalaya_series, phi3_series, xgen_mm_series, qwen2vl_series,
    slime_series, eagle_series, moondream_series, llama_series, molmo_series,
    kosmos_series, points_series, nvlm_series, vintern_series, h2ovl_series, aria_series,
    smolvlm_series, sail_series, valley_series, vita_series, ross_series, emu_series, ola_series, ursa_series
]

for grp in model_groups:
    supported_VLM.update(grp)
