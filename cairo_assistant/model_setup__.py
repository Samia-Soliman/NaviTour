# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:07:36 2026

@author: Samia
"""


import torch, os, json
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModelForSequenceClassification
from peft import PeftModel
from huggingface_hub import login, snapshot_download
from pathlib import Path

# HF login
HF_TOKEN = "HF_TOKEN"
login(HF_TOKEN)
# specify a fast SSD cache path
FAST_HF_CACHE = "C:/Users/samia/.cache/huggingface/hub"  
os.makedirs(FAST_HF_CACHE, exist_ok=True)


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Whisper ASR
    # -----------------------------
    whisper_model_id = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"
    # Ensure snapshot downloaded to fast cache
    local_whisper_path = snapshot_download(
        repo_id=whisper_model_id,
        cache_dir=FAST_HF_CACHE,
        local_files_only=True,
        #use_auth_token=HF_TOKEN
    )

    whisper_pipe = pipeline(
        task="automatic-speech-recognition",
        model=local_whisper_path,
        chunk_length_s=30,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
        #use_auth_token=HF_TOKEN,
    )

    # -----------------------------
    # NileChat 4B
    # -----------------------------
    base_model_id = "MBZUAI-Paris/Nile-Chat-4B"
    adapter_model_id = "hananelhosary8/Nile-Chat-4B-Cairo-Transit-Extractor"



    # Pre-download model to fast cache
    local_nile_path = snapshot_download(
        repo_id=base_model_id,
        cache_dir=FAST_HF_CACHE,
        local_files_only=True,
        #use_auth_token=HF_TOKEN
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading Cairo Assistant from Hugging Face... please wait.")
    tokenizer = AutoTokenizer.from_pretrained(local_nile_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Force GPU if available for speed
    device_map = "auto"
    if device == "cuda":
        device_map = {"": 0}  # load full model on GPU if enough memory

    base_model = AutoModelForCausalLM.from_pretrained(
        local_nile_path,
        quantization_config=bnb_config,
        device_map=device_map,
        low_cpu_mem_usage=True,  # load fully into GPU for speed
    )
    
    try:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_model_id
        )
        model.eval()
        print("All Models Loaded Successfully from the Cloud!")
    except Exception as e:
        print("Error: Could not load adapter from Hugging Face.")
        print(e)

# =============================================================================
#     # Load PEFT adapter
#     if os.path.exists(adapter_path):
#         model = PeftModel.from_pretrained(base_model, adapter_path, local_files_only=True)
#         model.eval()
#     else:
#         raise FileNotFoundError(f"{adapter_path} not found.")
#         
# =============================================================================
    # -----------------------------
    # Intent Model 
    # -----------------------------
    raw_path = r"D:\G4\graduation project\NaviTour\cairo_assistant\intent_model"
    intent_model_path = str(Path(raw_path).resolve())
    if not os.path.exists(intent_model_path):
        raise FileNotFoundError(f"{intent_model_path} not found")
    intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path, local_files_only=True)
    intent_model = AutoModelForSequenceClassification.from_pretrained(intent_model_path, local_files_only=True)

    intent_model.to(device)
    intent_model.eval()

    with open(os.path.join(intent_model_path, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id2label = {int(k): v for k, v in label_map["id2label"].items()}

    return {
    "whisper": whisper_pipe,
    "llm_tokenizer": tokenizer,
    "llm_model": model,
    "intent_tokenizer": intent_tokenizer,
    "intent_model": intent_model,
    "id2label": id2label,
    "device": device
}