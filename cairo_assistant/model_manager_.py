# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:11:14 2026

@author: Samia
"""

import torch
_models = None


def get_models():
    global _models

    if _models is None:
        print("Loading models for the first time...")
        from .model_setup__ import load_models
        _models = load_models()
        print("Models loaded and cached.")
    else:
        print("Using cached models.")

    return _models


def predict_intent(text: str):
    models = get_models()

    tokenizer = models["intent_tokenizer"]
    model = models["intent_model"]
    id2label = models["id2label"]
    device = models["device"]

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    pred_id = torch.argmax(outputs.logits, dim=1).item()

    return id2label[pred_id]