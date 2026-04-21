# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:36:47 2026

@author: Samia
"""

import torch, json


def ask_cairo_assistant(user_query, tokenizer, model):

    instruction = "استخرج محطات البداية والنهاية كملف JSON"
    prompt = f"### Instruction:\n{instruction}\n\nInput: {user_query}\n\n### Response:\n"


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )


    full_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    response = full_text.strip()
    print(f"full_text {full_text}")
    print(f"response {response}")
    response = json.loads(response)
    print(f"response {response}")

    return response

from cairo_assistant.model_manager_ import predict_intent, get_models


def handle_input(text):
    intent = predict_intent(text)

    print(f"[DEBUG] Intent: {intent}")

    # ---------------- NAVIGATION ----------------
    if intent == "navigation":
        models = get_models()
        tokenizer = models["llm_tokenizer"] 
        model = models["llm_model"]
        return ask_cairo_assistant(text, tokenizer, model), True
    # ---------------- GREETING ----------------
    elif intent == "greeting":
        return "أهلا بيك 👋", False

    # ---------------- SUPPORT ----------------
    elif intent == "support":
        return "قولّي مشكلتك", False

    # ---------------- FALLBACK  ----------------
    else:
        
        return "..........", False