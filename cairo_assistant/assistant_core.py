# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:36:47 2026

@author: Samia
"""

import torch, re

def ask_cairo_assistant(user_query, tokenizer, model):
    nav_triggers = ['من', 'لـ', 'إلى', 'رايح', 'اروح', 'وديني', 'اوصل', 'فين', 'طريق', 'محطة']
    is_nav = any(word in user_query for word in nav_triggers)

    system_msg = ("You are a Cairo Transport Extractor. Respond ONLY with a single JSON."
                  if is_nav else
                  "You are an expert Cairo transport assistant. Respond briefly in Egyptian Arabic.")

    messages = [{"role": "system", "content": system_msg},
                {"role": "user", "content": user_query}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=2,
            repetition_penalty=1.5,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

    if is_nav:
        json_match = re.search(r'\{.*\}', response.replace('\n', ''))
        return json_match.group() if json_match else response.strip(), True
    return response.split('!')[0].split('.')[0].strip() , False