from __future__ import annotations

import json
import pickle
import re
import uuid
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from raptor.output_translation import load_translations, print_legs, render_legs
from raptor.services.map_visualizer import RouteVisualizer
from raptor.services.raptor_service import run_raptor_from_assistant_json


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MAPS_DIR = BASE_DIR / "web" / "generated_maps"

INTENT_MODEL_DIR = BASE_DIR / "cairo_assistant" / "intent_model"
ADAPTER_DIR = BASE_DIR / "cairo_assistant" / "nilechat_cairo_final_v1"
NETWORK_PATH = DATA_DIR / "network.pkl"
TRANSLATIONS_PATH = DATA_DIR / "translations.txt"
BASE_MODEL_ID = "MBZUAI-Paris/Nile-Chat-4B"
adapter_model_id = "hananelhosary8/Nile-Chat-4B-Cairo-Transit-Extractor"
WHISPER_MODEL_ID = "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic"

_MODELS: dict[str, Any] | None = None
_NETWORK = None
_STOP_TRANSLATOR = None


def get_models() -> dict[str, Any]:
    global _MODELS

    if _MODELS is not None:
        return _MODELS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ.get("PATH", "")

    local_whisper_path = snapshot_download(
        repo_id=WHISPER_MODEL_ID,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
    )

    local_nile_path = snapshot_download(
        repo_id=BASE_MODEL_ID,
        cache_dir=str(CACHE_DIR),
        local_files_only=True,
    )

    whisper_pipe = pipeline(
        task="automatic-speech-recognition",
        model=local_whisper_path,
        chunk_length_s=30,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    intent_tokenizer = AutoTokenizer.from_pretrained(str(INTENT_MODEL_DIR), local_files_only=True)
    intent_model = AutoModelForSequenceClassification.from_pretrained(
        str(INTENT_MODEL_DIR),
        local_files_only=True,
    )
    intent_model.to(device)
    intent_model.eval()

    with open(INTENT_MODEL_DIR / "label_map.json", "r", encoding="utf-8") as file:
        label_map = json.load(file)

    llm_tokenizer = AutoTokenizer.from_pretrained(local_nile_path)
    if llm_tokenizer.pad_token is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if device == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["device_map"] = {"": 0}
    else:
        model_kwargs["device_map"] = "cpu"

    base_model = AutoModelForCausalLM.from_pretrained(local_nile_path, **model_kwargs)
    llm_model = PeftModel.from_pretrained(base_model, adapter_model_id, local_files_only=True)
    llm_model.eval()

    _MODELS = {
        "device": device,
        "whisper": whisper_pipe,
        "intent_tokenizer": intent_tokenizer,
        "intent_model": intent_model,
        "id2label": {int(k): v for k, v in label_map["id2label"].items()},
        "llm_tokenizer": llm_tokenizer,
        "llm_model": llm_model,
    }
    return _MODELS


def transcribe_audio_bytes(audio_bytes: bytes, suffix: str = ".webm") -> str:
    if not audio_bytes:
        return ""

    models = get_models()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_path = temp_audio.name

    try:
        prediction = models["whisper"](temp_path, generate_kwargs={"language": "arabic"})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    text = prediction.get("text", "")
    return text.strip()


def get_network():
    global _NETWORK

    if _NETWORK is None:
        with open(NETWORK_PATH, "rb") as file:
            _NETWORK = pickle.load(file)
    return _NETWORK


def get_stop_translator():
    global _STOP_TRANSLATOR

    if _STOP_TRANSLATOR is None:
        _STOP_TRANSLATOR = load_translations(str(TRANSLATIONS_PATH), get_network())
    return _STOP_TRANSLATOR


def predict_intent(text: str) -> str:
    models = get_models()
    inputs = models["intent_tokenizer"](text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(models["device"]) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = models["intent_model"](**inputs)

    prediction_id = torch.argmax(outputs.logits, dim=1).item()
    return models["id2label"][prediction_id]


def _extract_json_candidates(raw_response: str) -> list[str]:
    candidates: list[str] = []
    stack = 0
    start_idx: int | None = None

    for idx, char in enumerate(raw_response):
        if char == "{":
            if stack == 0:
                start_idx = idx
            stack += 1
        elif char == "}":
            if stack == 0:
                continue
            stack -= 1
            if stack == 0 and start_idx is not None:
                candidates.append(raw_response[start_idx : idx + 1])
                start_idx = None

    return candidates


def _looks_like_stop_mapping(value: Any) -> bool:
    return isinstance(value, dict) and any(
        isinstance(value.get(key), str) and value.get(key).strip()
        for key in ("official_name_ar", "official_name_en", "name", "value")
    )


def _coerce_stop_name(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip(" .,:;-/\n\t")
        return cleaned or None

    if _looks_like_stop_mapping(value):
        for key in ("official_name_ar", "official_name_en", "name", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None


def _normalize_extractor_output(parsed: Any) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None

    start_name = _coerce_stop_name(parsed.get("start_point"))
    end_name = _coerce_stop_name(parsed.get("end_point"))
    if not start_name or not end_name:
        return None

    normalized = dict(parsed)
    normalized["intent"] = parsed.get("intent") or "navigation"
    normalized["start_point"] = start_name
    normalized["end_point"] = end_name
    return normalized

# =============================================================================
# def _fallback_extract_navigation_json(user_query: str) -> dict[str, Any] | None:
#     text = " ".join(user_query.split())
#     patterns = [
#         r"(?:^|\\s)من\\s+(?P<start>.+?)\\s+(?:إلى|الى|لغاية|لحد|حتى|to)\\s+(?P<end>.+)$",
#         r"(?:^|\\s)(?:عايز اروح|عايز أروح|اروح|أروح|روحني|وديني|وصلني|take me)\\s+من\\s+(?P<start>.+?)\\s+(?:إلى|الى|لغاية|لحد|حتى|to)\\s+(?P<end>.+)$",
#         r"(?:^|\\s)(?:to)\\s+(?P<end>.+?)\\s+(?:from)\\s+(?P<start>.+)$",
#     ]
# 
#     for pattern in patterns:
#         match = re.search(pattern, text, flags=re.IGNORECASE)
#         if not match:
#             continue
# 
#         start_name = _coerce_stop_name(match.group("start"))
#         end_name = _coerce_stop_name(match.group("end"))
#         if start_name and end_name:
#             return {
#                 "intent": "navigation",
#                 "start_point": start_name,
#                 "end_point": end_name,
#             }
# 
#     return None
# 
# =============================================================================

def extract_navigation_json(user_query: str) -> dict[str, Any] | None:
    models = get_models()

    
    instruction = "استخرج محطات البداية والنهاية كملف JSON"
    prompt = f"### Instruction:\n{instruction}\n\nInput: {user_query}\n\n### Response:\n"
   
    inputs = models["llm_tokenizer"](prompt, return_tensors="pt").to(models["llm_model"].device)

    with torch.no_grad():
        outputs = models["llm_model"].generate(
            **inputs,
            max_new_tokens=120,
            num_beams=2,
            repetition_penalty=1.3,
            eos_token_id=models["llm_tokenizer"].eos_token_id,
        )

    raw_response = models["llm_tokenizer"].decode(
        outputs[0][len(inputs.input_ids[0]):],
        skip_special_tokens=True,
    ).strip()

    for candidate in [raw_response, *_extract_json_candidates(raw_response)]:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        normalized = _normalize_extractor_output(parsed)
        if normalized is not None:
            return normalized

    return None


def build_map(legs: list[dict[str, Any]]) -> str | None:
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    network = get_network()
    visualizer = RouteVisualizer(
        network.stops,
        network.shapes,
        stop_name_func=get_stop_translator(),
    )
    output_name = f"route_{uuid.uuid4().hex}.html"
    output_path = MAPS_DIR / output_name
    visualizer.plot_path(legs).save(str(output_path))
    return f"/maps/{output_name}"


def _route_message(extractor_output: dict[str, Any], formatted_legs: list[str]) -> str:
    start_name = extractor_output.get("start_point")
    end_name = extractor_output.get("end_point")
    steps_text = "\n".join(formatted_legs)
    return f"أفضل مسار من {start_name} إلى {end_name}:\n{steps_text}"


def process_chat_message(user_message: str) -> dict[str, Any]:
    text = user_message.strip()
    if not text:
        return {
            "intent": "empty",
            "assistant_message": "اكتب رسالتك الأول.",
            "extractor_output": None,
            "route": None,
            "map_url": None,
            "error": None,
        }

    intent = predict_intent(text)

    if intent == "greeting":
        return {
            "intent": intent,
            "assistant_message": "أهلا بيك، قول لي رايح منين لفين؟",
            "extractor_output": None,
            "route": None,
            "map_url": None,
            "error": None,
        }

    if intent != "navigation":
        return {
            "intent": intent,
            "assistant_message": "أقدر أساعدك في معرفة طريق المواصلات داخل القاهرة. اكتب المكان اللي هتتحرك منه وإلى فين.",
            "extractor_output": None,
            "route": None,
            "map_url": None,
            "error": None,
        }

    extractor_output = extract_navigation_json(text)
    if extractor_output is None:
        return {
            "intent": intent,
            "assistant_message": "فهمت إنك عايز مسار، لكن ماقدرتش أحدد البداية والنهاية بشكل واضح.",
            "extractor_output": None,
            "route": None,
            "map_url": None,
            "error": "Navigation JSON extraction failed.",
            "raw_model_output": extractor_output,
        }

    #extractor_output.setdefault("intent", "navigation")
    route_result = run_raptor_from_assistant_json(get_network(), extractor_output)

    if not isinstance(route_result, list):
        return {
            "intent": intent,
            "assistant_message": str(route_result),
            "extractor_output": extractor_output,
            "route": None,
            "map_url": None,
            "error": str(route_result),
        }

    formatted_legs = render_legs(route_result, get_stop_translator())
    map_url = build_map(route_result)
    return {
        "intent": intent,
        "assistant_message": _route_message(extractor_output, formatted_legs),
        "extractor_output": extractor_output,
        "route": {
            "legs": route_result,
            "formatted_legs": formatted_legs,
        },
        "map_url": map_url,
        "error": None,
    }


def _route_message(extractor_output: dict[str, Any], formatted_legs: list[str]) -> str:
    return "\n".join(formatted_legs)
