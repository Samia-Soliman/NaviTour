# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:08:26 2026

@author: Samia
"""

# raptor/output_translation.py
import pandas as pd

def get_translators(translations_path: str, network) -> dict:
    """
    Load Arabic translations for stops and routes.
    """
    translations = pd.read_csv(translations_path, encoding="utf-8-sig")

    stop_name_ar = (
        translations[
            (translations.table_name == "stops") &
            (translations.field_name == "stop_name") &
            (translations.language == "ar")
        ]
        .set_index("field_value")["translation"]
        .to_dict()
    )

    route_short_name_ar = (
        translations[
            (translations.table_name == "routes") &
            (translations.field_name == "route_short_name") &
            (translations.language == "ar")
        ]
        .set_index("field_value")["translation"]
        .to_dict()
    )

    route_long_name_ar = (
        translations[
            (translations.table_name == "routes") &
            (translations.field_name == "route_long_name") &
            (translations.language == "ar")
        ]
        .set_index("field_value")["translation"]
        .to_dict()
    )

    def stop_name(sid):
        en = network.stop_id_to_name.get(sid, sid)
        return stop_name_ar.get(en, en)

    def route_short_name(name):
        if not name:
            return name
        return route_short_name_ar.get(name, name)

    def route_long_name(name):
        if not name:
            return name
        return route_long_name_ar.get(name, name)

    return {
        "stop_name": stop_name,
        "route_short_name": route_short_name,
        "route_long_name": route_long_name,
    }


def load_translations(translations_path: str, network):
    """
    Backward-compatible stop-name translator.
    """
    return get_translators(translations_path, network)["stop_name"]


def translate_route_names(items, translators):
    """
    Return a copy of segments/legs with Arabic route names when available.
    """
    translated_items = []

    for item in items:
        translated_item = dict(item)
        translated_item["route_short"] = translators["route_short_name"](item.get("route_short"))
        translated_item["route_long"] = translators["route_long_name"](item.get("route_long"))
        translated_items.append(translated_item)

    return translated_items


def print_legs(legs, stop_name_func):
    """
    Pretty-print collapsed legs using a stop_name function.
    """
    for leg in legs:
        if leg['mode'] == 'WALK':
            print(
                f"WALK: {stop_name_func(leg['from_stop'])} "
                f"→ {stop_name_func(leg['to_stop'])}"
            )
        else:
            print(
                f"{leg['agency']} | {leg['route_short']} ({leg['route_long']})\n"
                f"  {stop_name_func(leg['from_stop'])} "
                f"→ {stop_name_func(leg['to_stop'])}"
            )


def print_segments(segments, stop_name_func):
    """
    Pretty-print full segments using a stop_name function.
    """
    for seg in segments:
        if seg['mode'] == 'WALK':
            print(
                f"WALK: {stop_name_func(seg['from_stop'])} "
                f"→ {stop_name_func(seg['to_stop'])}"
            )
        else:
            print(
                f"{seg['agency']} | {seg['route_short']} ({seg['route_long']})\n"
                f"  {stop_name_func(seg['from_stop'])} "
                f"→ {stop_name_func(seg['to_stop'])}"
            )
def render_legs(legs, stop_name_func):
    """
    Return the same text that print_legs should print.
    """
    lines = []

    for leg in legs:
        if leg["mode"] == "WALK":
            lines.append(
                f"WALK: {stop_name_func(leg['from_stop'])} "
                f"→ {stop_name_func(leg['to_stop'])}"
            )
        else:
            lines.append(
                f"{leg['agency']} | {leg['route_short']} ({leg['route_long']})\n"
                f"  {stop_name_func(leg['from_stop'])} "
                f"→ {stop_name_func(leg['to_stop'])}"
            )

    return lines


def print_legs(legs, stop_name_func):
    """
    Pretty-print collapsed legs using a stop_name function.
    """
    for line in render_legs(legs, stop_name_func):
        print(line)
