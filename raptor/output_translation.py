# -*- coding: utf-8 -*-
"""
Created on Thu Feb 26 16:08:26 2026

@author: Samia
"""

# raptor/output_translation.py
import pandas as pd

def load_translations(translations_path: str, network) -> dict:
    """
    Load translations 
    """
    translations = pd.read_csv(translations_path)

    stop_name_ar = (
        translations[
            (translations.table_name == "stops") &
            (translations.field_name == "stop_name") &
            (translations.language == "ar")
        ]
        .set_index("field_value")["translation"]
        .to_dict()
    )

    def stop_name(sid):
        en = network.stop_id_to_name.get(sid, sid)
        return stop_name_ar.get(en, en)

    return stop_name


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