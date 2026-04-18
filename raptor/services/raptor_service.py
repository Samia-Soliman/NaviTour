# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:17:18 2026

@author: Samia
"""

# raptor/services/raptor_service.py

from raptor.algorithm import mc_raptor
from raptor.utils import extract_solutions, reconstruct, collapse_to_legs, time_to_sec, sec_to_time
from raptor.output_translation import get_translators, print_legs, print_segments, translate_route_names
from raptor.services.stop_matcher import StopMatcher



translations_path =  r"D:\G4\graduation project\NaviTour\data\translations.txt"


def _coerce_stop_name(value):
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None

    if isinstance(value, dict):
        for key in ("official_name_ar", "official_name_en", "name", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return None

def _segments_are_contiguous(segments):
    for i in range(len(segments) - 1):
        if segments[i]["to_stop"] != segments[i + 1]["from_stop"]:
            return False
    return True


def _select_best_contiguous_path(solutions, network):
    valid_paths = []

    for sol in solutions:
        candidate = reconstruct(sol, network)
        if _segments_are_contiguous(candidate):
            valid_paths.append({
                "segments": candidate,
                "arrival_time": sol.time,
                "transfers": sol.transfers,
            })

    if not valid_paths:
        return None

    valid_paths.sort(key=lambda path: (path["arrival_time"], path["transfers"]))
    return valid_paths[0]["segments"]

def run_raptor_from_assistant_json(network ,assistant_json, departure_time="10:00:00"):
    """
    Runs RAPTOR from Cairo assistant JSON.
    Returns formatted legs or an error message.
    """

    if not isinstance(assistant_json, dict):
        return (
            "Error: Invalid navigation payload. Expected a JSON object with "
            "'start_point' and 'end_point'."
        )

    # -----------------------------
    # Initialize StopMatcher
    # -----------------------------
    stop_matcher = StopMatcher(network, translations_path)

    # -----------------------------
    # Extract origin & destination names
    # -----------------------------
    start_name = _coerce_stop_name(assistant_json.get("start_point"))
    end_name = _coerce_stop_name(assistant_json.get("end_point"))
    if not start_name or not end_name:
        return "Error: Missing origin or destination names"

    # -----------------------------
    # Match names to network stop IDs
    # -----------------------------
    origin_id = stop_matcher.match_with_fallback(start_name)
    destination_id = stop_matcher.match_with_fallback(end_name)

    if origin_id is None or destination_id is None:
        return f"Error: Could not find valid stops for '{start_name}' or '{end_name}'"

    print(f" Using stop {origin_id} for origin '{start_name}'")
    print(f" Using stop {destination_id} for destination '{end_name}'")
    

    # -----------------------------
    # Run RAPTOR
    # -----------------------------
    try:
        B, target = mc_raptor(network, origin_id, destination_id, departure_time)
    except KeyError as e:
        return f"Error: RAPTOR KeyError for stop {e}"


    # -----------------------------
    # Extract Pareto-optimal path
    # -----------------------------
    solutions = extract_solutions(B, target)
    if not solutions:
        return "Error: No solution found"

    segments = _select_best_contiguous_path(solutions, network)
    if segments is None:
        # Fallback: keep behavior deterministic even if all candidates are malformed.
        segments = reconstruct(solutions[0], network)

    legs = collapse_to_legs(segments)
    # -----------------------------
    # Load stop translations for readable output
    # -----------------------------
    translators = get_translators(translations_path, network)
    stop_name_func = translators["stop_name"]
    segments = translate_route_names(segments, translators)
    legs = translate_route_names(legs, translators)
    

    # Optional: print nicely
    print_legs(legs, stop_name_func)
    print_segments(segments, stop_name_func)

    return legs
