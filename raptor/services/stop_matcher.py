# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:16:49 2026

@author: Samia
"""

# raptor/services/stop_matcher.py

from rapidfuzz import process, fuzz
from shared.arabic_text import normalize_arabic
from raptor.output_translation import load_translations
from raptor.services.geo_utils import get_lat_lon_from_api, find_nearest_stop

class StopMatcher:
    """
    Match user Arabic input to stop_id using translations.
    Provides suggestions if no exact match.
    Ensures returned stop_id exists in network.stop_id_to_idx.
    """

    def __init__(self, network, translations_path):
        """
        network: RAPTOR network object
        translations_path: path to translations.txt
        """
        self.network = network
        self.stop_index = {}

        # Load Arabic translations
        stop_name_func = load_translations(translations_path, network)

        # Build Arabic index from the network DataFrame
        for _, stop in network.stops.iterrows():
            stop_id = stop['stop_id']  
            arabic_name = stop_name_func(stop_id)
            if arabic_name:
                norm_name = normalize_arabic(arabic_name)
                self.stop_index[norm_name] = stop_id

        # Keep list of normalized names for fuzzy matching
        self.stop_names = list(self.stop_index.keys())

    def match(self, user_input, threshold=80):
        """
        Returns stop_id if fuzzy match succeeds, else None
        """
        norm_input = normalize_arabic(user_input)

        match, score, _ = process.extractOne(
            norm_input,
            self.stop_names,
            scorer=fuzz.ratio
        )

        if score >= threshold:
            stop_id = self.stop_index[match]
            # Ensure stop exists in network
            if stop_id in self.network.stop_id_to_idx:
                return stop_id

        return None

    def match_with_fallback(self, user_input, threshold=80, max_distance_km=1.5):
        """
        Returns a valid stop_id:
        1. Fuzzy match in Arabic
        2. If not found, fallback to nearest stop using geocoding API
        """
        # Try fuzzy match
        stop_id = self.match(user_input, threshold)
        if stop_id:
            return stop_id

        # Fallback: get coordinates from API
        coords = get_lat_lon_from_api(user_input)
        if coords:
            nearest_stop_id = find_nearest_stop(self.network, coords, max_distance_km)
            if nearest_stop_id:
                print(f"⚠️ Fallback: using nearest stop '{nearest_stop_id}' for input '{user_input}'")
                return nearest_stop_id

        #Nothing found
        print(f"❌ Could not find stop for input '{user_input}'")
        return None

    def match_with_suggestions(self, user_input, threshold=80, max_suggestions=3):
        """
        Returns dict:
        - type: "matched" or "suggestions"
        - stop_id: if matched
        - suggestions: list of close matches if not
        """
        norm_input = normalize_arabic(user_input)

        matches = process.extract(
            norm_input,
            self.stop_names,
            scorer=fuzz.ratio,
            limit=max_suggestions
        )

        best_match, best_score, _ = matches[0]

        if best_score >= threshold:
            stop_id = self.stop_index[best_match]
            if stop_id in self.network.stop_id_to_idx:
                return {"type": "matched", "stop_id": stop_id}

        suggestions = [m[0] for m in matches]
        return {"type": "suggestions", "suggestions": suggestions}