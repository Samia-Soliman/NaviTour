# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:07:48 2026

@author: Samia
"""

import sys
import os
sys.path.insert(0, os.getcwd())

from raptor.services.raptor_service import run_raptor_from_assistant_json
from raptor.services.map_visualizer import RouteVisualizer
from raptor.output_translation import load_translations
from raptor.utils import format_legs
import pickle
import webbrowser


# Load network once
# 
with open("data/network.pkl", "rb") as f:
    network = pickle.load(f)
    
    
# =============================================================================
# assistant_json = {"intent": "navigation",
#              "start_point": {"official_name_ar": "رمسيس", "official_name_en": "Ramses"},
#              "end_point": {"official_name_ar": "سيتي سنتر", "official_name_en": "City Center Mall"}}
# =============================================================================
# =============================================================================
# assistant_json ={"intent": "navigation",
#     "start_point": {"official_name_ar":  "العباسية", "official_name_en": "Abbaseya"},
#     "end_point": {"official_name_ar": "رمسيس", "official_name_en": "Ramses"}}
# =============================================================================

assistant_json = {"intent": "navigation",
             "start_point": {"official_name_ar": "مدينة نصر", "official_name_en": "Ramses"},
             "end_point": {"official_name_ar": "المعادي", "official_name_en": "City Center Mall"}}

legs_or_error = run_raptor_from_assistant_json(network, assistant_json)

if isinstance(legs_or_error, dict) and "error" in legs_or_error:
    print("❗", legs_or_error["message"])

    if "suggestions" in legs_or_error:
        print("💡 Suggestions:", legs_or_error["suggestions"])

elif isinstance(legs_or_error, list):
    for leg in format_legs(legs_or_error):
        print(leg)

    # visualize only if valid route
    stop_name_ar = load_translations(
        r"D:\G4\graduation project\NaviTour\data\translations.txt",
        network
    )
    visualizer = RouteVisualizer(
        network.stops,
        network.shapes,
        stop_name_func=stop_name_ar
    )

    m = visualizer.plot_path(legs_or_error)
    m.save("cairo_real_route.html")
    webbrowser.open("cairo_real_route.html")

else:
    print("❗", legs_or_error)
