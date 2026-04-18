# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:45:02 2026

@author: Samia
"""

from KafrEL_sheikh.Kafr_router import KafrAdvancedRouter

router = KafrAdvancedRouter()
path = router.find_route("مسجد الاستاد", "موقف عبود")
router.print_itinerary(path)