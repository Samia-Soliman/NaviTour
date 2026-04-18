# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 14:32:55 2026

@author: Samia
"""

import pandas as pd
from collections import deque
from rapidfuzz import process, fuzz
from functools import lru_cache
import re
import requests
import math
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
class KafrAdvancedRouter:
    def __init__(self):
        self.stops = pd.read_csv(BASE_DIR +'/data/stops.txt').rename(columns=lambda x: x.strip())
        self.stop_times = pd.read_csv(BASE_DIR +'/data/stop_times.txt').rename(columns=lambda x: x.strip())
        self.trips = pd.read_csv(BASE_DIR +'/data/trips.txt').rename(columns=lambda x: x.strip())
        
        self.stop_to_trips = self.stop_times.groupby('stop_id')['trip_id'].apply(list).to_dict()

        # 🧠 build smart index
        self.stop_index = {}
        self.stop_coords = {}

        for _, row in self.stops.iterrows():
            sid = row['stop_id']
            name = str(row['stop_name'])

            norm = self.normalize_text(name)

            if norm not in self.stop_index:
                self.stop_index[norm] = []
            self.stop_index[norm].append(sid)

            # save coords
            self.stop_coords[sid] = (row.get('stop_lat'), row.get('stop_lon'))

        self.stop_names = tuple(self.stop_index.keys())

    # =========================
    # 🔤 NORMALIZATION
    # =========================
    def normalize_text(self, text):
        text = text.lower().strip()

        # Arabic normalization
        text = re.sub(r'[إأآا]', 'ا', text)
        text = re.sub(r'ى', 'ي', text)
        text = re.sub(r'ة', 'ه', text)

        # remove symbols
        text = re.sub(r'[^\w\s]', '', text)

        return text

    # =========================
    # ⚡ FUZZY MATCH (CACHED)
    # =========================
    @lru_cache(maxsize=500)
    def _fuzzy_match(self, text):
        return process.extractOne(text, self.stop_names, scorer=fuzz.ratio)

    # =========================
    # 🌍 GEO HELPERS
    # =========================
    def geocode(self, place_name):
      url = "https://nominatim.openstreetmap.org/search"
      
      params = {
          "q": f"{place_name}, Kafr El-Sheikh, Egypt",
          "format": "json",
          "limit": 1
      }
      
      headers = {
          "User-Agent": "kafr-router-app"
      }
      
      try:
          response = requests.get(url, params=params, headers=headers, timeout=5)
      
          if response.status_code != 200:
              return None
      
          data = response.json()
      
          if not data:
              return None
      
          return float(data[0]["lon"]), float(data[0]["lat"])
      
      except Exception:
          return None

    def distance(self, a, b):
        # Haversine
        lat1, lon1 = a
        lat2, lon2 = b

        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        x = (math.sin(dlat/2)**2 +
             math.cos(math.radians(lat1)) *
             math.cos(math.radians(lat2)) *
             math.sin(dlon/2)**2)

        return 2 * R * math.asin(math.sqrt(x))

    def nearest_stop(self, coords, max_km=2):
        best = None
        best_dist = 999

        for sid, sc in self.stop_coords.items():
            if None in sc:
                continue

            d = self.distance(coords, sc)
            if d < best_dist and d <= max_km:
                best = sid
                best_dist = d

        return best

    # =========================
    # 🎯 MAIN MATCH
    # =========================
    def get_stop_id(self, user_input, threshold=80):
        norm = self.normalize_text(user_input)

        result = self._fuzzy_match(norm)

        if result:
            match, score, _ = result
            if score >= threshold:
                return self.stop_index[match][0]

        # 🌍 fallback
        coords = self.geocode(user_input)
        print("DEBUG coords:", coords)
        if coords:
            sid = self.nearest_stop(coords)
            if sid:
                print("⚠️ using nearest stop (geo fallback)")
                return sid

        return None

    def suggest_stops(self, user_input, limit=3):
        norm = self.normalize_text(user_input)

        matches = process.extract(
            norm,
            self.stop_names,
            scorer=fuzz.ratio,
            limit=limit
        )

        return [m[0] for m in matches]

    # =========================
    # 🧭 ROUTING (UNCHANGED CORE)
    # =========================
    def find_route(self, start, end, max_stages=5):
        if isinstance(start, str):
            start = self.get_stop_id(start)
        if isinstance(end, str):
            end = self.get_stop_id(end)

        if start is None or end is None:
            print("❌ محطة غير معروفة")
            return None


        queue = deque([(start, [])])
        visited = {start: 0}

        while queue:
            curr_stop, path = queue.popleft()
            
            if len(path) >= max_stages:
                continue
                
            for tid in self.stop_to_trips.get(curr_stop, []):
                trip_stops = self.stop_times[self.stop_times['trip_id'] == tid].sort_values('stop_sequence')
                curr_seq = trip_stops[trip_stops['stop_id'] == curr_stop]['stop_sequence'].iloc[0]
                
                for _, row in trip_stops[trip_stops['stop_sequence'] > curr_seq].iterrows():
                    next_stop_id = row['stop_id']

                    new_leg = {
                        'trip_id': tid,
                        'from_stop': curr_stop,
                        'to_stop': next_stop_id,
                        'dep_time': trip_stops[trip_stops['stop_id'] == curr_stop]['departure_time'].iloc[0],
                        'arr_time': row['arrival_time']
                    }

                    new_path = path + [new_leg]

                    if next_stop_id == end:
                        return new_path

                    if next_stop_id not in visited or len(new_path) < visited[next_stop_id]:
                        visited[next_stop_id] = len(new_path)
                        queue.append((next_stop_id, new_path))

        return None

    def time_to_minutes(self, t):
        # "HH:MM:SS" → minutes
        h, m, s = map(int, t.split(":"))
        return h * 60 + m + s // 60
    
    def minutes_to_time(self, mins):
        h = mins // 60
        m = mins % 60
        return f"{h} ساعة و {m} دقيقة"
    
    def calculate_eta(self, path):
        if not path:
            return 0
    
        total = 0
    
        for leg in path:
            dep = self.time_to_minutes(leg['dep_time'])
            arr = self.time_to_minutes(leg['arr_time'])
    
            diff = arr - dep
    
            # handle overnight trips
            if diff < 0:
                diff += 24 * 60
    
            total += diff
    
        return total
    
    
    def print_itinerary(self, path):
        if not path:
            print("❌ لم يتم العثور على مسار.")
            return
    
        total_time = self.calculate_eta(path)
    
        print(f"✅ المسار ({len(path)} مراحل)")
        print(f"⏱️ الزمن المتوقع: {self.minutes_to_time(total_time)}")
        print("=" * 40)
    
        for i, leg in enumerate(path):
            from_n = self.stops[self.stops['stop_id'] == leg['from_stop']]['stop_name'].values[0]
            to_n = self.stops[self.stops['stop_id'] == leg['to_stop']]['stop_name'].values[0]
            t_info = self.trips[self.trips['trip_id'] == leg['trip_id']].iloc[0]
    
            leg_time = self.time_to_minutes(leg['arr_time']) - self.time_to_minutes(leg['dep_time'])
    
            print(f"مرحلة {i+1}: [{t_info['trip_headsign']}]")
            print(f"من: {from_n}")
            print(f"إلى: {to_n}")
            print(f"مدة المرحلة: {self.minutes_to_time(leg_time)}")
            print("-" * 30)