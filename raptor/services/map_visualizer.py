# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 21:35:34 2026

@author: Samia
"""

import folium
import numpy as np


class RouteVisualizer:
    def __init__(self, stops_df, shapes_df, stop_name_func=None):
        """
        stops_df -> GTFS stops DataFrame
        shapes_df -> GTFS shapes DataFrame
        """
        self.stops_df = stops_df
        self.shapes_df = shapes_df
        self.stop_coords = {
            row["stop_id"]: (row["stop_lat"], row["stop_lon"])
            for _, row in stops_df.iterrows()
        }
        self.stop_names = {
            row["stop_id"]: row.get("stop_name", row["stop_id"])
            for _, row in stops_df.iterrows()
        }
        self.stop_name_func = stop_name_func
        self._palette = [
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#ff7f0e",
            "#9467bd",
            "#17becf",
            "#8c564b",
            "#e377c2",
            "#bcbd22",
            "#7f7f7f",
        ]

    # ---------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------

    def _nearest_shape_index(self, shape_pts, lat, lon):
        d = (shape_pts[:, 0] - lat) ** 2 + (shape_pts[:, 1] - lon) ** 2
        return np.argmin(d)

    def _expand_leg_with_shape(self, leg):
        if leg["mode"] == "WALK":
            return [[
                self.stop_coords[leg["from_stop"]],
                self.stop_coords[leg["to_stop"]],
            ]]

        shape_id = leg.get("shape_id")
        if shape_id is None:
            return [[
                self.stop_coords[leg["from_stop"]],
                self.stop_coords[leg["to_stop"]],
            ]]

        shape = self.shapes_df[self.shapes_df.shape_id == shape_id]
        if shape.empty:
            return [[
                self.stop_coords[leg["from_stop"]],
                self.stop_coords[leg["to_stop"]],
            ]]

        shape = shape.sort_values("shape_pt_sequence")
        pts = shape[["shape_pt_lat", "shape_pt_lon"]].values

        lat1, lon1 = self.stop_coords[leg["from_stop"]]
        lat2, lon2 = self.stop_coords[leg["to_stop"]]
        i1 = self._nearest_shape_index(pts, lat1, lon1)
        i2 = self._nearest_shape_index(pts, lat2, lon2)
        if i1 > i2:
            i1, i2 = i2, i1

        return [pts[i1:i2 + 1].tolist()]

    def _color_for_leg(self, leg, leg_index):
        if leg.get("mode") == "WALK":
            return "gray"
        key = leg.get("trip_id") or f"leg_{leg_index}"
        return self._palette[abs(hash(str(key))) % len(self._palette)]

    def _stop_label(self, stop_id):
        if self.stop_name_func is not None:
            return self.stop_name_func(stop_id)
        return self.stop_names.get(stop_id, stop_id)

    # ---------------------------------------------------
    # Public function
    # ---------------------------------------------------

    def plot_path(self, path, center=(30.0444, 31.2357), zoom=12):
        m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

        for idx, leg in enumerate(path):
            leg_color = self._color_for_leg(leg, idx)
            segments = self._expand_leg_with_shape(leg)

            for seg in segments:
                color = "gray" if leg["mode"] == "WALK" else leg_color
                dash = "5,5" if leg["mode"] == "WALK" else None
                tooltip = (
                    "WALK"
                    if leg["mode"] == "WALK"
                    else f"{leg.get('route_short', '')} - {leg.get('route_long', '')}"
                )

                folium.PolyLine(
                    seg,
                    color=color,
                    weight=6,
                    smooth_factor=2,
                    dash_array=dash,
                    tooltip=tooltip,
                ).add_to(m)

            if leg.get("mode") != "WALK":
                for stop_id in leg.get("stops", []):
                    coord = self.stop_coords.get(stop_id)
                    if coord is None:
                        continue
                    folium.CircleMarker(
                        location=coord,
                        radius=5,
                        color=leg_color,
                        fill=True,
                        fill_opacity=0.95,
                        tooltip=self._stop_label(stop_id),
                    ).add_to(m)

        if path:
            first = path[0].get("from_stop")
            last = path[-1].get("to_stop")
            if first in self.stop_coords:
                folium.Marker(
                    location=self.stop_coords[first],
                    tooltip=f"Start: {self._stop_label(first)}",
                    icon=folium.Icon(color="green", icon="play"),
                ).add_to(m)
            if last in self.stop_coords:
                folium.Marker(
                    location=self.stop_coords[last],
                    tooltip=f"End: {self._stop_label(last)}",
                    icon=folium.Icon(color="red", icon="stop"),
                ).add_to(m)

        return m
