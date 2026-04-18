import datetime
from raptor.config import MAX_ROUNDS

# ------------------ Time utils ------------------
def time_to_sec(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def sec_to_time(sec):
    return str(datetime.timedelta(seconds=int(sec)))

def extract_solutions(B, target):
    solutions = []
    for r in range(MAX_ROUNDS + 1):
        solutions.extend(B[r][target])
    # Keep results deterministic: earliest arrival first, then fewer transfers.
    solutions.sort(key=lambda l: (l.time, l.transfers))
    return solutions

# =============================================================================
# 
# def extract_solutions(B, target):
#     final_bag = B[len(B)-1].get(target, [])
#     seen = set()
#     solutions = []
#     for lbl in final_bag:
#         key = (lbl.time, lbl.transfers)
#         if key not in seen:
#             seen.add(key)
#             solutions.append(lbl)
#     solutions.sort(key=lambda l: l.time)
#     return solutions
# =============================================================================

def reconstruct(label, network):
    """
    Reconstruct full path from mcRAPTOR Label with all intermediate stops.
    Returns a list of dicts:
    {
        'from_stop': stop_id,
        'to_stop': stop_id,
        'mode': 'WALK' or 'TRANSIT',
        'agency': agency_id or None,
        'route_short': route_short_name or None,
        'route_long': route_long_name or None,
        'trip_id': trip_id or None,
        'shape_id': shape_id or None
    }
    """
    path = []
    cur = label

    while cur.prev is not None:
        prev_label = cur.prev
        from_stop_idx = prev_label.stop
        to_stop_idx   = cur.stop

        if cur.mode == "WALK":
            path.append({
                "from_stop": network.idx_to_stop_id[from_stop_idx],
                "to_stop": network.idx_to_stop_id[to_stop_idx],
                "mode": "WALK",
                "agency": None,
                "route_short": None,
                "route_long": None,
                "trip_id": None,
                "shape_id": None
            })
        else:
            # This is a transit trip, expand all stops in between
            trip_id = cur.mode
            seq = network.trip_stop_times[trip_id]
            # Find indices of from_stop and to_stop in seq
            idx_from = next(i for i, (s, _, _) in enumerate(seq) if s == from_stop_idx)
            idx_to   = next(i for i, (s, _, _) in enumerate(seq) if s == to_stop_idx)

            # We are traversing labels backward (destination -> source), so we must
            # append this leg's segments in reverse order here; final path[::-1]
            # will then restore global source -> destination continuity.
            for i in range(idx_to - 1, idx_from - 1, -1):
                s_from = seq[i][0]
                s_to   = seq[i+1][0]
                route_id = network.trip_to_route[trip_id]
                info = network.route_info.get(route_id, {})
                shape_id = network.trip_to_shape.get(trip_id)  # optional, if you have it
                path.append({
                    "from_stop": network.idx_to_stop_id[s_from],
                    "to_stop": network.idx_to_stop_id[s_to],
                    "mode": "TRANSIT",
                    "agency": info.get("agency_id"),
                    "route_short": info.get("route_short_name"),
                    "route_long": info.get("route_long_name"),
                    "trip_id": trip_id,
                    "shape_id": shape_id
                })

        cur = prev_label

    return path[::-1]

## ---------- output utils ------- ##
def collapse_to_legs(segments):
    legs = []
    current = None

    for seg in segments:
        key = (seg["mode"], seg["trip_id"])

        if current is None:
            # start first leg
            current = {
                "mode": seg["mode"],
                "agency": seg["agency"],
                "route_short": seg["route_short"],
                "route_long": seg["route_long"],
                "trip_id": seg["trip_id"],
                "shape_id": seg["shape_id"],
                "from_stop": seg["from_stop"],
                "to_stop": seg["to_stop"],
                "stops": [seg["from_stop"], seg["to_stop"]],
            }
            continue

        # same trip or same walk → extend leg
        if key == (current["mode"], current["trip_id"]):
            current["to_stop"] = seg["to_stop"]
            current["stops"].append(seg["to_stop"])
        else:
            # close previous leg
            legs.append(current)

            # start new leg
            current = {
                "mode": seg["mode"],
                "agency": seg["agency"],
                "route_short": seg["route_short"],
                "route_long": seg["route_long"],
                "trip_id": seg["trip_id"],
                "shape_id": seg["shape_id"],
                "from_stop": seg["from_stop"],
                "to_stop": seg["to_stop"],
                "stops": [seg["from_stop"], seg["to_stop"]],
            }

    if current is not None:
        legs.append(current)

    return legs

def format_legs(legs):
    lines = []

    for leg in legs:
        if leg["mode"] == "WALK":
            lines.append(
                f"WALK: {leg['from_stop']} → {leg['to_stop']}"
            )
        else:
            route_name = f"{leg['route_short']} ({leg['route_long']})"
            lines.append(
                f"{leg['agency']} | {route_name}: "
                f"{leg['from_stop']} → {leg['to_stop']}"
            )

    return lines


def format_legs(legs, stop_name_func=None):
    lines = []

    for leg in legs:
        from_stop = stop_name_func(leg["from_stop"]) if stop_name_func else leg["from_stop"]
        to_stop = stop_name_func(leg["to_stop"]) if stop_name_func else leg["to_stop"]

        if leg["mode"] == "WALK":
            lines.append(f"WALK: {from_stop} → {to_stop}")
        else:
            route_name = f"{leg['route_short']} ({leg['route_long']})"
            lines.append(
                f"{leg['agency']} | {route_name}\n"
                f"  {from_stop} → {to_stop}"
            )

    return lines

