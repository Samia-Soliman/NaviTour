# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 20:50:55 2026

@author: Samia
"""
from raptor.utils import time_to_sec

class Label:
    __slots__ = ("stop", "time", "transfers", "prev", "mode")

    def __init__(self, stop, time, transfers, prev=None, mode=None):
        self.stop = stop          # stop index of this label
        self.time = time          # arrival time at this stop
        self.transfers = transfers
        self.prev = prev          # previous Label object
        self.mode = mode          # trip_id or "WALK"
        
def dominates(a, b):
    return (a.time <= b.time and a.transfers <= b.transfers and
            (a.time < b.time or a.transfers < b.transfers))

def pareto_insert(bag, new_label):
    for l in bag:
        if dominates(l, new_label):
            return False
    bag[:] = [l for l in bag if not dominates(new_label, l)]
    bag.append(new_label)
    return True

def mc_raptor(network, source_id, target_id, departure_time, max_rounds=5):
    stop_id_to_idx  = network.stop_id_to_idx
    stop_to_trips   = network.stop_to_trips
    trip_stop_times = network.trip_stop_times
    trip_stop_index = network.trip_stop_index
    walk_transfers  = network.walk_transfers

    source = stop_id_to_idx[source_id]
    target = stop_id_to_idx[target_id]
    dep    = time_to_sec(departure_time)

    B = [{} for _ in range(max_rounds + 1)]
    B[0][source] = [Label(stop=source, time=dep, transfers=0)]
    marked = {source}

    best_target_time = float('inf')

    for r in range(1, max_rounds + 1):
        for p, labels in B[r-1].items():
            B[r][p] = labels.copy()
        new_marked = set()

        # Update best known target time
        for lbl in B[r].get(target, []):
            if lbl.time < best_target_time:
                best_target_time = lbl.time

        # -------- Transit phase --------
        Q = {}
        for p in marked:
            for trip in stop_to_trips.get(p, []):
                idx = trip_stop_index[trip].get(p)
                if idx is None:
                    continue
                seq = trip_stop_times[trip]
                board_time = seq[idx][1]

                # ← SAFE PRUNING: only prune after first solution found
                if best_target_time < float('inf') and board_time > best_target_time:
                    continue

                feasible = [lbl for lbl in B[r-1].get(p, [])
                            if board_time >= lbl.time]
                if not feasible:
                    continue

                if trip not in Q or idx < Q[trip][0]:
                    Q[trip] = (idx, feasible, seq)

        for trip, (idx, feasible_labels, seq) in Q.items():
            for s, arr_t, _ in seq[idx + 1:]:
                # ← SAFE PRUNING: only prune after first solution found
                if best_target_time < float('inf') and arr_t > best_target_time:
                    break

                if s not in B[r]:
                    B[r][s] = []
                bag = B[r][s]

                for lbl in feasible_labels:
                    new_label = Label(
                        stop=s,
                        time=arr_t,
                        transfers=lbl.transfers + 1,
                        prev=lbl,
                        mode=trip
                    )
                    if pareto_insert(bag, new_label):
                        new_marked.add(s)
                        if s == target and arr_t < best_target_time:
                            best_target_time = arr_t

        # -------- Walking phase --------
        for p in list(new_marked):
            bag_p = B[r].get(p)
            if not bag_p:
                continue
            for to_p, wt in walk_transfers.get(p, []):
                best = min(bag_p, key=lambda l: l.time)
                new_arr = best.time + wt

                # ← SAFE PRUNING: only prune after first solution found
                if best_target_time < float('inf') and new_arr > best_target_time:
                    continue

                if to_p not in B[r]:
                    B[r][to_p] = []
                bag = B[r][to_p]

                new_label = Label(
                    stop=to_p,
                    time=new_arr,
                    transfers=best.transfers,
                    prev=best,
                    mode="WALK"
                )
                if pareto_insert(bag, new_label):
                    new_marked.add(to_p)
                    if to_p == target and new_arr < best_target_time:
                        best_target_time = new_arr

        marked = new_marked
        if not marked:
            break

    return B, target

