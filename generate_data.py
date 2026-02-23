"""
Generate synthetic ride request, depot, vehicle type, and incompatibility data
for the Uber Network Routing Demo.

Creates:
    data/ride_requests.csv      -- 70 ride requests with priority levels
    data/depots.csv             -- 5 depot locations with max vehicle counts
    data/vehicle_types.csv      -- 3 vehicle types (Sedan, SUV, Van)
    data/incompatible_pairs.csv -- ~25 incompatible request pairs

Coordinates cover Manhattan plus Williamsburg/DUMBO/Roosevelt Island
(lat ~40.70-40.81, lng ~-74.01 to -73.94).
Time windows span a morning rush period: 6:00 AM - 11:00 AM (minutes 360-660).
"""

import csv
import math
import os
import random

SEED = 42
NUM_REQUESTS = 70

# Neighborhoods: (name, center_lat, center_lng, demand_weight)
NEIGHBORHOODS = [
    ("Financial District", 40.708, -74.011, 10),
    ("Tribeca",            40.716, -74.008,  5),
    ("SoHo",               40.724, -73.997,  8),
    ("East Village",       40.728, -73.985,  7),
    ("Chelsea",            40.745, -73.998,  8),
    ("Midtown East",       40.752, -73.972, 12),
    ("Midtown West",       40.757, -73.988, 10),
    ("Upper East Side",    40.774, -73.957,  8),
    ("Upper West Side",    40.783, -73.974,  7),
    ("Harlem",             40.811, -73.946,  5),
    ("Williamsburg",       40.713, -73.960,  6),
    ("DUMBO",              40.703, -73.987,  4),
    ("Roosevelt Island",   40.762, -73.950,  4),
]

TIME_START = 360   # 6:00 AM
TIME_END   = 660   # 11:00 AM
WINDOW_MIN = 30    # minutes (wider than before to reduce presolve effectiveness)
WINDOW_MAX = 60    # minutes

# Priority distribution: 15% VIP, 60% standard, 25% economy
PRIORITY_WEIGHTS = [15, 60, 25]
PRIORITY_VALUES = [1, 2, 3]  # 1=VIP, 2=standard, 3=economy


def haversine_miles(lat1, lng1, lat2, lng2):
    """Distance in miles between two lat/lng points."""
    R_earth = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return 2 * R_earth * math.asin(math.sqrt(a))


def generate_ride_requests(rng):
    """Generate ride requests with locations, time windows, passenger counts, and priority."""
    weights = [n[3] for n in NEIGHBORHOODS]
    requests = []

    for i in range(NUM_REQUESTS):
        pickup_hood = rng.choices(NEIGHBORHOODS, weights=weights)[0]
        dropoff_hood = rng.choices(NEIGHBORHOODS, weights=weights)[0]

        pickup_lat = round(pickup_hood[1] + rng.gauss(0, 0.004), 6)
        pickup_lng = round(pickup_hood[2] + rng.gauss(0, 0.004), 6)
        dropoff_lat = round(dropoff_hood[1] + rng.gauss(0, 0.004), 6)
        dropoff_lng = round(dropoff_hood[2] + rng.gauss(0, 0.004), 6)

        earliest = rng.randint(TIME_START, TIME_END - WINDOW_MAX)
        window = rng.randint(WINDOW_MIN, WINDOW_MAX)
        latest = earliest + window

        passengers = rng.choices([1, 2, 3, 4], weights=[45, 30, 18, 7])[0]
        priority = rng.choices(PRIORITY_VALUES, weights=PRIORITY_WEIGHTS)[0]

        requests.append({
            "request_id": i,
            "pickup_lat": pickup_lat,
            "pickup_lng": pickup_lng,
            "dropoff_lat": dropoff_lat,
            "dropoff_lng": dropoff_lng,
            "earliest_pickup": earliest,
            "latest_pickup": latest,
            "passengers": passengers,
            "priority": priority,
        })
    return requests


def generate_depots():
    """Generate depot locations spread across the service area with vehicle limits."""
    return [
        {"depot_id": 0, "lat": 40.710, "lng": -74.008, "name": "Financial District", "max_vehicles": 6},
        {"depot_id": 1, "lat": 40.752, "lng": -73.978, "name": "Midtown", "max_vehicles": 6},
        {"depot_id": 2, "lat": 40.782, "lng": -73.971, "name": "Upper West Side", "max_vehicles": 5},
        {"depot_id": 3, "lat": 40.714, "lng": -73.962, "name": "Williamsburg", "max_vehicles": 5},
        {"depot_id": 4, "lat": 40.762, "lng": -73.952, "name": "Roosevelt Island", "max_vehicles": 5},
    ]


def generate_vehicle_types():
    """Generate 3 vehicle types with different capacities and costs."""
    return [
        {
            "type_id": 0,
            "name": "Sedan",
            "capacity": 4,
            "max_route_minutes": 90,
            "fixed_cost": 40,
            "per_minute_cost": 1.0,
            "count": 6,
        },
        {
            "type_id": 1,
            "name": "SUV",
            "capacity": 6,
            "max_route_minutes": 120,
            "fixed_cost": 60,
            "per_minute_cost": 1.5,
            "count": 6,
        },
        {
            "type_id": 2,
            "name": "Van",
            "capacity": 10,
            "max_route_minutes": 150,
            "fixed_cost": 80,
            "per_minute_cost": 2.0,
            "count": 6,
        },
    ]


def generate_incompatible_pairs(requests, rng, target_pairs=25):
    """
    Generate incompatible request pairs based on two criteria:
    1. Time-overlap + geographic distance: requests whose time windows overlap
       but pickup locations are far apart (hard to serve on same route).
    2. VIP-economy conflicts: VIP (priority=1) and economy (priority=3) requests
       with overlapping time windows shouldn't share a vehicle.
    """
    pairs = set()

    # Criterion 1: Time-overlap + distance conflicts
    for i in range(len(requests)):
        for j in range(i + 1, len(requests)):
            ri, rj = requests[i], requests[j]

            # Check time overlap
            overlap = (ri["earliest_pickup"] <= rj["latest_pickup"]
                       and rj["earliest_pickup"] <= ri["latest_pickup"])
            if not overlap:
                continue

            # Check distance between pickups
            dist = haversine_miles(
                ri["pickup_lat"], ri["pickup_lng"],
                rj["pickup_lat"], rj["pickup_lng"],
            )
            if dist > 3.0:  # More than 3 miles apart
                pairs.add((ri["request_id"], rj["request_id"]))

    # Criterion 2: VIP-economy conflicts with time overlap
    vip_ids = [r["request_id"] for r in requests if r["priority"] == 1]
    econ_ids = [r["request_id"] for r in requests if r["priority"] == 3]
    req_by_id = {r["request_id"]: r for r in requests}

    for v in vip_ids:
        for e in econ_ids:
            rv, re = req_by_id[v], req_by_id[e]
            overlap = (rv["earliest_pickup"] <= re["latest_pickup"]
                       and re["earliest_pickup"] <= rv["latest_pickup"])
            if overlap:
                a, b = min(v, e), max(v, e)
                pairs.add((a, b))

    # Trim to target count if needed
    pairs = sorted(pairs)
    if len(pairs) > target_pairs:
        pairs = rng.sample(pairs, target_pairs)
        pairs.sort()

    return pairs


def main():
    rng = random.Random(SEED)
    os.makedirs("data", exist_ok=True)

    # Ride requests
    requests = generate_ride_requests(rng)
    with open("data/ride_requests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "request_id", "pickup_lat", "pickup_lng",
            "dropoff_lat", "dropoff_lng",
            "earliest_pickup", "latest_pickup", "passengers", "priority",
        ])
        writer.writeheader()
        writer.writerows(requests)
    print(f"Wrote {len(requests)} ride requests to data/ride_requests.csv")

    # Priority breakdown
    for p in PRIORITY_VALUES:
        count = sum(1 for r in requests if r["priority"] == p)
        label = {1: "VIP", 2: "standard", 3: "economy"}[p]
        print(f"  Priority {p} ({label}): {count}")

    # Depots
    depots = generate_depots()
    with open("data/depots.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "depot_id", "lat", "lng", "name", "max_vehicles",
        ])
        writer.writeheader()
        writer.writerows(depots)
    print(f"Wrote {len(depots)} depots to data/depots.csv")

    # Vehicle types
    vtypes = generate_vehicle_types()
    with open("data/vehicle_types.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "type_id", "name", "capacity", "max_route_minutes",
            "fixed_cost", "per_minute_cost", "count",
        ])
        writer.writeheader()
        writer.writerows(vtypes)
    total_vehicles = sum(v["count"] for v in vtypes)
    print(f"Wrote {len(vtypes)} vehicle types to data/vehicle_types.csv "
          f"({total_vehicles} vehicles total)")

    # Incompatible pairs
    pairs = generate_incompatible_pairs(requests, rng)
    with open("data/incompatible_pairs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["request_a", "request_b"])
        writer.writeheader()
        for a, b in pairs:
            writer.writerow({"request_a": a, "request_b": b})
    print(f"Wrote {len(pairs)} incompatible pairs to data/incompatible_pairs.csv")


if __name__ == "__main__":
    main()
