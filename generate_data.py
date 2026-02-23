"""
Generate synthetic ride request and depot data for the Uber Network Routing Demo.

Creates:
    data/ride_requests.csv  -- 60 ride requests across Manhattan and nearby areas
    data/depots.csv         -- 4 depot locations

Coordinates cover Manhattan plus Williamsburg/DUMBO (lat ~40.70-40.81, lng ~-74.01 to -73.94).
Time windows span a morning rush period: 6:00 AM - 10:00 AM (minutes 360-600 from midnight).
"""

import csv
import os
import random

SEED = 42
NUM_REQUESTS = 60

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
]

TIME_START = 360   # 6:00 AM
TIME_END   = 600   # 10:00 AM
WINDOW_MIN = 20    # minutes
WINDOW_MAX = 45    # minutes


def generate_ride_requests(rng):
    """Generate ride requests with locations, time windows, and passenger counts."""
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

        requests.append({
            "request_id": i,
            "pickup_lat": pickup_lat,
            "pickup_lng": pickup_lng,
            "dropoff_lat": dropoff_lat,
            "dropoff_lng": dropoff_lng,
            "earliest_pickup": earliest,
            "latest_pickup": latest,
            "passengers": passengers,
        })
    return requests


def generate_depots():
    """Generate depot locations spread across the service area."""
    return [
        {"depot_id": 0, "lat": 40.710, "lng": -74.008, "name": "Financial District"},
        {"depot_id": 1, "lat": 40.752, "lng": -73.978, "name": "Midtown"},
        {"depot_id": 2, "lat": 40.782, "lng": -73.971, "name": "Upper West Side"},
        {"depot_id": 3, "lat": 40.714, "lng": -73.962, "name": "Williamsburg"},
    ]


def main():
    rng = random.Random(SEED)
    os.makedirs("data", exist_ok=True)

    requests = generate_ride_requests(rng)
    with open("data/ride_requests.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "request_id", "pickup_lat", "pickup_lng",
            "dropoff_lat", "dropoff_lng",
            "earliest_pickup", "latest_pickup", "passengers",
        ])
        writer.writeheader()
        writer.writerows(requests)
    print(f"Wrote {len(requests)} ride requests to data/ride_requests.csv")

    depots = generate_depots()
    with open("data/depots.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["depot_id", "lat", "lng", "name"])
        writer.writeheader()
        writer.writerows(depots)
    print(f"Wrote {len(depots)} depots to data/depots.csv")


if __name__ == "__main__":
    main()
