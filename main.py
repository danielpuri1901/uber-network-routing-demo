"""
Uber Network Routing Demo -- Multi-Depot Capacitated Vehicle Routing
with Time Windows (MDCVRPTW)

Solves a single monolithic MILP for routing a heterogeneous fleet of vehicles
from multiple depots to serve ride requests across Manhattan. Inspired by
Uber's Gurobi case study on urban aerial ridesharing network design.

Reference: https://www.gurobi.com/case_studies/uber-shaping-urban-aerial-ridesharing/

===============================================================================
MATHEMATICAL FORMULATION
===============================================================================

Sets:
    R               Ride requests, |R| = 40
    D               Depots, |D| = 5
    K               Vehicles, |K| = 12 (4 Sedan + 4 SUV + 4 Van)
    T               Vehicle types, |T| = 3 (Sedan, SUV, Van)
    N = R ∪ Ds ∪ De Extended node set
        Ds          Depot start nodes (one per depot)
        De          Depot end nodes (one per depot)
    A               All arcs (i, j) for i, j ∈ N, i ≠ j
    I               Incompatible request pairs

Parameters:
    c[i,j]      Travel time (minutes) between nodes i and j
    q[i]        Passenger count for request i (0 for depots)
    e[i]        Earliest service time for node i
    l[i]        Latest service time for node i
    s[i]        Service duration at node i (pickup + ride + dropoff)
    p[i]        Priority of request i (1=VIP, 2=standard, 3=economy)
    Q[k]        Capacity of vehicle k (type-dependent: 4/6/10)
    R_max[k]    Maximum route duration for vehicle k (type-dependent: 90/120/150)
    F[k]        Fixed activation cost for vehicle k (type-dependent: $40/$60/$80)
    V[k]        Per-minute travel cost for vehicle k (type-dependent: $1.0/$1.5/$2.0)
    W[d]        Maximum vehicles departing from depot d
    M = 1000000 Big-M constant for time window linearization
    P[p]        Priority penalty: {1: 2000, 2: 1000, 3: 500}

Decision Variables:
    x[i,j,k] ∈ {0,1}   1 if vehicle k travels arc (i,j)
    y[i,k]   ∈ {0,1}   1 if vehicle k serves request i
    z[k]     ∈ {0,1}   1 if vehicle k is activated
    t[i,k]   ≥ 0       Arrival time of vehicle k at node i

Objective:
    Minimize  Σ V[k]·c[i,j]·x[i,j,k]  +  Σ F[k]·z[k]
            + Σ P[p[i]]·(1 - Σk y[i,k])

Constraints:
    (1) Each request served at most once:
        Σk y[i,k] ≤ 1                                      ∀ i ∈ R
    (2) Assignment-arc consistency:
        y[i,k] = Σj x[i,j,k]                               ∀ i ∈ R, k ∈ K
    (3) Flow conservation:
        Σj x[j,i,k] = Σj x[i,j,k]                         ∀ i ∈ R, k ∈ K
    (4) Depart from at most one depot:
        Σ(ds,j) x[ds,j,k] ≤ 1                              ∀ k ∈ K
    (5) Return to at most one depot:
        Σ(i,de) x[i,de,k] ≤ 1                              ∀ k ∈ K
    (6) No incoming arcs to depot starts:
        Σj x[j,ds,k] = 0                                   ∀ ds, k
    (7) No outgoing arcs from depot ends:
        Σj x[de,j,k] = 0                                   ∀ de, k
    (8) Activation linking:
        y[i,k] ≤ z[k]                                      ∀ i ∈ R, k ∈ K
    (9) Depot start activation:
        Σj x[ds,j,k] ≤ z[k]                                ∀ ds, k
    (10) Capacity (type-dependent):
        Σi q[i]·y[i,k] ≤ Q[k]·z[k]                        ∀ k ∈ K
    (11) Time windows (Big-M):
        t[i,k] + s[i] + c[i,j] - M·(1 - x[i,j,k]) ≤ t[j,k]
                                                            ∀ (i,j) ∈ A, k ∈ K
    (12) Time bounds:
        e[i] ≤ t[i,k] ≤ l[i]                              ∀ i ∈ N, k ∈ K
    (13) Maximum route duration (Big-M):
        t[de,k] - t[ds,k] ≤ R_max[k] + M·(1 - Σj x[ds,j,k])
                                                            ∀ depot pairs (ds,de), k
    (14) Depot vehicle capacity:
        Σk (Σj x[ds,j,k]) ≤ W[d]                          ∀ depot d
    (15) VIP must-serve:
        Σk y[i,k] = 1                                      ∀ i where p[i] = 1
    (16) Incompatible pairs:
        y[a,k] + y[b,k] ≤ 1                               ∀ (a,b) ∈ I, k ∈ K
    (17) Minimum utilization:
        Σi y[i,k] ≥ 2·z[k]                                ∀ k ∈ K

===============================================================================
DELIBERATE INEFFICIENCIES (for optimization agent demo)
===============================================================================

| # | Inefficiency                            | Agent Fix                              | Expected Speedup |
|---|----------------------------------------|----------------------------------------|-----------------|
| 1 | Global Big-M = 1,000,000 (actual ~20-250) | Per-constraint tight M values       | ~30-40%         |
| 2 | No arc pre-filtering (all N*(N-1) arcs)  | Filter time-infeasible arcs          | ~15-20%         |
| 3 | No symmetry breaking (4 identical vehicles × 3 types = (4!)^3 equiv.) | z[k] >= z[k+1] within type | ~40-60%  |
| 4 | No branching priorities (all vars default) | Prioritize z[k] and VIP y[i,k]    | ~15-25%         |
| 5 | Default Symmetry=-1                     | Set Symmetry=2 (aggressive)           | ~10-20%         |
| 6 | Default MIPFocus=0                      | Set MIPFocus=1 (feasibility)          | ~10-15%         |
| 7 | Default Cuts=-1                         | Set Cuts=2 (aggressive)               | ~10-20%         |

===============================================================================
"""

import csv
import math
import time as time_module

import gurobipy as gp
from gurobipy import GRB


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ride_requests(path="data/ride_requests.csv"):
    """Load ride requests from CSV, including priority column."""
    requests = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            requests.append({
                "id": int(row["request_id"]),
                "pickup_lat": float(row["pickup_lat"]),
                "pickup_lng": float(row["pickup_lng"]),
                "dropoff_lat": float(row["dropoff_lat"]),
                "dropoff_lng": float(row["dropoff_lng"]),
                "earliest": int(row["earliest_pickup"]),
                "latest": int(row["latest_pickup"]),
                "passengers": int(row["passengers"]),
                "priority": int(row["priority"]),
            })
    return requests


def load_depots(path="data/depots.csv"):
    """Load depot locations from CSV, including max_vehicles column."""
    depots = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            depots.append({
                "id": int(row["depot_id"]),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "name": row["name"],
                "max_vehicles": int(row["max_vehicles"]),
            })
    return depots


def load_vehicle_types(path="data/vehicle_types.csv"):
    """Load vehicle type definitions from CSV."""
    vtypes = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vtypes.append({
                "type_id": int(row["type_id"]),
                "name": row["name"],
                "capacity": int(row["capacity"]),
                "max_route_minutes": int(row["max_route_minutes"]),
                "fixed_cost": float(row["fixed_cost"]),
                "per_minute_cost": float(row["per_minute_cost"]),
                "count": int(row["count"]),
            })
    return vtypes


def load_incompatible_pairs(path="data/incompatible_pairs.csv"):
    """Load incompatible request pairs from CSV."""
    pairs = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((int(row["request_a"]), int(row["request_b"])))
    return pairs


# ---------------------------------------------------------------------------
# Distance / travel time
# ---------------------------------------------------------------------------

def haversine_minutes(lat1, lng1, lat2, lng2, speed_mph=15.0):
    """
    Travel time in minutes between two lat/lng points.
    Uses haversine distance with an average urban speed of 15 mph.
    """
    R_earth = 3958.8  # miles
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    dist_miles = 2 * R_earth * math.asin(math.sqrt(a))
    return (dist_miles / speed_mph) * 60.0


def build_travel_times(requests, depots):
    """
    Build a complete travel-time matrix for the extended node set.

    Node indexing:
        0 .. N_R-1                : request nodes (pickup location)
        N_R .. N_R+N_D-1          : depot start nodes
        N_R+N_D .. N_R+2*N_D-1    : depot end nodes

    Travel between request nodes accounts for the ride: the departure
    point of request i is its dropoff location, so travel(i -> j) is
    computed as dropoff(i) -> pickup(j).
    """
    N_R = len(requests)
    N_D = len(depots)
    N = N_R + 2 * N_D

    arrive_coords = []
    depart_coords = []
    service_times = []

    for r in requests:
        arrive_coords.append((r["pickup_lat"], r["pickup_lng"]))
        depart_coords.append((r["dropoff_lat"], r["dropoff_lng"]))
        ride_time = haversine_minutes(
            r["pickup_lat"], r["pickup_lng"],
            r["dropoff_lat"], r["dropoff_lng"],
        )
        service_times.append(2.0 + ride_time + 1.0)

    for d in depots:
        arrive_coords.append((d["lat"], d["lng"]))
        depart_coords.append((d["lat"], d["lng"]))
        service_times.append(0.0)
    for d in depots:
        arrive_coords.append((d["lat"], d["lng"]))
        depart_coords.append((d["lat"], d["lng"]))
        service_times.append(0.0)

    travel = {}
    for i in range(N):
        for j in range(N):
            if i == j:
                travel[i, j] = 0.0
            else:
                travel[i, j] = haversine_minutes(
                    depart_coords[i][0], depart_coords[i][1],
                    arrive_coords[j][0], arrive_coords[j][1],
                )

    return travel, service_times, N_R, N_D, N


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def build_model(requests, depots, vehicle_types, incompatible_pairs):
    """
    Build the monolithic Multi-Depot CVRPTW model with heterogeneous fleet.

    Creates arc variables x[i,j,k] for every pair of nodes (i,j) with i != j
    and every vehicle k. A single global Big-M constant (M = 1,000,000) is
    used for all time-window and route-duration constraints.

    The arc set `arcs` contains all (i,j) pairs used for variable and
    constraint creation. Adjacency lists `out_nbrs` and `in_nbrs` are
    derived from this arc set and used throughout the constraint code.
    """
    travel, service, N_R, N_D, N = build_travel_times(requests, depots)

    # --- Vehicle type mappings ---
    # Build per-vehicle properties from vehicle types
    vehicles = []
    vehicle_type = {}   # k -> type_id
    vehicle_cap = {}    # k -> capacity
    vehicle_max_route = {}  # k -> max route minutes
    vehicle_fixed_cost = {}  # k -> fixed cost
    vehicle_per_min_cost = {}  # k -> per-minute cost
    type_groups = {}    # type_id -> [vehicle indices]

    k_idx = 0
    for vt in vehicle_types:
        tid = vt["type_id"]
        type_groups[tid] = []
        for _ in range(vt["count"]):
            vehicles.append(k_idx)
            vehicle_type[k_idx] = tid
            vehicle_cap[k_idx] = vt["capacity"]
            vehicle_max_route[k_idx] = vt["max_route_minutes"]
            vehicle_fixed_cost[k_idx] = vt["fixed_cost"]
            vehicle_per_min_cost[k_idx] = vt["per_minute_cost"]
            type_groups[tid].append(k_idx)
            k_idx += 1

    N_K = len(vehicles)
    M = 1_000_000      # Big-M for time window and route duration constraints

    # Priority penalties for unserved requests
    priority_penalty = {1: 2000, 2: 1000, 3: 500}
    req_priority = {r["id"]: r["priority"] for r in requests}

    depot_start = list(range(N_R, N_R + N_D))
    depot_end = list(range(N_R + N_D, N_R + 2 * N_D))
    request_nodes = list(range(N_R))
    all_nodes = list(range(N))

    # Depot max vehicles mapping (depot index -> max_vehicles)
    depot_max_vehicles = {d["id"]: d["max_vehicles"] for d in depots}

    # Time windows
    earliest = {}
    latest = {}
    for r in requests:
        earliest[r["id"]] = r["earliest"]
        latest[r["id"]] = r["latest"]
    for d_s in depot_start:
        earliest[d_s] = 0
        latest[d_s] = M
    for d_e in depot_end:
        earliest[d_e] = 0
        latest[d_e] = M

    passengers = {r["id"]: r["passengers"] for r in requests}

    # VIP requests (priority=1) that must be served
    vip_requests = [r["id"] for r in requests if r["priority"] == 1]

    # -----------------------------------------------------------------------
    # Build arc set: all (i, j) pairs with i != j
    # -----------------------------------------------------------------------
    arcs = [(i, j) for i in all_nodes for j in all_nodes if i != j]

    # Adjacency lists derived from arc set
    out_nbrs = {i: [] for i in all_nodes}
    in_nbrs = {j: [] for j in all_nodes}
    for i, j in arcs:
        out_nbrs[i].append(j)
        in_nbrs[j].append(i)

    print(f"  Arc pairs: {len(arcs):,} (of {N * (N - 1):,} possible)")
    print(f"  Vehicles:  {N_K} ({', '.join(f'{vt['count']} {vt['name']}' for vt in vehicle_types)})")

    # -----------------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------------
    model = gp.Model("uber_mdcvrptw")
    model.setParam("LogFile", "gurobi.log")
    model.setParam("MIPGap", 0.01)  # 1.0% optimality gap

    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # x[i,j,k]: binary -- vehicle k traverses arc (i,j)
    # Objective coefficient includes per-minute cost of vehicle type
    print("Creating arc variables x[i,j,k] ...")
    x = {}
    for k in vehicles:
        cost_k = vehicle_per_min_cost[k]
        for i, j in arcs:
            x[i, j, k] = model.addVar(
                vtype=GRB.BINARY, obj=cost_k * travel[i, j],
                name=f"x_{i}_{j}_{k}",
            )

    # y[i,k]: binary -- request i is served by vehicle k
    print("Creating assignment variables y[i,k] ...")
    y = {}
    for i in request_nodes:
        for k in vehicles:
            y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")

    # z[k]: binary -- vehicle k is activated
    z = {}
    for k in vehicles:
        z[k] = model.addVar(vtype=GRB.BINARY, name=f"z_{k}")

    # t[i,k]: continuous -- arrival time of vehicle k at node i
    print("Creating time variables t[i,k] ...")
    t = {}
    for i in all_nodes:
        for k in vehicles:
            t[i, k] = model.addVar(
                lb=earliest.get(i, 0),
                ub=latest.get(i, M),
                vtype=GRB.CONTINUOUS,
                name=f"t_{i}_{k}",
            )

    model.update()

    # -----------------------------------------------------------------------
    # Objective: type-dependent costs + priority-dependent unserved penalties
    # -----------------------------------------------------------------------
    # Travel cost is already in x[i,j,k] obj coefficients.
    # Add: fixed activation costs + priority-weighted unserved penalties
    fixed_cost_expr = gp.quicksum(
        vehicle_fixed_cost[k] * z[k] for k in vehicles
    )
    unserved_penalty_expr = gp.quicksum(
        priority_penalty[req_priority[i]] * (1 - gp.quicksum(y[i, k] for k in vehicles))
        for i in request_nodes
    )
    model.setObjective(
        model.getObjective() + fixed_cost_expr + unserved_penalty_expr,
        GRB.MINIMIZE,
    )

    # -----------------------------------------------------------------------
    # Constraints
    # -----------------------------------------------------------------------
    print("Adding constraints ...")

    # (1) Each request served at most once
    for i in request_nodes:
        model.addConstr(
            gp.quicksum(y[i, k] for k in vehicles) <= 1,
            f"serve_once_{i}",
        )

    # (2) Assignment-arc consistency: y[i,k] = sum of outgoing arcs from i
    for i in request_nodes:
        for k in vehicles:
            model.addConstr(
                y[i, k] == gp.quicksum(x[i, j, k] for j in out_nbrs[i]),
                f"assign_arc_{i}_{k}",
            )

    # (3) Flow conservation at request nodes
    for i in request_nodes:
        for k in vehicles:
            model.addConstr(
                gp.quicksum(x[j, i, k] for j in in_nbrs[i])
                == gp.quicksum(x[i, j, k] for j in out_nbrs[i]),
                f"flow_{i}_{k}",
            )

    # (4) Vehicle departs from at most one depot start node
    for k in vehicles:
        model.addConstr(
            gp.quicksum(
                x[d_s, j, k] for d_s in depot_start for j in out_nbrs[d_s]
            ) <= 1,
            f"depart_depot_{k}",
        )

    # (5) Vehicle returns to at most one depot end node
    for k in vehicles:
        model.addConstr(
            gp.quicksum(
                x[i, d_e, k] for d_e in depot_end for i in in_nbrs[d_e]
            ) <= 1,
            f"return_depot_{k}",
        )

    # (6) No incoming arcs to depot start nodes
    for k in vehicles:
        for d_s in depot_start:
            if in_nbrs[d_s]:
                model.addConstr(
                    gp.quicksum(x[j, d_s, k] for j in in_nbrs[d_s]) == 0,
                    f"no_in_start_{d_s}_{k}",
                )

    # (7) No outgoing arcs from depot end nodes
    for k in vehicles:
        for d_e in depot_end:
            if out_nbrs[d_e]:
                model.addConstr(
                    gp.quicksum(x[d_e, j, k] for j in out_nbrs[d_e]) == 0,
                    f"no_out_end_{d_e}_{k}",
                )

    # (8) Activation linking
    for i in request_nodes:
        for k in vehicles:
            model.addConstr(y[i, k] <= z[k], f"activate_{i}_{k}")

    # (9) Depot start activation
    for k in vehicles:
        for d_s in depot_start:
            if out_nbrs[d_s]:
                model.addConstr(
                    gp.quicksum(x[d_s, j, k] for j in out_nbrs[d_s]) <= z[k],
                    f"depot_act_{d_s}_{k}",
                )

    # (10) Capacity (type-dependent)
    for k in vehicles:
        model.addConstr(
            gp.quicksum(passengers[i] * y[i, k] for i in request_nodes)
            <= vehicle_cap[k] * z[k],
            f"capacity_{k}",
        )

    # (11) Time window feasibility (Big-M linearization)
    # Uses the arc set and a single global M for all constraints.
    print("Adding time-window Big-M constraints ...")
    for k in vehicles:
        for i, j in arcs:
            model.addConstr(
                t[i, k] + service[i] + travel[i, j]
                - M * (1 - x[i, j, k])
                <= t[j, k],
                f"tw_{i}_{j}_{k}",
            )
    print(f"  {len(arcs) * N_K:,} time-window constraints added")

    # (13) Maximum route duration (Big-M linearization)
    # For each depot pair (start, end) and each vehicle: route duration ≤ max
    print("Adding route duration constraints ...")
    n_route_dur = 0
    for d_idx in range(N_D):
        d_s = depot_start[d_idx]
        d_e = depot_end[d_idx]
        for k in vehicles:
            model.addConstr(
                t[d_e, k] - t[d_s, k]
                <= vehicle_max_route[k] + M * (1 - gp.quicksum(x[d_s, j, k] for j in out_nbrs[d_s])),
                f"route_dur_{d_idx}_{k}",
            )
            n_route_dur += 1
    print(f"  {n_route_dur:,} route duration constraints added")

    # (14) Depot vehicle capacity
    print("Adding depot vehicle capacity constraints ...")
    for d_idx in range(N_D):
        d_s = depot_start[d_idx]
        model.addConstr(
            gp.quicksum(
                gp.quicksum(x[d_s, j, k] for j in out_nbrs[d_s])
                for k in vehicles
            ) <= depot_max_vehicles[d_idx],
            f"depot_cap_{d_idx}",
        )

    # (15) VIP must-serve: priority=1 requests must be served
    print(f"Adding VIP must-serve constraints ({len(vip_requests)} VIP requests) ...")
    for i in vip_requests:
        model.addConstr(
            gp.quicksum(y[i, k] for k in vehicles) == 1,
            f"vip_serve_{i}",
        )

    # (16) Incompatible request pairs: cannot be on same vehicle
    print(f"Adding incompatible pair constraints ({len(incompatible_pairs)} pairs) ...")
    n_incompat = 0
    for a, b in incompatible_pairs:
        for k in vehicles:
            model.addConstr(
                y[a, k] + y[b, k] <= 1,
                f"incompat_{a}_{b}_{k}",
            )
            n_incompat += 1
    print(f"  {n_incompat:,} incompatibility constraints added")

    # (17) Minimum utilization: active vehicles serve >= 2 requests
    print("Adding minimum utilization constraints ...")
    for k in vehicles:
        model.addConstr(
            gp.quicksum(y[i, k] for i in request_nodes) >= 2 * z[k],
            f"min_util_{k}",
        )

    model.update()

    # Print model statistics
    print(f"\nModel statistics:")
    print(f"  Variables:    {model.NumVars:,}")
    print(f"    Binary:     {model.NumBinVars:,}")
    print(f"    Continuous: {model.NumVars - model.NumBinVars:,}")
    print(f"  Constraints:  {model.NumConstrs:,}")
    print(f"  Non-zeros:    {model.NumNZs:,}")

    return (model, x, y, z, t, requests, depots, vehicles, request_nodes,
            N_R, N_D, vehicle_types, vehicle_type, vehicle_cap,
            vehicle_max_route, vehicle_fixed_cost, vehicle_per_min_cost,
            type_groups)


# ---------------------------------------------------------------------------
# Solution reporting
# ---------------------------------------------------------------------------

def report_solution(model, x, y, z, t, requests, depots, vehicles,
                    request_nodes, N_R, N_D, vehicle_types, vehicle_type_map,
                    vehicle_cap, vehicle_max_route, vehicle_fixed_cost,
                    vehicle_per_min_cost, type_groups):
    """Print a detailed summary of the optimized solution."""
    if model.SolCount == 0:
        print("\nNo feasible solution found.")
        return

    all_nodes = list(range(N_R + 2 * N_D))
    depot_start = list(range(N_R, N_R + N_D))

    print(f"\n{'='*60}")
    print("SOLUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Objective:       {model.ObjVal:.2f}")
    print(f"MIP gap:         {model.MIPGap:.4%}")

    active = [k for k in vehicles if z[k].X > 0.5]
    served = sum(
        1 for i in request_nodes if any(y[i, k].X > 0.5 for k in vehicles)
    )
    print(f"Active vehicles: {len(active)} / {len(vehicles)}")
    print(f"Requests served: {served} / {len(request_nodes)}")

    # Vehicle type breakdown
    type_names = {vt["type_id"]: vt["name"] for vt in vehicle_types}
    print(f"\nVehicle type breakdown:")
    total_fixed = 0.0
    total_travel = 0.0
    for vt in vehicle_types:
        tid = vt["type_id"]
        active_of_type = [k for k in type_groups[tid] if z[k].X > 0.5]
        type_fixed = sum(vehicle_fixed_cost[k] for k in active_of_type)
        total_fixed += type_fixed
        print(f"  {vt['name']:6s}: {len(active_of_type)} active / {vt['count']} available "
              f"(fixed cost: ${type_fixed:.0f})")

    # Route details
    print(f"\nRoute details:")
    for k in active:
        current = None
        for d_s in depot_start:
            for j in all_nodes:
                if j != d_s and (d_s, j, k) in x and x[d_s, j, k].X > 0.5:
                    current = d_s
                    break
            if current is not None:
                break
        if current is None:
            continue

        route = [current]
        visited = {current}
        while current is not None:
            next_node = None
            for j in all_nodes:
                if (j != current and j not in visited
                        and (current, j, k) in x and x[current, j, k].X > 0.5):
                    next_node = j
                    break
            if next_node is not None:
                route.append(next_node)
                visited.add(next_node)
                current = next_node
            else:
                current = None

        reqs = [n for n in route if n < N_R]
        d_idx = route[0] - N_R
        d_name = depots[d_idx]["name"] if 0 <= d_idx < len(depots) else "?"
        tname = type_names.get(vehicle_type_map[k], "?")

        # Compute route duration
        start_time = t[route[0], k].X if (route[0], k) in t else 0
        end_time = t[route[-1], k].X if (route[-1], k) in t else 0
        duration = end_time - start_time

        pax = sum(requests[n]["passengers"] for n in reqs) if reqs else 0
        print(f"  Vehicle {k:2d} [{tname:5s}] ({d_name}): "
              f"{len(reqs)} reqs, {pax} pax, {duration:.0f} min "
              f"-> {' -> '.join(str(n) for n in route)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Uber Network Routing Demo -- MDCVRPTW")
    print("=" * 60)

    print("\nLoading data ...")
    requests = load_ride_requests()
    depots = load_depots()
    vehicle_types = load_vehicle_types()
    incompatible_pairs = load_incompatible_pairs()
    total_vehicles = sum(vt["count"] for vt in vehicle_types)
    print(f"  {len(requests)} ride requests, {len(depots)} depots, "
          f"{total_vehicles} vehicles ({len(vehicle_types)} types), "
          f"{len(incompatible_pairs)} incompatible pairs")

    print("\nBuilding model ...")
    t0 = time_module.time()
    result = build_model(requests, depots, vehicle_types, incompatible_pairs)
    (model, x, y, z, t_var, requests, depots, vehicles, request_nodes,
     N_R, N_D, vehicle_types, vehicle_type_map, vehicle_cap,
     vehicle_max_route, vehicle_fixed_cost, vehicle_per_min_cost,
     type_groups) = result
    build_time = time_module.time() - t0
    print(f"Model built in {build_time:.1f}s")

    print("\nSolving ...")
    t0 = time_module.time()
    model.optimize()
    solve_time = time_module.time() - t0
    print(f"\nSolve time: {solve_time:.1f}s")

    report_solution(model, x, y, z, t_var, requests, depots, vehicles,
                    request_nodes, N_R, N_D, vehicle_types, vehicle_type_map,
                    vehicle_cap, vehicle_max_route, vehicle_fixed_cost,
                    vehicle_per_min_cost, type_groups)

    print(f"\nTotal time: {build_time + solve_time:.1f}s")
    print(f"Log written to gurobi.log")


if __name__ == "__main__":
    main()
