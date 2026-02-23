"""
Uber Network Routing Demo -- Multi-Depot Capacitated Vehicle Routing
with Time Windows (MDCVRPTW)

Solves a single monolithic MILP for routing a fleet of vehicles from multiple
depots to serve ride requests across Manhattan. Inspired by Uber's Gurobi case
study on urban aerial ridesharing network design.

Reference: https://www.gurobi.com/case_studies/uber-shaping-urban-aerial-ridesharing/

===============================================================================
MATHEMATICAL FORMULATION
===============================================================================

Sets:
    R               Ride requests, |R| = 60
    D               Depots, |D| = 4
    K               Vehicles, |K| = 16
    N = R ∪ Ds ∪ De Extended node set
        Ds          Depot start nodes (one per depot)
        De          Depot end nodes (one per depot)
    A               All arcs (i, j) for i, j ∈ N, i ≠ j

Parameters:
    c[i,j]      Travel time (minutes) between nodes i and j
    q[i]        Passenger count for request i (0 for depots)
    e[i]        Earliest service time for node i
    l[i]        Latest service time for node i
    s[i]        Service duration at node i (pickup + ride + dropoff)
    Q = 4       Vehicle capacity (seats)
    M = 1000000 Big-M constant for time window linearization

Decision Variables:
    x[i,j,k] ∈ {0,1}   1 if vehicle k travels arc (i,j)
    y[i,k]   ∈ {0,1}   1 if vehicle k serves request i
    z[k]     ∈ {0,1}   1 if vehicle k is activated
    t[i,k]   ≥ 0       Arrival time of vehicle k at node i

Objective:
    Minimize  Σ c[i,j]·x[i,j,k]  +  50·Σ z[k]  +  1000·(|R| - Σ y[i,k])

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
    (10) Capacity:
        Σi q[i]·y[i,k] ≤ Q·z[k]                           ∀ k ∈ K
    (11) Time windows (Big-M):
        t[i,k] + s[i] + c[i,j] - M·(1 - x[i,j,k]) ≤ t[j,k]
                                                            ∀ (i,j) ∈ A, k ∈ K
    (12) Time bounds:
        e[i] ≤ t[i,k] ≤ l[i]                              ∀ i ∈ N, k ∈ K

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
    """Load ride requests from CSV."""
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
            })
    return requests


def load_depots(path="data/depots.csv"):
    """Load depot locations from CSV."""
    depots = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            depots.append({
                "id": int(row["depot_id"]),
                "lat": float(row["lat"]),
                "lng": float(row["lng"]),
                "name": row["name"],
            })
    return depots


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

def build_model(requests, depots):
    """
    Build the monolithic Multi-Depot CVRPTW model.

    Creates arc variables x[i,j,k] for every pair of nodes (i,j) with i != j
    and every vehicle k. A single global Big-M constant (M = 1,000,000) is
    used for all time-window linearization constraints.

    The arc set `arcs` contains all (i,j) pairs used for variable and
    constraint creation. Adjacency lists `out_nbrs` and `in_nbrs` are
    derived from this arc set and used throughout the constraint code.
    """
    travel, service, N_R, N_D, N = build_travel_times(requests, depots)

    N_K = 16           # number of vehicles
    Q = 4              # vehicle capacity (seats)
    M = 500            # Big-M for time window constraints (max time span + max travel time)

    depot_start = list(range(N_R, N_R + N_D))
    depot_end = list(range(N_R + N_D, N_R + 2 * N_D))
    request_nodes = list(range(N_R))
    all_nodes = list(range(N))
    vehicles = list(range(N_K))

    # Build complete arc set
    arcs = [(i, j) for i in all_nodes for j in all_nodes if i != j]
    
    # Adjacency lists
    out_nbrs = {i: [j for i2, j in arcs if i2 == i] for i in all_nodes}
    in_nbrs = {j: [i for i, j2 in arcs if j2 == j] for j in all_nodes}
    
    model = gp.Model("MDCVRPTW")
    model.setParam('MIPGap', 0.001)
    
    # Decision variables
    x = model.addVars(arcs, vehicles, vtype=GRB.BINARY, name="x")
    y = model.addVars(request_nodes, vehicles, vtype=GRB.BINARY, name="y")
    z = model.addVars(vehicles, vtype=GRB.BINARY, name="z")
    t = model.addVars(all_nodes, vehicles, lb=0, ub=M, name="t")
    
    # Objective: minimize travel cost + vehicle activation cost + unserved penalty
    obj_travel = gp.quicksum(travel[i, j] * x[i, j, k] 
                            for i, j in arcs for k in vehicles)
    obj_vehicles = gp.quicksum(50 * z[k] for k in vehicles)
    obj_unserved = gp.quicksum(1000 * (1 - gp.quicksum(y[i, k] for k in vehicles))
                              for i in request_nodes)
    model.setObjective(obj_travel + obj_vehicles + obj_unserved, GRB.MINIMIZE)
    
    # Symmetry breaking: force vehicles to be used in order
    for k in range(N_K - 1):
        model.addConstr(z[k] >= z[k + 1], name=f"symm_break_{k}")
    
    # Continue with other constraints = list(range(N_K))

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

    # -----------------------------------------------------------------------
    # Create model
    # -----------------------------------------------------------------------
    model = gp.Model("uber_mdcvrptw")
    model.setParam("LogFile", "gurobi.log")
    model.setParam("MIPGap", 0.001)  # 0.1% optimality gap

    # -----------------------------------------------------------------------
    # Decision variables
    # -----------------------------------------------------------------------

    # x[i,j,k]: binary -- vehicle k traverses arc (i,j)
    print("Creating arc variables x[i,j,k] ...")
    x = {}
    for k in vehicles:
        for i, j in arcs:
            x[i, j, k] = model.addVar(
                vtype=GRB.BINARY, obj=travel[i, j],
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
    # Objective
    # -----------------------------------------------------------------------
    served = gp.quicksum(y[i, k] for i in request_nodes for k in vehicles)
    model.setObjective(
        model.getObjective()
        + 50.0 * gp.quicksum(z[k] for k in vehicles)
        + 1000.0 * (N_R - served),
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

    # (10) Capacity
    for k in vehicles:
        model.addConstr(
            gp.quicksum(passengers[i] * y[i, k] for i in request_nodes)
            <= Q * z[k],
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

    model.update()

    # Print model statistics
    print(f"\nModel statistics:")
    print(f"  Variables:    {model.NumVars:,}")
    print(f"    Binary:     {model.NumBinVars:,}")
    print(f"    Continuous: {model.NumVars - model.NumBinVars:,}")
    print(f"  Constraints:  {model.NumConstrs:,}")
    print(f"  Non-zeros:    {model.NumNZs:,}")

    return model, x, y, z, t, requests, depots, vehicles, request_nodes, N_R, N_D


# ---------------------------------------------------------------------------
# Solution reporting
# ---------------------------------------------------------------------------

def report_solution(model, x, y, z, t, requests, depots, vehicles,
                    request_nodes, N_R, N_D):
    """Print a summary of the optimized solution."""
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
        print(f"  Vehicle {k:2d} ({d_name}): "
              f"{len(reqs)} requests -> {' -> '.join(str(n) for n in route)}")


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
    print(f"  {len(requests)} ride requests, {len(depots)} depots")

    print("\nBuilding model ...")
    t0 = time_module.time()
    result = build_model(requests, depots)
    model, x, y, z, t_var, requests, depots, vehicles, request_nodes, N_R, N_D = result
    build_time = time_module.time() - t0
    print(f"Model built in {build_time:.1f}s")

    print("\nSolving ...")
    t0 = time_module.time()
    model.optimize()
    solve_time = time_module.time() - t0
    print(f"\nSolve time: {solve_time:.1f}s")

    report_solution(model, x, y, z, t_var, requests, depots, vehicles,
                    request_nodes, N_R, N_D)

    print(f"\nTotal time: {build_time + solve_time:.1f}s")
    print(f"Log written to gurobi.log")


if __name__ == "__main__":
    main()
