# Uber Network Routing Demo

**Multi-Depot Capacitated Vehicle Routing with Time Windows (MDCVRPTW)**

A monolithic MILP that routes a heterogeneous fleet of vehicles from multiple depots to serve ride requests across Manhattan. Inspired by [Uber's Gurobi case study](https://www.gurobi.com/case_studies/uber-shaping-urban-aerial-ridesharing/) on urban aerial ridesharing network design.

## Problem

Given 50 ride requests with time windows and priorities, 5 depots with vehicle limits, and 15 vehicles of 3 types:

- **Minimize** type-weighted travel costs + vehicle activation costs + priority-weighted unserved penalties
- **Subject to** type-dependent capacity (4/6/10 seats), pickup time windows (30-60 min), route duration limits (90/120/150 min), depot vehicle caps, VIP must-serve, incompatible pairs, and minimum utilization

## Model

| Component | Details |
|-----------|---------|
| Requests | 50 rides across Manhattan, Williamsburg, DUMBO, Roosevelt Island |
| Depots | 5 (Financial District, Midtown, Upper West Side, Williamsburg, Roosevelt Island) |
| Vehicles | 15 (5 Sedan cap=4, 5 SUV cap=6, 5 Van cap=10) |
| Binary vars | ~55,000 arc + assignment + activation variables |
| Constraints | ~57,000 (time windows, flow, capacity, route duration, VIP, incompatible pairs, utilization) |
| Solve time | ~5 minutes (default settings) |

### Sets

| Set | Description | Size |
|-----|-------------|------|
| R | Ride requests | 50 |
| D | Depots | 5 |
| K | Vehicles | 15 (5 per type) |
| T | Vehicle types (Sedan, SUV, Van) | 3 |
| N | Extended nodes (R + depot starts + depot ends) | 60 |
| A | Arcs (all node pairs) | ~3,540 |
| I | Incompatible request pairs | ~15 |

### Vehicle Types

| Type | Capacity | Max Route | Fixed Cost | Per-Min Cost | Count |
|------|----------|-----------|------------|--------------|-------|
| Sedan | 4 | 90 min | $40 | $1.0/min | 5 |
| SUV | 6 | 120 min | $60 | $1.5/min | 5 |
| Van | 10 | 150 min | $80 | $2.0/min | 5 |

### Formulation

- **Arc variables** `x[i,j,k]`: vehicle k traverses arc (i,j)
- **Assignment variables** `y[i,k]`: vehicle k serves request i
- **Activation variables** `z[k]`: vehicle k is used
- **Time variables** `t[i,k]`: arrival time at node i

**Constraints (1-12):** Standard MDCVRPTW (serve once, assignment-arc consistency, flow conservation, depot departure/return, activation linking, type-dependent capacity, Big-M time windows, time bounds)

**Constraint (13) Route duration:** `t[de,k] - t[ds,k] <= max_route[k] + M*(1 - Σj x[ds,j,k])`

**Constraint (14) Depot vehicle cap:** `Σk (Σj x[ds,j,k]) <= max_vehicles[d]`

**Constraint (15) VIP must-serve:** `Σk y[i,k] = 1` for priority-1 requests

**Constraint (16) Incompatible pairs:** `y[a,k] + y[b,k] <= 1` for each pair and vehicle

**Constraint (17) Minimum utilization:** `Σi y[i,k] >= 2*z[k]`

### Deliberate Inefficiencies

This model includes 7 deliberate inefficiencies for an optimization agent to discover and fix:

| # | Inefficiency | Agent Fix | Expected Speedup |
|---|-------------|-----------|-----------------|
| 1 | Global Big-M = 1,000,000 (actual ~20-250) | Per-constraint tight M values | ~30-40% |
| 2 | No arc pre-filtering (all N*(N-1) arcs) | Filter time-infeasible arcs | ~15-20% |
| 3 | No symmetry breaking (5 identical vehicles × 3 types) | `z[k] >= z[k+1]` within type | ~40-60% |
| 4 | No branching priorities | Prioritize `z[k]` and VIP `y[i,k]` | ~15-25% |
| 5 | Default Symmetry=-1 | Set Symmetry=2 (aggressive) | ~10-20% |
| 6 | Default MIPFocus=0 | Set MIPFocus=1 (feasibility) | ~10-15% |
| 7 | Default Cuts=-1 | Set Cuts=2 (aggressive) | ~10-20% |

## Quick Start

```bash
pip install gurobipy
python generate_data.py   # regenerate data (optional)
python main.py
```

Requires a valid Gurobi license ([free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/)).

## Files

| File | Description |
|------|-------------|
| `main.py` | Model formulation, solver, and solution reporter |
| `generate_data.py` | Data generator (re-run to regenerate CSVs) |
| `data/ride_requests.csv` | 50 ride requests with coordinates, time windows, and priorities |
| `data/depots.csv` | 5 depot locations with max vehicle counts |
| `data/vehicle_types.csv` | 3 vehicle types with capacities and costs |
| `data/incompatible_pairs.csv` | ~15 incompatible request pairs |

## License

MIT
