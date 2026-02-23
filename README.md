# Uber Network Routing Demo

**Multi-Depot Capacitated Vehicle Routing with Time Windows (MDCVRPTW)**

A monolithic MILP that routes a fleet of vehicles from multiple depots to serve ride requests across Manhattan. Inspired by [Uber's Gurobi case study](https://www.gurobi.com/case_studies/uber-shaping-urban-aerial-ridesharing/) on urban aerial ridesharing network design.

## Problem

Given 60 ride requests with time windows, 4 depots, and 16 vehicles:

- **Minimize** total travel time + vehicle activation costs + unserved request penalties
- **Subject to** vehicle capacity (4 seats), pickup time windows (20-45 min), depot assignments, and flow conservation

## Model

| Component | Details |
|-----------|---------|
| Requests | 60 rides across Manhattan, Williamsburg, DUMBO |
| Depots | 4 (Financial District, Midtown, Upper West Side, Williamsburg) |
| Vehicles | 16 (capacity: 4 passengers each) |
| Variables | ~74,000 binary arc variables + assignment + time |
| Constraints | ~76,000 (time windows, flow, capacity) |

### Formulation

- **Arc variables** `x[i,j,k]`: vehicle k traverses arc (i,j)
- **Assignment variables** `y[i,k]`: vehicle k serves request i
- **Activation variables** `z[k]`: vehicle k is used
- **Time variables** `t[i,k]`: arrival time at node i

Time window constraints use Big-M linearization:

```
t[i,k] + service[i] + travel[i,j] - M*(1 - x[i,j,k]) <= t[j,k]
```

## Quick Start

```bash
pip install gurobipy
python main.py
```

Requires a valid Gurobi license ([free academic license](https://www.gurobi.com/academia/academic-program-and-licenses/)).

## Files

| File | Description |
|------|-------------|
| `main.py` | Model formulation, solver, and solution reporter |
| `generate_data.py` | Data generator (re-run to regenerate CSVs) |
| `data/ride_requests.csv` | 60 ride requests with coordinates and time windows |
| `data/depots.csv` | 4 depot locations |

## License

MIT
