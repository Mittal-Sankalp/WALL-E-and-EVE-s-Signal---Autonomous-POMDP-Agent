# WALL-E and EVE's Signal - Autonomous POMDP Agent

## Overview
This repository contains an autonomous agent built to operate within a Partially Observable Markov Decision Process (POMDP) framework. It must explore a 2D grid to locate EVE while strictly managing a limited resource budget (battery and time).

## Architecture & Strategy
* **Environment:** A custom simulated grid where every action (`move`, `scan`, `recharge`) consumes battery and time.
* **Belief State (Bayesian Updating):** Since EVE's location is unknown, the agent maintains a probability distribution matrix over the entire map. The config allows for an `initial_signal` to bias this prior belief. When the agent scans and finds clues, it uses Bayesian logic to update the likelihood of EVE's position based on signal strength and Manhattan distance.
* **Decision Policy:** The agent follows a greedy-heuristic policy. It continuously navigates toward the coordinate with the highest belief probability. It incorporates stochastic scanning to gather environmental data and possesses a safety override to navigate to a `recharge` state if its battery drops below a critical threshold (25%).

## Deliverables Included
1. **`main.py`**: The core simulation, containing the `WalleEnvironment` and `WalleAgent`.
2. **`config.json`**: The mission configuration defining map size, clue seeds, budget, and initial signal intelligence.
3. **`Mission_Report.xlsx`**: A full tabular action trace detailing every step, observation, battery level, and reasoning.
4. **`mission_report.txt`**: A summarized final report of the mission's outcome, evidence gathered, and inferred coordinates.
5. **`mission_trajectory.png`**: A heatmap and pathing visualization of the agent's belief matrix and movement history.

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Execute the simulation: `python main.py`
4. The script will output the terminal trace and automatically generate the Excel log, text report, and visualization image in the root directory.
