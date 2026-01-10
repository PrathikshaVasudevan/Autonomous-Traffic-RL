# Autonomous Traffic Optimization using Reinforcement Learning

## Problem Statement

Traditional traffic signals operate on static or fixed timers, which cannot adapt to real-time traffic conditions. This project implements an AI-powered traffic controller using Reinforcement Learning (DQN) to dynamically optimize traffic flow and minimize cumulative waiting time.

---

## Project Overview

This system simulates a 4-way intersection using PyGame. A Deep Q-Network (DQN) agent learns to control traffic signals based on real-time vehicle density and waiting times. The system also supports an emergency override mechanism for ambulances.

---

## Key Features

* 4-way intersection simulation (PyGame)
* DQN-based adaptive signal control
* Real-time state observation
* Reward-based learning
* Ambulance priority override
* Static vs AI performance comparison
* Model saving (.pth)
* Analytics graph generation

---

## State, Action, Reward

### State

```
[cars_N, cars_S, cars_E, cars_W, avg_wait_time]
```

### Actions

* 0 → North-South Green, East-West Red
* 1 → East-West Green, North-South Red

### Reward

```
Reward = - Total Waiting Time
```

---

## How to Run

### Install Dependencies

```bash
pip install pygame torch matplotlib numpy
```

### Train the AI Model

```bash
python simulator.py
```

This will generate:

```
traffic_dqn.pth
```

### Generate Performance Graph

```bash
python experiment.py
```

This will create:

```
performance_comparison.png
```

---

## Results

The AI-based traffic controller significantly reduces average waiting time compared to a fixed-timer system, especially at higher traffic densities.

---

## Future Enhancements

* Multi-intersection support
* Pedestrian modeling
* PPO-based agent
* Real-world map integration
* Cloud dashboard

---

## Demo Video
[Watch the demo here]https://youtu.be/ycu4LJ_EJ0g


## Author

Prathiksha Vasudevan
