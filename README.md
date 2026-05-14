# PPO-Based Adaptive Locomotion for Quadruped Robots

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch)
![Isaac Lab](https://img.shields.io/badge/NVIDIA-Isaac%20Lab-76B900?style=flat-square&logo=nvidia)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Ubuntu-lightgrey?style=flat-square)

**A deep reinforcement learning framework for training quadruped robots to achieve stable, adaptive locomotion in simulation using Proximal Policy Optimization (PPO) on the NVIDIA Isaac Lab platform.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Core Algorithm](#core-algorithm)
- [Simulation Platform](#simulation-platform)
- [Supported Robots](#supported-robots)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Monitoring Training](#monitoring-training)
- [Keyboard Control](#keyboard-control)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## Overview

Quadruped robots require intelligent control strategies to walk, balance, recover from disturbances, and navigate uneven terrain. Traditional model-based controllers demand extensive manual tuning and struggle to generalize across varied terrains.

This project leverages **Deep Reinforcement Learning (DRL)** to train locomotion policies that emerge directly through interaction with a physics-accurate simulation environment. The trained PPO agent learns complex motor behaviors end-to-end without hand-crafted rules.

### What the Agent Learns

| Behavior | Description |
|---|---|
| Stable Walking | Smooth, energy-efficient forward locomotion |
| Velocity Tracking | Following target velocity commands accurately |
| Terrain Adaptation | Adjusting gait for rough, uneven surfaces |
| Balance Recovery | Recovering from pushes and disturbances |
| Gait Generation | Discovering efficient contact patterns |
| Rough Surface Navigation | Robust movement over obstacles |

---

## Features

- **Sim-to-Real Inspired Pipeline** — Domain randomization and noise injection for policy robustness
- **Parallel Training** — Massively parallel environment rollouts via Isaac Lab
- **Modular Design** — Easily swap robots, reward functions, and terrain configs
- **TensorBoard Integration** — Real-time training metrics and reward tracking
- **Checkpoint System** — Save and resume training; deploy specific checkpoints
- **Keyboard Teleoperation** — Interactive control of the trained agent in simulation

---

## Core Algorithm

### Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement learning algorithm designed for stable, sample-efficient training with continuous action spaces — well-suited for legged locomotion.

**Key components used in this project:**

| Component | Role |
|---|---|
| Actor-Critic Architecture | Separate networks for policy and value estimation |
| Clipped Surrogate Objective | Prevents destructively large policy updates |
| Generalized Advantage Estimation (GAE) | Reduces variance in gradient estimates |
| Entropy Regularization | Encourages exploration during training |
| Value Function Learning | Improves baseline estimates for advantage computation |
| Parallel Environment Rollouts | Efficient data collection across many simulated instances |

---

## Simulation Platform

| Component | Technology |
|---|---|
| Simulation Engine | NVIDIA Isaac Lab / Isaac Sim |
| RL Framework | RSL-RL |
| Deep Learning | PyTorch |
| Language | Python 3.10 |

---

## Supported Robots

| Robot | Description |
|---|---|
| **Deep Robotics Lite3** | Lightweight quadruped, ideal for agile locomotion |
| **Deep Robotics X30** | Mid-size platform with strong terrain traversal |
| **Deep Robotics M20** | Compact model for fast-paced locomotion tasks |

---

## System Requirements

### Minimum

| Requirement | Specification |
|---|---|
| OS | Windows 10/11 or Ubuntu 20.04+ |
| Python | 3.10 |
| RAM | 16 GB |
| GPU | NVIDIA GPU (any CUDA-capable) |

### Recommended

| Requirement | Specification |
|---|---|
| GPU | NVIDIA RTX series with CUDA support |
| RAM | 32 GB |
| Storage | SSD |

### Verify GPU

```bash
nvidia-smi
```

Install the latest NVIDIA drivers from [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) if needed.

---

## Installation

### 1. Install Visual Studio Code

Download from [code.visualstudio.com](https://code.visualstudio.com/)

**Windows:** Run the installer and enable:
- Add to PATH
- Register Code as editor
- Add "Open with Code" context menu action

**Ubuntu:**
```bash
sudo snap install code --classic
```

---

### 2. Install Miniconda or Anaconda

- Anaconda: [anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- Miniconda (lighter): [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

---

### 3. Clone the Repository

```bash
git clone https://github.com/avi9482/ppo-quadruped-isaaclab.git
cd ppo-quadruped-isaaclab
```

---

### 4. Create and Activate Conda Environment

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab
```

---

### 5. Install PyTorch with CUDA Support

Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to find the command matching your CUDA version.

Example for CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### 6. Install Project Dependencies

```bash
pip install -e .
pip install tensorboard gymnasium matplotlib numpy
```

---

### 7. Install NVIDIA Isaac Lab

Follow the official pip installation guide:
[isaac-sim.github.io/IsaacLab — pip installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab
```

> **Note:** Follow all steps in the official guide carefully, as Isaac Lab has additional CUDA and driver dependencies.

---

## Usage

### List Available Environments

```bash
python rl_training/scripts/tools/list_envs.py
```

---

### Play Trained Agent on Rough Terrain

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-Lite3-v0
```

---

### Play Using a Saved Checkpoint

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task=Rough-Deeprobotics-Lite3-v0 \
    --keyboard \
    --checkpoint .\logs\rsl_rl\deeprobotics_lite3_rough\2026-04-12_19-05-07\model_29999.pt
```

Replace the checkpoint path with your own saved model path.

---

## Monitoring Training

Launch TensorBoard to track rewards, losses, and training metrics in real time:

```bash
tensorboard --logdir logs
```

Then open your browser at `http://localhost:6006`.

**Key metrics to monitor:**
- `Episode/Reward` — Total reward per episode
- `Train/ValueLoss` — Critic loss over time
- `Train/Entropy` — Exploration entropy
- `Episode/EpLen` — Episode length (longer = more stable walking)

---

## Keyboard Control

After launching `play.py` with a checkpoint and the `--keyboard` flag, control the robot interactively:

| Key | Action |
|---|---|
| `↑` Up Arrow | Move Forward |
| `↓` Down Arrow | Move Backward |
| `←` Left Arrow | Turn Left |
| `→` Right Arrow | Turn Right |
| `Space` | Stop |

---

## Project Structure

```
ppo-quadruped-isaaclab/
├── rl_training/
│   └── scripts/
│       └── tools/
│           └── list_envs.py          # List registered environments
├── scripts/
│   └── reinforcement_learning/
│       └── rsl_rl/
│           └── play.py               # Deploy and visualize trained agent
├── logs/                             # Training logs and checkpoints
│   └── rsl_rl/
│       └── deeprobotics_lite3_rough/
├── setup.py                          # Package setup
└── README.md
```


## Acknowledgements

- [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — Simulation and training infrastructure
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) — PPO implementation for legged robots

---

<div align="center">
Made with PyTorch · Trained in Isaac Lab · Deployed on Quadrupeds
</div>
