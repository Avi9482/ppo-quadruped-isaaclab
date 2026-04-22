# PPO-Based Adaptive Locomotion Framework for Quadruped Robots in Simulation

A reinforcement learning framework for training quadruped robots to achieve stable and adaptive locomotion in simulation using **Proximal Policy Optimization (PPO)** on the **NVIDIA Isaac Lab** platform.

This project focuses on robust locomotion control, terrain adaptation, policy learning, and sim-to-real inspired training pipelines for modern legged robots.

---

## 🚀 Project Overview

Quadruped robots require intelligent control strategies to walk, balance, recover, and navigate uneven terrain. Traditional controllers often need extensive manual tuning.

This project uses **Deep Reinforcement Learning (DRL)** to train locomotion policies that learn directly through interaction with the simulation environment.

The trained PPO agent learns:

- Stable walking behavior
- Velocity tracking
- Terrain adaptation
- Balance recovery
- Efficient gait generation
- Robust movement on rough surfaces

---

## 🧠 Core Algorithm

### Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement learning algorithm designed for stable and efficient training.

Features used in this project:

- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Entropy regularization
- Value function learning
- Parallel environment rollouts

---

## 🦾 Simulation Platform

Built using:

- **NVIDIA Isaac Lab**
- **Isaac Sim**
- **PyTorch**
- **RSL-RL**
- **Python**

---

## 🤖 Supported Robot Models

- Deep Robotics Lite3
- Deep Robotics X30
- Deep Robotics M20

---

## 📁 Project Structure

```text
MTP3/
│── rl_training/
│   ├── scripts/
│   ├── source/
│   ├── config/
│   ├── deep_robotics_model/
│   └── training_logs/
│
│── README.md
