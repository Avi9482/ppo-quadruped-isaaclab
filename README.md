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

## 💻 System Requirements

### Minimum

- Windows 10 / 11 or Ubuntu 20.04+
- Python 3.10
- 16 GB RAM
- NVIDIA GPU

### Recommended

- RTX GPU with CUDA support
- 32 GB RAM
- SSD Storage

---

## 🖥️ NVIDIA Driver Requirement

Install the latest NVIDIA GPU drivers.

### Check GPU

```bash
nvidia-smi

---
## ⚙️ Installation (For New Laptop / New PC)

## 💻 Install Visual Studio Code (VS Code)

Download VS Code:

https://code.visualstudio.com/

### Windows Installation

1. Download **Windows Installer**  
2. Run setup file  
3. Enable these options during installation:

- Add to PATH  
- Register Code as editor  
- Add "Open with Code" action  

4. Click **Install**

### Ubuntu Installation

```bash
sudo snap install code --classic

---


### 1️⃣ Install Miniconda / Anaconda

Download:

- https://www.anaconda.com/products/distribution  
- https://docs.conda.io/en/latest/miniconda.html

---

### 2️⃣ Clone Repository

```bash
git clone https://github.com/avi9482/ppo-quadruped-isaaclab.git
cd ppo-quadruped-isaaclab

---

### 3️⃣ Create Environment

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab

---

### 4️⃣ Install PyTorch (CUDA)

Visit official site:

https://pytorch.org/get-started/locally/

Example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


---

### 5️⃣ Install Project

```bash
pip install -e .

pip install tensorboard
pip install gymnasium
pip install matplotlib
pip install numpy

---
---

## 🤖 Install NVIDIA Isaac Lab

Follow the official Isaac Lab pip installation guide:

https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html

### Create Environment

```bash
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab

---

## ▶️ Important Commands

### 🔹 Check Available Environments

```bash
python rl_training/scripts/tools/list_envs.py

---

### 🔹 Play Trained Agent (Rough Terrain)

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-Lite3-v0 


---

## 📊 TensorBoard Monitoring

Launch TensorBoard:

```bash
tensorboard --logdir logs

---

### 🔹 Play Using Saved Checkpoint

```bash
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-Lite3-v0 --keyboard --checkpoint .\logs\rsl_rl\deeprobotics_lite3_rough\2026-04-12_19-05-07\model_29999.pt

---

---

## 🎮 Manual Keyboard Control (After Running `play.py`)

After launching `play.py` with a checkpoint, you can manually control the robot using keyboard arrow keys.

| Key | Action |
|-----|--------|
| ↑ Up Arrow | Move Forward |
| ↓ Down Arrow | Move Backward |
| ← Left Arrow | Turn Left |
| → Right Arrow | Turn Right |
| Space | Stop |

