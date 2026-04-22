import argparse
import os
import sys
import time
import torch


from isaaclab.app import AppLauncher

# Add local path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL (DEBUG MODE).")

parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=20000)
parser.add_argument("--distributed", action="store_true", default=False)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()


sys.argv = [sys.argv[0]] + hydra_args


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg):


    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    env_cfg.scene.num_envs = args_cli.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations

    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device


    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)


    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, log_dir)

    

    try:
        env = gym.make(
            args_cli.task,
            cfg=env_cfg,
            render_mode="rgb_array" if args_cli.video else None
        )
    except Exception as e:
        sys.exit(1)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:

        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }

        print_dict(video_kwargs)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)


    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


    try:
        runner = OnPolicyRunner(
            env,
            agent_cfg.to_dict(),
            log_dir=log_dir,
            device=agent_cfg.device
        )
       
    except Exception as e:
        print("[ERROR] Failed to create OnPolicyRunner:", e)
        sys.exit(1)

    dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "agent.yaml"), agent_cfg)


    start_time = time.time()

    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True
        )
    except Exception as e:
        print("[ERROR] Training crashed:", e)
        sys.exit(1)

    elapsed = time.time() - start_time

    env.close()
    


if __name__ == "__main__":
    main()
    simulation_app.close()

