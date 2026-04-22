
import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from rl_utils import camera_follow

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import rl_training.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 50

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels = None

    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1]/2,
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.tensor(controller.advance(), dtype=torch.float32).unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Environment observation space - Actor: {env.num_obs}, Critic: {env.num_privileged_obs}")
    print(f"[INFO]: Environment action space: {env.num_actions}")    
    # load previously trained model
    cfg = agent_cfg.to_dict()

    if "runner" not in cfg or not cfg.get("runner", {}).get("policy_class_name"):

        runner_fields = {
            "num_steps_per_env", "max_iterations", "save_interval", 
            "empirical_normalization", "experiment_name", "run_name",
            "logger", "device", "seed", "clip_actions", "check_for_nan",
            "neptune_project", "wandb_project", "resume", "load_run", "load_checkpoint"
        }
        
        runner_cfg = {}
        for field in runner_fields:
            if field in cfg:
                runner_cfg[field] = cfg[field]
        
        # Add class names
        runner_cfg["policy_class_name"] = cfg["policy"]["class_name"]
        runner_cfg["algorithm_class_name"] = cfg["algorithm"]["class_name"]
        
        cfg = {
            "runner": runner_cfg,
            "algorithm": cfg["algorithm"],
            "policy": cfg["policy"]
        }

    ppo_runner = OnPolicyRunner(env, cfg, log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # Inspect checkpoint and attempt a robust partial load: print detailed key/shape
    # information and load any compatible parameters by exact name+shape match.
    try:
        # infer policy module from runner for backward compatibility
        try:
            _policy_nn = ppo_runner.alg.policy
        except AttributeError:
            _policy_nn = ppo_runner.alg.actor_critic

        # load checkpoint safely when possible
        try:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        except TypeError:
            # older torch versions don't support weights_only
            ckpt = torch.load(resume_path, map_location="cpu")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            ckpt = None

        if ckpt is None:
            print("[WARNING] No checkpoint loaded for inspection.")
        else:
            # locate candidate state-dict inside the checkpoint
            cand_sd = None
            if isinstance(ckpt, dict):
                for key in ("state_dict", "model_state_dict", "policy_state_dict", "actor_critic_state_dict", "model"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        cand_sd = ckpt[key]
                        break
                if cand_sd is None:
                    # fallback: find first dict-like value
                    for k, v in ckpt.items():
                        if isinstance(v, dict):
                            cand_sd = v
                            break
            # fallback if checkpoint itself looks like a state-dict
            if cand_sd is None and isinstance(ckpt, dict):
                cand_sd = ckpt

            if cand_sd is None:
                print("[WARNING] No state-dict found in checkpoint for inspection.")
            else:
                model_sd = _policy_nn.state_dict()
                ckpt_keys = list(cand_sd.keys())
                model_keys = list(model_sd.keys())
                print(f"[INFO] Checkpoint contains {len(ckpt_keys)} params; model expects {len(model_keys)} params.")

                exact_matches = []
                shape_mismatches = []
                missing_in_ckpt = []

                for mk in model_keys:
                    if mk in cand_sd:
                        v_ck = cand_sd[mk]
                        v_mo = model_sd[mk]
                        shape_ck = getattr(v_ck, "shape", None)
                        shape_mo = getattr(v_mo, "shape", None)
                        if shape_ck == shape_mo:
                            exact_matches.append(mk)
                        else:
                            shape_mismatches.append((mk, shape_ck, shape_mo))
                    else:
                        missing_in_ckpt.append(mk)

                extra_in_ckpt = [k for k in ckpt_keys if k not in model_sd]

                # Adapter insertion disabled: prefer padding/trimming checkpoint weights
                # to match current model architecture. This avoids changing module structure
                # which can break downstream exporters (ONNX/JIT).
                pass

                print(f"[INFO] Exact key+shape matches: {len(exact_matches)}")
                if exact_matches:
                    print("[INFO] Sample matched keys:", ", ".join(exact_matches[:10]))

                print(f"[INFO] Keys with shape mismatches: {len(shape_mismatches)}")
                for k, s_ck, s_mo in shape_mismatches[:20]:
                    print(f"  - {k}: checkpoint shape={s_ck}, model shape={s_mo}")

                print(f"[INFO] Keys missing in checkpoint: {len(missing_in_ckpt)}")
                if missing_in_ckpt:
                    print("  Sample missing:", ", ".join(missing_in_ckpt[:10]))

                print(f"[INFO] Extra keys in checkpoint: {len(extra_in_ckpt)}")
                if extra_in_ckpt:
                    print("  Sample extras:", ", ".join(extra_in_ckpt[:10]))

                # perform robust partial load: exact matches, pad/trim first-layer weights, map log_std->std,
                # and fallback to shape-based mapping when needed.
                compatible = {}
                skipped = []

                # Helper: try to map log_std -> std (convert with exp)
                if 'log_std' in cand_sd and 'std' in model_sd:
                    try:
                        v_log = cand_sd['log_std']
                        if getattr(v_log, 'shape', None) == getattr(model_sd['std'], 'shape', None):
                            compatible['std'] = torch.exp(v_log)
                            print("[INFO] Mapped 'log_std' -> 'std' using exp().")
                        else:
                            print("[WARNING] 'log_std' shape does not match model 'std'; skipping mapping.")
                    except Exception as e:
                        print(f"[WARNING] Failed mapping log_std->std: {e}")

                # Exact-name matches with same shape
                for k, v in cand_sd.items():
                    if k in model_sd:
                        target = model_sd[k]
                        shape_ck = getattr(v, 'shape', None)
                        shape_mo = getattr(target, 'shape', None)
                        if shape_ck == shape_mo:
                            compatible[k] = v
                        else:
                            # attempt safe pad/trim for 2D weight matrices (Linear layers)
                            if isinstance(v, torch.Tensor) and isinstance(target, torch.Tensor) and getattr(v, 'ndim', None) == 2 and getattr(target, 'ndim', None) == 2:
                                out_ck, in_ck = v.shape
                                out_mo, in_mo = target.shape
                                if out_ck == out_mo:
                                    if in_ck < in_mo:
                                        # pad columns with zeros to match model input size
                                        new = torch.zeros((out_ck, in_mo), dtype=v.dtype)
                                        new[:, :in_ck] = v
                                        compatible[k] = new
                                        print(f"[INFO] Padded checkpoint '{k}' from in_features={in_ck} to {in_mo}.")
                                    elif in_ck > in_mo:
                                        # trim extra input columns
                                        new = v[:, :in_mo].clone()
                                        compatible[k] = new
                                        print(f"[INFO] Trimmed checkpoint '{k}' from in_features={in_ck} to {in_mo}.")
                                    else:
                                        skipped.append(k)
                                else:
                                    skipped.append(k)
                            else:
                                skipped.append(k)

                # If no compatible keys yet, try mapping by identical shapes (name changed)
                if not compatible:
                    used_model_keys = set()
                    mapped = {}
                    for ck in ckpt_keys:
                        v_ck = cand_sd[ck]
                        shape_ck = getattr(v_ck, 'shape', None)
                        if shape_ck is None:
                            continue
                        for mk in model_keys:
                            if mk in used_model_keys or mk in compatible:
                                continue
                            v_mo = model_sd[mk]
                            shape_mo = getattr(v_mo, 'shape', None)
                            if shape_ck == shape_mo:
                                mapped[ck] = mk
                                used_model_keys.add(mk)
                                break
                    if mapped:
                        print(f"[INFO] Mapped {len(mapped)} checkpoint keys to model keys by shape.")
                        for ck, mk in mapped.items():
                            compatible[mk] = cand_sd[ck]
                        print("[INFO] Sample mappings:")
                        for ck, mk in list(mapped.items())[:10]:
                            print(f"  {ck} --> {mk}")

                if compatible:
                    # load into model state dict (compatible keys are model_key: tensor)
                    model_sd.update(compatible)
                    try:
                        _policy_nn.load_state_dict(model_sd)
                        print(f"[INFO] Partially loaded {len(compatible)} parameters into policy (from checkpoint).")
                    except Exception as e:
                        print(f"[WARNING] Failed to load state_dict after compatibility processing: {e}")
                else:
                    print("[WARNING] No compatible parameters found to load into policy.")
    except Exception as e:
        print(f"[WARNING] Checkpoint inspection/partial load failed: {e}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.onnx",
    )
    export_policy_as_jit(
        policy=policy_nn,
        normalizer=None,
        path=export_model_dir,
        filename="policy.pt",
    )

    dt = env.unwrapped.step_dt
    # print(dt, "dt")
    # reset environment
    
    obs = env.get_observations()
    
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)

            # env stepping
            # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
            #obs, _, terminated, truncated, _ = env.step(actions)
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if args_cli.keyboard:
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
