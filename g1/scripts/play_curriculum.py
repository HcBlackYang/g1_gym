# play_curriculum.py
import sys
import os
import isaacgym
import torch
import numpy as np
import argparse # Use argparse for play-specific args
import copy # For deep copying config

# --- 导入 G1 项目模块 ---
G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(G1_ROOT_DIR)
import g1.envs # 触发注册
from g1.utils import get_args as get_train_args, task_registry, set_seed # Use training arg parser as base
from g1.envs.configs.curriculum.curriculum_manager_config import CurriculumManagerConfig # Needed potentially if task configs depend on it

# --- DotDict ---
class DotDict(dict):
    """一个支持点访问的字典类"""
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = DotDict(v) if isinstance(v, dict) else v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = DotDict(v) if isinstance(v, dict) else v
    def __getattr__(self, attr): return self.get(attr)
    def __setattr__(self, key, value): self.__setitem__(key, value)
    def __deepcopy__(self, memo): return DotDict(deepcopy(dict(self), memo=memo))

def parse_play_args():
    """解析 Play 模式的命令行参数"""
    # Start with training args parser to get common arguments like --task, --seed, etc.
    parser = get_train_args(parse_known=True) # Use parse_known=True if get_args supports it
    if isinstance(parser, tuple): # If get_args returns (args, unknown)
         args, _ = parser
         # Need to create a new parser for play-specific args
         play_parser = argparse.ArgumentParser(description='Play policy for G1 Curriculum Task')
    else: # If get_args returns the parser object itself
         play_parser = parser
         args = None # Parse later

    # Add/override arguments specific to play mode
    play_parser.add_argument('--checkpoint', required=True, type=str, help='Path to the trained model checkpoint (.pt file)')
    play_parser.add_argument('--num_envs', type=int, default=16, help='Number of environments to visualize')
    play_parser.add_argument('--export_policy', action='store_true', default=False, help='Export the policy as a JIT module')
    play_parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of simulation steps to run')
    play_parser.add_argument('--task', required=True, type=str, help='Name of the task to run (e.g., G1BasicLocomotion)')

    # Ensure headless is False for visualization
    play_parser.set_defaults(headless=False)
    # Remove arguments not relevant for playing
    # parser.remove_argument('--save_interval') # Example if using standard parser
    # parser.remove_argument('--resume')
    # ... remove other training-specific args if necessary

    if args is None: # If get_train_args returned the parser
         args = play_parser.parse_args()
    else: # If get_train_args returned parsed args, parse the rest
         # Parse only known args from play_parser to avoid conflicts if get_train_args didn't use parse_known
         play_args, unknown = play_parser.parse_known_args()
         # Update the main args object
         for key, value in vars(play_args).items():
              setattr(args, key, value)

    # Final overrides
    args.headless = False # Force GUI
    args.play = True # Add a flag indicating play mode
    args.resume = False # Ensure resume is off

    return args


def play_curriculum(args):
    """加载指定任务的模型并在环境中运行进行评估"""
    print("="*50)
    print(f"▶️ 开始运行 Play 模式 - Task: {args.task}")
    print(f"  Checkpoint: {args.checkpoint}")
    print("="*50)

    # --- 1. 配置加载和参数覆盖 ---
    print("\n--- 1. 加载和修改配置 ---")
    try:
        # Get the *base* registered configs for the task
        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
        # Make deep copies to modify safely
        env_cfg = copy.deepcopy(env_cfg)
        # train_cfg is needed by make_alg_runner, copy it too
        train_cfg = copy.deepcopy(train_cfg)
        print(f"  加载任务 '{args.task}' 的基础配置成功。")
    except Exception as e:
        print(f"❌ 加载任务 '{args.task}' 的基础配置失败: {e}")
        print(f"   可用任务: {task_registry.list_tasks()}")
        return

    print("  为 Play 模式覆盖配置参数...")
    # General play overrides
    env_cfg.env.num_envs = args.num_envs # Use num_envs from command line
    env_cfg.env.test = True # Set test flag if environment uses it
    if hasattr(env_cfg, 'terrain'):
        env_cfg.terrain.num_rows = 5
        env_cfg.terrain.num_cols = 5
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_init_terrain_level = None # Use default terrain
        print("    - 关闭地形课程，使用小型默认地形。")
    if hasattr(env_cfg, 'noise'):
        env_cfg.noise.add_noise = False
        print("    - 关闭观测噪声。")
    if hasattr(env_cfg, 'domain_rand'):
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.randomize_base_mass = False
        env_cfg.domain_rand.push_robots = False
        print("    - 关闭领域随机化 (摩擦、质量、推力)。")

    # Specific overrides for Stage 1 might be needed if its config has unique settings
    # Example: Ensure arm usage matches the trained model if Stage1LocomotionConfig had such a flag
    # if args.task == "G1BasicLocomotion" and hasattr(env_cfg.env, 'use_arm'):
    #     env_cfg.env.use_arm = True # Or False, depending on how Stage 1 was trained


    # --- 2. 创建环境 ---
    print("\n--- 2. 创建环境 ---")
    try:
        # Pass the modified env_cfg
        env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
        print(f"  ✅ 环境创建成功 (Num Envs: {env.num_envs})")
    except Exception as e:
        print(f"❌ 创建环境时出错: {e}")
        if args.debug: import traceback; traceback.print_exc()
        return

    # --- 3. 加载策略 ---
    print("\n--- 3. 加载策略 ---")
    try:
        # Create the Runner - it will load the policy based on args.checkpoint
        # Pass the (potentially modified) train_cfg
        train_cfg.runner.resume = True # Tell runner to load
        # args already contains the checkpoint path
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

        # Get the policy function for inference
        policy = ppo_runner.get_inference_policy(device=env.device)
        print(f"  ✅ 策略从 {args.checkpoint} 加载成功。")
    except FileNotFoundError:
         print(f"❌ 错误: 检查点文件未找到: {args.checkpoint}")
         env.close()
         return
    except Exception as e:
         print(f"❌ 创建 Runner 或加载策略时出错: {e}")
         if args.debug: import traceback; traceback.print_exc()
         env.close()
         return

    # --- 4. 导出策略 (可选) ---
    if args.export_policy:
        print("\n--- 4. 导出 JIT 策略 ---")
        export_dir = os.path.join(os.path.dirname(args.checkpoint), 'exported')
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f'{os.path.basename(args.checkpoint).replace(".pt", "")}_jit.pt')
        try:
            # Assumes actor_critic model is accessible via runner.alg.actor_critic
            model_to_export = ppo_runner.alg.actor_critic
            # Get observation shape from env
            obs_shape = (env.num_observations,)
            # Create dummy input matching the observation shape
            dummy_input = torch.randn((1,) + obs_shape, device=env.device) # Batch size 1

            # Trace the model
            traced_script_module = torch.jit.trace(model_to_export.actor, dummy_input) # Export only the actor part
            traced_script_module.save(export_path)
            print(f'  ✅ 策略 Actor 已导出为 JIT 脚本: {export_path}')
        except AttributeError:
             print("  ❌ 错误: 无法访问 runner.alg.actor_critic.actor 进行导出。")
        except Exception as e:
             print(f"  ❌ 导出策略时出错: {e}")


    # --- 5. 运行仿真循环 ---
    print("\n--- 5. 开始仿真循环 ---")
    obs = env.get_observations()
    if isinstance(obs, dict): # Handle dictionary observations if used
         obs = obs['obs']

    num_steps = 0
    total_reward = torch.zeros(env.num_envs, device=env.device)
    total_episodes = 0

    try:
        while num_steps < args.max_steps:
            with torch.no_grad():
                actions = policy(obs.detach())
            obs, _, rews, dones, infos = env.step(actions.detach())
            if isinstance(obs, dict): obs = obs['obs'] # Handle dict obs

            total_reward += rews
            num_steps += 1

            # Count completed episodes and print stats
            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            if len(done_indices) > 0:
                completed_episodes = len(done_indices)
                total_episodes += completed_episodes
                avg_reward = torch.mean(total_reward[done_indices]).item()
                print(f"  Step {num_steps:>6} | Episodes Completed: {completed_episodes:>3} | Avg Reward: {avg_reward:>8.2f}")
                # Reset reward for completed envs
                total_reward[done_indices] = 0


            # Check for window close events (handled by env.step -> env.render)
            if env.viewer and env.gym.query_viewer_has_closed(env.viewer):
                print("检测到查看器关闭，退出 Play 模式。")
                break

            # Optional delay
            # import time
            # time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n🛑 Play 模式被用户中断 (KeyboardInterrupt)")
    except Exception as e:
         print(f"\n❌ 仿真循环中发生错误: {e}")
         if args.debug: import traceback; traceback.print_exc()

    finally:
        print("\n--- 6. Play 模式结束 ---")
        print(f"  总运行步数: {num_steps}")
        print(f"  总完成回合数: {total_episodes}")
        env.close()
        print("  环境已关闭。")

if __name__ == '__main__':
    # 解析 Play 模式的参数
    args = parse_play_args()

    # 设置种子
    if args.seed is None: args.seed = 42
    set_seed(args.seed)
    print(f"🎲 Play 模式种子设置为: {args.seed}")

    # 运行 Play 函数
    play_curriculum(args)