# train.py (Modified from train_curriculum.py for direct training)
import traceback
import argparse
import numpy as np
from datetime import datetime, timedelta
import copy
import inspect
import os
import time
import yaml # For loading state (optional, kept for consistency)

# 导入IsaacGym
import isaacgym
from isaacgym import gymapi, gymutil # Import necessary gym modules
import torch

# 导入 G1 相关 (按需保留)
import g1.envs # Keep this for task registration trigger
# from g1.envs.curriculum.curriculum_manager import CurriculumManager # REMOVED
# from g1.envs.curriculum.model_transfer import ModelTransfer # REMOVED
# from g1.envs.curriculum.reward_scheduler import RewardScheduler # REMOVED
# from g1.envs.configs.curriculum.curriculum_manager_config import CurriculumManagerConfig # REMOVED

# 导入工具函数和注册表
from g1.utils import get_args, task_registry, set_seed
from g1.utils.helpers import update_cfg_from_args, class_to_dict, parse_sim_params, DotDict

# --- 解析参数 ---
def parse_args():
    """解析命令行参数 (简化版)"""
    # 使用 get_args() 获取基础 RL 参数
    args = get_args()
    # 添加必要的训练脚本参数 (如果 get_args() 中没有)
    parser = argparse.ArgumentParser(description='Train G1 Robot', add_help=False)
    if not hasattr(args, 'debug'):
        parser.add_argument('--debug', action='store_true', default=False, help='启用调试模式，打印更多信息')

    script_args, _ = parser.parse_known_args()
    for key, value in vars(script_args).items():
        setattr(args, key, value)

    if args.headless is None: args.headless = True # Default headless for training
    # 确保 --task 参数存在
    if args.task is None:
        raise ValueError("必须通过 --task 指定要训练的环境任务 (例如: --task G1FullLocomotion)")
    return args


# --- 验证任务兼容性 ---
# (保持 validate_task_compatibility 函数不变)
def validate_task_compatibility(task_name):
    """验证任务是否已在 task_registry 中注册"""
    if hasattr(task_registry, 'task_classes') and isinstance(task_registry.task_classes, dict):
        available_tasks = list(task_registry.task_classes.keys())
        if task_name not in available_tasks:
            print(f"❌ 任务 '{task_name}' 未在 task_registry 中注册!")
            print(f"   可用任务: {available_tasks}")
            return False
        return True
    else:
        print("❌ 错误: 无法访问 task_registry.task_classes 来验证任务兼容性。")
        return False

# --- 加载检查点信息 (简化版) ---
def load_checkpoint_simple(checkpoint_path, device):
    """加载检查点并返回模型状态和环境维度 (简化版)"""
    if not os.path.exists(checkpoint_path):
         print(f"  ❌ 加载错误: 检查点文件未找到: {checkpoint_path}")
         return None, None, 0
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy_state_dict = None
        possible_policy_keys = ['policy_state_dict', 'model_state_dict', 'actor_critic_state_dict', 'state_dict']
        for key in possible_policy_keys:
            if key in checkpoint: policy_state_dict = checkpoint[key]; break
        if policy_state_dict is None and isinstance(checkpoint, dict) and any('weight' in k for k in checkpoint.keys()):
             print("  ⚠️ 无法找到明确的策略状态键，假设检查点根目录是 state_dict。")
             policy_state_dict = checkpoint

        # 尝试获取环境维度
        env_dims = None
        if 'env_config_dims' in checkpoint and isinstance(checkpoint['env_config_dims'], dict):
             env_dims = checkpoint['env_config_dims']
        elif 'env_dims' in checkpoint and isinstance(checkpoint['env_dims'], dict):
             env_dims = checkpoint['env_dims']

        # 获取步数
        loaded_steps = 0
        possible_step_keys = ['total_env_steps', 'total_steps', 'env_steps']
        for key in possible_step_keys:
             if key in checkpoint and isinstance(checkpoint[key], (int, float)): loaded_steps = int(checkpoint[key]); break
        if loaded_steps == 0: # Fallback using iterations
             iter_count = 0
             possible_iter_keys = ['iterations', 'iter', 'current_learning_iteration']
             for key in possible_iter_keys:
                  if key in checkpoint and isinstance(checkpoint[key], int): iter_count = checkpoint[key]; break
             if iter_count > 0: print(f"  ⚠️ 无法从检查点获取环境步数，找到迭代次数 {iter_count}。")

        print(f"  ✅ 检查点数据加载: Steps={loaded_steps:,}, EnvDims={env_dims}")
        return policy_state_dict, env_dims, loaded_steps

    except Exception as e:
        print(f"  ❌ 加载检查点 '{checkpoint_path}' 失败: {e}")
        return None, None, 0

# ==============================================================================
# 主训练函数
# ==============================================================================
def train(args):
    """直接训练指定任务的主函数"""
    print("="*50); print(f"🚀 开始 G1 机器人训练 (任务: {args.task}) 🚀"); print("="*50)

    gym = None
    sim = None
    env = None
    runner = None
    sim_params = None
    total_env_steps = 0

    try:
        # --- 1. 初始化核心组件 ---
        print("\n--- 1. 初始化核心组件 ---")
        # 验证任务名称
        if not validate_task_compatibility(args.task): return

        # 加载任务配置
        try:
            env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)


            print("-" * 20)
            print(f"DEBUG: Type of 'args' before update: {type(args)}")
            print(f"DEBUG: Value of 'args' before update: {args}")
            print("-" * 20)

            # 更新配置以匹配命令行参数 (例如 num_envs, seed 等)
            env_cfg = update_cfg_from_args(env_cfg, args)
            train_cfg = update_cfg_from_args(train_cfg, args)
            print(f"✅ 加载任务 '{args.task}' 的配置成功。")
            print(f"  将使用的环境数量: {env_cfg.env.num_envs}")
        except Exception as e:
            print(f"❌ 加载任务 '{args.task}' 的配置失败: {e}")
            if args.debug: traceback.print_exc()
            return

        # 初始化 Gym 和 Sim
        print("\n--- 初始化 Isaac Gym 和 Simulation ---")
        gym = gymapi.acquire_gym()
        try:
            sim_params_dict = {"sim": class_to_dict(env_cfg.sim)}
            sim_params = parse_sim_params(args, sim_params_dict)
        except Exception as e:
             print(f"❌ 解析 Sim 参数失败: {e}"); return

        physics_engine = gymapi.SIM_PHYSX
        sim_device_type, sim_device_id = gymutil.parse_device_str(args.sim_device)
        graphics_device_id = sim_device_id if not args.headless else -1

        sim = gym.create_sim(sim_device_id, graphics_device_id, physics_engine, sim_params)
        if sim is None: raise RuntimeError("Failed to create sim!")
        print(f"✅ Gym 和 Sim 创建成功 (Sim Handle: {sim})")

        # --- 2. 创建环境和 Runner ---
        print("\n--- 2. 创建环境和 Runner ---")
        policy_state_dict = None
        loaded_env_dims = None
        loaded_steps = 0

        try:
            print(f"  准备创建环境实例: {args.task}")
            env, env_cfg = task_registry.make_env(
                name=args.task, args=args, env_cfg=env_cfg,
                gym_handle=gym, sim_handle=sim, sim_params=sim_params
            )
            print(f"  环境创建成功。Obs: {env.num_observations}, Act: {env.num_actions}")

            # 加载检查点 (如果指定)
            checkpoint_to_load = args.checkpoint
            if args.resume and checkpoint_to_load and os.path.exists(checkpoint_to_load):
                 print(f"  准备从检查点恢复: {checkpoint_to_load}")
                 policy_state_dict, loaded_env_dims, loaded_steps = load_checkpoint_simple(checkpoint_to_load, args.rl_device)
                 if policy_state_dict is None:
                      print("  ❌ 加载检查点数据失败，将随机初始化模型。")
                      args.resume = False; checkpoint_to_load = None; loaded_steps = 0;
                 else:
                      print(f"  ✅ 检查点数据加载成功 (@ {loaded_steps:,} 步)")
            elif args.checkpoint: # Checkpoint specified but not resume
                 print(f"  准备加载检查点（不恢复训练状态）: {args.checkpoint}")
                 policy_state_dict, loaded_env_dims, _ = load_checkpoint_simple(args.checkpoint, args.rl_device)
                 if policy_state_dict is None: print("  ❌ 加载检查点数据失败，将随机初始化模型。")
                 args.resume = False # Ensure resume is off
            else:
                 print("  将随机初始化模型。")
                 args.resume = False

            print(f"  准备创建 Runner (Task: {args.task})...")
            train_cfg.runner.resume = args.resume # 设置 Runner 的恢复状态
            runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

            # 加载模型状态 (如果从检查点加载了)
            if policy_state_dict:
                 target_policy = runner.alg.actor_critic
                 # 检查维度是否匹配
                 current_obs_dim = env.num_observations
                 current_act_dim = env.num_actions
                 ckpt_obs_dim = loaded_env_dims.get('num_observations', -1) if loaded_env_dims else -1
                 ckpt_act_dim = loaded_env_dims.get('num_actions', -1) if loaded_env_dims else -1

                 if loaded_env_dims and (ckpt_obs_dim != current_obs_dim or ckpt_act_dim != current_act_dim):
                      print(f"❌ 致命错误: 检查点维度 (Obs: {ckpt_obs_dim}, Act: {ckpt_act_dim}) 与当前环境维度 (Obs: {current_obs_dim}, Act: {current_act_dim}) 不匹配！")
                      print("   对于直接训练，检查点必须与环境配置兼容。请使用匹配的检查点或不加载检查点。")
                      raise ValueError("Checkpoint dimension mismatch in direct training mode.")
                 else:
                      print("  检查点维度匹配或无法验证，尝试直接加载...")
                      try:
                           missing_keys, unexpected_keys = target_policy.load_state_dict(policy_state_dict, strict=False)
                           if missing_keys: print(f"    - 警告: 加载时缺少键: {[k for k in missing_keys if not k.startswith('optimizer')]}")
                           if unexpected_keys: print(f"    - 警告: 加载时发现意外键: {unexpected_keys}")
                           print("    ✅ 状态字典加载成功。")
                      except Exception as e:
                           print(f"    ❌ 直接加载状态字典失败: {e}。")

                 if args.resume: # 恢复训练状态
                      runner.global_step = loaded_steps
                      steps_per_iter = getattr(train_cfg.runner, 'num_steps_per_env', 24) * env.num_envs
                      runner.current_learning_iteration = int(loaded_steps / steps_per_iter) if steps_per_iter > 0 else 0
                      total_env_steps = loaded_steps # 更新全局步数
                      print(f"    恢复 Runner 状态: Global Steps={runner.global_step:,}, Approx Iteration={runner.current_learning_iteration}")

        except Exception as e:
            print(f"❌ 环境或训练运行器初始化失败: {str(e)}")
            if args.debug: traceback.print_exc();
            if sim: gym.destroy_sim(sim); gym = None
            if env: env.close(); env = None
            return


        # --- 3. 训练循环 ---
        print("\n--- 3. 开始训练循环 ---")
        max_iterations = getattr(train_cfg.runner, 'max_iterations', 1500)
        # 从顶层配置获取 max_env_steps (如果存在)
        max_env_steps = getattr(train_cfg, 'max_env_steps', 100_000_000) # 检查 train_cfg

        print(f"最大环境步数: {max_env_steps:,}")
        print(f"最大迭代次数: {max_iterations:,}")
        print(f"当前总环境步数: {total_env_steps:,}")
        print(f"当前迭代次数: {runner.current_learning_iteration}")

        start_time_ts = time.time();
        last_save_time_ts = start_time_ts;
        last_log_time_ts = start_time_ts
        if not hasattr(runner, 'current_learning_iteration'): runner.current_learning_iteration = 0

        try:
            while runner.current_learning_iteration < max_iterations and total_env_steps < max_env_steps:
                current_iter = runner.current_learning_iteration
                iter_start_time_ts = time.time()

                # --- 3.1 运行一个学习迭代 ---
                try:
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e): print("\n❌❌❌ CUDA Out of Memory! ❌❌❌");torch.cuda.empty_cache(); raise e
                    elif "tensor a" in str(e) and "tensor b" in str(e) or "mat1 and mat2 shapes cannot be multiplied" in str(e):
                         print("\n\n" + "="*50); print(f"❌ 运行时错误 (张量形状不匹配): {e}"); print("="*50)
                         if hasattr(runner, 'alg') and hasattr(runner.alg, 'actor_critic'): actor = runner.alg.actor_critic.actor; print(f"模型输入层 In: {getattr(actor[0], 'in_features', 'N/A')}, Out: {getattr(actor[0], 'out_features', 'N/A')}")
                         print(f"环境观察维度: Actual={env.obs_buf.shape}, Configured={env.num_observations}"); print("="*50)
                         raise e
                    else: print(f"❌ 训练迭代运行时错误: {str(e)}"); raise e
                except Exception as e: print(f"❌ 训练迭代中发生未知异常: {str(e)}"); raise e

                # --- 3.2 获取统计数据和更新步数 ---
                train_info = runner.get_inference_stats(); train_info = train_info or {}
                new_total_env_steps = runner.global_step
                steps_this_iter = new_total_env_steps - total_env_steps
                total_env_steps = new_total_env_steps

                # --- 3.3 日志记录 ---
                iter_time_sec = time.time() - iter_start_time_ts
                elapsed_time_sec = time.time() - start_time_ts
                elapsed_timedelta = timedelta(seconds=int(elapsed_time_sec))

                if time.time() - last_log_time_ts > 30 or current_iter % 50 == 0:
                     mean_reward = train_info.get('mean_reward', float('nan')); mean_reward = float(mean_reward) if not isinstance(mean_reward, (int, float)) else mean_reward
                     mean_ep_length = train_info.get('mean_episode_length', float('nan')); mean_ep_length = float(mean_ep_length) if not isinstance(mean_ep_length, (int, float)) else mean_ep_length
                     # 直接使用 Runner 的统计数据
                     success_rate = train_info.get('success_rate', 0.0) # Runner 中可能包含此项

                     log_msg = (f"It {current_iter+1:>6}/{max_iterations} | "
                                f"Steps {total_env_steps/1e6:>6.1f}M/{max_env_steps/1e6:.1f}M | "
                                f"Rew {mean_reward:>6.2f} | Len {mean_ep_length:>5.1f} | "
                                f"SR {success_rate:.3f} | iter time {iter_time_sec:.2f}s | Elap {str(elapsed_timedelta)}")
                     print(log_msg)
                     last_log_time_ts = time.time()

                # --- 3.4 保存检查点 ---
                save_freq_iters = getattr(train_cfg.runner, 'save_interval', 50)
                time_based_save = (time.time() - last_save_time_ts) > 900
                is_last_iter = (current_iter + 1 >= max_iterations) or (total_env_steps >= max_env_steps)

                if iter_based_save or time_based_save or is_last_iter:
                     print(f"\n--- Saving Checkpoint (Iteration {current_iter+1}) ---")
                     try:
                         if runner.log_dir: os.makedirs(runner.log_dir, exist_ok=True)
                         save_filename = os.path.join(runner.log_dir, f'model_{current_iter+1}.pt')
                         model_save_path = runner.save(save_filename)
                         if model_save_path:
                             print(f"✅ Checkpoint saved successfully to {model_save_path}")
                         else: print("❌ Runner save returned None.")
                         last_save_time_ts = time.time()
                     except Exception as e: print(f"❌ 保存检查点失败: {str(e)}")

                # --- 移除课程推进逻辑 ---

            # --- End of While Loop ---

        except KeyboardInterrupt: print("\n🛑 训练被用户中断")
        except Exception as train_loop_err:
            print(f"\n❌❌❌ 训练循环中发生未捕获的严重错误: {train_loop_err}")
            traceback.print_exc()

    finally:
        # --- 4. 收尾工作 ---
        print("\n--- 4. 训练结束，执行收尾 ---")
        current_steps = total_env_steps
        try:
            # 保存最终模型
            if 'runner' in locals() and runner and runner.log_dir:
                 final_iter = getattr(runner, 'current_learning_iteration', 'final')
                 final_model_path = os.path.join(runner.log_dir, f'model_{final_iter}.pt')
                 runner.save(final_model_path)
                 print(f"  💾 最终模型保存路径: {final_model_path}")
            # 关闭环境
            if 'env' in locals() and env is not None:
                 env.close()
                 print("  ✅ 环境已关闭。")
        except Exception as e:
            print(f"  ❌ 保存最终状态或关闭环境失败: {str(e)}")
        # 销毁 Sim
        if sim is not None and gym is not None:
            gym.destroy_sim(sim)
            print("  ✅ Simulation 已销毁。")
        print("\n🏁 训练流程结束 🏁")


# ==============================================================================
if __name__ == "__main__":
    args = parse_args()
    if args.seed is None: args.seed = int(time.time() * 1000) % 2**32
    set_seed(args.seed)
    print(f"🎲 使用随机种子: {args.seed}")

    # --- 注册任务 (确保目标任务已注册) ---
    # 假设 G1FullLocomotionEnv 等已在 g1.envs.__init__ 中注册
    print("--- 检查任务注册 ---")
    if hasattr(task_registry, 'task_classes'):
         print(f"Available Tasks: {list(task_registry.task_classes.keys())}")
    else:
         print("⚠️ 无法访问 task_registry.task_classes")

    # --- 运行训练 ---
    train(args) # 调用修改后的 train 函数