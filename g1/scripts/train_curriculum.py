
import argparse
import numpy as np
from datetime import datetime, timedelta
import copy
import inspect
import os
import time
import yaml # For loading curriculum state

# 导入IsaacGym
import isaacgym
from isaacgym import gymapi, gymutil # Import necessary gym modules
import torch

# 导入 G1 相关
import g1.envs # Keep this for task registration trigger
from g1.envs.curriculum.curriculum_manager import CurriculumManager
from g1.envs.curriculum.model_transfer import ModelTransfer
from g1.envs.curriculum.reward_scheduler import RewardScheduler
from g1.envs.configs.curriculum.curriculum_manager_config import CurriculumManagerConfig

# 导入工具函数和注册表
from g1.utils import get_args, task_registry, set_seed
# Need helpers for parsing sim params and updating cfg
from g1.utils.helpers import update_cfg_from_args, class_to_dict, parse_sim_params

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


# --- 解析参数 ---
def parse_args():
    """解析命令行参数"""
    args = get_args()
    parser = argparse.ArgumentParser(description='Train with curriculum', add_help=False)
    # Add args only if not already present from get_args
    if not hasattr(args, 'config_class'):
        parser.add_argument('--config_class', type=str, default='CurriculumManagerConfig', help='课程学习配置类名称')
    if not hasattr(args, 'resume_curriculum'):
        parser.add_argument('--resume_curriculum', type=str, default=None, help='恢复训练的课程状态文件 (YAML)')
    if not hasattr(args, 'debug'):
        parser.add_argument('--debug', action='store_true', default=False, help='启用调试模式，打印更多信息')

    curriculum_args, _ = parser.parse_known_args()
    for key, value in vars(curriculum_args).items():
        setattr(args, key, value)

    if args.headless is None: args.headless = True # Default headless for training
    return args


# --- 加载课程配置 ---
def load_curriculum_config(config_class_name):
    """加载课程学习配置类实例，并转换为 DotDict"""
    try:
        # TODO: Make dynamic based on name if needed
        if config_class_name == 'CurriculumManagerConfig':
            config_obj = CurriculumManagerConfig()
            # Convert class instance attributes to DotDict structure
            config_dict = {}
            for key, value in inspect.getmembers(config_obj):
                 if not key.startswith('_') and not inspect.ismethod(value):
                      if isinstance(value, dict):
                           config_dict[key] = DotDict(value)
                      else:
                           config_dict[key] = value
            # Nest under 'curriculum' and 'output' keys for consistency with original access
            nested_config = DotDict({
                 'curriculum': DotDict({k: v for k, v in config_dict.items() if k.startswith('stage') or k in ['initial_stage', 'initial_sub_stage', 'max_stages', 'max_sub_stages', 'success_threshold', 'evaluation_window', 'min_steps_between_eval', 'model_transfer']}),
                 'output': config_dict.get('output', DotDict()), # Handle missing output
                 'max_env_steps': config_dict.get('max_env_steps', 100_000_000)
            })
            print(f"✅ 成功加载课程学习配置: {config_class_name}")
            return nested_config
        else:
            raise ValueError(f"未知的配置类名称: {config_class_name}")
    except Exception as e:
        print(f"❌ 无法加载课程学习配置 {config_class_name}: {str(e)}")
        if args.debug: import traceback; traceback.print_exc()
        return None


# --- 获取任务信息 ---
def get_task_info(curriculum_config, stage):
    """根据课程阶段获取任务名称和该阶段的特定参数"""
    stage_key = f'stage{stage}'
    if stage_key not in curriculum_config.curriculum:
        raise ValueError(f"未找到阶段 {stage} 的配置 (Key: '{stage_key}')")

    stage_config = curriculum_config.curriculum[stage_key] # Access as attribute/key
    task_name = stage_config.get('env_class')
    if not task_name:
        raise ValueError(f"阶段 {stage} 配置中未指定 'env_class'")
    return {'task_name': task_name, 'stage_params': dict(stage_config)}


# --- 验证任务兼容性 ---
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


# --- 更新环境奖励 ---
def update_env_rewards(env, reward_scheduler, stage, sub_stage):
    """根据课程阶段使用 RewardScheduler 更新环境实例的奖励系数"""
    print(f"--- Updating reward scales for Stage {stage}.{sub_stage} ---")
    if not hasattr(env, 'cfg') or not hasattr(env.cfg, 'rewards') or not hasattr(env.cfg.rewards, 'scales'):
        print("  ⚠️ Cannot update rewards: env.cfg.rewards.scales structure missing.")
        return False
    try:
        reward_scales_dict = reward_scheduler.get_reward_scales(stage, sub_stage)
        scales_target = env.cfg.rewards.scales # Should be an object or dict
        updated_count = 0
        applied_scales = {}

        # Get available scale names from the target object/dict
        if isinstance(scales_target, dict): defined_in_env = list(scales_target.keys())
        else: defined_in_env = [attr for attr in dir(scales_target) if not attr.startswith('_') and not callable(getattr(scales_target, attr))]

        for reward_name, scale_value in reward_scales_dict.items():
            if reward_name in defined_in_env:
                try:
                    if isinstance(scales_target, dict): scales_target[reward_name] = scale_value
                    else: setattr(scales_target, reward_name, scale_value)
                    applied_scales[reward_name] = f"{scale_value:.4f}" # Format for printing
                    updated_count += 1
                except Exception as e: print(f"  ⚠️ Error updating scale '{reward_name}': {e}")
            # else: print(f"  - Scale '{reward_name}' not found in env config scales.") # Optional warning

        if updated_count > 0:
            print(f"  ✅ Updated {updated_count} reward scales. Applied: {applied_scales}")
            # Re-prepare reward functions to use new scales
            if hasattr(env, '_prepare_reward_function'):
                 print("    - Calling env._prepare_reward_function()")
                 env._prepare_reward_function()
            return True
        else:
            print(f"  ⚠️ No matching reward scales found to update for Stage {stage}.{sub_stage}.")
            print(f"    - Available scales in env config: {defined_in_env}")
            print(f"    - Scales provided by scheduler: {list(reward_scales_dict.keys())}")
            return False
    except Exception as e:
        print(f"❌ Error during reward update: {str(e)}")
        if args.debug: import traceback; traceback.print_exc()
        return False


# ==============================================================================
# 主训练函数
# ==============================================================================
def train_curriculum(args):
    """课程学习训练主函数"""
    print("="*50); print("🚀 开始 G1 课程学习训练 🚀"); print("="*50)

    # --- 全局 Gym 和 Sim ---
    gym = None
    sim = None

    try:
        # --- 1. 初始化核心组件 ---
        print("\n--- 1. 初始化核心组件 ---")
        curriculum_config = load_curriculum_config(args.config_class)
        if curriculum_config is None: return

        curriculum_mgr = CurriculumManager(curriculum_config)
        print(f"✅ 课程管理器创建成功 (Output: {curriculum_mgr.output_dir})")

        device_arg = getattr(args, 'rl_device', 'cuda:0')
        model_transfer_cfg = curriculum_config.curriculum.model_transfer
        if not isinstance(model_transfer_cfg, DotDict): model_transfer_cfg = DotDict(model_transfer_cfg)
        model_transfer_cfg.device = device_arg
        model_transfer = ModelTransfer(model_transfer_cfg)
        print(f"✅ 模型迁移工具创建成功")

        reward_scheduler = RewardScheduler(curriculum_config)
        print(f"✅ 奖励调度器创建成功")

        # --- !!! 在脚本开始时创建 Gym 和 Sim !!! ---
        print("\n--- 初始化 Isaac Gym 和 Simulation ---")
        gym = gymapi.acquire_gym()
        # 解析 sim_params
        try:
            # Need initial config to get sim params structure
            initial_stage = curriculum_mgr.current_stage # Get initial stage
            initial_task_info = get_task_info(curriculum_config, initial_stage)
            initial_env_cfg_cls = task_registry.env_cfgs[initial_task_info['task_name']] # Get class
            initial_env_cfg = initial_env_cfg_cls() # Create instance
            sim_params_dict = {"sim": class_to_dict(initial_env_cfg.sim)}
            sim_params = parse_sim_params(args, sim_params_dict) # Parse into SimParams object
        except Exception as e:
             print(f"❌ 获取初始 Sim 参数失败: {e}")
             if args.debug: import traceback; traceback.print_exc()
             return

        physics_engine = gymapi.SIM_PHYSX
        sim_device_type, sim_device_id = gymutil.parse_device_str(args.sim_device)
        graphics_device_id = sim_device_id if not args.headless else -1

        sim = gym.create_sim(sim_device_id, graphics_device_id, physics_engine, sim_params)
        if sim is None: raise RuntimeError("Failed to create sim!")
        print(f"✅ Gym 和 Sim 创建成功 (Sim Handle: {sim})")
        # ---------------------------------------------

        # --- 2. 恢复状态 (如果需要) ---
        print("\n--- 2. 恢复状态检查 ---")
        loaded_model_path = None
        # ... (恢复逻辑保持不变, 会设置 args.checkpoint 和 args.resume) ...
        if args.resume_curriculum:
            print(f"尝试从课程状态文件恢复: {args.resume_curriculum}")
            success, loaded_model_path = curriculum_mgr.load_curriculum_state(args.resume_curriculum)
            if success:
                print(f"✅ 已恢复课程状态。当前阶段: {curriculum_mgr.current_stage}.{curriculum_mgr.current_sub_stage}")
                if loaded_model_path:
                     print(f"  - 从课程状态获取模型路径: {loaded_model_path}")
                     args.checkpoint = loaded_model_path
                     args.resume = True
                else: print("  - 课程状态中未找到模型路径。")
            else:
                print(f"❌ 无法恢复课程状态，将从头开始。")
                args.resume_curriculum = None; args.checkpoint = None; args.resume = False
        elif args.checkpoint or args.resume:
             print("使用命令行 --resume 或 --checkpoint。")
             loaded_model_path = args.checkpoint


        # --- 3. 设置初始阶段和任务 ---
        print("\n--- 3. 设置初始阶段和任务 ---")
        stage, sub_stage = curriculum_mgr.get_current_stage_info()
        print(f"当前课程阶段: {stage}.{sub_stage}")

        task_name = None; stage_params = None
        try:
            task_info = get_task_info(curriculum_config, stage)
            task_name = task_info.get('task_name')
            stage_params = task_info.get('stage_params')
            if not task_name: raise ValueError("获取的 task_name 为空")
            print(f"获取阶段 {stage} 任务信息: Task='{task_name}', Params={list(stage_params.keys()) if stage_params else 'N/A'}")

            if hasattr(task_registry, 'task_classes') and isinstance(task_registry.task_classes, dict):
                print("--- DEBUG train_curriculum: 可用任务:", list(task_registry.task_classes.keys()))
            if not validate_task_compatibility(task_name): return
            print(f"✅ 任务 '{task_name}' 验证通过。")
        except Exception as e:
            print(f"❌ 获取或验证任务配置失败: {str(e)}")
            if args.debug: import traceback; traceback.print_exc(); return

        # Update args
        args.task = task_name
        cmd_line_num_envs = args.num_envs # Store command line value before override
        args.num_envs = stage_params.get('num_envs', 4096)
        if cmd_line_num_envs is not None: args.num_envs = cmd_line_num_envs
        print(f"  将使用的环境数量: {args.num_envs}")


        # --- 4. 创建初始环境和 Runner ---
        print("\n--- 4. 创建初始环境和 Runner ---")
        env, env_cfg = None, None
        runner, train_cfg = None, None
        policy_state_dict = None
        loaded_env_dims = None

        try:
            # 4.1 Load base configs
            base_env_cfg_cls = task_registry.env_cfgs[args.task] # Get class
            base_train_cfg_cls = task_registry.train_cfgs[args.task]
            env_cfg = base_env_cfg_cls() # Create instance
            train_cfg = base_train_cfg_cls()
            print(f"  加载任务 '{args.task}' 的基础 env_cfg (类型: {type(env_cfg)}) 和 train_cfg (类型: {type(train_cfg)})")

            # 4.2 Merge curriculum/stage params
            if not hasattr(env_cfg, 'curriculum'): env_cfg.curriculum = DotDict()
            elif not isinstance(env_cfg.curriculum, (dict, DotDict)): env_cfg.curriculum = DotDict() # Force DotDict
            env_cfg.curriculum.stage = stage
            env_cfg.curriculum.sub_stage = sub_stage
            stage_params_attr = f'stage{stage}_params'
            setattr(env_cfg.curriculum, stage_params_attr, DotDict(stage_params)) # Use setattr
            env_cfg.env.num_envs = args.num_envs # Ensure num_envs matches args
            print(f"  已将阶段 {stage} 参数注入到 env_cfg.curriculum")

            # 4.3 Create environment instance (pass gym, sim, sim_params)
            print(f"  准备创建环境实例...")
            env, env_cfg = task_registry.make_env(
                name=args.task, args=args, env_cfg=env_cfg,
                gym_handle=gym, sim_handle=sim, sim_params=sim_params
            )

            # 4.4 Update rewards
            update_env_rewards(env, reward_scheduler, stage, sub_stage)

            # 4.5 Prepare for checkpoint loading (load data)
            if args.checkpoint and os.path.exists(args.checkpoint):
                 print(f"  准备加载检查点: {args.checkpoint}")
                 policy_state_dict, loaded_env_dims, loaded_steps, loaded_stage = model_transfer.load_checkpoint(args.checkpoint)
                 if policy_state_dict is None:
                      print("  ❌ 加载检查点失败，将随机初始化模型。")
                      args.checkpoint = None; args.resume = False
                 else:
                      print(f"  ✅ 检查点加载成功 (来自阶段 {loaded_stage} @ {loaded_steps:,} 步)")
                      # Dimension check happens *after* runner creation
            else:
                 if args.checkpoint: print(f"  ⚠️ 指定的检查点文件不存在: {args.checkpoint}")
                 print("  将随机初始化模型。")
                 args.checkpoint = None; args.resume = False


            # 4.6 Create Runner
            print(f"  准备创建 Runner (Task: {args.task})...")
            # Pass the train_cfg instance created from registry
            train_cfg.runner.resume = args.resume # Ensure resume flag is correct
            runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)

            # 4.7 Execute model loading/transfer *after* runner is created
            if policy_state_dict: # If loaded from checkpoint
                 target_policy = runner.alg.actor_critic
                 current_env_cfg_for_dims = env_cfg # Config for current env
                 old_env_cfg_for_dims = DotDict({'env': DotDict(loaded_env_dims)}) # Create temp cfg for old dims

                 # Check dimensions before transfer
                 if loaded_env_dims['num_observations'] != env.num_observations or \
                    loaded_env_dims['num_actions'] != env.num_actions:
                     print("  ⚠️ 执行模型迁移 (维度不匹配)...")
                     model_transfer.transfer_policy(
                          old_policy_state_dict=policy_state_dict,
                          old_cfg=old_env_cfg_for_dims,
                          new_cfg=current_env_cfg_for_dims,
                          target_policy=target_policy
                     )
                 else:
                      print("  检查点维度匹配，直接加载状态字典...")
                      try:
                           target_policy.load_state_dict(policy_state_dict)
                           print("    ✅ 状态字典加载成功。")
                      except Exception as e:
                           print(f"    ❌ 直接加载状态字典失败: {e}。 Runner 可能已内部加载或需重新训练。")

                 # Update runner step/iteration counters if resuming
                 if args.resume:
                      if hasattr(runner, 'global_step'): runner.global_step = loaded_steps
                      if hasattr(runner, 'current_learning_iteration'):
                          steps_per_iter = getattr(train_cfg.runner, 'num_steps_per_env', 24) * env.num_envs
                          runner.current_learning_iteration = int(loaded_steps / steps_per_iter) if steps_per_iter > 0 else 0
                          print(f"    恢复 Runner 迭代计数到约: {runner.current_learning_iteration}")

            # Set curriculum manager's model path if resuming
            if args.checkpoint: curriculum_mgr.set_latest_model_path(args.checkpoint)

        except Exception as e:
            print(f"❌ 环境或训练运行器初始化失败: {str(e)}")
            if args.debug: import traceback; traceback.print_exc();
            # Cleanup Sim if initialization failed
            if sim: gym.destroy_sim(sim)
            return

        # --- 5. 训练循环 ---
        print("\n--- 5. 开始训练循环 ---")
        total_env_steps = getattr(runner, 'global_step', 0)

        # --- !!! 检查 max_iterations 和 max_env_steps !!! ---
        if not hasattr(train_cfg, 'runner') or not hasattr(train_cfg.runner, 'max_iterations') or not isinstance(
                train_cfg.runner.max_iterations, int):
            print(f"⚠️ 警告: train_cfg.runner.max_iterations 无效或缺失。使用默认值 1500。")
            max_iterations = 1500
        else:
            max_iterations = train_cfg.runner.max_iterations

        if not isinstance(curriculum_config.max_env_steps, int):
            print(f"⚠️ 警告: curriculum_config.max_env_steps 无效或缺失。使用默认值 100,000,000。")
            max_env_steps = 100_000_000
        else:
            max_env_steps = curriculum_config.max_env_steps
        # --- 结束检查 ---

        print(f"最大环境步数: {max_env_steps:,}")
        print(f"最大迭代次数: {max_iterations:,}")  # 打印出来确认
        if total_env_steps > 0: print(f"从检查点恢复，当前总环境步数: {total_env_steps:,}")

        start_time_ts = time.time();
        last_save_time_ts = start_time_ts;
        last_log_time_ts = start_time_ts
        if not hasattr(runner, 'current_learning_iteration'): runner.current_learning_iteration = 0


        try:
            while runner.current_learning_iteration < train_cfg.runner.max_iterations and total_env_steps < max_env_steps:
                current_iter = runner.current_learning_iteration
                iter_start_time_ts = time.time()

                # --- 5.1 运行一个学习迭代 ---
                # try: runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
                # except RuntimeError as e: # ... (错误处理保持不变) ...
                #     if "CUDA out of memory" in str(e): print("\n❌❌❌ CUDA Out of Memory! ❌❌❌");torch.cuda.empty_cache(); raise e
                #     elif "tensor a" in str(e) and "tensor b" in str(e): print(f"❌ 运行时错误 (张量形状不匹配): {e}"); raise e
                #     else: print(f"❌ 训练迭代运行时错误: {str(e)}"); continue
                # except Exception as e: print(f"❌ 训练迭代中发生未知异常: {str(e)}"); continue # Catch other potential errors

                try:
                    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
                except RuntimeError as e:  # ... (错误处理保持不变) ...
                    if "CUDA out of memory" in str(e):
                        print("\n❌❌❌ CUDA Out of Memory! ❌❌❌")
                        torch.cuda.empty_cache()
                        raise e
                    elif "tensor a" in str(e) and "tensor b" in str(
                            e) or "mat1 and mat2 shapes cannot be multiplied" in str(e):
                        print("\n\n" + "=" * 50)
                        print(f"❌ 运行时错误 (张量形状不匹配): {e}")
                        print("=" * 50)
                        # 打印核心模型信息
                        if hasattr(runner, 'alg') and hasattr(runner.alg, 'actor_critic'):
                            actor = runner.alg.actor_critic.actor
                            if hasattr(actor, '0'):
                                print(f"模型第一层输入维度: {actor[0].in_features}")
                                print(f"模型第一层输出维度: {actor[0].out_features}")
                        # 打印环境观察维度
                        print(f"环境观察维度: {env.obs_buf.shape}")
                        print(f"环境配置观察维度: {env.num_observations}")
                        print("=" * 50)
                        # 立即终止程序
                        import sys
                        sys.exit(1)
                    else:
                        print(f"❌ 训练迭代运行时错误: {str(e)}")
                        # 立即终止程序
                        import sys
                        sys.exit(1)
                except Exception as e:
                    print(f"❌ 训练迭代中发生未知异常: {str(e)}")
                    # 立即终止程序
                    import sys
                    sys.exit(1)



                # --- 5.2 获取统计数据和更新步数 ---
                train_info = runner.current_statistics
                steps_this_iter = runner.num_steps_per_env * env.num_envs
                total_env_steps += steps_this_iter


                # --- 5.3 日志记录 ---
                iter_time_sec = time.time() - iter_start_time_ts
                elapsed_time_sec = time.time() - start_time_ts
                elapsed_timedelta = timedelta(seconds=elapsed_time_sec)

                if time.time() - last_log_time_ts > 30: # Log every 30 seconds
                     mean_reward = train_info.get('Mean/reward', float('nan'))
                     mean_ep_length = train_info.get('Mean/episode_length', float('nan'))
                     success_rate = train_info.get('success_rate', 0.0) # Assumes runner calculates this

                     log_msg = (f"S{stage}.{sub_stage} | It {current_iter+1:>5}/{train_cfg.runner.max_iterations} | "
                                f"Steps {total_env_steps/1e6:>6.1f}M/{max_env_steps/1e6:.1f}M | "
                                f"Rew {mean_reward:>6.2f} | Len {mean_ep_length:>5.1f} | "
                                f"SR {success_rate:.3f} | iter time {iter_time_sec:.2f}s | total time {str(elapsed_timedelta).split('.')[0]}")
                     print(log_msg)
                     last_log_time_ts = time.time()

                     curriculum_mgr.update_statistics(success_rate, mean_reward, steps_this_iter)


                # --- 5.4 保存检查点 ---
                time_based_save = (time.time() - last_save_time_ts) > 900
                iter_based_save = (current_iter + 1) % train_cfg.runner.save_interval == 0
                if iter_based_save or time_based_save:
                     print(f"\n--- Saving Checkpoint (Iteration {current_iter+1}) ---")
                     try:
                         model_save_path = runner.save(os.path.join(runner.log_dir, f'model_{current_iter+1}.pt'))
                         curriculum_mgr.set_latest_model_path(model_save_path)
                         curriculum_mgr.save_curriculum_state(total_env_steps)
                         print(f"✅ Checkpoint saved successfully.")
                         last_save_time_ts = time.time()
                     except Exception as e: print(f"❌ 保存检查点或课程状态失败: {str(e)}")


                # --- !!! 临时修改：强制进入 Stage 2 !!! ---
                force_advance_to_stage2 = False
                num_iters_to_test_stage1 = 10
                if stage == 1 and current_iter + 1 >= num_iters_to_test_stage1:
                     print("\n" + "="*15 + f" 强制进阶测试 (Iteration {current_iter+1}) " + "="*15)
                     print(f"达到 Stage 1 的 {num_iters_to_test_stage1} 次迭代，强制触发进入 Stage 2...")
                     force_advance_to_stage2 = True
                # --- 结束临时修改 ---


                # --- 5.5 检查并推进课程 ---
                should_advance = curriculum_mgr.should_advance_curriculum(total_env_steps)
                if force_advance_to_stage2 or should_advance:
                    if not force_advance_to_stage2:
                         print("\n" + "="*20 + " CURRICULUM ADVANCEMENT CHECK " + "="*20)
                         print(f"条件满足 (SR >= Threshold)，准备从阶段 {stage}.{sub_stage} 推进...")

                    # 5.5.1 保存当前模型
                    print("  强制保存当前模型...")
                    # 在阶段切换前，确保有一个可用的模型文件
                    print("  强制保存当前模型...")
                    try:
                        model_save_path = runner.save(
                            os.path.join(runner.log_dir, f'model_{current_iter + 1}_pre_transition.pt'))
                        if model_save_path is None or not os.path.exists(model_save_path):
                            raise ValueError("模型保存返回无效路径")
                        curriculum_mgr.set_latest_model_path(model_save_path)
                        print(f"  ✅ 模型已保存: {model_save_path}")
                    except Exception as e:
                        print(f"  ❌ 课程推进前保存模型失败: {str(e)}")
                        # 尝试寻找最近的模型文件
                        model_save_path = None
                        log_dir = getattr(runner, 'log_dir', None)
                        if log_dir and os.path.exists(log_dir):
                            model_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if
                                           f.startswith('model_') and f.endswith('.pt')]
                            if model_files:
                                # 按修改时间排序，获取最新的
                                model_save_path = sorted(model_files, key=os.path.getmtime)[-1]
                                print(f"  找到备选模型: {model_save_path}")


                    # 检查是否有可用的模型文件
                    if model_save_path is None or not os.path.exists(model_save_path):
                        print("  ⚠️ 无可用模型文件。将使用随机初始化策略。")
                        old_policy_state_dict = None
                        old_env_dims = {'num_observations': env.num_observations, 'num_actions': env.num_actions}
                    else:
                        print(f"  准备从 {model_save_path} 加载模型...")
                        old_policy_state_dict, old_env_dims, _, _ = model_transfer.load_checkpoint(model_save_path)
                        if old_policy_state_dict is None:
                            print("  ⚠️ 模型加载失败。将使用随机初始化策略。")
                            old_env_dims = {'num_observations': env.num_observations, 'num_actions': env.num_actions}

                        # 只有在model_save_path不为None时才尝试加载
                    if model_save_path is not None:
                        print("  准备加载/迁移模型到新 Runner...")
                        old_policy_state_dict, old_env_dims, _, _ = model_transfer.load_checkpoint(model_save_path)
                        if old_policy_state_dict is None:
                            print(f"  ⚠️ 无法从 {model_save_path} 加载模型，将初始化新模型")
                    else:
                        print("  ⚠️ 跳过模型加载/迁移步骤")
                        old_policy_state_dict = None
                        old_env_dims = None

                    # 5.5.2 推进/设置课程状态
                    target_stage = 2; target_sub_stage = 1
                    if not force_advance_to_stage2:
                         new_stage_str = curriculum_mgr.advance_curriculum(total_env_steps)
                         new_stage, new_sub_stage = map(int, new_stage_str.split('.'))
                    else:
                         print(f"  手动设置 Curriculum Manager 到 Stage {target_stage}.{target_sub_stage}")
                         curriculum_mgr.current_stage = target_stage
                         curriculum_mgr.current_sub_stage = target_sub_stage
                         curriculum_mgr.reset_evaluation()
                         # 记录强制转换
                         curriculum_mgr.stage_history.append({
                             "old_stage": f"{stage}.{sub_stage}", "new_stage": f"{target_stage}.{target_sub_stage}",
                             "total_env_steps": total_env_steps, "steps_in_stage": curriculum_mgr.total_env_steps_in_stage,
                             "success_rate_at_transition": curriculum_mgr.get_smoothed_success_rate(),
                             "average_reward_at_transition": curriculum_mgr.get_average_reward(),
                             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                             "model_checkpoint": model_save_path, "forced_transition": True
                         })
                         curriculum_mgr._save_curriculum_progress_plot()
                         curriculum_mgr.save_curriculum_state(total_env_steps)
                         new_stage, new_sub_stage = target_stage, target_sub_stage

                    print(f"🎓 课程已推进/设置为新阶段: {new_stage}.{new_sub_stage}")

                    # 5.5.3 阶段切换处理
                    try:
                         print("  开始阶段切换流程...")
                         new_task_info = get_task_info(curriculum_config, new_stage)
                         new_task_name = new_task_info['task_name']
                         new_stage_params = new_task_info['stage_params']
                         print(f"  新阶段任务: '{new_task_name}'")

                         new_base_env_cfg_cls = task_registry.env_cfgs[new_task_name]
                         new_base_train_cfg_cls = task_registry.train_cfgs[new_task_name]
                         new_env_cfg = new_base_env_cfg_cls()
                         new_train_cfg = new_base_train_cfg_cls()

                         # Merge params
                         if not hasattr(new_env_cfg, 'curriculum'): new_env_cfg.curriculum = DotDict()
                         new_env_cfg.curriculum.stage = new_stage
                         new_env_cfg.curriculum.sub_stage = new_sub_stage
                         setattr(new_env_cfg.curriculum, f'stage{new_stage}_params', DotDict(new_stage_params))

                         # Update args
                         args.task = new_task_name
                         new_num_envs = new_stage_params.get('num_envs', 1024)
                         if cmd_line_num_envs is not None: new_num_envs = cmd_line_num_envs
                         args.num_envs = new_num_envs
                         new_env_cfg.env.num_envs = new_num_envs
                         print(f"  新环境数量: {args.num_envs}")

                         # --- !!! 清理旧环境和 Runner !!! ---
                         print("  清理旧环境逻辑实例和 Runner...")

                         # 在"5.5.3 阶段切换处理"部分
                         print("  清理旧环境逻辑实例和 Runner...")
                         if env is not None:
                             env.close()  # 这会调用destroy_viewer
                             print("  - 环境已关闭（查看器已销毁）")
                             del env
                             env = None

                         if runner is not None:
                             del runner
                             runner = None

                         # 清理GPU内存
                         torch.cuda.empty_cache()

                         # 重要：解析新环境的sim参数
                         new_sim_params_dict = {"sim": class_to_dict(new_env_cfg.sim)}
                         new_sim_params = parse_sim_params(args, new_sim_params_dict)

                         # 重新创建sim
                         print("  重新创建Simulation以适应新环境...")
                         if sim is not None:
                             gym.destroy_sim(sim)
                             sim = None

                         sim = gym.create_sim(sim_device_id, graphics_device_id, physics_engine, new_sim_params)
                         if sim is None:
                             raise RuntimeError("Failed to create new simulation!")
                         print("  ✅ 新的Sim创建成功")

                         # Create new environment (pass existing gym, sim, sim_params)
                         print("  创建新环境实例...")
                         env, env_cfg = task_registry.make_env(
                             name=args.task, args=args, env_cfg=new_env_cfg,
                             gym_handle=gym, sim_handle=sim, sim_params=sim_params
                         )

                         # Update rewards
                         update_env_rewards(env, reward_scheduler, new_stage, new_sub_stage)

                         # Load old model state for transfer
                         print("  准备加载/迁移模型到新 Runner...")
                         old_policy_state_dict, old_env_dims, _, _ = model_transfer.load_checkpoint(model_save_path)
                         if old_policy_state_dict is None: raise RuntimeError(f"无法从 {model_save_path} 加载模型！")

                         # Create new Runner
                         # 创建新Runner
                         print("  创建新 Runner...")
                         new_train_cfg.runner.resume = (old_policy_state_dict is not None)  # 只在有模型时恢复
                         args.resume = (old_policy_state_dict is not None)
                         args.checkpoint = model_save_path  # 即使为None也没关系，runner会处理

                         runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args,
                                                                           train_cfg=new_train_cfg)

                         # 执行模型迁移或初始化
                         if old_policy_state_dict is not None:
                             target_policy = runner.alg.actor_critic
                             old_env_cfg_for_dims = DotDict({'env': DotDict(old_env_dims)})
                             current_env_cfg_for_dims = env_cfg

                             print("  执行模型状态迁移/加载...")
                             # 在train_curriculum.py中，修改模型保存逻辑（约第687行）
                             try:
                                 # 确保日志目录存在
                                 os.makedirs(runner.log_dir, exist_ok=True)

                                 model_save_path = runner.save(
                                     os.path.join(runner.log_dir, f'model_{current_iter + 1}_pre_transition.pt'))
                                 if model_save_path is None or not os.path.exists(model_save_path):
                                     # 添加明确的保存路径
                                     model_save_path = os.path.join(runner.log_dir,
                                                                    f'model_{current_iter + 1}_pre_transition.pt')
                                     print(f"警告: runner.save返回None，使用显式路径: {model_save_path}")

                                     # 再次确保目录存在
                                     os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

                                     # 手动保存模型
                                     if hasattr(runner, 'alg') and hasattr(runner.alg, 'actor_critic'):
                                         state_dict = {
                                             'model_state_dict': runner.alg.actor_critic.state_dict(),
                                             'optimizer_state_dict': runner.alg.optimizer.state_dict() if hasattr(
                                                 runner.alg, 'optimizer') else None,
                                             'iter': current_iter,
                                             'env_dims': {
                                                 'num_observations': env.num_observations,
                                                 'num_privileged_obs': env.num_privileged_obs if hasattr(env,
                                                                                                         'num_privileged_obs') else None,
                                                 'num_actions': env.num_actions,
                                             },
                                             'stage': stage,
                                             'total_steps': total_env_steps
                                         }
                                         # 打印路径信息进行调试
                                         print(f"保存模型到: {model_save_path}")
                                         print(f"目录存在?: {os.path.exists(os.path.dirname(model_save_path))}")

                                         torch.save(state_dict, model_save_path)
                                         print(f"  ✅ 手动保存模型到: {model_save_path}")
                                 curriculum_mgr.set_latest_model_path(model_save_path)
                                 print(f"  ✅ 模型已保存: {model_save_path}")
                             except Exception as e:
                                 print(f"  ❌ 课程推进前保存模型失败: {str(e)}")
                                 import traceback
                                 traceback.print_exc()
                                 # 查找替代模型...
                         else:
                             print("  没有旧模型可加载，使用随机初始化策略")

                         # 设置计数器
                         runner.global_step = total_env_steps
                         runner.current_learning_iteration = current_iter + 1
                         print(f"    Runner 状态已更新/恢复。")

                         print(f"✅ 阶段切换成功完成！")

                    except Exception as e:
                         print(f"❌❌❌ 阶段切换失败: {str(e)} ❌❌❌")
                         if args.debug: import traceback; traceback.print_exc()
                         print("无法安全恢复，停止训练。")
                         raise e # Re-raise

                    # Update stage tracking vars
                    stage = new_stage
                    sub_stage = new_sub_stage

                    # 在强制阶段转换部分（约第700行）
                    print("\n--- 详细调试信息: 阶段转换 ---")
                    print(f"当前目录: {os.getcwd()}")
                    print(f"Runner日志目录: {runner.log_dir}")
                    print(f"目录是否存在: {os.path.exists(runner.log_dir)}")
                    print(
                        f"环境统计: num_envs={env.num_envs}, num_observations={env.num_observations}, num_actions={env.num_actions}")
                    if hasattr(env, 'num_privileged_obs'):
                        print(f"特权观察维度: {env.num_privileged_obs}")
                    print(f"阶段信息: 当前={stage}.{sub_stage}, 目标={target_stage}.{target_sub_stage}")
                    print("--- 详细调试结束 ---\n")

            # --- End of Advancement Check ---

        # --- End of While Loop ---

        except KeyboardInterrupt: print("\n🛑 训练被用户中断")
        # except Exception as train_loop_err: print(f"\n❌❌❌ 训练循环错误: {train_loop_err}")

        except Exception as train_loop_err:
            import traceback  # 导入 traceback 模块
            print(f"\n❌❌❌ 训练循环错误: {train_loop_err}")
            print("--- 完整错误堆栈 (Traceback) ---")
            traceback.print_exc()  # 打印详细错误信息
            print("-------------------------------")
            # 可以选择在这里退出程序，以便清晰地看到错误
            import sys
            sys.exit(1)


    finally:
        # --- 6. 收尾工作 ---
        print("\n--- 6. 训练结束，执行收尾 ---")
        current_steps = getattr(runner, 'global_step', total_env_steps)
        try:
            if 'curriculum_mgr' in locals() and curriculum_mgr: curriculum_mgr.save_curriculum_state(current_steps); print(f"  💾 最终课程状态已保存。")
            if 'runner' in locals() and runner: final_iter = getattr(runner, 'current_learning_iteration', 'final'); final_model_path = os.path.join(runner.log_dir, f'model_{final_iter}.pt'); runner.save(final_model_path); print(f"  💾 最终模型保存路径: {final_model_path}")
            if env is not None: env.close() # Close viewer if open
        except Exception as e: print(f"  ❌ 保存最终状态失败: {str(e)}")
        # --- !!! 销毁 Sim !!! ---
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
    train_curriculum(args)
