# # play_curriculum.py
# import sys
# import os
# import isaacgym
# import torch
# import numpy as np
#
# # --- 修改导入路径以匹配你的项目结构 ---
# # 假设 G1_ROOT_DIR 指向 g1_gym 目录
# G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 推断项目根目录
# sys.path.append(G1_ROOT_DIR) # 将项目根目录添加到 Python 路径
#
# # 从你的项目中导入必要的模块
# import g1.envs # 确保环境和任务被注册
# from g1.utils import get_args, task_registry, set_seed # 导入你的 get_args, task_registry, set_seed
# # from g1.utils.helpers import export_policy_as_jit # 假设你有类似的导出函数
# # --- 结束修改导入路径 ---
#
# # --- Helper function to load a specific stage config (optional but cleaner) ---
# # You might need a way to load the specific config object used for training that stage,
# # especially if StageXLocomotionConfig has structure beyond what's in LeggedRobotCfg.
# # For now, we'll rely on task_registry.get_cfgs and override parameters.
#
#
# def play_curriculum(args):
#     """加载指定任务的模型并在环境中运行进行评估"""
#
#     # --- 1. 配置加载和参数覆盖 ---
#     print(f"准备加载任务 '{args.task}' 的配置...")
#     try:
#         # 从 task_registry 获取与该任务关联的基础配置
#         # 注意：这获取的是注册时的 *基础* 配置，可能不包含课程合并的参数
#         # 但对于 play 模式，我们通常使用修改后的基础配置
#         env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
#     except ValueError:
#         print(f"错误: 任务 '{args.task}' 未在 task_registry 中注册。")
#         print(f"可用任务: {list(task_registry.task_classes.keys())}")
#         return
#     except Exception as e:
#          print(f"加载任务 '{args.task}' 的配置时出错: {e}")
#          return
#
#
#     print("为测试模式覆盖配置参数...")
#     # 覆盖环境配置以进行测试
#     env_cfg.env.num_envs = min(getattr(env_cfg.env, 'num_envs', 64), args.num_envs if args.num_envs is not None else 64) # 减少环境数量以便观察
#     if hasattr(env_cfg, 'terrain'): # 检查是否有 terrain 配置
#         env_cfg.terrain.num_rows = 5    # 减少地形复杂度 (如果使用地形)
#         env_cfg.terrain.num_cols = 5
#         env_cfg.terrain.curriculum = False # 关闭地形课程
#     if hasattr(env_cfg, 'noise'): # 检查是否有 noise 配置
#         env_cfg.noise.add_noise = False   # 关闭观测噪声
#     if hasattr(env_cfg, 'domain_rand'): # 检查是否有 domain_rand 配置
#         env_cfg.domain_rand.randomize_friction = False # 关闭摩擦随机化
#         env_cfg.domain_rand.push_robots = False      # 关闭机器人推动
#
#     env_cfg.env.test = True # 设置测试标志 (如果你的环境类使用它)
#
#
#     # --- 2. 创建环境 ---
#     print(f"创建环境 '{args.task}' (数量: {env_cfg.env.num_envs})...")
#     try:
#         # 使用修改后的 env_cfg 创建环境
#         # 注意：我们不传入课程合并的配置，因为 play 通常在固定配置下进行
#         env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
#         print("环境创建成功。")
#     except Exception as e:
#         print(f"创建环境时出错: {e}")
#         import traceback
#         traceback.print_exc()
#         return
#
#     obs = env.get_observations()
#
#     # --- 3. 加载策略 ---
#     print("加载策略模型...")
#     # 设置加载检查点所需的训练配置参数
#     # train_cfg.runner.resume = True # make_alg_runner 会根据 args.checkpoint 处理
#     # train_cfg.runner.load_run = args.load_run # 由 args 控制
#     # train_cfg.runner.checkpoint = args.checkpoint # 由 args 控制
#
#     # 创建 Runner (主要目的是为了获取策略加载功能和推理接口)
#     # 注意：这里 env_cfg 已经是修改过的测试配置
#     try:
#         # 不需要训练配置 train_cfg，因为我们只加载模型
#         # 但 make_alg_runner 可能需要它，或者我们可以只创建策略对象
#         # 为了与样板一致，我们创建 runner
#         # 确保 args.checkpoint 指向你要加载的模型文件
#         if not args.checkpoint:
#              print("错误：请使用 --checkpoint 指定要加载的模型文件路径。")
#              env.close()
#              return
#
#         print(f"使用检查点: {args.checkpoint}")
#         # 创建 Runner，它内部会根据 args.checkpoint 加载模型
#         ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
#
#         # 获取用于推理的策略函数
#         policy = ppo_runner.get_inference_policy(device=env.device)
#         print("策略加载成功。")
#
#     except FileNotFoundError:
#          print(f"错误: 检查点文件未找到: {args.checkpoint}")
#          env.close()
#          return
#     except Exception as e:
#          print(f"创建 Runner 或加载策略时出错: {e}")
#          import traceback
#          traceback.print_exc()
#          env.close()
#          return
#
#
#     # --- 4. 导出策略 (可选) ---
#     if args.export_policy:
#         export_path = os.path.join(os.path.dirname(args.checkpoint), 'exported') # 保存在模型同目录下的 exported 文件夹
#         # 假设你的 export_policy_as_jit 函数已导入或定义
#         try:
#             from g1.utils.helpers import export_policy_as_jit # 尝试导入
#             export_policy_as_jit(ppo_runner.alg.actor_critic, export_path)
#             print(f'策略已导出为 JIT 脚本到: {export_path}')
#         except ImportError:
#              print("警告: 未找到 export_policy_as_jit 函数，跳过导出。")
#         except Exception as e:
#              print(f"导出策略时出错: {e}")
#
#
#     # --- 5. 运行仿真循环 ---
#     print("开始运行仿真...")
#     # 设置一个合理的循环次数，例如运行 N 个回合
#     num_episodes_to_run = 10
#     max_steps = num_episodes_to_run * env.max_episode_length
#
#     for i in range(max_steps):
#         with torch.no_grad(): # 推理时不需要计算梯度
#              actions = policy(obs.detach())
#         obs, _, rews, dones, infos = env.step(actions.detach())
#
#         # 可选：添加延迟以便观察
#         # import time
#         # time.sleep(0.01)
#
#         # 检查是否有环境完成（用于可能的统计或提前退出）
#         # if torch.any(dones):
#         #     print(f"Step {i+1}: 有环境完成。")
#
#         # 检查是否需要退出 (例如按 ESC) - render() 方法内部处理
#         # env.render() # render 通常在 step 内部调用
#
#     print("仿真运行结束。")
#     env.close()
#
#
# if __name__ == '__main__':
#     # --- 设置 Play 模式的参数 ---
#     # 不要导出策略、录制视频或移动相机（除非需要）
#     # EXPORT_POLICY = False # 通过命令行参数控制
#     # RECORD_FRAMES = False
#     # MOVE_CAMERA = False
#
#     # 解析命令行参数
#     args = parse_args() # 使用你的 parse_args
#
#     # --- 为 Play 模式添加/修改特定参数 ---
#     parser_play = argparse.ArgumentParser(description='Play policy', add_help=False)
#     parser_play.add_argument('--export_policy', action='store_true', default=False, help='Export the policy as a JIT module')
#     # --task 参数应该由 get_args 处理，确保它存在
#     if not hasattr(args, 'task') or args.task is None:
#          print("错误: 请使用 --task 指定要运行的任务名称 (例如 G1BasicLocomotion)。")
#          sys.exit(1)
#     # --checkpoint 参数应该由 get_args 处理，确保它指向有效的模型文件
#     if not hasattr(args, 'checkpoint') or args.checkpoint is None:
#         print("错误: 请使用 --checkpoint 指定要加载的模型文件路径 (例如 logs/.../model_1000.pt)。")
#         sys.exit(1)
#     # --headless 参数由 get_args 处理，确保为 False 以便看到可视化
#     args.headless = False # 强制显示 GUI
#
#     # 解析新增的 play 特定参数
#     play_args, _ = parser_play.parse_known_args()
#     # 添加到主 args 对象
#     for key, value in vars(play_args).items():
#         setattr(args, key, value)
#
#     # 设置随机种子（可选，但在评估时设置固定种子可能更好）
#     if args.seed is None:
#          args.seed = 42 # 设置一个固定的种子
#     set_seed(args.seed)
#     print(f"Play 模式种子设置为: {args.seed}")
#
#
#     # 调用 play 函数
#     play_curriculum(args)

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