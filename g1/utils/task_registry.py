# import os
# from datetime import datetime
# from typing import Tuple
# import torch
# import numpy as np
# import sys
#
# from rsl_rl.env import VecEnv
# from rsl_rl.runners import OnPolicyRunner
#
#
# G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
# from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
#
#
#
# class TaskRegistry():
#     def __init__(self):
#         self.task_classes = {}
#         self.env_cfgs = {}
#         self.train_cfgs = {}
#
#     def register(self, name: str, task_class: VecEnv, env_cfg: LeggedRobotCfg, train_cfg: LeggedRobotCfgPPO):
#         self.task_classes[name] = task_class
#         self.env_cfgs[name] = env_cfg
#         self.train_cfgs[name] = train_cfg
#
#     def get_task_class(self, name: str) -> VecEnv:
#         return self.task_classes[name]
#
#     def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
#         train_cfg = self.train_cfgs[name]
#         env_cfg = self.env_cfgs[name]
#         # copy seed
#         env_cfg.seed = train_cfg.seed
#         return env_cfg, train_cfg
#
#     # def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
#     #     """ Creates an environment either from a registered namme or from the provided config file.
#     #
#     #     Args:
#     #         name (string): Name of a registered env.
#     #         args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
#     #         env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.
#     #
#     #     Raises:
#     #         ValueError: Error if no registered env corresponds to 'name'
#     #
#     #     Returns:
#     #         isaacgym.VecTaskPython: The created environment
#     #         Dict: the corresponding config file
#     #     """
#     #
#     #     print("当前注册的任务:", self.task_classes.keys())  # 打印已注册的任务
#     #
#     #     # if no args passed get command line arguments
#     #     if args is None:
#     #         args = get_args()
#     #     # check if there is a registered env with that name
#     #     if name in self.task_classes:
#     #         task_class = self.get_task_class(name)
#     #     else:
#     #         raise ValueError(f"Task with name: {name} was not registered")
#     #     if env_cfg is None:
#     #         # load config files
#     #         env_cfg, _ = self.get_cfgs(name)
#     #     # override cfg from args (if specified)
#     #     env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
#     #     set_seed(env_cfg.seed)
#     #     # parse sim params (convert to dict first)
#     #     sim_params = {"sim": class_to_dict(env_cfg.sim)}
#     #     sim_params = parse_sim_params(args, sim_params)
#     #     env = task_class(   cfg=env_cfg,
#     #                         sim_params=sim_params,
#     #                         physics_engine=args.physics_engine,
#     #                         sim_device=args.sim_device,
#     #                         headless=args.headless)
#     #     return env, env_cfg
#
#     def make_env(self, name, args=None, env_cfg=None) -> Tuple[VecEnv, LeggedRobotCfg]:
#         print("--- DEBUG make_env: 函数开始 ---") # 确保函数被调用
#         print("当前注册的任务:", self.task_classes.keys())
#
#         try: # 添加 try...except 块来捕获内部错误
#             if args is None:
#                 args = get_args()
#             if name in self.task_classes:
#                 task_class = self.get_task_class(name)
#                 print(f"--- DEBUG make_env: 获取到 task_class: {task_class.__name__}")
#             else:
#                 raise ValueError(f"Task with name: {name} was not registered")
#
#             if env_cfg is None:
#                 # load config files
#                 original_env_cfg, _ = self.get_cfgs(name) # 获取原始配置对象
#                 print(f"--- DEBUG make_env: 从 get_cfgs 获取原始 env_cfg 类型: {type(original_env_cfg)}")
#             else:
#                  original_env_cfg = env_cfg # 如果传入了 env_cfg
#
#             # override cfg from args (if specified)
#             env_cfg, _ = update_cfg_from_args(original_env_cfg, None, args) # update_cfg_from_args 修改对象
#             print(f"--- DEBUG make_env: update_cfg_from_args 后 env_cfg 类型: {type(env_cfg)}")
#
#             # ---- 关键检查点：在调用 task_class 之前 ----
#             if not hasattr(env_cfg, 'env') or not hasattr(env_cfg.env, 'num_envs'):
#                  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#                  print(f"--- DEBUG make_env: CRITICAL: 在调用 task_class 之前, env_cfg (类型: {type(env_cfg)}) 缺少 env.num_envs 属性!")
#                  # 打印 env_cfg 的内容帮助分析
#                  try:
#                      import pprint
#                      print("env_cfg 内容:")
#                      pprint.pprint(vars(env_cfg)) # 尝试打印对象的属性字典
#                  except TypeError:
#                      print("无法打印 env_cfg 的 vars()")
#                  print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#                  raise AttributeError("env_cfg object is missing 'env.num_envs' before calling task constructor") # 主动抛出错误
#
#             print(f"--- DEBUG make_env: 准备调用 {task_class.__name__}.__init__ ---")
#             print(f"--- DEBUG make_env: 将要传入的 cfg 类型: {type(env_cfg)}")
#             # --------------------------------------------
#
#             set_seed(env_cfg.seed)
#             sim_params_dict = {"sim": class_to_dict(env_cfg.sim)}
#             sim_params_obj = parse_sim_params(args, sim_params_dict) # 获取 SimParams 对象
#             print(f"--- DEBUG make_env: parse_sim_params 返回类型: {type(sim_params_obj)}")
#
#             # 调用环境构造函数
#             env = task_class(   cfg=env_cfg, # 传入配置对象
#                                 sim_params=sim_params_obj, # 传入 SimParams 对象
#                                 physics_engine=args.physics_engine,
#                                 sim_device=args.sim_device,
#                                 headless=args.headless)
#
#             print(f"--- DEBUG make_env: {task_class.__name__}.__init__ 调用完成 ---")
#             print(f"--- DEBUG make_env: 返回的 env 类型: {type(env)}")
#             print(f"--- DEBUG make_env: 返回的 env_cfg 类型: {type(env_cfg)}")
#             return env, env_cfg
#
#         except Exception as e_inner:
#             # 捕获 make_env 内部发生的任何异常
#             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             print(f"--- DEBUG make_env: 内部发生错误: {e_inner}")
#             import traceback
#             traceback.print_exc()
#             print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#             raise # 重新抛出异常，让外部知道
#
#
#
#
#     def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
#         """ Creates the training algorithm  either from a registered namme or from the provided config file.
#
#         Args:
#             env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
#             name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
#             args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
#             train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
#             log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example).
#                                       Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.
#
#         Raises:
#             ValueError: Error if neither 'name' or 'train_cfg' are provided
#             Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored
#
#         Returns:
#             PPO: The created algorithm
#             Dict: the corresponding config file
#         """
#         # if no args passed get command line arguments
#         if args is None:
#             args = get_args()
#         # if config files are passed use them, otherwise load from the name
#         if train_cfg is None:
#             if name is None:
#                 raise ValueError("Either 'name' or 'train_cfg' must be not None")
#             # load config files
#             _, train_cfg = self.get_cfgs(name)
#         else:
#             if name is not None:
#                 print(f"'train_cfg' provided -> Ignoring 'name={name}'")
#         # override cfg from args (if specified)
#         _, train_cfg = update_cfg_from_args(None, train_cfg, args)
#
#         if log_root=="default":
#             log_root = os.path.join(G1_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
#             log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
#         elif log_root is None:
#             log_dir = None
#         else:
#             log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
#
#         train_cfg_dict = class_to_dict(train_cfg)
#         runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
#         #save resume path before creating a new log_dir
#         resume = train_cfg.runner.resume
#         if resume:
#             # load previously trained model
#             resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
#             print(f"Loading model from: {resume_path}")
#             runner.load(resume_path)
#         return runner, train_cfg
#
# # make global task registry
# task_registry = TaskRegistry()

# task_registry.py
import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np
import sys

# Isaac Gym imports
from isaacgym import gymapi, gymutil
# RSL-RL imports
from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

# Project imports (ensure paths are correct)
try:
    # Define G1_ROOT_DIR robustly if not already set globally
    if 'G1_ROOT_DIR' not in globals():
         G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes task_registry.py is in g1/utils/
    from g1.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
    from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
except ImportError as e:
    print(f"❌ ERROR importing project modules in task_registry.py: {e}")
    print("   Please ensure G1_ROOT_DIR is correct and PYTHONPATH includes the project root.")
    sys.exit(1)


class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        print("TaskRegistry initialized.")

    def register(self, name: str, task_class: type, env_cfg_class: type, train_cfg_class: type):
        """ Registers a task with its class and config classes. """
        # Ensure task_class is a class, not an instance
        if not isinstance(task_class, type):
             raise TypeError(f"task_class for '{name}' must be a class, not an instance ({type(task_class)}).")
        # Same for config classes
        if not issubclass(env_cfg_class, LeggedRobotCfg): # Check base class
             raise TypeError(f"env_cfg_class for '{name}' must be a subclass of LeggedRobotCfg.")
        if not issubclass(train_cfg_class, LeggedRobotCfgPPO):
             raise TypeError(f"train_cfg_class for '{name}' must be a subclass of LeggedRobotCfgPPO.")

        self.task_classes[name] = task_class
        # Store the classes themselves, not instances yet
        self.env_cfgs[name] = env_cfg_class
        self.train_cfgs[name] = train_cfg_class
        print(f"  Registered task: '{name}' (Env: {task_class.__name__}, Cfg: {env_cfg_class.__name__})")


    def get_task_class(self, name: str) -> type:
        """ Gets the registered task class by name. """
        if name not in self.task_classes:
             raise ValueError(f"Task '{name}' not found in registry. Available: {list(self.task_classes.keys())}")
        return self.task_classes[name]


    def get_cfgs(self, name) -> Tuple[LeggedRobotCfg, LeggedRobotCfgPPO]:
        """ Creates instances of the registered config classes for a task. """
        if name not in self.env_cfgs or name not in self.train_cfgs:
             raise ValueError(f"Config classes for task '{name}' not found.")

        EnvCfgClass = self.env_cfgs[name]
        TrainCfgClass = self.train_cfgs[name]

        env_cfg = EnvCfgClass()
        train_cfg = TrainCfgClass()

        # --- 确保 train_cfg 有 seed (从 LeggedRobotCfgPPO 继承) ---
        if not hasattr(train_cfg, 'seed'):
             print(f"⚠️ Warning: train_cfg for task '{name}' has no 'seed' attribute. Using default 1.")
             train_cfg.seed = 1 # Set a default seed

        # --- 确保 env_cfg 有 seed 属性并赋值 ---
        env_cfg.seed = train_cfg.seed
        # ---------------------------------------

        return env_cfg, train_cfg

    # !!! 修改 make_env 签名以接收 gym, sim, sim_params !!!
    def make_env(self, name, args=None, env_cfg=None, gym_handle=None, sim_handle=None, sim_params=None) -> Tuple[VecEnv, LeggedRobotCfg]:
        """ Creates an environment instance for the given task name.
            Requires pre-initialized gym and sim handles.
        """
        print(f"--- task_registry.make_env for task '{name}' ---")
        if gym_handle is None or sim_handle is None or sim_params is None:
             raise ValueError("make_env requires valid 'gym_handle', 'sim_handle', and 'sim_params'.")

        try:
            # 1. Get arguments if not provided
            if args is None: args = get_args()

            # 2. Get task class
            task_class = self.get_task_class(name) # Raises ValueError if not found

            # 3. Get or use provided env_cfg instance
            if env_cfg is None:
                # Create a fresh instance from the registered class
                env_cfg, _ = self.get_cfgs(name)
                print(f"  Created fresh env_cfg instance from registry (Type: {type(env_cfg)}).")
            else:
                # Use the provided env_cfg instance (already copied/modified by caller)
                print(f"  Using provided env_cfg instance (Type: {type(env_cfg)}).")

            # 4. Override config from command line arguments
            # Important: update_cfg_from_args modifies env_cfg in-place
            env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
            print(f"  env_cfg updated from args.")

            # 5. Check critical attributes before calling constructor
            if not hasattr(env_cfg, 'env') or not hasattr(env_cfg.env, 'num_envs'):
                 raise AttributeError("env_cfg object is missing 'env.num_envs' before calling task constructor")
            # Add check for num_observations which BaseTask now needs directly
            if not hasattr(env_cfg.env, 'num_observations'):
                 raise AttributeError("env_cfg object is missing 'env.num_observations'")


            # 6. Set seed (using the potentially updated seed in env_cfg)
            env_seed = getattr(env_cfg, 'seed', 1)  # Default to 1 if missing
            set_seed(env_seed)

            # 7. Call environment constructor, passing handles
            print(f"  Instantiating environment: {task_class.__name__}...")
            env = task_class(
                cfg=env_cfg,              # Pass the potentially modified config object
                sim_params=sim_params,    # Pass the SimParams object
                physics_engine=args.physics_engine,
                sim_device=args.sim_device,
                headless=args.headless,
                gym_handle=gym_handle,    # Pass the existing Gym API handle
                sim_handle=sim_handle     # Pass the existing Sim handle
            )
            print(f"  ✅ Environment instance created.")

            # 8. Return the env instance and the final config used
            return env, env_cfg

        except Exception as e_inner:
            print(f"❌❌❌ ERROR in task_registry.make_env: {e_inner}")
            import traceback
            traceback.print_exc()
            raise # Re-raise the exception


    def make_alg_runner(self, env: VecEnv, name=None, args=None, train_cfg=None, log_root="default") -> Tuple[OnPolicyRunner, LeggedRobotCfgPPO]:
        """ Creates the OnPolicyRunner instance. """
        print(f"--- task_registry.make_alg_runner for task '{name or args.task}' ---")
        # 1. Get arguments
        if args is None: args = get_args()

        # 2. Get or use provided train_cfg instance
        if train_cfg is None:
            if name is None: raise ValueError("Either 'name' or 'train_cfg' must be provided")
            # Create a fresh instance from the registered class
            _, train_cfg = self.get_cfgs(name)
            print(f"  Created fresh train_cfg instance from registry (Type: {type(train_cfg)}).")
        else:
            # Use the provided train_cfg instance
             if name is not None: print(f"  Using provided train_cfg instance (Ignoring 'name={name}').")
             else: print(f"  Using provided train_cfg instance.")

        # 3. Override config from command line arguments
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)
        print(f"  train_cfg updated from args.")

        # 4. Determine log directory
        if log_root=="default":
            if not hasattr(train_cfg, 'runner') or not hasattr(train_cfg.runner, 'experiment_name'):
                 raise AttributeError("train_cfg.runner.experiment_name is not defined.")
            log_root_path = os.path.join(G1_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        elif log_root is None:
            log_root_path = None # No logging
        else:
            log_root_path = log_root # Use custom root

        log_dir = None
        if log_root_path is not None:
            run_name = getattr(train_cfg.runner, 'run_name', 'run') # Default run name
            log_dir = os.path.join(log_root_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + run_name)
            print(f"  Log directory set to: {log_dir}")


        # 5. Create the runner instance
        try:
            # Convert train_cfg object to dict for OnPolicyRunner
            train_cfg_dict = class_to_dict(train_cfg)
            # Ensure env has necessary attributes for runner
            required_env_attrs = ['num_envs', 'num_actions', 'num_observations', 'device']
            for attr in required_env_attrs:
                if not hasattr(env, attr):
                     raise AttributeError(f"Environment instance is missing required attribute: '{attr}'")

            print(f"  Instantiating OnPolicyRunner...")
            runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=args.rl_device)
            print(f"  ✅ OnPolicyRunner instance created.")

        except Exception as e:
             print(f"❌❌❌ ERROR Creating OnPolicyRunner: {e}")
             import traceback; traceback.print_exc()
             raise

        # 6. Handle checkpoint loading if resume is enabled
        resume = getattr(train_cfg.runner, 'resume', False)
        if resume and log_root_path is not None: # Need log_root to find previous runs
             # load previously trained model
             load_run = getattr(train_cfg.runner, 'load_run', -1)
             checkpoint = getattr(train_cfg.runner, 'checkpoint', -1)
             # args.checkpoint overrides train_cfg settings if provided
             resume_path = args.checkpoint if args.checkpoint else get_load_path(log_root_path, load_run=load_run, checkpoint=checkpoint)
             if resume_path:
                 print(f"  Attempting to load model from: {resume_path}")
                 try:
                     runner.load(resume_path)
                     print(f"    ✅ Runner loaded checkpoint successfully.")
                 except FileNotFoundError:
                      print(f"    ❌ Checkpoint file not found: {resume_path}. Starting fresh.")
                 except Exception as e:
                      print(f"    ❌ Error loading checkpoint {resume_path}: {e}. Starting fresh.")
             else:
                  print(f"  Resume enabled but no valid checkpoint path found (load_run={load_run}, checkpoint={checkpoint}). Starting fresh.")
        elif resume and log_root_path is None:
             print("⚠️ Resume requested but log_root is None. Cannot load checkpoint.")


        return runner, train_cfg

# --- Global Task Registry Instance ---
task_registry = TaskRegistry()