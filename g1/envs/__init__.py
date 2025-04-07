#
# # g1/envs/__init__.py
# try:
#     print("envs.__init__: 尝试导入 task_registry")
#     from g1.utils.task_registry import task_registry
#     print(f"envs.__init__: task_registry 已导入, id: {id(task_registry)}") # 打印对象ID
#
#     print("envs.__init__: 正在导入环境类...")
#     from g1.envs.g1_basic_locomotion import G1BasicLocomotion
#     from g1.envs.g1_kitchen_navigation import G1KitchenNavigation
#     from g1.envs.g1_kitchen_interaction import G1KitchenInteraction
#     from g1.envs.g1_kitchen_full_task import G1KitchenFullTask
#     print("envs.__init__: 环境类导入完成。")
#
#     print("envs.__init__: 正在导入配置类...")
#     from g1.envs.configs.curriculum.stage1_locomotion_config import Stage1LocomotionConfig
#     from g1.envs.configs.curriculum.stage2_kitchen_nav_config import Stage2KitchenNavConfig
#     from g1.envs.configs.curriculum.stage3_kitchen_interaction_config import Stage3KitchenInteractionConfig
#     from g1.envs.configs.curriculum.stage4_full_task_config import Stage4FullTaskConfig
#     from g1.envs.base.legged_robot_config import LeggedRobotCfgPPO # 确保这个也导入了
#     print("envs.__init__: 配置类导入完成。")
#
#     print("envs.__init__: 正在创建配置实例...")
#     locomotion_cfg = Stage1LocomotionConfig()
#     kitchen_nav_cfg = Stage2KitchenNavConfig()
#     kitchen_interaction_cfg = Stage3KitchenInteractionConfig()
#     kitchen_full_task_cfg = Stage4FullTaskConfig()
#     train_cfg = LeggedRobotCfgPPO()
#     print("envs.__init__: 配置实例创建完成。")
#
#     print("envs.__init__: 开始注册任务...")
#     task_registry.register("G1BasicLocomotion", G1BasicLocomotion, locomotion_cfg, train_cfg)
#     print("envs.__init__: 已注册 G1BasicLocomotion")
#     task_registry.register("G1KitchenNavigation", G1KitchenNavigation, kitchen_nav_cfg, train_cfg)
#     print("envs.__init__: 已注册 G1KitchenNavigation")
#     task_registry.register("G1KitchenInteraction", G1KitchenInteraction, kitchen_interaction_cfg, train_cfg)
#     print("envs.__init__: 已注册 G1KitchenInteraction")
#     task_registry.register("G1KitchenFullTask", G1KitchenFullTask, kitchen_full_task_cfg, train_cfg)
#     print("envs.__init__: 已注册 G1KitchenFullTask")
#
#     print("envs.__init__: 任务注册完成！")
#     print("envs.__init__: 当前可用任务:", list(task_registry.task_classes.keys()))
# except Exception as e:
#     # 捕获并打印任何在 __init__.py 中发生的异常
#     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     print(f"错误发生在 g1.envs.__init__ 中: {e}")
#     import traceback
#     traceback.print_exc() # 打印详细的错误堆栈
#     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     raise # 重新抛出异常，确保程序能看到错误并停止

# g1/envs/__init__.py (精确修改后)
import os
import sys
import traceback

try:
    print("envs.__init__: 尝试导入 task_registry")
    from g1.utils.task_registry import task_registry
    print(f"envs.__init__: task_registry 已导入, id: {id(task_registry)}")

    print("envs.__init__: 正在导入环境类...")
    from g1.envs.g1_basic_locomotion import G1BasicLocomotion
    from g1.envs.g1_kitchen_navigation import G1KitchenNavigation
    from g1.envs.g1_kitchen_interaction import G1KitchenInteraction
    from g1.envs.g1_kitchen_full_task import G1KitchenFullTask
    print("envs.__init__: 环境类导入完成。")

    print("envs.__init__: 正在导入配置类...")
    # --- 只导入类，不在此处实例化 ---
    from g1.envs.configs.curriculum.stage1_locomotion_config import Stage1LocomotionConfig,Stage1LocomotionConfigPPO
    from g1.envs.configs.curriculum.stage2_kitchen_nav_config import Stage2KitchenNavConfig,Stage2KitchenNavCfgPPO # 假设PPO配置在此
    from g1.envs.configs.curriculum.stage3_kitchen_interaction_config import Stage3KitchenInteractionConfig #, Stage3KitchenInteractionCfgPPO
    from g1.envs.configs.curriculum.stage4_full_task_config import Stage4FullTaskConfig #, Stage4FullTaskCfgPPO
    from g1.envs.base.legged_robot_config import LeggedRobotCfgPPO # 基础 PPO 配置
    print("envs.__init__: 配置类导入完成。")

    # --- !!! 不再需要在这里创建配置实例 !!! ---
    # print("envs.__init__: 正在创建配置实例...")
    # locomotion_cfg = Stage1LocomotionConfig()
    # kitchen_nav_cfg = Stage2KitchenNavConfig()
    # kitchen_interaction_cfg = Stage3KitchenInteractionConfig()
    # kitchen_full_task_cfg = Stage4FullTaskConfig()
    # train_cfg = LeggedRobotCfgPPO()
    # print("envs.__init__: 配置实例创建完成。")
    # ----------------------------------------

    print("envs.__init__: 开始注册任务...")
    # --- 传递类本身给 register ---
    task_registry.register(
        "G1BasicLocomotion",
        G1BasicLocomotion,          # 环境类
        Stage1LocomotionConfig,     # 环境配置类
        Stage1LocomotionConfigPPO           # 训练配置类 (使用基础PPO)
    )
    print("envs.__init__: 已注册 G1BasicLocomotion")

    task_registry.register(
        "G1KitchenNavigation",
        G1KitchenNavigation,        # 环境类
        Stage2KitchenNavConfig,     # 环境配置类
        Stage2KitchenNavCfgPPO       # 训练配置类 (Stage 2 特定)
    )
    print("envs.__init__: 已注册 G1KitchenNavigation")

    # # --- 假设 Stage 3 和 4 也使用基础 PPO 配置 ---
    # # --- 如果它们有自己的 PPO 配置，需要导入并传递对应的类 ---
    # task_registry.register(
    #     "G1KitchenInteraction",
    #     G1KitchenInteraction,         # 环境类
    #     Stage3KitchenInteractionConfig, # 环境配置类
    #     LeggedRobotCfgPPO             # 训练配置类 (使用基础PPO)
    # )
    # print("envs.__init__: 已注册 G1KitchenInteraction")
    #
    # task_registry.register(
    #     "G1KitchenFullTask",
    #     G1KitchenFullTask,            # 环境类
    #     Stage4FullTaskConfig,         # 环境配置类
    #     LeggedRobotCfgPPO             # 训练配置类 (使用基础PPO)
    # )
    # print("envs.__init__: 已注册 G1KitchenFullTask")
    # # -------------------------------------------------

    print("envs.__init__: 任务注册完成！")
    # 打印可用任务列表 (task_registry 内部现在应该有 task_classes)
    if hasattr(task_registry, 'task_classes'):
        print("envs.__init__: 当前可用任务:", list(task_registry.task_classes.keys()))
    else:
        print("envs.__init__: 警告 - 无法访问 task_registry.task_classes")

except ImportError as e: # 更具体地捕获导入错误
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"导入错误发生在 g1.envs.__init__ 中: {e}")
    print(f"请确保所有引用的模块/文件存在且路径在 PYTHONPATH 中。")
    traceback.print_exc()
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise
except Exception as e:
    # 捕获并打印任何在 __init__.py 中发生的其他异常
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"错误发生在 g1.envs.__init__ 中: {e}")
    traceback.print_exc() # 打印详细的错误堆栈
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    raise # 重新抛出异常，确保程序能看到错误并停止