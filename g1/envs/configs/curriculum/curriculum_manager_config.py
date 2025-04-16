# # curriculum_manager.py
# # 课程学习管理器配置

# class CurriculumManagerConfig:
#     def __init__(self):
#         self.initial_stage = 1  # 初始主阶段
#         self.initial_sub_stage = 1  # 初始子阶段
#         self.max_stages = 5  # 最大主阶段
#         self.max_sub_stages = 5  # 每个主阶段的最大子阶段数

#         # 进阶条件
#         self.success_threshold = 0.8  # 成功率阈值
#         self.min_episodes = 1000  # 最小训练回合数
#         self.evaluation_window = 100  # 评估窗口大小
#         self.max_env_steps = 100000000
#         self.min_steps_between_eval = 100

#         # 模型迁移设置
#         self.model_transfer = {
#             "transfer_weights": True,  # 是否迁移权重
#             "init_scale": 0.01  # 新权重初始化范围
#         }

#         # 课程阶段定义
#         self.stage1 = {  # 基础运动技能
#             "name": "基础运动技能",
#             "env_class": "G1BasicLocomotion",
#             "num_envs": 4096,  # 大规模并行
#             "base_lin_vel_range": 2.0,  # 线速度命令范围
#             "base_ang_vel_range": 1.0  # 角速度命令范围
#         }

#         self.stage2 = {  # 厨房导航
#             "name": "厨房导航",
#             "env_class": "G1KitchenNavigation",
#             "num_envs": 64,  # 较小规模并行
#             "use_fixed_kitchen": True  # 使用固定厨房布局
#         }

#         self.stage3 = {  # 厨房交互
#             "name": "厨房交互",
#             "env_class": "G1KitchenInteraction",
#             "num_envs": 256  # 较小规模并行
#         }

#         self.stage4 = {  # 完整任务
#             "name": "完整厨房任务",
#             "env_class": "G1KitchenFullTask",
#             "num_envs": 128  # 小规模并行
#         }

#         self.output = {
#             "output_dir": "output/curriculum",  # 输出目录
#             "save_frequency": 50,  # 保存频率（以环境步数为单位）
#             "plot_frequency": 20  # 绘图频率
#         }

#     def get_stage_config(self, stage_num):
#         """获取指定阶段的配置"""
#         stages = {
#             1: self.stage1,
#             2: self.stage2,
#             3: self.stage3,
#             4: self.stage4
#         }
#         return stages.get(stage_num)


class CurriculumManagerConfig:
    def __init__(self):
        self.initial_stage = 1          # 初始主阶段
        self.initial_sub_stage = 1      # 初始子阶段
        self.max_stages = 4             # 最大主阶段 (1: Loco, 2: Nav, 3: Interact, 4: FullTask)
        # !!! 增加 Stage 1 的子阶段数量 !!!
        self.max_sub_stages_per_stage = { # 为每个主阶段定义最大子阶段数
            1: 5,  # Stage 1 (Locomotion) 有 5 个子阶段用于嵌套课程
            2: 3,  # Stage 2 (Navigation) 示例 3 个子阶段
            3: 3,  # Stage 3 (Interaction) 示例 3 个子阶段
            4: 3   # Stage 4 (Full Task) 示例 3 个子阶段
        }
        # 旧的 max_sub_stages (保留用于向后兼容或简化配置)
        self.max_sub_stages = 5 # 可以设置为所有阶段中子阶段数量的最大值

        # 进阶条件
        self.success_threshold = 0.75 # 成功率阈值 (可以根据阶段调整，但这里用全局)
        # self.min_episodes = 1000     # 不再使用 episodes，改用 steps
        self.evaluation_window = 50    # 评估窗口大小 (改为使用平滑历史记录的长度)
        self.min_steps_between_eval = 2_000_000 # 每次检查进阶之间所需的最小环境步数 (例如 2M)
        self.max_env_steps = 200_000_000 # 训练总步数上限

        # 模型迁移设置
        self.model_transfer = {
            "transfer_weights": True,  # 是否迁移权重
            "init_scale": 0.01,      # 新权重初始化范围
            "device": "cuda:0"       # 可以在 train_curriculum.py 中覆盖
        }

        # 课程阶段定义 (保持简洁，具体参数移到各阶段的 Config 文件)
        self.stage1 = {
            "name": "基础运动技能",
            "env_class": "G1BasicLocomotion",
            # num_envs 和其他特定参数在 Stage1LocomotionConfig 中定义
        }
        self.stage2 = {
            "name": "厨房导航",
            "env_class": "G1KitchenNavigation",
            # num_envs 和其他特定参数在 Stage2KitchenNavConfig 中定义
        }
        self.stage3 = {
            "name": "厨房交互",
            "env_class": "G1KitchenInteraction",
            # num_envs 和其他特定参数在 Stage3KitchenInteractionConfig 中定义
        }
        self.stage4 = {
            "name": "完整厨房任务",
            "env_class": "G1KitchenFullTask",
            # num_envs 和其他特定参数在 Stage4FullTaskConfig 中定义
        }
        # 可以添加 Stage 5 等...

        # 输出配置 (保持不变)
        self.output = {
            "output_dir": "output/curriculum_nested", # 可以改个名字
            "save_frequency": 100,  # 保存频率 (以 PPO 迭代次数为单位)
            "plot_frequency": 50   # 绘图频率 (以 PPO 迭代次数为单位)
        }

    def get_stage_config(self, stage_num):
        """获取指定阶段的配置"""
        stages = {
            1: self.stage1,
            2: self.stage2,
            3: self.stage3,
            4: self.stage4
            # 添加更多阶段...
        }
        return stages.get(stage_num)
