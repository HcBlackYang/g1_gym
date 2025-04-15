# curriculum_manager.py
# 课程学习管理器配置

class CurriculumManagerConfig:
    def __init__(self):
        self.initial_stage = 1  # 初始主阶段
        self.initial_sub_stage = 1  # 初始子阶段
        self.max_stages = 5  # 最大主阶段
        self.max_sub_stages = 5  # 每个主阶段的最大子阶段数

        # 进阶条件
        self.success_threshold = 0.8  # 成功率阈值
        self.min_episodes = 1000  # 最小训练回合数
        self.evaluation_window = 100  # 评估窗口大小
        self.max_env_steps = 100000000
        self.min_steps_between_eval = 100

        # 模型迁移设置
        self.model_transfer = {
            "transfer_weights": True,  # 是否迁移权重
            "init_scale": 0.01  # 新权重初始化范围
        }

        # 课程阶段定义
        self.stage1 = {  # 基础运动技能
            "name": "基础运动技能",
            "env_class": "G1BasicLocomotion",
            # "num_envs": 16384,  # 大规模并行
            # "base_lin_vel_range": 2.0,  # 线速度命令范围
            # "base_ang_vel_range": 1.0  # 角速度命令范围
        }

        self.stage2 = {  # 厨房导航
            "name": "厨房导航",
            "env_class": "G1KitchenNavigation",
            "num_envs": 64,  # 较小规模并行
            "use_fixed_kitchen": True  # 使用固定厨房布局
        }

        self.stage3 = {  # 厨房交互
            "name": "厨房交互",
            "env_class": "G1KitchenInteraction",
            "num_envs": 256  # 较小规模并行
        }

        self.stage4 = {  # 完整任务
            "name": "完整厨房任务",
            "env_class": "G1KitchenFullTask",
            "num_envs": 128  # 小规模并行
        }

        self.output = {
            "output_dir": "output/curriculum",  # 输出目录
            "save_frequency": 50,  # 保存频率（以环境步数为单位）
            "plot_frequency": 20  # 绘图频率
        }

    def get_stage_config(self, stage_num):
        """获取指定阶段的配置"""
        stages = {
            1: self.stage1,
            2: self.stage2,
            3: self.stage3,
            4: self.stage4
        }
        return stages.get(stage_num)