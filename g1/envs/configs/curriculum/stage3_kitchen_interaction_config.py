# stage3_kitchen_interaction.py
# 厨房交互训练配置

from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Stage3KitchenInteractionConfig:
    def __init__(self, curriculum_config=None):
        # 从课程管理器读取阶段4的配置
        stage4_config = None
        if curriculum_config:
            stage4_config = curriculum_config.stage4

        # 环境配置
        self.env = {
            "num_envs": self._get_curriculum_value(stage4_config, "num_envs", 256),  # 较小规模并行
            "rollout_horizon": 24,  # rollout窗口大小
            "test": False,
            "episode_length_s": 20,  # 每个episode的最大时长（秒）
        }

        # 领域随机化（在厨房环境中进一步降低随机化）
        self.domain_rand = {
            "push_robots": True,  # 随机推动机器人
            "push_interval": 25,  # 推动间隔增加
            "max_push_vel_xy": 0.3,  # 最大推动速度降低
        }

        # 奖励配置
        self.rewards = {
            "scales": {
                "alive": 1.0,
                "base_height": 1.0,  # 基础高度
                "feet_swing_height": 0.5,  # 脚部摆动高度
                "navigation_scale": 0.5,  # 导航奖励缩放因子
                "waypoint_distance": 1.0,  # 导航点距离奖励
                "waypoint_reached": 5.0,  # 导航点达成奖励
                "kitchen_collision": 5.0,  # 厨房碰撞惩罚
                "interaction_progress": 2.0,  # 交互进度奖励
                "interaction_success": 10.0,  # 交互成功奖励
            },
            "only_positive_rewards": False,  # 允许负奖励
        }

        # 命令生成
        self.commands = {
            "resampling_time": 10.0,  # 命令重采样时间
            "num_commands": 4,  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]

            # 命令范围 - 交互阶段更小
            "lin_vel_x": [-0.6, 0.6],
            "lin_vel_y": [-0.6, 0.6],
            "ang_vel_yaw": [-0.6, 0.6],
            "heading": [-3.14, 3.14],
            "heading_command": True,  # 使用朝向命令
        }

        # 交互配置
        self.interaction = {
            # 交互对象
            "interactive_objects": [
                "fridge",
                "drawer",
                "cabinet"
            ],

            # 交互任务
            "interaction_tasks": [
                "open_fridge",
                "close_fridge",
                "open_drawer",
                "close_drawer"
            ],

            # 交互目标状态
            "interaction_targets": {
                "open_fridge": 1.0,
                "close_fridge": 0.0,
                "open_drawer": 0.4,
                "close_drawer": 0.0
            }
        }

        # 训练配置
        self.train = {
            "max_iterations": 1000,  # 最大训练迭代数
            "save_interval": 100  # 保存间隔
        }

    def _get_curriculum_value(self, config, key, default_value):
        """从课程配置中获取值，如果不存在则返回默认值"""
        if config is None:
            return default_value
        return config.get(key, default_value)
