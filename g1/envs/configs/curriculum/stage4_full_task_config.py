# stage4_full_task.py
# 完整厨房任务训练配置


from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Stage4FullTaskConfig:
    def __init__(self, curriculum_config=None):
        # 从课程管理器读取阶段5的配置
        stage5_config = None
        if curriculum_config:
            stage5_config = curriculum_config.stage5

        # 环境配置
        self.env = {
            "num_envs": self._get_curriculum_value(stage5_config, "num_envs", 128),  # 较小规模并行
            "rollout_horizon": 24,  # rollout窗口大小
            "test": False,
            "episode_length_s": 30,  # 延长最大时长（秒）以完成复杂任务
        }

        # 领域随机化（最小化，关注任务执行）
        self.domain_rand = {
            "push_robots": True,  # 随机推动机器人
            "push_interval": 30,  # 更长的推动间隔
            "max_push_vel_xy": 0.2,  # 最小推动速度
        }

        # 奖励配置
        self.rewards = {
            "scales": {
                "alive": 1.0,
                "base_height": 1.0,  # 基础高度
                "feet_swing_height": 0.5,  # 脚部摆动高度
                "waypoint_distance": 0.5,  # 导航点距离奖励（减小权重）
                "waypoint_reached": 3.0,  # 导航点达成奖励（减小权重）
                "kitchen_collision": 3.0,  # 厨房碰撞惩罚
                "interaction_progress": 1.5,  # 交互进度奖励
                "interaction_success": 8.0,  # 交互成功奖励
                "step_progress": 2.0,  # 步骤进度奖励
                "grab_success": 5.0,  # 抓取成功奖励
                "sequence_completion": 20.0,  # 序列完成奖励（高权重）
            },
            "only_positive_rewards": False,  # 允许负奖励
        }

        # 命令生成
        self.commands = {
            "resampling_time": 10.0,  # 命令重采样时间
            "num_commands": 4,  # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]

            # 命令范围 - 完整任务阶段最小
            "lin_vel_x": [-0.5, 0.5],
            "lin_vel_y": [-0.5, 0.5],
            "ang_vel_yaw": [-0.5, 0.5],
            "heading": [-3.14, 3.14],

            "heading_command": True,  # 使用朝向命令
        }

        # 任务序列配置
        self.task_sequences = {
            # 序列1: 从冰箱取牛奶
            "fridge_milk_sequence": [
                {"type": "navigate", "target": "fridge_front", "name": "导航到冰箱"},
                {"type": "interact", "action": "open_fridge", "name": "打开冰箱门"},
                {"type": "grab", "object": "milk", "name": "抓取牛奶"},
                {"type": "interact", "action": "close_fridge", "name": "关闭冰箱门"},
                {"type": "navigate", "target": "table", "name": "将牛奶放到桌子上"}
            ],

            # 序列2: 从抽屉取餐具
            "drawer_utensil_sequence": [
                {"type": "navigate", "target": "counter", "name": "导航到柜台"},
                {"type": "interact", "action": "open_drawer", "name": "打开抽屉"},
                {"type": "grab", "object": "utensil", "name": "抓取餐具"},
                {"type": "interact", "action": "close_drawer", "name": "关闭抽屉"},
                {"type": "navigate", "target": "table", "name": "将餐具放到桌子上"}
            ]
        }

        # 可抓取物品配置
        self.grab_objects = [
            {
                "name": "milk",
                "urdf": "milk.urdf",
                "grab_offset": [0, 0, 0.1]
            },
            {
                "name": "utensil",
                "urdf": "utensil.urdf",
                "grab_offset": [0, 0, 0.05]
            },
            {
                "name": "plate",
                "urdf": "plate.urdf",
                "grab_offset": [0, 0, 0.02]
            },
            {
                "name": "cup",
                "urdf": "cup.urdf",
                "grab_offset": [0, 0, 0.08]
            }
        ]

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