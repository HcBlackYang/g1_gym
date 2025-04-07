# g1_kitchen_interaction.py
from g1.envs.g1_kitchen_navigation import G1KitchenNavigation
import torch
import numpy as np


class G1KitchenInteraction(G1KitchenNavigation):
    """第三阶段：厨房交互训练"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 初始化交互相关属性
        self.interactive_objects = {}
        self.interactive_joints = {}
        self.interaction_targets = {}
        self.current_task = None
        self.task_progress = None

        # 调用父类初始化
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # 交互任务设置
        self._setup_interaction_tasks()

        # 初始化任务状态
        self.current_task = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.task_progress = torch.zeros(self.num_envs, device=self.device)
        self.interaction_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _load_kitchen_assets(self):
        """重写加载厨房资产方法，识别可交互对象"""
        # 先调用父类方法加载所有厨房资产
        super()._load_kitchen_assets()

        # 识别可交互对象
        self._identify_interactive_objects()

    def _identify_interactive_objects(self):
        """识别和记录可交互对象"""
        # 这里需要根据您的厨房模型结构进行修改
        # 以下是示例，假设冰箱、抽屉等有关节信息

        # 收集所有可交互对象的信息
        for urdf_path, asset in self.kitchen_assets.items():
            # 检查是否有可动关节
            dof_count = self.gym.get_asset_dof_count(asset)

            if dof_count > 0:
                # 有关节的资产可能是可交互的
                dof_names = [self.gym.get_asset_dof_name(asset, i) for i in range(dof_count)]

                # 记录可交互对象
                object_name = urdf_path.split('/')[-1].replace('.urdf', '')
                self.interactive_objects[object_name] = {
                    'urdf_path': urdf_path,
                    'dof_count': dof_count,
                    'dof_names': dof_names
                }

                print(f"✅ 发现可交互对象: {object_name}，关节数: {dof_count}")

    def _create_envs(self):
        """重写环境创建方法，记录可交互对象的关节索引"""
        # 先调用父类方法创建环境
        super()._create_envs()

        # 在所有环境创建完成后，记录关节索引
        self._record_joint_indices()

    def _record_joint_indices(self):
        """记录各环境中交互对象的关节索引"""
        # 初始化交互关节索引存储
        self.interactive_joints = {}

        # 对每个环境遍历
        for env_idx in range(self.num_envs):
            env_handle = self.envs[env_idx]
            env_joints = {}

            # 遍历该环境中所有kitchen actor
            for kitchen_actor in self.kitchen_actors_by_env[env_idx]:
                # 获取actor的名称
                actor_name = self.gym.get_actor_name(env_handle, kitchen_actor)

                # 检查是否是可交互对象
                for obj_name, obj_info in self.interactive_objects.items():
                    if obj_name in actor_name:
                        # 获取该actor的关节数
                        dof_count = self.gym.get_actor_dof_count(env_handle, kitchen_actor)

                        if dof_count > 0:
                            # 记录每个关节的索引
                            joint_indices = []
                            for j in range(dof_count):
                                dof_name = self.gym.get_actor_dof_name(env_handle, kitchen_actor, j)
                                # 获取全局DOF索引
                                global_idx = self.gym.get_actor_dof_index(env_handle, kitchen_actor, j,
                                                                          gymapi.DOF_STATE_ALL)
                                joint_indices.append(global_idx)

                            # 存储关节索引
                            env_joints[obj_name] = {
                                'actor_handle': kitchen_actor,
                                'dof_count': dof_count,
                                'joint_indices': joint_indices
                            }

                            print(f"环境 {env_idx}: 记录 {obj_name} 的 {dof_count} 个关节")

            # 将该环境的关节信息存储到全局字典
            self.interactive_joints[env_idx] = env_joints

    def _setup_interaction_tasks(self):
        """设置交互任务"""
        # 定义任务类型
        self.task_types = ["open_fridge", "close_fridge", "open_drawer", "close_drawer"]

        # 根据子阶段调整可用任务
        sub_stage = self.cfg.curriculum.sub_stage
        if sub_stage <= 1:
            # 阶段4.1: 只需打开冰箱
            self.available_tasks = ["open_fridge"]
        elif sub_stage <= 2:
            # 阶段4.2: 可以打开和关闭冰箱
            self.available_tasks = ["open_fridge", "close_fridge"]
        else:
            # 阶段4.3+: 所有交互任务
            self.available_tasks = self.task_types

        # 为每个任务定义目标状态
        self.interaction_targets = {
            "open_fridge": 1.0,  # 完全打开为1.0
            "close_fridge": 0.0,  # 完全关闭为0.0
            "open_drawer": 0.4,  # 抽屉开到0.4位置
            "close_drawer": 0.0  # 抽屉完全关闭
        }

    def _sample_interaction_task(self, env_ids=None):
        """为指定环境采样交互任务"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 为每个环境随机选择一个可用任务
        for env_id in env_ids:
            task_name = np.random.choice(self.available_tasks)
            self.current_task[env_id] = self.task_types.index(task_name)

        # 重置任务进度和成功状态
        self.task_progress[env_ids] = 0.0
        self.interaction_success[env_ids] = False

    def _get_joint_state(self, env_idx, object_name, joint_idx=0):
        """获取指定环境、对象和关节的当前状态"""
        if env_idx not in self.interactive_joints:
            return 0.0

        if object_name not in self.interactive_joints[env_idx]:
            return 0.0

        obj_info = self.interactive_joints[env_idx][object_name]
        if joint_idx >= len(obj_info['joint_indices']):
            return 0.0

        # 获取全局DOF索引
        global_idx = obj_info['joint_indices'][joint_idx]

        # 获取关节位置
        dof_state = self.dof_state[global_idx].view(-1)
        joint_pos = dof_state[0].item()  # 位置是第一个元素

        return joint_pos

    def _update_task_progress(self):
        """更新任务进度"""
        for env_idx in range(self.num_envs):
            task_idx = self.current_task[env_idx].item()
            task_name = self.task_types[task_idx]

            # 获取目标对象和关节
            if "fridge" in task_name:
                object_name = "fridge"  # 根据您的具体模型调整
            elif "drawer" in task_name:
                object_name = "drawer"  # 根据您的具体模型调整
            else:
                continue

            # 获取当前关节状态
            current_state = self._get_joint_state(env_idx, object_name)

            # 获取目标状态
            target_state = self.interaction_targets[task_name]

            # 计算进度（范围0-1）

            if "open" in task_name:
            # 打开任务：当前/目标
                progress = min(1.0, max(0.0, current_state / target_state))
            else:
            # 关闭任务：1 - 当前/初始
                initial_state = 1.0 if object_name == "fridge" else 0.4  # 根据对象调整初始完全打开状态
            progress = min(1.0, max(0.0, 1.0 - current_state / initial_state))

            # 更新进度
            self.task_progress[env_idx] = progress

            # 检查是否完成任务（进度达到95%以上）
            if progress > 0.95:
                self.interaction_success[env_idx] = True

    def compute_observations(self):
        """计算观察，包含交互任务信息"""
        # 调用父类观察计算
        super().compute_observations()

        # 创建任务编码（one-hot）
        task_onehot = torch.zeros((self.num_envs, len(self.task_types)), device=self.device)
        for i in range(self.num_envs):
            task_onehot[i, self.current_task[i]] = 1.0

        # 添加任务进度
        task_progress = self.task_progress.unsqueeze(1)

        # 构建任务观察
        task_obs = torch.cat([task_onehot, task_progress], dim=1)

        # 合并到观察向量
        self.obs_buf = torch.cat([self.obs_buf, task_obs], dim=1)

        # 特权观察（如果使用）
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, task_obs], dim=1)

            # 添加更多特权信息（如关节状态）
            joint_states = torch.zeros((self.num_envs, 4), device=self.device)  # 假设最多4个重要关节

            # 在实际实现中，填充真实的关节状态
            for env_idx in range(self.num_envs):
                joint_states[env_idx, 0] = self._get_joint_state(env_idx, "fridge", 0)
                joint_states[env_idx, 1] = self._get_joint_state(env_idx, "drawer", 0)
                # ... 其他关节

            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, joint_states], dim=1)

    def compute_reward(self):
        """计算奖励，包含交互奖励"""
        # 初始化奖励
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        # 更新任务进度
        self._update_task_progress()

        # 包含导航的基础奖励
        self.rew_buf += self._reward_base_height() * self.reward_scales["base_height"]
        self.rew_buf += self._reward_alive() * self.reward_scales["alive"]

        # 导航奖励（缩小权重，因为现在主要关注交互）
        nav_scale = self.reward_scales.get("navigation_scale", 0.5)

        robot_pos = self.robot_root_states[:, :3]
        target_positions = self.waypoint_positions[self.current_waypoint_idx]
        distances = torch.norm(robot_pos[:, :2] - target_positions[:, :2], dim=1)

        distance_reward = torch.exp(-0.2 * distances)
        self.rew_buf += distance_reward * self.reward_scales.get("waypoint_distance", 1.0) * nav_scale

        # 导航点到达奖励
        reached = self._check_waypoint_reached()
        self.rew_buf += reached.float() * self.reward_scales.get("waypoint_reached", 5.0) * nav_scale

        # 交互奖励 - 进度奖励和成功奖励
        for env_idx in range(self.num_envs):
            # 进度奖励（平方进度可以鼓励更快完成）
            progress_reward = self.task_progress[env_idx] ** 2
            self.rew_buf[env_idx] += progress_reward * self.reward_scales.get("interaction_progress", 2.0)

            # 成功完成奖励
            if self.interaction_success[env_idx]:
                self.rew_buf[env_idx] += self.reward_scales.get("interaction_success", 10.0)

        # 碰撞惩罚（仍然保留，但调整权重）
        collision = self._check_collision()
        self.rew_buf -= collision.float() * self.reward_scales.get("kitchen_collision", 3.0)

    def compute_success_flags(self):
        """计算成功标志"""
        # 交互任务的成功条件：到达交互点并成功完成交互任务
        success = self.reached_waypoint & self.interaction_success & ~self.collision_detected
        return success

    def post_physics_step(self):
        """后处理步骤，更新任务和成功率"""
        # 调用父类方法
        super().post_physics_step()

        # 更新任务进度
        self._update_task_progress()

        # 为已完成任务的环境重新采样任务
        completed = self.interaction_success.clone()
        if completed.any():
            env_ids = torch.nonzero(completed).squeeze(-1)
            self._sample_interaction_task(env_ids)
            # 同时重新采样导航点
            self._sample_navigation_targets(env_ids)

        # 计算成功标志并更新成功率
        success_flags = self.compute_success_flags()
        if success_flags.any():
            self.update_success_rate(success_flags)