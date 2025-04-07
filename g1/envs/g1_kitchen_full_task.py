# g1_kitchen_full_task.py
from g1.envs.g1_kitchen_interaction import G1KitchenInteraction
import torch
import numpy as np


class G1KitchenFullTask(G1KitchenInteraction):
    """第四阶段：完整厨房任务序列训练"""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # 任务序列相关
        self.task_sequence = []
        self.current_sequence_step = None
        self.sequence_progress = None

        # 物品交互
        self.grabable_objects = {}
        self.grabbed_object = None

        # 调用父类初始化
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # 设置任务序列
        self._setup_task_sequences()

        # 初始化序列状态
        self.current_sequence_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.sequence_progress = torch.zeros((self.num_envs, self.max_sequence_length), device=self.device)

        # 初始化物品抓取状态
        self.grabbed_object = torch.ones(self.num_envs, dtype=torch.long, device=self.device) * -1  # -1表示没有抓取

    def _setup_task_sequences(self):
        """设置任务序列"""
        # 定义可用的任务序列
        self.task_sequences = [
            # 序列1: 从冰箱取牛奶
            [
                {"type": "navigate", "target": "fridge_front", "name": "导航到冰箱"},
                {"type": "interact", "action": "open_fridge", "name": "打开冰箱门"},
                {"type": "grab", "object": "milk", "name": "抓取牛奶"},
                {"type": "interact", "action": "close_fridge", "name": "关闭冰箱门"},
                {"type": "navigate", "target": "table", "name": "将牛奶放到桌子上"}
            ],

            # 序列2: 从抽屉取餐具
            [
                {"type": "navigate", "target": "counter", "name": "导航到柜台"},
                {"type": "interact", "action": "open_drawer", "name": "打开抽屉"},
                {"type": "grab", "object": "utensil", "name": "抓取餐具"},
                {"type": "interact", "action": "close_drawer", "name": "关闭抽屉"},
                {"type": "navigate", "target": "table", "name": "将餐具放到桌子上"}
            ]
        ]

        # 确定最长序列长度
        self.max_sequence_length = max(len(seq) for seq in self.task_sequences)

        # 根据子阶段选择可用序列
        sub_stage = self.cfg.curriculum.sub_stage
        if sub_stage <= 2:
            # 阶段5.1-5.2: 只使用冰箱-牛奶序列
            self.available_sequences = [0]  # 序列1索引
        else:
            # 阶段5.3+: 所有序列
            self.available_sequences = list(range(len(self.task_sequences)))

    def _load_kitchen_assets(self):
        """重写加载厨房资产方法，识别可抓取对象"""
        # 先调用父类方法加载厨房和交互对象
        super()._load_kitchen_assets()

        # 识别可抓取对象
        self._identify_grabable_objects()

    def _identify_grabable_objects(self):
        """识别和记录可抓取物品"""
        # 这里需要根据您的厨房模型结构进行修改
        # 以下是示例，假设某些模型是可抓取的小物品

        # 预定义的可抓取对象
        grabable_names = ["milk", "utensil", "plate", "cup"]

        # 查找匹配的资产
        for urdf_path, asset in self.kitchen_assets.items():
            # 检查是否匹配可抓取对象名称
            for grabable_name in grabable_names:
                if grabable_name in urdf_path.lower():
                    # 记录可抓取对象
                    object_name = urdf_path.split('/')[-1].replace('.urdf', '')
                    self.grabable_objects[grabable_name] = {
                        'urdf_path': urdf_path,
                        'name': object_name
                    }

                    print(f"✅ 发现可抓取物品: {grabable_name} - {object_name}")

    def _sample_task_sequence(self, env_ids=None):
        """为指定环境采样任务序列"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # 为每个环境随机选择一个可用序列
        for env_id in env_ids:
            seq_idx = np.random.choice(self.available_sequences)
            # 设置序列的第一步
            self.current_sequence_step[env_id] = 0
            # 记录当前正在执行的序列
            self.current_task[env_id] = seq_idx

        # 重置序列进度
        self.sequence_progress[env_ids] = 0.0
        # 重置抓取状态
        self.grabbed_object[env_ids] = -1
        # 重置交互成功和到达状态
        self.interaction_success[env_ids] = False
        self.reached_waypoint[env_ids] = False

    def _update_current_step_target(self, env_idx):
        """更新当前步骤的目标（导航点或交互任务）"""
        # 获取当前序列和步骤
        seq_idx = self.current_task[env_idx].item()
        step_idx = self.current_sequence_step[env_idx].item()

        # 确保索引有效
        if seq_idx < 0 or seq_idx >= len(self.task_sequences):
            return

        sequence = self.task_sequences[seq_idx]
        if step_idx < 0 or step_idx >= len(sequence):
            return

        # 获取当前步骤
        step = sequence[step_idx]
        step_type = step["type"]

        if step_type == "navigate":
            # 设置导航目标
            target_name = step["target"]
            if target_name in self.waypoint_names:
                wp_idx = self.waypoint_names.index(target_name)
                self.current_waypoint_idx[env_idx] = wp_idx
                # 重置到达状态
                self.reached_waypoint[env_idx] = False

        elif step_type == "interact":
            # 设置交互任务
            action_name = step["action"]
            if action_name in self.task_types:
                task_idx = self.task_types.index(action_name)
                # 更新任务类型，但保持同样的序列
                old_seq = self.current_task[env_idx].item()
                self.current_task[env_idx] = task_idx
                # 重置交互成功状态
                self.interaction_success[env_idx] = False

    def _update_sequence_progress(self):
        """更新任务序列进度"""
        for env_idx in range(self.num_envs):
            # 获取当前序列和步骤
            seq_idx = self.current_task[env_idx].item()
            step_idx = self.current_sequence_step[env_idx].item()

            # 确保索引有效
            if seq_idx < 0 or seq_idx >= len(self.task_sequences) or step_idx < 0:
                continue

            sequence = self.task_sequences[seq_idx]
            if step_idx >= len(sequence):
                continue

            # 获取当前步骤
            step = sequence[step_idx]
            step_type = step["type"]

            # 根据步骤类型检查进度
            if step_type == "navigate":
                # 导航任务：检查是否到达
                progress = 1.0 if self.reached_waypoint[env_idx] else 0.0

                # 如果已到达，准备进入下一步
                if progress > 0.95 and step_idx < len(sequence) - 1:
                    self.current_sequence_step[env_idx] = step_idx + 1
                    # 更新下一步的目标
                    self._update_current_step_target(env_idx)

            elif step_type == "interact":
                # 交互任务：使用交互进度
                progress = 1.0 if self.interaction_success[env_idx] else self.task_progress[env_idx]

                # 如果交互成功，准备进入下一步
                if progress > 0.95 and step_idx < len(sequence) - 1:
                    self.current_sequence_step[env_idx] = step_idx + 1
                    # 更新下一步的目标
                    self._update_current_step_target(env_idx)

            elif step_type == "grab":
                # 抓取任务：如果在正确位置且已交互成功，则视为已抓取
                object_name = step["object"]
                object_idx = list(self.grabable_objects.keys()).index(
                    object_name) if object_name in self.grabable_objects else -1

                # 检查是否已经抓取了物体
                if self.grabbed_object[env_idx].item() == object_idx:
                    progress = 1.0
                else:
                    # 未抓取，但如果位置正确且前一步是交互（打开容器），则可以抓取
                    if self.reached_waypoint[env_idx] and step_idx > 0 and self.task_sequences[seq_idx][step_idx - 1][
                        "type"] == "interact":
                        progress = 0.5  # 位置正确但未完成抓取

                        # 模拟抓取动作（在真实实现中，需要物理交互）
                        # 这里简化为：如果位置正确并且前一步成功完成，则认为可以抓取
                        if self.interaction_success[env_idx]:
                            self.grabbed_object[env_idx] = object_idx
                            progress = 1.0
                    else:
                        progress = 0.0

                # 如果抓取成功，准备进入下一步
                if progress > 0.95 and step_idx < len(sequence) - 1:
                    self.current_sequence_step[env_idx] = step_idx + 1
                    # 更新下一步的目标
                    self._update_current_step_target(env_idx)

            # 更新序列进度
            self.sequence_progress[env_idx, step_idx] = progress

        def _check_sequence_completed(self):
            """检查任务序列是否完成"""
            completed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            for env_idx in range(self.num_envs):
                # 获取当前序列和步骤
                seq_idx = self.current_task[env_idx].item()
                step_idx = self.current_sequence_step[env_idx].item()

                # 确保索引有效
                if seq_idx < 0 or seq_idx >= len(self.task_sequences):
                    continue

                sequence = self.task_sequences[seq_idx]

                # 检查是否到达序列末尾且最后一步完成
                if step_idx == len(sequence) - 1:
                    # 检查最后一步进度
                    last_step_progress = self.sequence_progress[env_idx, step_idx]
                    if last_step_progress > 0.95:
                        completed[env_idx] = True

            return completed

        def compute_observations(self):
            """计算观察，包含序列任务信息"""
            # 调用父类观察计算
            super().compute_observations()

            # 添加序列信息
            # 1. 当前序列步骤（归一化）
            current_step_normalized = self.current_sequence_step.float() / self.max_sequence_length
            current_step_normalized = current_step_normalized.unsqueeze(1)

            # 2. 抓取状态（one-hot）
            grab_state = torch.zeros((self.num_envs, len(self.grabable_objects) + 1), device=self.device)  # +1表示未抓取状态
            for i in range(self.num_envs):
                obj_idx = self.grabbed_object[i].item()
                if obj_idx < 0:
                    grab_state[i, -1] = 1.0  # 未抓取
                else:
                    grab_state[i, obj_idx] = 1.0  # 抓取特定物体

            # 合并序列观察
            sequence_obs = torch.cat([current_step_normalized, grab_state], dim=1)

            # 合并到观察向量
            self.obs_buf = torch.cat([self.obs_buf, sequence_obs], dim=1)

            # 特权观察（如果使用）
            if self.privileged_obs_buf is not None:
                # 添加完整序列进度
                self.privileged_obs_buf = torch.cat([
                    self.privileged_obs_buf,
                    sequence_obs,
                    self.sequence_progress
                ], dim=1)

        def compute_reward(self):
            """计算奖励，包含序列任务奖励"""
            # 更新序列进度
            self._update_sequence_progress()

            # 初始化奖励
            self.rew_buf = torch.zeros(self.num_envs, device=self.device)

            # 基础生存奖励
            self.rew_buf += self._reward_alive() * self.reward_scales["alive"]

            # 每个环境单独计算任务奖励
            for env_idx in range(self.num_envs):
                # 获取当前序列和步骤
                seq_idx = self.current_task[env_idx].item()
                step_idx = self.current_sequence_step[env_idx].item()

                # 确保索引有效
                if seq_idx < 0 or seq_idx >= len(self.task_sequences) or step_idx < 0:
                    continue

                sequence = self.task_sequences[seq_idx]
                if step_idx >= len(sequence):
                    continue

                # 获取当前步骤
                step = sequence[step_idx]
                step_type = step["type"]

                # 根据步骤类型给予奖励
                step_progress = self.sequence_progress[env_idx, step_idx]

                # 进度奖励（鼓励更快完成当前步骤）
                progress_reward = step_progress ** 2  # 平方可以增强完成的重要性
                self.rew_buf[env_idx] += progress_reward * self.reward_scales.get("step_progress", 2.0)

                # 根据步骤类型给予额外奖励
                if step_type == "navigate":
                    # 导航奖励
                    target_name = step["target"]
                    if target_name in self.waypoint_names:
                        wp_idx = self.waypoint_names.index(target_name)
                        target_pos = self.waypoint_positions[wp_idx]

                        # 距离奖励
                        robot_pos = self.robot_root_states[env_idx, :3]
                        distance = torch.norm(robot_pos[:2] - target_pos[:2])
                        distance_reward = torch.exp(-0.2 * distance)

                        self.rew_buf[env_idx] += distance_reward * self.reward_scales.get("waypoint_distance", 0.5)

                elif step_type == "interact":
                    # 交互奖励已在父类中处理
                    pass

                elif step_type == "grab":
                    # 抓取奖励
                    object_name = step["object"]
                    object_idx = list(self.grabable_objects.keys()).index(
                        object_name) if object_name in self.grabable_objects else -1

                    # 如果成功抓取，给予额外奖励
                    if self.grabbed_object[env_idx].item() == object_idx:
                        self.rew_buf[env_idx] += self.reward_scales.get("grab_success", 5.0)

                # 序列完成奖励
                if step_idx == len(sequence) - 1 and step_progress > 0.95:
                    # 最后一步且完成
                    sequence_reward = self.reward_scales.get("sequence_completion", 20.0)
                    self.rew_buf[env_idx] += sequence_reward

            # 碰撞惩罚（仍然保留）
            collision = self._check_collision()
            self.rew_buf -= collision.float() * self.reward_scales.get("kitchen_collision", 2.0)

        def compute_success_flags(self):
            """计算成功标志"""
            # 完整任务序列的成功条件：完成整个序列
            return self._check_sequence_completed()

        def post_physics_step(self):
            """后处理步骤，更新序列状态和成功率"""
            # 调用父类方法
            super().post_physics_step()

            # 更新序列进度
            self._update_sequence_progress()

            # 检查哪些环境完成了序列
            completed = self._check_sequence_completed()

            # 为已完成序列的环境重新采样序列
            if completed.any():
                env_ids = torch.nonzero(completed).squeeze(-1)
                self._sample_task_sequence(env_ids)

            # 计算成功标志并更新成功率
            success_flags = self.compute_success_flags()
            if success_flags.any():
                self.update_success_rate(success_flags)