
# # g1_basic_locomotion.py
# from g1.envs.curriculum.curriculum_base import G1CurriculumBase # Use the updated base class

# from isaacgym.torch_utils import *
# import torch
# import numpy as np

# class G1BasicLocomotion(G1CurriculumBase):
#     """第一阶段：基础运动技能训练.
#     继承自 G1CurriculumBase，实现阶段特定逻辑。
#     """

#     def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
#         # --- Stage 1 Specific Config Processing ---
#         # Stage params should be accessed *after* super().__init__ if they modify base cfg behavior
#         # But we need base_lin_vel_range *before* command resampling might happen in reset.
#         # Let's read them here, but ensure super init happens correctly.
#         stage_params_attr = f'stage{cfg.curriculum.stage}_params' # Access cfg directly before super init
#         stage_params = {}
#         if hasattr(cfg.curriculum, stage_params_attr):
#             stage_params = getattr(cfg.curriculum, stage_params_attr, {})
#         else:
#              print(f"⚠️ G1BasicLocomotion: Warning - Could not find '{stage_params_attr}' in cfg.curriculum during pre-init.")

#         self.base_lin_vel_range = stage_params.get('base_lin_vel_range', 1.0)
#         self.base_ang_vel_range = stage_params.get('base_ang_vel_range', 0.5)
#         print(f"--- G1BasicLocomotion Pre-init: lin_vel_range={self.base_lin_vel_range}, ang_vel_range={self.base_ang_vel_range}")

#         # --- 调用父类初始化 ---
#         super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)
#         print(f"--- G1BasicLocomotion Post-super().__init__ ---")


#         # --- Stage 1 Specific State Initialization (after buffers are ready) ---
#         # 使用 episode_length_buf 来判断命令周期
#         # self.command_duration = 0 # No longer needed

#         # Resetting streak counter logic moved to _resample_commands and reset_idx
#         self.successful_command_tracking_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
#         # Calculate steps needed based on final dt
#         self.resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt)
#         self.min_tracking_steps_for_success = int(self.resampling_interval_steps * 0.8) # 80% of interval
#         print(f"  Resampling interval: {self.resampling_interval_steps} steps")
#         print(f"  Min tracking steps for success: {self.min_tracking_steps_for_success}")


#     def _resample_commands(self, env_ids):
#         """重写命令采样方法以应用课程子阶段进度"""
#         if len(env_ids) == 0: return

#         sub_stage = getattr(self.cfg.curriculum, 'sub_stage', 1)
#         # progress_factor = min(1.0, sub_stage / 5.0) # Linear
#         progress_factor = np.clip(sub_stage / 5.0, 0.1, 1.0) # Clipped

#         current_vel_range = self.base_lin_vel_range * progress_factor
#         current_ang_range = self.base_ang_vel_range * progress_factor

#         # --- 调用父类 LeggedRobot 的 _resample_commands ---
#         # 它处理基础的速度和朝向命令采样
#         super()._resample_commands(env_ids)

#         # --- 如果需要覆盖父类的采样逻辑或进行特定调整，在此处进行 ---
#         # 例如，强制 Stage 1 不使用侧向速度 (如果父类采样了 lin_vel_y)
#         # self.commands[env_ids, 1] = 0.0

#         # --- 重置这些环境的成功跟踪计数器 ---
#         if hasattr(self, 'successful_command_tracking_streak'): # Ensure buffer exists
#             self.successful_command_tracking_streak[env_ids] = 0


#     # --- Stage 1 Specific Reward Overrides ---
#     # 这些函数会覆盖 G1CurriculumBase 中同名的函数
#     def _reward_tracking_lin_vel(self):
#         """(Stage 1 Override) 线速度跟踪奖励 - 更关注精确跟踪"""
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         # 使用更严格的 sigma (例如 0.15 或 0.1)
#         sigma = getattr(self.cfg.rewards, 'tracking_sigma_stage1', 0.15) # Allow config override
#         return torch.exp(-lin_vel_error / sigma)

#     def _reward_tracking_ang_vel(self):
#         """(Stage 1 Override) 角速度跟踪奖励 - 更关注精确跟踪"""
#         ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#         sigma = getattr(self.cfg.rewards, 'tracking_sigma_stage1', 0.15) # Allow config override
#         return torch.exp(-ang_vel_error / sigma)


#     # --- Stage 1 Specific Success Criteria ---
#     def compute_success_criteria(self):
#         """计算每个环境是否在当前命令周期内达到了连续成功跟踪的条件"""
#         # 1. 判断当前步骤是否在跟踪容差内
#         # lin_vel_threshold = getattr(self.cfg.curriculum, 'stage1_success_lin_tol', 0.3)
#         # ang_vel_threshold = getattr(self.cfg.curriculum, 'stage1_success_ang_tol', 0.2)
#         #
#         # lin_vel_close = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
#         # ang_vel_close = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]) < ang_vel_threshold
#         lin_vel_threshold = 0.3 # m/s error tolerance
#         ang_vel_threshold = 0.2 # rad/s error tolerance

#         lin_vel_close = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
#         ang_vel_close = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]) < ang_vel_threshold

#         current_step_tracking = lin_vel_close & ang_vel_close

#         # 2. 更新连续跟踪计数器
#         # 仅当当前步骤在跟踪时才增加计数，否则重置
#         self.successful_command_tracking_streak = (self.successful_command_tracking_streak + 1) * current_step_tracking

#         # 3. 判断是否达到连续成功的步数阈值
#         reached_threshold = self.successful_command_tracking_streak >= self.min_tracking_steps_for_success

#         # 4. 检查是否是命令周期的最后一步 (近似)
#         # 如果 episode_length_buf 能被 resampling_interval_steps 整除 (或余数很小)，认为是周期结束
#         is_end_of_command_cycle = (self.episode_length_buf % self.resampling_interval_steps == (self.resampling_interval_steps - 1))

#         # 5. 成功标志：当达到阈值 *并且* 处于命令周期结束时，才标记为成功
#         #    这样可以确保在一个完整的命令周期内评估跟踪性能
#         # success_flags = reached_threshold & is_end_of_command_cycle
#         # --- 或者更简单的：只要达到阈值就认为是成功（允许在一个周期内多次成功？）---
#         success_flags = reached_threshold

#         # 如果一个环境刚刚成功，重置其计数器，以便它可以为下一个命令周期重新计数
#         # self.successful_command_tracking_streak[success_flags] = 0 # Reset counter immediately on success?

#         return success_flags


#     def post_physics_step(self):
#         """Stage 1 后处理: 计算成功标志并放入 extras"""
#         # --- 1. 调用父类 (G1CurriculumBase) 的后处理 ---
#         # 这会处理相位计算、命令重采样、调用 LeggedRobot 的后处理等
#         super().post_physics_step()

#         # --- 2. 计算本阶段的成功标准 ---
#         # compute_success_criteria 更新了 streak 并返回了哪些环境达到了连续成功阈值
#         success_flags = self.compute_success_criteria()

#         # --- 3. 将成功标志放入 extras 字典供 Runner 使用 ---

#         self.extras["success_flags"] = success_flags.clone() # Send a copy

#         # --- 4. 重置成功计数器 (可选，取决于你的成功定义) ---
#         # 如果成功是一次性事件（达到N次就算成功），则需要重置计数器
#         # 如果 Runner 期望每一步都报告是否处于“成功状态”，则不需要在这里重置
#         # 假设 Runner 会统计 True 的比例，我们在成功时重置计数器，以便开始下一个周期的计数
#         if success_flags.any():
#             self.successful_command_tracking_streak[success_flags] = 0

#         # --- 5. 重置因环境 reset 而中断的计数器 ---
#         # (父类的 post_physics_step 调用了 reset_idx，reset_idx 会重置 streak)
#         # 所以这里不需要再次处理 reset_buf


#     def reset_idx(self, env_ids):
#          """ 重写 reset_idx 以重置 Stage 1 特定状态 """
#          if len(env_ids) == 0: return

#          # 调用父类 reset (重置机器人状态、命令、基础缓冲区等)
#          super().reset_idx(env_ids)

#          # 重置本阶段特定的状态
#          if hasattr(self, 'successful_command_tracking_streak'): # Ensure buffer exists
#               self.successful_command_tracking_streak[env_ids] = 0
#          # print(f"  G1BasicLocomotion: Resetting success streak for {len(env_ids)} environments.")


# from g1.envs.curriculum.curriculum_base import G1CurriculumBase # Use the updated base class

# from isaacgym.torch_utils import *
# import torch
# import numpy as np

# class G1BasicLocomotion(G1CurriculumBase):
#     """第一阶段：基础运动技能训练 (支持嵌套课程学习).
#     继承自 G1CurriculumBase，实现阶段特定逻辑，特别是子阶段处理。
#     """

#     def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
#         # --- 1. 预处理配置，特别是嵌套课程参数 ---
#         self.is_nested_curriculum = getattr(cfg, 'nested_locomotion_curriculum', False)
#         self.current_sub_stage = getattr(cfg.curriculum, 'sub_stage', 1)
#         self.active_joint_indices = None
#         self.num_total_dofs = 43 # G1 的总自由度

#         if self.is_nested_curriculum and hasattr(cfg, 'sub_stage_params'):
#             print(f"--- G1BasicLocomotion (Nested): Initializing for Sub-Stage 1.{self.current_sub_stage} ---")
#             sub_stage_cfg = cfg.sub_stage_params.get(self.current_sub_stage)
#             if sub_stage_cfg:
#                 print(f"  Loading params for sub-stage {self.current_sub_stage}: {sub_stage_cfg.get('name', 'N/A')}")
#                 # !!! 关键: 在 super().__init__ 之前覆盖主 cfg 的 obs/action 维度 !!!
#                 self.num_observations_override = sub_stage_cfg.get('num_observations')
#                 self.num_actions_override = sub_stage_cfg.get('num_actions')
#                 self.num_privileged_obs_override = sub_stage_cfg.get('num_privileged_obs')

#                 if self.num_observations_override is None or self.num_actions_override is None:
#                     print(f"  ⚠️ Warning: Sub-stage {self.current_sub_stage} config missing num_observations or num_actions. Using defaults from main config.")
#                     self.num_observations_override = cfg.env.num_observations
#                     self.num_actions_override = cfg.env.num_actions
#                     self.num_privileged_obs_override = cfg.env.num_privileged_obs
#                 else:
#                      # 更新主配置，以便 BaseTask 初始化正确的缓冲区大小
#                      cfg.env.num_observations = self.num_observations_override
#                      cfg.env.num_actions = self.num_actions_override
#                      if self.num_privileged_obs_override is not None:
#                           cfg.env.num_privileged_obs = self.num_privileged_obs_override
#                      else: # 如果子阶段没定义特权观测，则跟随普通观测+3
#                           cfg.env.num_privileged_obs = self.num_observations_override + 3

#                 print(f"  Overriding env dimensions: Obs={cfg.env.num_observations}, PrivObs={cfg.env.num_privileged_obs}, Act={cfg.env.num_actions}")

#                 # 存储子阶段特定的速度范围
#                 self.base_lin_vel_range = sub_stage_cfg.get('base_lin_vel_range', cfg.commands.ranges.lin_vel_x[1])
#                 self.base_ang_vel_range = sub_stage_cfg.get('base_ang_vel_range', cfg.commands.ranges.ang_vel_yaw[1])
#                 print(f"  Sub-stage velocity ranges: Lin={self.base_lin_vel_range}, Ang={self.base_ang_vel_range}")

#                 # 存储活跃关节信息 (将在 _init_buffers 或之后使用)
#                 self.active_joint_keywords = sub_stage_cfg.get('active_joints', ["all"])
#                 print(f"  Active joint keywords: {self.active_joint_keywords}")

#             else:
#                 print(f"⚠️ G1BasicLocomotion: Warning - Could not find sub_stage_params for sub-stage {self.current_sub_stage}. Using main config defaults.")
#                 # 使用主配置的速度范围
#                 self.base_lin_vel_range = cfg.commands.ranges.lin_vel_x[1]
#                 self.base_ang_vel_range = cfg.commands.ranges.ang_vel_yaw[1]
#                 self.active_joint_keywords = ["all"]
#         else:
#             print(f"--- G1BasicLocomotion: Initializing (No Nested Curriculum or Config) ---")
#             # 非嵌套课程或无配置，使用主配置
#             self.base_lin_vel_range = cfg.commands.ranges.lin_vel_x[1]
#             self.base_ang_vel_range = cfg.commands.ranges.ang_vel_yaw[1]
#             self.active_joint_keywords = ["all"]
#             # 确保维度设置正确
#             self.num_actions_override = cfg.env.num_actions
#             self.num_observations_override = cfg.env.num_observations
#             self.num_privileged_obs_override = cfg.env.num_privileged_obs


#         # --- 2. 调用父类初始化 (父类会处理基础 G1 设置和课程阶段属性) ---
#         # 父类会使用上面更新后的 cfg.env.num_observations/actions 初始化缓冲区
#         super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)
#         print(f"--- G1BasicLocomotion Post-super().__init__ ---")
#         print(f"  Final env dimensions used: Obs={self.num_observations}, PrivObs={self.num_privileged_obs}, Act={self.num_actions}")

#         # --- 3. 嵌套课程相关的状态初始化 (在缓冲区和 DoF 名称可用后) ---
#         if self.is_nested_curriculum:
#             # 获取活跃关节的索引 (基于父类加载的 self.dof_names)
#             self._compute_active_joint_indices()
#             print(f"  Active DoF indices ({len(self.active_joint_indices)}): {self.active_joint_indices.tolist()}")
#             # 验证计算出的 active_joint_indices 数量是否与配置的 num_actions 匹配
#             if len(self.active_joint_indices) != self.num_actions_override:
#                  print(f"  ❌ CRITICAL WARNING: Number of active joint indices ({len(self.active_joint_indices)}) does not match configured num_actions ({self.num_actions_override}) for sub-stage {self.current_sub_stage}!")
#                  print(f"     Active keywords: {self.active_joint_keywords}")
#                  print(f"     All DoF names: {self.dof_names}")


#         # --- 4. Stage 1 特定的状态变量 (与原版类似) ---
#         self.successful_command_tracking_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
#         # 计算成功所需的步数
#         self.resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt) if self.dt > 0 else 0
#         self.min_tracking_steps_for_success = int(self.resampling_interval_steps * 0.8) if self.resampling_interval_steps > 0 else 0 # 80% of interval
#         print(f"  Resampling interval: {self.resampling_interval_steps} steps")
#         print(f"  Min tracking steps for success: {self.min_tracking_steps_for_success}")

#         # --- 5. 存储锁定关节的 PD 增益 ---
#         self.locked_stiffness = getattr(self.cfg.control, 'locked_stiffness', 500.0)
#         self.locked_damping = getattr(self.cfg.control, 'locked_damping', 50.0)
#         print(f"  Locked joint gains: P={self.locked_stiffness}, D={self.locked_damping}")


#     def _compute_active_joint_indices(self):
#         """根据 self.active_joint_keywords 计算活跃关节的索引列表。"""
#         if "all" in self.active_joint_keywords:
#             self.active_joint_indices = torch.arange(self.num_total_dofs, device=self.device, dtype=torch.long)
#             return

#         active_indices = []
#         if hasattr(self, 'dof_names') and self.dof_names:
#             for idx, name in enumerate(self.dof_names):
#                 is_active = False
#                 for keyword in self.active_joint_keywords:
#                     if keyword in name:
#                         is_active = True
#                         break
#                 if is_active:
#                     active_indices.append(idx)
#             self.active_joint_indices = torch.tensor(sorted(list(set(active_indices))), device=self.device, dtype=torch.long)
#         else:
#             print("⚠️ G1BasicLocomotion: Warning - self.dof_names not available during _compute_active_joint_indices. Assuming all joints active.")
#             self.active_joint_indices = torch.arange(self.num_total_dofs, device=self.device, dtype=torch.long)


#     def _process_dof_props(self, props, env_id):
#         """ 重写 DoF 属性处理，根据子阶段锁定非活跃关节。"""
#         # 调用父类处理基础限制和存储 (确保父类方法存在且被调用)
#         processed_props = super()._process_dof_props(props, env_id)

#         # 如果是嵌套课程，则应用锁定逻辑
#         if self.is_nested_curriculum and self.active_joint_indices is not None:
#             # 创建活跃关节索引的集合，以便快速查找
#             active_set = set(self.active_joint_indices.cpu().tolist())

#             for i in range(len(processed_props)): # 遍历所有 43 个 DoF
#                 if i not in active_set: # 如果当前 DoF 不活跃
#                     # 设置高刚度和高阻尼来锁定关节
#                     processed_props["stiffness"][i] = self.locked_stiffness
#                     processed_props["damping"][i] = self.locked_damping
#                     # 驱动模式可以保持 EFFORT，高 PD 增益会使其难以移动
#                     # processed_props["driveMode"][i] = gymapi.DOF_MODE_POS # 或者强制位置模式? 实验确定
#                 # else: # 如果 DoF 活跃，保持父类或配置中设置的正常 PD 增益
#                     # (父类的 _process_dof_props 已经根据 cfg.control 设置了活跃关节的增益)
#                     pass
#         return processed_props


#     def _resample_commands(self, env_ids):
#         """重写命令采样方法以应用课程子阶段进度或固定范围。"""
#         if len(env_ids) == 0:
#             return

#         # 使用存储在 self 中的当前子阶段的速度范围
#         current_vel_range = self.base_lin_vel_range
#         current_ang_range = self.base_ang_vel_range

#         # --- 重采样命令 (使用当前范围) ---
#         self.commands[env_ids, 0] = torch_rand_float(
#             -current_vel_range, current_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
#         self.commands[env_ids, 1] = torch_rand_float(
#             -current_vel_range, current_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
#         # 使用 heading command 或 yaw command
#         if self.cfg.commands.heading_command:
#             # 采样 heading 目标
#             self.commands[env_ids, 3] = torch_rand_float(
#                 self.cfg.commands.ranges.heading[0], self.cfg.commands.ranges.heading[1], (len(env_ids), 1), device=self.device).squeeze(1)
#             # yaw 速度命令 (index 2) 会在 _post_physics_step_callback 中根据 heading error 计算
#         else:
#             # 直接采样 yaw 速度命令
#             self.commands[env_ids, 2] = torch_rand_float(
#                 -current_ang_range, current_ang_range, (len(env_ids), 1), device=self.device).squeeze(1)


#         # set small commands to zero (copied from base LeggedRobot)
#         self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

#         # 重置这些环境的成功跟踪计数器
#         if hasattr(self, 'successful_command_tracking_streak'): # Ensure buffer exists
#             self.successful_command_tracking_streak[env_ids] = 0


#     def compute_observations(self):
#         """ 构建观测向量，只包含活跃关节的信息。"""
#         # 调用父类计算基础观测 (会产生一个包含所有 43 个关节信息的 obs_buf)
#         super().compute_observations() # G1CurriculumBase.compute_observations

#         # 如果是嵌套课程，需要根据活跃关节进行调整或屏蔽
#         # 当前父类实现已经是正确的：它基于 self.num_observations，
#         # 而我们在 __init__ 中已经根据子阶段修改了 self.num_observations。
#         # 父类的 compute_observations 会使用 self.dof_pos[:, :self.num_actions]
#         # 和 self.actions[:, :self.num_actions] 等，这天然地只使用了“活跃”的部分，
#         # 因为 self.num_actions 本身就代表了活跃关节的数量。
#         # 所以理论上不需要在这里做额外的屏蔽。

#         # 唯一需要确认的是噪声：_get_noise_scale_vec 是否正确处理了维度变化？
#         # 它应该在 G1CurriculumBase 中基于当前的 self.num_observations 来构建，
#         # 所以也应该是正确的。

#         # 验证最终 obs_buf 的维度是否与预期一致
#         if self.obs_buf.shape[1] != self.num_observations:
#              print(f"❌ CRITICAL ERROR in G1BasicLocomotion.compute_observations: Final obs_buf dim ({self.obs_buf.shape[1]}) != self.num_observations ({self.num_observations}) for sub-stage {self.current_sub_stage}!")
#              # 尝试修复维度，但这通常表示上游逻辑错误
#              if self.obs_buf.shape[1] > self.num_observations:
#                   self.obs_buf = self.obs_buf[:, :self.num_observations]
#              else:
#                   padding = torch.zeros((self.num_envs, self.num_observations - self.obs_buf.shape[1]), device=self.device)
#                   self.obs_buf = torch.cat([self.obs_buf, padding], dim=-1)
#              print(f"   Attempted to fix obs_buf shape to ({self.num_envs}, {self.num_observations}).")

#         if self.privileged_obs_buf is not None and self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
#             print(f"❌ CRITICAL ERROR in G1BasicLocomotion.compute_observations: Final privileged_obs_buf dim ({self.privileged_obs_buf.shape[1]}) != self.num_privileged_obs ({self.num_privileged_obs}) for sub-stage {self.current_sub_stage}!")
#              # 尝试修复维度
#             if self.privileged_obs_buf.shape[1] > self.num_privileged_obs:
#                   self.privileged_obs_buf = self.privileged_obs_buf[:, :self.num_privileged_obs]
#             else:
#                   padding = torch.zeros((self.num_envs, self.num_privileged_obs - self.privileged_obs_buf.shape[1]), device=self.device)
#                   self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, padding], dim=-1)
#             print(f"   Attempted to fix privileged_obs_buf shape to ({self.num_envs}, {self.num_privileged_obs}).")


#     def _compute_torques(self, actions):
#         """计算扭矩，只为活跃关节计算控制力矩，非活跃关节用力矩强制锁定。"""
#         # actions 的维度应该是当前子阶段的 self.num_actions
#         if actions.shape[1] != self.num_actions:
#              print(f"❌ ERROR _compute_torques: Received actions shape {actions.shape} does not match current env num_actions {self.num_actions} (Sub-stage {self.current_sub_stage})")
#              # 返回零力矩可能导致不稳定，返回一个全尺寸的零力矩
#              return torch.zeros(self.num_envs, self.num_total_dofs, device=self.device)

#         # 初始化一个全尺寸 (43 DOF) 的力矩张量
#         torques_full = torch.zeros(self.num_envs, self.num_total_dofs, device=self.device)

#         # 如果是嵌套课程并且有活跃关节索引
#         if self.is_nested_curriculum and self.active_joint_indices is not None:
#             # 1. 计算活跃关节的控制力矩
#             active_actions_scaled = actions * self.cfg.control.action_scale # actions 已经是活跃部分的了
#             control_type = self.cfg.control.control_type

#             # 获取活跃关节的当前状态和目标
#             active_dof_pos = self.dof_pos[:, self.active_joint_indices]
#             active_dof_vel = self.dof_vel[:, self.active_joint_indices]
#             active_default_pos = self.default_dof_pos[:, self.active_joint_indices] # default_dof_pos 是 [1, 43]
#             active_p_gains = self.p_gains[self.active_joint_indices] # p_gains 是 [43]
#             active_d_gains = self.d_gains[self.active_joint_indices] # d_gains 是 [43]
#             active_torque_limits = self.torque_limits[self.active_joint_indices] # torque_limits 是 [43]

#             if control_type == "P":
#                 active_torques = active_p_gains * (active_actions_scaled + active_default_pos - active_dof_pos) - active_d_gains * active_dof_vel
#             # elif control_type == "V": # 保持与 LeggedRobot 一致
#             #     if self.last_dof_vel.shape[1] == self.num_total_dofs: # 确保 last_dof_vel 是全尺寸的
#             #         active_last_dof_vel = self.last_dof_vel[:, self.active_joint_indices]
#             #     else: # 如果 last_dof_vel 维度不匹配，则假设为零
#             #          active_last_dof_vel = torch.zeros_like(active_dof_vel)
#             #     dt = self.dt; if dt <= 0: dt = 1e-5
#             #     active_torques = active_p_gains * (active_actions_scaled - active_dof_vel) - active_d_gains * ((active_dof_vel - active_last_dof_vel) / dt)
#             elif control_type == "T":
#                 active_torques = active_actions_scaled
#             else:
#                 raise NameError(f"Unknown controller type: {control_type}")

#             active_torques = torch.clip(active_torques, -active_torque_limits, active_torque_limits)

#             # 将计算出的活跃力矩填充到全尺寸力矩张量中
#             torques_full[:, self.active_joint_indices] = active_torques

#             # 2. 计算非活跃关节的锁定力矩
#             locked_indices = torch.tensor([i for i in range(self.num_total_dofs) if i not in self.active_joint_indices.cpu().tolist()], device=self.device, dtype=torch.long)
#             if len(locked_indices) > 0:
#                  locked_dof_pos = self.dof_pos[:, locked_indices]
#                  locked_dof_vel = self.dof_vel[:, locked_indices]
#                  # 锁定目标通常是默认初始姿态
#                  locked_target_pos = self.default_dof_pos[:, locked_indices]
#                  # 使用锁定 PD 增益
#                  locked_torques = self.locked_stiffness * (locked_target_pos - locked_dof_pos) - self.locked_damping * locked_dof_vel
#                  # 获取非活跃关节的力矩限制
#                  locked_torque_limits = self.torque_limits[locked_indices]
#                  locked_torques = torch.clip(locked_torques, -locked_torque_limits, locked_torque_limits)
#                  # 填充到全尺寸力矩张量
#                  torques_full[:, locked_indices] = locked_torques

#         else: # 如果不是嵌套课程或无活跃关节信息，则假定所有关节都活跃
#             # 直接使用父类的方法计算所有关节的力矩
#             torques_full = super()._compute_torques(actions) # actions 应该是 43 维

#         # 返回全尺寸的力矩张量 (num_envs, 43)
#         return torques_full

#     # --- Stage 1 Specific Reward Overrides ---
#     # 这些奖励函数基于 self.commands 和 self.base_*_vel，这些不受嵌套课程影响
#     # 只需要确保 sigma 值是合理的。
#     def _reward_tracking_lin_vel(self):
#         """(Stage 1 Override) 线速度跟踪奖励 - 更关注精确跟踪"""
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         # 可以根据子阶段调整 sigma
#         sub_stage_cfg = self.cfg.sub_stage_params.get(self.current_sub_stage, {})
#         sigma = sub_stage_cfg.get('tracking_sigma', 0.15) # 从子阶段配置获取，默认为 0.15
#         return torch.exp(-lin_vel_error / sigma)

#     def _reward_tracking_ang_vel(self):
#         """(Stage 1 Override) 角速度跟踪奖励 - 更关注精确跟踪"""
#         # 根据是否使用 heading command 确定目标角速度
#         if self.cfg.commands.heading_command:
#             # 目标角速度在 self.commands[:, 2] 中 (由 callback 计算)
#             target_ang_vel = self.commands[:, 2]
#         else:
#             # 目标角速度直接是命令 (也在 self.commands[:, 2])
#             target_ang_vel = self.commands[:, 2]

#         ang_vel_error = torch.square(target_ang_vel - self.base_ang_vel[:, 2])
#         sub_stage_cfg = self.cfg.sub_stage_params.get(self.current_sub_stage, {})
#         sigma = sub_stage_cfg.get('tracking_sigma', 0.15)
#         return torch.exp(-ang_vel_error / sigma)

#     # --- Stage 1 Specific Success Criteria ---
#     def compute_success_criteria(self):
#         """计算每个环境在当前步骤是否满足成功跟踪命令的条件"""
#         # 可以根据子阶段调整阈值
#         sub_stage_cfg = self.cfg.sub_stage_params.get(self.current_sub_stage, {})
#         lin_vel_threshold = sub_stage_cfg.get('success_lin_tol', 0.3)
#         ang_vel_threshold = sub_stage_cfg.get('success_ang_tol', 0.2)
#         min_track_steps = sub_stage_cfg.get('min_tracking_steps_for_success', self.min_tracking_steps_for_success)

#         # 判断是否成功跟踪命令 (使用较宽松的阈值判断单步成功)
#         lin_vel_close_enough = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
#         # 使用正确的角速度目标
#         if self.cfg.commands.heading_command: target_ang_vel = self.commands[:, 2]
#         else: target_ang_vel = self.commands[:, 2]
#         ang_vel_close_enough = torch.abs(target_ang_vel - self.base_ang_vel[:, 2]) < ang_vel_threshold

#         current_step_success = lin_vel_close_enough & ang_vel_close_enough

#         # 更新连续成功计数
#         self.successful_command_tracking_streak = (self.successful_command_tracking_streak + 1) * current_step_success

#         # 判断是否达到了连续成功的阈值
#         # 使用从子阶段配置读取的或默认的步数阈值
#         episode_success_flags = self.successful_command_tracking_streak >= min_track_steps

#         return episode_success_flags


#     def post_physics_step(self):
#         """重写后处理步骤以计算并将成功标志放入 extras"""
#         # 调用父类的 post_physics_step (处理终止、重置、计算基础奖励、观测等)
#         # 父类的 post_physics_step 会调用 _post_physics_step_callback
#         # _post_physics_step_callback 会调用 _resample_commands
#         # compute_reward 会调用 _reward_* 函数
#         super().post_physics_step()

#         # 计算本阶段的成功标准 (使用可能已更新的 streak 计数器)
#         success_flags = self.compute_success_criteria()

#         # --- 将成功标志放入 extras 字典 ---
#         self.extras["success_flags"] = success_flags.clone() # 发送副本

#         # --- 重置成功计数器 ---
#         # 如果 Runner 基于 extras["success_flags"] 的 True 比例计算成功率，
#         # 那么在成功时重置计数器，以便为下一个命令周期计数。
#         if success_flags.any():
#              # 只重置那些刚刚达到成功的环境的计数器
#              self.successful_command_tracking_streak[success_flags] = 0

#         # 当环境被重置时 (由父类的 post_physics_step -> reset_idx 处理 reset_buf)
#         # 相应的成功跟踪计数器也需要重置，这在 reset_idx 中完成。


#     def reset_idx(self, env_ids):
#          """ 重写 reset_idx 以重置 Stage 1 特定状态 """
#          if len(env_ids) == 0: return

#          # 调用父类 reset (重置机器人状态、命令、基础缓冲区等)
#          super().reset_idx(env_ids)

#          # 重置本阶段特定的状态
#          if hasattr(self, 'successful_command_tracking_streak'): # Ensure buffer exists
#               self.successful_command_tracking_streak[env_ids] = 0

# g1_basic_locomotion.py
from g1.envs.curriculum.curriculum_base import G1CurriculumBase
from isaacgym.torch_utils import *
import torch
import numpy as np
from g1.utils.helpers import DotDict # <--- ADD THIS IMPORT

class G1BasicLocomotion(G1CurriculumBase):
    """第一阶段：基础运动技能训练 (支持嵌套课程学习).
    继承自 G1CurriculumBase，实现阶段特定逻辑，特别是子阶段处理。
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        # --- 1. 预处理配置，特别是嵌套课程参数 ---
        self.is_nested_curriculum = getattr(cfg, 'nested_locomotion_curriculum', False)
        self.current_sub_stage = getattr(cfg.curriculum, 'sub_stage', 1) # Get sub_stage BEFORE super init
        self.active_joint_indices = None
        self.num_total_dofs = 43 # G1 的总自由度

        if self.is_nested_curriculum and hasattr(cfg, 'sub_stage_params'):
            print(f"--- G1BasicLocomotion (Nested): Initializing for Sub-Stage 1.{self.current_sub_stage} ---")
            # Get sub-stage params as a standard dict first
            sub_stage_cfg_dict = cfg.sub_stage_params.get(self.current_sub_stage)
            if sub_stage_cfg_dict:
                # Now convert to DotDict for easier access within this method
                sub_stage_cfg = DotDict(sub_stage_cfg_dict)
                print(f"  Loading params for sub-stage {self.current_sub_stage}: {sub_stage_cfg.get('name', 'N/A')}")

                # Get dimensions from sub-stage config
                self.num_observations_override = sub_stage_cfg.get('num_observations')
                self.num_actions_override = sub_stage_cfg.get('num_actions')
                self.num_privileged_obs_override = sub_stage_cfg.get('num_privileged_obs')

                if self.num_observations_override is None or self.num_actions_override is None:
                    print(f"  ⚠️ Warning: Sub-stage {self.current_sub_stage} config missing num_observations or num_actions. Using parent config defaults.")
                    self.num_observations_override = cfg.env.num_observations
                    self.num_actions_override = cfg.env.num_actions
                    self.num_privileged_obs_override = getattr(cfg.env, 'num_privileged_obs', None)
                # else: # Dimensions found in sub-stage config
                    # Override main cfg dimensions BEFORE calling super().__init__
                cfg.env.num_observations = self.num_observations_override
                cfg.env.num_actions = self.num_actions_override
                if self.num_privileged_obs_override is not None:
                    cfg.env.num_privileged_obs = self.num_privileged_obs_override
                else: # Infer privileged obs dim if not specified
                    cfg.env.num_privileged_obs = self.num_observations_override + 3 # Assume base lin vel added

                print(f"  Overriding main cfg dimensions: Obs={cfg.env.num_observations}, PrivObs={cfg.env.num_privileged_obs}, Act={cfg.env.num_actions}")

                # Store sub-stage specific velocity ranges
                self.base_lin_vel_range = sub_stage_cfg.get('base_lin_vel_range', cfg.commands.ranges.lin_vel_x[1])
                self.base_ang_vel_range = sub_stage_cfg.get('base_ang_vel_range', cfg.commands.ranges.ang_vel_yaw[1])
                print(f"  Sub-stage velocity ranges: Lin={self.base_lin_vel_range}, Ang={self.base_ang_vel_range}")

                # Store active joint info (keywords for now, indices computed later)
                self.active_joint_keywords = sub_stage_cfg.get('active_joints', ["all"])
                print(f"  Active joint keywords: {self.active_joint_keywords}")

            else:
                print(f"⚠️ G1BasicLocomotion: Warning - Could not find sub_stage_params for sub-stage {self.current_sub_stage}. Using main config defaults.")
                self.base_lin_vel_range = cfg.commands.ranges.lin_vel_x[1]
                self.base_ang_vel_range = cfg.commands.ranges.ang_vel_yaw[1]
                self.active_joint_keywords = ["all"]
                # Ensure override vars reflect the actual dimensions used
                self.num_actions_override = cfg.env.num_actions
                self.num_observations_override = cfg.env.num_observations
                self.num_privileged_obs_override = getattr(cfg.env, 'num_privileged_obs', None)
        else:
            # (Keep non-nested logic as before)
            print(f"--- G1BasicLocomotion: Initializing (No Nested Curriculum or Config) ---")
            self.base_lin_vel_range = cfg.commands.ranges.lin_vel_x[1]
            self.base_ang_vel_range = cfg.commands.ranges.ang_vel_yaw[1]
            self.active_joint_keywords = ["all"]
            self.num_actions_override = cfg.env.num_actions
            self.num_observations_override = cfg.env.num_observations
            self.num_privileged_obs_override = getattr(cfg.env, 'num_privileged_obs', None)


        # --- 2. 调用父类初始化 ---
        # (Keep super().__init__ call as before)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)
        print(f"--- G1BasicLocomotion Post-super().__init__ ---")
        # (Keep dimension verification as before)
        if self.num_actions != self.num_actions_override: print(f"❌ CRITICAL WARNING: Final self.num_actions ({self.num_actions}) != expected override ({self.num_actions_override})!")
        if self.num_observations != self.num_observations_override: print(f"❌ CRITICAL WARNING: Final self.num_observations ({self.num_observations}) != expected override ({self.num_observations_override})!")
        print(f"  Final instance dimensions: Obs={self.num_observations}, PrivObs={self.num_privileged_obs}, Act={self.num_actions}")


        # --- 3. 嵌套课程相关的状态初始化 ---
        # (Keep _compute_active_joint_indices call and checks as before)
        if not hasattr(self, 'dof_names') or not self.dof_names: print("❌ CRITICAL ERROR: self.dof_names not initialized by base class!")
        self._compute_active_joint_indices()
        if self.active_joint_indices is not None:
            print(f"  Active DoF indices ({len(self.active_joint_indices)}): {self.active_joint_indices.tolist()}")
            if len(self.active_joint_indices) != self.num_actions_override: print(f"❌ CRITICAL WARNING: Computed active joint indices ({len(self.active_joint_indices)}) != configured num_actions ({self.num_actions_override})!")


        # --- 4. Stage 1 特定的状态变量 ---
        # (Keep streak, steps calculation as before)
        self.successful_command_tracking_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt) if self.dt > 0 else 0
        self.min_tracking_steps_for_success = int(self.resampling_interval_steps * 0.8) if self.resampling_interval_steps > 0 else 0
        print(f"  Resampling interval: {self.resampling_interval_steps} steps")
        print(f"  Min tracking steps for success: {self.min_tracking_steps_for_success}")

        # --- 5. 存储锁定关节的 PD 增益 ---
        # (Keep locked gain storage as before)
        self.locked_stiffness = getattr(self.cfg.control, 'locked_stiffness', 500.0)
        self.locked_damping = getattr(self.cfg.control, 'locked_damping', 50.0)
        print(f"  Locked joint gains: P={self.locked_stiffness}, D={self.locked_damping}")


    # --- _compute_active_joint_indices ---
    # (Keep implementation as before)
    def _compute_active_joint_indices(self):
        """根据 self.active_joint_keywords 计算活跃关节的索引列表 (使用全 43 DoF)。"""
        if "all" in self.active_joint_keywords:
            self.active_joint_indices = torch.arange(self.num_dof, device=self.device, dtype=torch.long)
            return
        active_indices = []
        if hasattr(self, 'dof_names') and self.dof_names:
            for idx, name in enumerate(self.dof_names):
                is_active = any(keyword in name for keyword in self.active_joint_keywords)
                if is_active: active_indices.append(idx)
            self.active_joint_indices = torch.tensor(sorted(list(set(active_indices))), device=self.device, dtype=torch.long)
        else:
            print("⚠️ G1BasicLocomotion: Warning - self.dof_names not available. Cannot determine active indices.")
            self.active_joint_indices = None


    # --- _process_dof_props ---
    # (Keep implementation as before)
    def _process_dof_props(self, props, env_id):
        processed_props = super()._process_dof_props(props, env_id)
        if self.is_nested_curriculum and self.active_joint_indices is not None:
            active_set = set(self.active_joint_indices.cpu().tolist())
            num_props = len(processed_props['stiffness'])
            if num_props != self.num_dof: print(f"❌ ERROR _process_dof_props: Prop count {num_props} != self.num_dof {self.num_dof}"); return processed_props
            for i in range(num_props):
                if i not in active_set:
                    processed_props["stiffness"][i] = self.locked_stiffness
                    processed_props["damping"][i] = self.locked_damping
        return processed_props

    # --- _resample_commands ---
    # (Keep implementation as before)
    def _resample_commands(self, env_ids):
        if len(env_ids) == 0: return
        current_lin_vel_range = self.base_lin_vel_range
        current_ang_vel_range = self.base_ang_vel_range
        commands_buf = self.commands; cmd_cfg = self.cfg.commands
        commands_buf[env_ids, 0] = torch_rand_float(-current_lin_vel_range, current_lin_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
        commands_buf[env_ids, 1] = torch_rand_float(-current_lin_vel_range, current_lin_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
        if cmd_cfg.heading_command and commands_buf.shape[1] >= 4: commands_buf[env_ids, 3] = torch_rand_float(cmd_cfg.ranges.heading[0], cmd_cfg.ranges.heading[1], (len(env_ids), 1), device=self.device).squeeze(1)
        elif commands_buf.shape[1] >= 3: commands_buf[env_ids, 2] = torch_rand_float(-current_ang_vel_range, current_ang_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
        commands_buf[env_ids, :2] *= (torch.norm(commands_buf[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
        if hasattr(self, 'successful_command_tracking_streak'): self.successful_command_tracking_streak[env_ids] = 0


    # --- compute_observations ---
    # (Keep implementation as before)
    def compute_observations(self):
        super().compute_observations()
        # Optional final checks
        if self.obs_buf.shape[1] != self.num_observations: print(f"❌ ERROR obs shape mismatch: {self.obs_buf.shape[1]} vs {self.num_observations}")
        if self.privileged_obs_buf is not None and self.privileged_obs_buf.shape[1] != self.num_privileged_obs: print(f"❌ ERROR priv_obs shape mismatch: {self.privileged_obs_buf.shape[1]} vs {self.num_privileged_obs}")

    # --- _compute_torques ---
    # (Keep implementation as before)
    def _compute_torques(self, actions):
        if actions.shape[1] != self.num_actions: print(f"❌ ERROR _compute_torques: actions shape {actions.shape} != num_actions {self.num_actions}"); return torch.zeros(self.num_envs, self.num_dof, device=self.device)
        torques_full = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        if self.is_nested_curriculum and self.active_joint_indices is not None:
            if len(self.active_joint_indices) != self.num_actions: print(f"❌ ERROR _compute_torques: index/action mismatch"); return torques_full
            active_actions_scaled = actions * self.cfg.control.action_scale; control_type = self.cfg.control.control_type
            active_dof_pos = self.dof_pos[:, self.active_joint_indices]; active_dof_vel = self.dof_vel[:, self.active_joint_indices]
            active_default_pos = self.default_dof_pos[:, self.active_joint_indices]; active_p_gains = self.p_gains[self.active_joint_indices]
            active_d_gains = self.d_gains[self.active_joint_indices]; active_torque_limits = self.torque_limits[self.active_joint_indices]
            if control_type == "P": active_torques = active_p_gains * (active_actions_scaled + active_default_pos - active_dof_pos) - active_d_gains * active_dof_vel
            elif control_type == "T": active_torques = active_actions_scaled
            else: raise NameError(f"Unknown controller type: {control_type}")
            active_torques = torch.clip(active_torques, -active_torque_limits, active_torque_limits)
            torques_full[:, self.active_joint_indices] = active_torques
            is_active_mask = torch.zeros(self.num_dof, dtype=torch.bool, device=self.device); is_active_mask[self.active_joint_indices] = True
            locked_indices = torch.nonzero(~is_active_mask).squeeze(-1)
            if len(locked_indices) > 0:
                 locked_dof_pos = self.dof_pos[:, locked_indices]; locked_dof_vel = self.dof_vel[:, locked_indices]
                 locked_target_pos = self.default_dof_pos[:, locked_indices]; locked_torque_limits = self.torque_limits[locked_indices]
                 locked_torques = self.locked_stiffness * (locked_target_pos - locked_dof_pos) - self.locked_damping * locked_dof_vel
                 locked_torques = torch.clip(locked_torques, -locked_torque_limits, locked_torque_limits)
                 torques_full[:, locked_indices] = locked_torques
        else:
            if self.num_actions != self.num_dof: print(f"❌ ERROR non-nested torque: num_actions {self.num_actions} != num_dof {self.num_dof}"); return torques_full
            torques_full = super()._compute_torques(actions)
        return torques_full

    # --- Stage 1 Specific Reward Overrides ---
    # (Keep implementation as before)
    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1); sigma = 0.15
        if self.is_nested_curriculum and hasattr(self.cfg, 'sub_stage_params'): sub_cfg = self.cfg.sub_stage_params.get(self.current_sub_stage, {}); sigma = sub_cfg.get('tracking_sigma', sigma)
        return torch.exp(-lin_vel_error / sigma)

    def _reward_tracking_ang_vel(self):
        if self.cfg.commands.heading_command: target_ang_vel = self.commands[:, 2]
        else: target_ang_vel = self.commands[:, 2]
        ang_vel_error = torch.square(target_ang_vel - self.base_ang_vel[:, 2]); sigma = 0.15
        if self.is_nested_curriculum and hasattr(self.cfg, 'sub_stage_params'): sub_cfg = self.cfg.sub_stage_params.get(self.current_sub_stage, {}); sigma = sub_cfg.get('tracking_sigma', sigma)
        return torch.exp(-ang_vel_error / sigma)


    # --- Stage 1 Specific Success Criteria ---
    # (Keep implementation as before)
    def compute_success_criteria(self):
        lin_vel_threshold = 0.3; ang_vel_threshold = 0.2; min_track_steps = self.min_tracking_steps_for_success
        if self.is_nested_curriculum and hasattr(self.cfg, 'sub_stage_params'):
             sub_cfg = self.cfg.sub_stage_params.get(self.current_sub_stage, {})
             lin_vel_threshold = sub_cfg.get('success_lin_tol', lin_vel_threshold)
             ang_vel_threshold = sub_cfg.get('success_ang_tol', ang_vel_threshold)
             min_track_steps = sub_cfg.get('min_tracking_steps_for_success', min_track_steps)
        lin_vel_close_enough = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
        if self.cfg.commands.heading_command: target_ang_vel = self.commands[:, 2]
        else: target_ang_vel = self.commands[:, 2]
        ang_vel_close_enough = torch.abs(target_ang_vel - self.base_ang_vel[:, 2]) < ang_vel_threshold
        current_step_success = lin_vel_close_enough & ang_vel_close_enough
        self.successful_command_tracking_streak = (self.successful_command_tracking_streak + 1) * current_step_success
        episode_success_flags = self.successful_command_tracking_streak >= min_track_steps
        return episode_success_flags


    # --- post_physics_step ---
    # (Keep implementation as before)
    def post_physics_step(self):
        super().post_physics_step()
        success_flags = self.compute_success_criteria()
        self.extras["success_flags"] = success_flags.clone()
        if success_flags.any(): self.successful_command_tracking_streak[success_flags] = 0

    # --- reset_idx ---
    # (Keep implementation as before)
    def reset_idx(self, env_ids):
         if len(env_ids) == 0: return
         super().reset_idx(env_ids)
         if hasattr(self, 'successful_command_tracking_streak'): self.successful_command_tracking_streak[env_ids] = 0

    # --- Add helper for stage transition internal update (optional but good practice) ---
    def update_sub_stage_parameters(self, new_sub_stage):
        """ Updates internal parameters when the sub-stage changes without full env recreation.
            Called by train_curriculum.py when only sub-stage advances but dimensions don't change.
        """
        print(f"--- G1BasicLocomotion: Updating internal state for sub-stage 1.{new_sub_stage} ---")
        self.current_sub_stage = new_sub_stage

        # Re-read parameters that might change (like velocity ranges, tolerances)
        if self.is_nested_curriculum and hasattr(self.cfg, 'sub_stage_params'):
             sub_stage_cfg_dict = self.cfg.sub_stage_params.get(self.current_sub_stage)
             if sub_stage_cfg_dict:
                  sub_stage_cfg = DotDict(sub_stage_cfg_dict)
                  self.base_lin_vel_range = sub_stage_cfg.get('base_lin_vel_range', self.cfg.commands.ranges.lin_vel_x[1])
                  self.base_ang_vel_range = sub_stage_cfg.get('base_ang_vel_range', self.cfg.commands.ranges.ang_vel_yaw[1])
                  print(f"  Updated velocity ranges: Lin={self.base_lin_vel_range}, Ang={self.base_ang_vel_range}")
                  # Update success criteria steps if needed (though compute_success_criteria already reads dynamically)
                  # self.min_tracking_steps_for_success = sub_stage_cfg.get('min_tracking_steps_for_success', self.min_tracking_steps_for_success)

             else: print(f"  Warning: Could not find config for sub-stage {new_sub_stage} during update.")
        # No need to recompute active joints or change PD gains if dimensions didn't change