# g1_full_locomotion.py
import torch
import numpy as np
from isaacgym.torch_utils import *
from isaacgym import gymtorch

# 假设你的 legged_robot.py 在 g1.envs.base 目录下
# 如果你的 legged_robot.py 是 legged_gym 库里的，则从那里导入
try:
    # 尝试从你的项目结构导入修正后的 LeggedRobot
    from g1.envs.base.legged_robot import LeggedRobot
except ImportError:
    # 如果上面失败，尝试从 legged_gym 库导入 (但你需要确保它也被修正了!)
    print("⚠️ 未找到 g1.envs.base.legged_robot，尝试从 legged_gym.envs 导入 (请确保该文件已修复DoF覆盖bug!)")
    from legged_gym.envs.base.legged_robot import LeggedRobot

from g1.utils.helpers import class_to_dict # 确保导入

class G1FullLocomotionEnv(LeggedRobot):
    """
    G1 43DoF 机器人基础移动环境 (无课程学习)。
    继承自 (修正后的) LeggedRobot，并整合了 G1 特定的观测、相位和奖励。
    """
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        print(f"--- G1FullLocomotionEnv.__init__ ---")

        # --- 调用父类 LeggedRobot 初始化 ---
        # 父类会处理:
        # - BaseTask 初始化 (gym, sim, device, base buffers based on cfg.env.num_obs/act)
        # - _parse_cfg
        # - _create_envs (加载 URDF, 获取 self.num_dof=43, 验证 self.num_actions=43)
        # - prepare_sim
        # - _init_buffers (创建基础 state tensors, 调用 _get_noise_scale_vec)
        # - _prepare_reward_function (查找本类及父类中的 _reward_* 方法)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

        # --- G1 特定的初始化 ---
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device) # 假设用于相位奖励

        # --- 获取手臂和手部关节索引 (用于特定奖励) ---
        self.arm_dof_indices = []
        self.hand_dof_indices = []
        if hasattr(self, 'dof_names') and self.dof_names:
             arm_keywords = ["shoulder", "elbow", "wrist"]
             hand_keywords = ["hand"] # 假设手部关节名包含 "hand"
             for i, name in enumerate(self.dof_names):
                  if any(keyword in name for keyword in arm_keywords):
                       self.arm_dof_indices.append(i)
                  if any(keyword in name for keyword in hand_keywords):
                       self.hand_dof_indices.append(i)
             self.arm_dof_indices = torch.tensor(self.arm_dof_indices, device=self.device, dtype=torch.long)
             self.hand_dof_indices = torch.tensor(self.hand_dof_indices, device=self.device, dtype=torch.long)
             print(f"  Found {len(self.arm_dof_indices)} arm DoFs and {len(self.hand_dof_indices)} hand DoFs.")
        else:
             print("⚠️ G1FullLocomotionEnv Warning: Could not find dof_names to determine arm/hand indices.")
        # ----------------------------------------

        # 可以在这里再次验证维度，确保所有初始化都使用了正确的值
        if self.num_observations != 140: print(f"❌ ERROR: Final self.num_observations is {self.num_observations}, expected 140")
        if self.num_actions != 43: print(f"❌ ERROR: Final self.num_actions is {self.num_actions}, expected 43")
        if self.num_dof != 43: print(f"❌ ERROR: Final self.num_dof is {self.num_dof}, expected 43")


        print(f"--- G1FullLocomotionEnv.__init__ Done ---")

    # --- 重写 G1 特定的方法 ---

    def _init_buffers(self):
        """ 初始化缓冲区，调用父类后初始化足部状态 """
        super()._init_buffers() # 父类会初始化基础缓冲区并调用 _get_noise_scale_vec
        self._init_foot()       # 初始化足部相关的缓冲区

    def _init_foot(self):
        """ 初始化足部状态缓冲区 (来自 curriculum_base.py) """
        if not hasattr(self, 'feet_indices') or self.feet_indices is None or len(self.feet_indices) == 0:
             print("⚠️ G1FullLocomotionEnv Warning: feet_indices not set. Cannot initialize foot states.")
             self.feet_num = 0; self.feet_state = None; self.feet_pos = None; self.feet_vel = None; return

        self.feet_num = len(self.feet_indices)

        # rigid_body_states 和 view 应该已在父类的 _init_buffers 中创建
        if not hasattr(self, 'rigid_body_states_view') or self.rigid_body_states_view is None:
             print("⚠️ _init_foot: rigid_body_states_view not available. Foot state tracking disabled."); self.feet_num = 0; return

        if torch.any(self.feet_indices >= self.num_bodies):
             print(f"❌ ERROR _init_foot: feet_indices ({self.feet_indices}) out of bounds for num_bodies ({self.num_bodies})."); self.feet_num = 0; return

        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        print("  ✅ Initialized foot state views.")

    def update_feet_state(self):
        """ 更新足部状态 (来自 curriculum_base.py) """
        if self.feet_num > 0 and self.feet_state is not None and hasattr(self, 'rigid_body_states_view'):
            if torch.any(self.feet_indices >= self.num_bodies): return # Avoid index error
            self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
            self.feet_pos = self.feet_state[:, :, :3]
            self.feet_vel = self.feet_state[:, :, 7:10]

    def _post_physics_step_callback(self):
        """ 计算相位并更新足部状态 (来自 curriculum_base.py) """
        super()._post_physics_step_callback() # 调用 LeggedRobot 的回调 (处理命令重采样、航向等)
        self.update_feet_state() # 更新足部状态

        # 计算相位
        period = getattr(self.cfg.control, 'gait_period', 0.8)
        offset = getattr(self.cfg.control, 'gait_offset', 0.5)
        if self.dt > 0 and period > 0:
             elapsed_time_in_episode = self.episode_length_buf * self.dt
             self.phase = (elapsed_time_in_episode % period) / period
             self.phase_left = self.phase
             self.phase_right = (self.phase + offset) % 1.0
             if self.feet_num >= 2: # 确保 leg_phase 缓冲区存在
                 if not hasattr(self, 'leg_phase') or self.leg_phase.shape != (self.num_envs, 2):
                     self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
                 self.leg_phase[:, 0] = self.phase_left
                 self.leg_phase[:, 1] = self.phase_right
        elif not hasattr(self, 'phase'): # 确保 phase 存在
             self.phase = torch.zeros(self.num_envs, device=self.device)

    def _get_noise_scale_vec(self, cfg):
        """ 重写以确保匹配 G1 的 140 维观测空间 (来自 curriculum_base.py) """
        expected_obs_dim = 140 # 直接使用目标值
        noise_vec = torch.zeros(expected_obs_dim, device=self.device)

        self.add_noise = getattr(cfg.noise, 'add_noise', True)
        if not self.add_noise: return noise_vec

        noise_scales = getattr(cfg.noise, 'noise_scales', {})
        if not isinstance(noise_scales, dict): noise_scales = class_to_dict(noise_scales)
        noise_level = getattr(cfg.noise, 'noise_level', 1.0)

        obs_scales = getattr(cfg.normalization, 'obs_scales', {})
        if not isinstance(obs_scales, dict): obs_scales = class_to_dict(obs_scales)

        num_act = 43 # 直接使用目标值
        current_idx = 0
        # 顺序: 3(ang_vel) + 3(gravity) + 3(commands) + 43(dof_pos) + 43(dof_vel) + 43(actions) + 2(phase) = 140
        # Ang Vel (3)
        noise_vec[current_idx:current_idx+3] = noise_scales.get('ang_vel', 0.0) * noise_level * obs_scales.get('ang_vel', 1.0); current_idx += 3
        # Gravity (3)
        noise_vec[current_idx:current_idx+3] = noise_scales.get('gravity', 0.0) * noise_level * 1.0; current_idx += 3
        # Commands (3)
        noise_vec[current_idx:current_idx+3] = 0.0; current_idx += 3
        # DoF Pos (43)
        noise_vec[current_idx:current_idx+num_act] = noise_scales.get('dof_pos', 0.0) * noise_level * obs_scales.get('dof_pos', 1.0); current_idx += num_act
        # DoF Vel (43)
        noise_vec[current_idx:current_idx+num_act] = noise_scales.get('dof_vel', 0.0) * noise_level * obs_scales.get('dof_vel', 1.0); current_idx += num_act
        # Actions (43)
        noise_vec[current_idx:current_idx+num_act] = 0.0; current_idx += num_act
        # Phase (2)
        noise_vec[current_idx:current_idx+2] = 0.0; current_idx += 2

        if current_idx != expected_obs_dim:
            print(f"❌ G1FullLocomotionEnv ERROR: _get_noise_scale_vec dim mismatch! Index={current_idx}, Expected={expected_obs_dim}")
            # 尝试修正尺寸
            if current_idx > expected_obs_dim: noise_vec = noise_vec[:expected_obs_dim]
            else: noise_vec = torch.cat((noise_vec[:current_idx], torch.zeros(expected_obs_dim - current_idx, device=self.device)))

        return noise_vec

    def compute_observations(self):
        """ 计算 G1 的 140 维观测向量 (来自 curriculum_base.py) """
        # --- 确保所有依赖的缓冲区都已正确初始化和更新 ---
        if not hasattr(self, 'phase'): self._post_physics_step_callback() # 计算相位如果缺失
        if not hasattr(self, 'actions'): self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device) # 初始化 actions

        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)

        # 检查 DoF 缓冲区维度是否为 43
        if self.dof_pos.shape[1] != 43 or self.dof_vel.shape[1] != 43:
             print(f"❌ FATAL ERROR compute_observations: DOF buffers have wrong shape ({self.dof_pos.shape[1]}), expected 43! Check URDF loading.")
             # 返回零以避免崩溃，但这表明严重问题
             self.obs_buf.zero_()
             if self.privileged_obs_buf is not None: self.privileged_obs_buf.zero_()
             return

        # --- 使用完整的 43 DoF 数据 ---
        dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
        dof_vel_scaled = self.dof_vel * self.obs_scales.dof_vel
        # -----------------------------

        base_ang_vel_scaled = self.base_ang_vel * self.obs_scales.ang_vel
        commands_scaled = self.commands[:, :3] * self.commands_scale

        # actions 缓冲区应为 [N, 43]
        actions_to_include = self.actions
        if actions_to_include.shape[1] != 43:
             print(f"⚠️ WARNING compute_observations: self.actions shape ({actions_to_include.shape}) mismatch, expected 43. Using zeros.")
             actions_to_include = torch.zeros((self.num_envs, 43), device=self.device)

        # 组装观测列表 (确保顺序和维度正确: 3+3+3+43+43+43+2 = 140)
        obs_list = [
            base_ang_vel_scaled,    # 3
            self.projected_gravity, # 3
            commands_scaled,        # 3
            dof_pos_scaled,         # 43
            dof_vel_scaled,         # 43
            actions_to_include,     # 43
            sin_phase,              # 1
            cos_phase               # 1
        ]

        try:
            self.obs_buf = torch.cat(obs_list, dim=-1)
            if self.obs_buf.shape[1] != 140: # 再次验证
                 raise ValueError(f"Concatenated obs shape is {self.obs_buf.shape[1]}, expected 140")
        except Exception as e:
            print(f"❌ G1FullLocomotionEnv ERROR concatenating observation buffer: {e}")
            for i, item in enumerate(obs_list): print(f"  Item {i}: shape={item.shape if hasattr(item,'shape') else 'N/A'}")
            self.obs_buf = torch.zeros(self.num_envs, 140, device=self.device, dtype=torch.float) # Fallback

        # 组装特权观测 (3 + 140 = 143)
        if self.privileged_obs_buf is not None:
            # 确保 privileged_obs_buf 缓冲区大小正确 (BaseTask 创建时应为 143)
            if self.privileged_obs_buf.shape[1] != 143:
                 print(f"❌ ERROR compute_observations: privileged_obs_buf has wrong shape ({self.privileged_obs_buf.shape[1]}), expected 143.")
                 self.privileged_obs_buf = torch.zeros(self.num_envs, 143, device=self.device, dtype=torch.float) # Fallback
            else:
                 priv_obs_list = [ self.base_lin_vel * self.obs_scales.lin_vel, self.obs_buf ]
                 try:
                     self.privileged_obs_buf = torch.cat(priv_obs_list, dim=-1)
                     if self.privileged_obs_buf.shape[1] != 143: # 再次验证
                         raise ValueError(f"Concatenated priv_obs shape is {self.privileged_obs_buf.shape[1]}, expected 143")
                 except Exception as e:
                     print(f"❌ G1FullLocomotionEnv ERROR concatenating privileged observation buffer: {e}")
                     self.privileged_obs_buf.zero_() # Fallback to zeros

        # 添加噪声
        if self.add_noise and hasattr(self, 'noise_scale_vec') and self.noise_scale_vec is not None:
            if self.noise_scale_vec.shape[0] == self.num_observations:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec


    # --- 实现或继承所有需要的奖励函数 ---
    # (复制/合并之前提供的所有 _reward_* 实现到这里)

    # --- 运动平滑性与效率 ---
    def _reward_action_rate(self):
        if not hasattr(self, 'last_actions') or self.last_actions.shape != self.actions.shape:
             self.last_actions = torch.zeros_like(self.actions)
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=1) # self.torques is [N, 43]

    def _reward_dof_acc(self):
        if not hasattr(self, 'last_dof_vel') or self.last_dof_vel.shape != self.dof_vel.shape:
             self.last_dof_vel = torch.zeros_like(self.dof_vel)
        dt = self.dt
        if dt <= 0: dt = 1e-5
        dof_acc = (self.dof_vel - self.last_dof_vel) / dt # Uses full [N, 43] buffers
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=1) # Uses full [N, 43] buffer

    # --- 姿态与稳定性 ---
    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        height_target = getattr(self.cfg.rewards, 'base_height_target', 0.78)
        return torch.square(self.root_states[:, 2] - height_target)

    # --- 约束与安全 ---
    def _reward_dof_pos_limits(self):
        if not hasattr(self, 'dof_pos_limits') or self.dof_pos_limits.shape[0] != self.num_dof:
            print("⚠️ _reward_dof_pos_limits: Limits not initialized correctly."); return torch.zeros_like(self.rew_buf)
        # Uses full [N, 43] dof_pos and [43, 2] limits
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_collision(self):
        if not hasattr(self, 'penalised_contact_indices') or self.penalised_contact_indices.numel() == 0: return torch.zeros_like(self.rew_buf)
        if not hasattr(self, 'contact_forces') or self.contact_forces is None or self.penalised_contact_indices.max() >= self.contact_forces.shape[1]:
             print("⚠️ _reward_collision: Contact forces or indices invalid."); return torch.zeros_like(self.rew_buf)
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_torque_limits(self):
        if not hasattr(self, 'torques') or not hasattr(self, 'torque_limits') or self.torque_limits.shape[0] != self.num_dof:
             print("⚠️ _reward_torque_limits: Buffers not available or shape mismatch."); return torch.zeros_like(self.rew_buf)
        soft_limit_factor = getattr(self.cfg.rewards, 'soft_torque_limit', 0.9)
        exceeding_soft_limit = torch.abs(self.torques) > soft_limit_factor * self.torque_limits # Compare [N, 43] with [43]
        return torch.sum(exceeding_soft_limit.float(), dim=1)

    # --- 足部与步态 ---
    def _reward_feet_air_time(self):
        if self.feet_air_time is None or self.last_contacts is None or self.feet_indices.numel() == 0: return torch.zeros_like(self.rew_buf)
        if self.contact_forces is None or self.contact_forces.shape[1] < self.feet_indices.max()+1: return torch.zeros_like(self.rew_buf)
        try:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1.
            if self.last_contacts.shape != contact.shape: self.last_contacts = torch.zeros_like(contact)
            contact_filt = torch.logical_or(contact, self.last_contacts); self.last_contacts = contact
            if self.feet_air_time.shape != contact.shape: self.feet_air_time = torch.zeros_like(contact)
            first_contact = (self.feet_air_time > 0.) * contact_filt; self.feet_air_time += self.dt
            max_air_time = getattr(self.cfg.rewards, 'max_air_time', 1.0)
            reward_contribution = (self.feet_air_time - 0.5) * first_contact
            air_time_valid_mask = torch.ones_like(first_contact)
            if first_contact.any(): air_time_valid_mask[first_contact] = (self.feet_air_time[first_contact] < max_air_time)
            valid_reward_contribution = reward_contribution * air_time_valid_mask
            rew_airTime = torch.sum(valid_reward_contribution, dim=1)
            self.feet_air_time *= (~contact_filt)
            if rew_airTime.shape != self.rew_buf.shape: return torch.zeros_like(self.rew_buf)
            return rew_airTime
        except Exception as e: print(f"❌ ERROR feet_air_time: {e}"); return torch.zeros_like(self.rew_buf)

    def _reward_stand_still(self):
        if not hasattr(self, 'default_dof_pos') or self.default_dof_pos.shape[1] != self.num_dof:
            print("⚠️ _reward_stand_still: Default DoF Pos incorrect shape."); return torch.zeros_like(self.rew_buf)
        # Use full DoF pos buffer
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_stumble(self):
        if self.feet_indices.numel() == 0 or self.contact_forces is None or self.contact_forces.shape[1] < self.feet_indices.max()+1:
            return torch.zeros_like(self.rew_buf)
        vertical_contact_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        return torch.sum(1.*(vertical_contact_force > 0.1), dim=1)

    # --- G1 特有奖励 (来自官方 g1_env.py 或你的 curriculum_base.py) ---
    def _reward_contact(self): # Phase-contact correlation
        if not hasattr(self, 'leg_phase') or self.leg_phase is None or self.leg_phase.shape[1] < 2 or self.feet_num < 2: return torch.zeros_like(self.rew_buf)
        if self.contact_forces.shape[1] < self.feet_indices.max()+1: return torch.zeros_like(self.rew_buf)
        res = torch.zeros_like(self.rew_buf)
        for i in range(min(2, self.feet_num)): # Iterate safely
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            res += torch.eq(is_stance, contact)
        return res / 2.0

    def _reward_feet_swing_height(self): # Penalize swing foot height deviation
        if self.feet_pos is None or self.feet_num == 0: return torch.zeros_like(self.rew_buf)
        target_swing_height = getattr(self.cfg.rewards, 'target_swing_height', 0.08)
        if self.contact_forces.shape[1] < self.feet_indices.max()+1: return torch.zeros_like(self.rew_buf)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        is_swing = ~contact
        if self.feet_pos.shape[1] != is_swing.shape[1]: return torch.zeros_like(self.rew_buf)
        height_error_sq = torch.square(self.feet_pos[:, :, 2] - target_swing_height)
        swing_height_error = height_error_sq * is_swing
        return torch.sum(swing_height_error, dim=1)

    def _reward_contact_no_vel(self): # Penalize foot velocity during contact
        if self.feet_vel is None or self.feet_num == 0: return torch.zeros_like(self.rew_buf)
        if self.contact_forces.shape[1] < self.feet_indices.max()+1: return torch.zeros_like(self.rew_buf)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        if self.feet_vel.shape[1] != contact.shape[1]: return torch.zeros_like(self.rew_buf)
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize_sq_vel = torch.sum(torch.square(contact_feet_vel), dim=2) # Sum over x,y,z
        return torch.sum(penalize_sq_vel, dim=1) # Sum over feet

    def _reward_hip_pos(self): # Penalize hip roll/pitch deviation from default
        try:
             if not hasattr(self, 'dof_names') or not self.dof_names: return torch.zeros_like(self.rew_buf)
             left_roll_idx = self.dof_names.index('left_hip_roll_joint'); left_pitch_idx = self.dof_names.index('left_hip_pitch_joint')
             right_roll_idx = self.dof_names.index('right_hip_roll_joint'); right_pitch_idx = self.dof_names.index('right_hip_pitch_joint')
             indices = torch.tensor([left_roll_idx, left_pitch_idx, right_roll_idx, right_pitch_idx], device=self.device)
             if indices.max() >= self.dof_pos.shape[1]: return torch.zeros_like(self.rew_buf) # Check bounds
             default_hip_pos = self.default_dof_pos[:, indices]
             hip_pos_error_sq = torch.square(self.dof_pos[:, indices] - default_hip_pos)
             return torch.sum(hip_pos_error_sq, dim=1)
        except (ValueError, AttributeError, IndexError): return torch.zeros_like(self.rew_buf)

    # --- 特定部位惩罚 (来自你的配置) ---
    def _reward_arm_dof_vel(self):
        """Penalize arm joint velocities."""
        if self.arm_dof_indices.numel() > 0 and self.arm_dof_indices.max() < self.dof_vel.shape[1]:
            arm_vels = self.dof_vel[:, self.arm_dof_indices]
            return torch.sum(torch.square(arm_vels), dim=1)
        return torch.zeros_like(self.rew_buf)

    def _reward_arm_dof_acc(self):
        """Penalize arm joint accelerations."""
        if (self.arm_dof_indices.numel() > 0 and hasattr(self, 'last_dof_vel') and
            self.arm_dof_indices.max() < self.dof_vel.shape[1] and
            self.last_dof_vel.shape == self.dof_vel.shape):
            dt = self.dt
            if dt <= 0: dt = 1e-5
            arm_acc = (self.dof_vel[:, self.arm_dof_indices] - self.last_dof_vel[:, self.arm_dof_indices]) / dt
            return torch.sum(torch.square(arm_acc), dim=1)
        return torch.zeros_like(self.rew_buf)

    def _reward_hand_dof_vel(self):
        """Penalize hand joint velocities."""
        if self.hand_dof_indices.numel() > 0 and self.hand_dof_indices.max() < self.dof_vel.shape[1]:
            hand_vels = self.dof_vel[:, self.hand_dof_indices]
            return torch.sum(torch.square(hand_vels), dim=1)
        return torch.zeros_like(self.rew_buf)

    # --- (可选)添加在配置文件中但你代码里没有的 arm_pose_penalty 和 hand_pose_penalty ---
    def _reward_arm_pose_penalty(self):
        """Penalize deviation of arm joints from default pose."""
        if (self.arm_dof_indices.numel() > 0 and
            self.arm_dof_indices.max() < self.dof_pos.shape[1] and
            self.default_dof_pos.shape[1] == self.num_dof): # Check default_dof_pos is full size
            arm_pos_error = self.dof_pos[:, self.arm_dof_indices] - self.default_dof_pos[:, self.arm_dof_indices]
            return torch.sum(torch.square(arm_pos_error), dim=1)
        return torch.zeros_like(self.rew_buf)

    def _reward_hand_pose_penalty(self):
        """Penalize deviation of hand joints from default pose."""
        if (self.hand_dof_indices.numel() > 0 and
            self.hand_dof_indices.max() < self.dof_pos.shape[1] and
            self.default_dof_pos.shape[1] == self.num_dof):
            hand_pos_error = self.dof_pos[:, self.hand_dof_indices] - self.default_dof_pos[:, self.hand_dof_indices]
            return torch.sum(torch.square(hand_pos_error), dim=1)
        return torch.zeros_like(self.rew_buf)

    # --- 继承自 LeggedRobot 但可能需要确认的 ---
    def _reward_tracking_lin_vel(self):
        """(G1 Override Option 1: Use Base) Tracking of linear velocity commands (xy axes)."""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        sigma = getattr(self.cfg.rewards, 'tracking_sigma', 0.25) # Use base sigma from config
        return torch.exp(-lin_vel_error / sigma)

    def _reward_tracking_ang_vel(self):
        """(G1 Override Option 1: Use Base) Tracking of angular velocity commands (yaw)."""
        if self.cfg.commands.heading_command: target_ang_vel = self.commands[:, 2]
        else: target_ang_vel = self.commands[:, 2]
        ang_vel_error = torch.square(target_ang_vel - self.base_ang_vel[:, 2])
        sigma = getattr(self.cfg.rewards, 'tracking_sigma', 0.25) # Use base sigma from config
        return torch.exp(-ang_vel_error / sigma)

    def _reward_termination(self):
        """(G1 Override Option 1: Use Base) Terminal reward/penalty."""
        # Penalize resets that are NOT timeouts
        return self.reset_buf * (~self.time_out_buf) * -1.0