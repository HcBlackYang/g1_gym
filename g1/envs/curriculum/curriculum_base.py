
# # curriculum_base.py
# from g1.envs.base.legged_robot import LeggedRobot # Import the modified LeggedRobot

# from isaacgym.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil
# import torch
# import numpy as np


# class G1CurriculumBase(LeggedRobot):
#     """G1 机器人所有课程学习环境的基类.
#     继承自修改后的 LeggedRobot，应自动继承正确的初始化流程。
#     主要负责实现 G1 特有的逻辑，如相位计算和相关奖励。
#     """

#     def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
#         # --- 课程学习相关属性 (从 cfg 读取) ---
#         # 这些属性由 train_curriculum.py 在创建环境前设置到 cfg 中
#         self.curriculum_stage = getattr(cfg.curriculum, 'stage', 1)
#         self.curriculum_sub_stage = getattr(cfg.curriculum, 'sub_stage', 1)
#         print(f"--- G1CurriculumBase.__init__: Initializing for Stage {self.curriculum_stage}.{self.curriculum_sub_stage} ---")

#         # --- 调用父类 LeggedRobot 初始化 ---
#         # 传递所有必要的参数，包括 gym_handle 和 sim_handle
#         super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

#         # --- G1CurriculumBase 特定的初始化 (如果需要) ---
#         # 例如，确保相位变量存在
#         self.phase = torch.zeros(self.num_envs, device=self.device)
#         self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device) # Assuming 2 legs for phase

#         # 验证 DOF 数量是否符合预期 (例如 43)
#         if self.num_dof != 43: # Or get expected value from cfg more dynamically
#              print(f"⚠️ G1CurriculumBase Warning: Expected 43 DoFs based on config, but found {self.num_dof} from asset.")
#              # This might indicate the wrong URDF is being loaded or config mismatch

#         print(f"--- G1CurriculumBase.__init__ Done ---")


#     # --- 重写或添加 G1 特定的方法 ---

#     # _get_noise_scale_vec: 根据最终的观测空间结构调整
#     def _get_noise_scale_vec(self, cfg):
#         """ 设置用于缩放添加到观测值的噪声的向量。
#             *** 需要根据最终的观测结构 (包括导航等) 仔细调整 ***
#         Args:
#             cfg (Dict): 环境配置文件
#         Returns:
#             [torch.Tensor]: 噪声尺度向量
#         """
#         expected_obs_dim = self.num_observations # Use the final number of observations
#         noise_vec = torch.zeros(expected_obs_dim, device=self.device)
#         # Check if obs_buf was initialized correctly before accessing shape
#         # if not hasattr(self, 'obs_buf') or self.obs_buf.shape[1] != expected_obs_dim:
#         #      print(f"⚠️ _get_noise_scale_vec: obs_buf not ready or shape mismatch. Returning zero noise vector.")
#         #      return noise_vec # Return zeros if obs_buf isn't right

#         self.add_noise = getattr(self.cfg.noise, 'add_noise', True) # Default to True
#         if not self.add_noise: return noise_vec # Return zeros if noise is disabled

#         noise_scales = self.cfg.noise.noise_scales
#         noise_level = self.cfg.noise.noise_level
#         obs_scales = self.cfg.normalization.obs_scales

#         current_idx = 0
#         # --- Base observation noise (match compute_observations structure) ---
#         if hasattr(noise_scales, 'ang_vel'): noise_vec[current_idx:current_idx+3] = noise_scales.ang_vel * noise_level * obs_scales.ang_vel; current_idx += 3
#         else: current_idx +=3
#         if hasattr(noise_scales, 'gravity'): noise_vec[current_idx:current_idx+3] = noise_scales.gravity * noise_level; current_idx += 3 # No obs_scale for gravity
#         else: current_idx +=3
#         noise_vec[current_idx:current_idx+3] = 0.; current_idx += 3 # Commands
#         if hasattr(noise_scales, 'dof_pos'): noise_vec[current_idx:current_idx+self.num_actions] = noise_scales.dof_pos * noise_level * obs_scales.dof_pos; current_idx += self.num_actions
#         else: current_idx += self.num_actions
#         if hasattr(noise_scales, 'dof_vel'): noise_vec[current_idx:current_idx+self.num_actions] = noise_scales.dof_vel * noise_level * obs_scales.dof_vel; current_idx += self.num_actions
#         else: current_idx += self.num_actions
#         noise_vec[current_idx:current_idx+self.num_actions] = 0.; current_idx += self.num_actions # Actions
#         noise_vec[current_idx:current_idx+2] = 0.; current_idx += 2 # Phase

#         # --- Noise for additional observations (e.g., navigation) ---
#         # Example: Add noise for target relative position (next 3 dims)
#         if expected_obs_dim >= current_idx + 3: # Check if space exists
#             if hasattr(noise_scales, 'target_pos') and hasattr(obs_scales, 'target_pos'):
#                  noise_vec[current_idx:current_idx+3] = noise_scales.target_pos * noise_level * obs_scales.target_pos
#             current_idx += 3

#         # --- Final check ---
#         if current_idx != expected_obs_dim:
#             print(f"❌ G1CurriculumBase ERROR: Final noise vector index ({current_idx}) != expected obs dim ({expected_obs_dim}).")

#         return noise_vec

#     # _init_buffers: No longer needed here if LeggedRobot handles it correctly for single robot
#     # def _init_buffers(self): ...

#     # _init_foot: Inherited from LeggedRobot, should work if indices are correct
#     # def _init_foot(self): ...

#     # update_feet_state: Inherited from LeggedRobot
#     # def update_feet_state(self): ...

#     def _init_foot(self):
#         """Initializes buffers for foot states using rigid body states."""
#         # Check if feet_indices are valid before proceeding
#         if self.feet_indices is None or len(self.feet_indices) == 0:
#              print("⚠️ G1CurriculumBase Warning: feet_indices are not set or empty. Cannot initialize foot states.")
#              self.feet_num = 0
#              self.feet_state = None
#              self.feet_pos = None
#              self.feet_vel = None
#              return

#         self.feet_num = len(self.feet_indices)

#         # Acquire rigid body state tensor (check if already acquired by parent)
#         if not hasattr(self, 'rigid_body_states'):
#              rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
#              self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
#              # Reshape based on num_envs and num_bodies
#              self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
#         elif self.rigid_body_states is None: # If parent acquired but it's None
#              rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
#              self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
#              self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)


#         # Ensure rigid_body_states_view is correctly shaped
#         if self.rigid_body_states_view.shape[0] != self.num_envs or self.rigid_body_states_view.shape[1] != self.num_bodies:
#              print(f"⚠️ G1CurriculumBase Warning: Reshaping rigid_body_states_view. Expected ({self.num_envs}, {self.num_bodies}, 13), Got {self.rigid_body_states_view.shape}")
#              self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)


#         # Get foot states using the valid indices
#         # Ensure feet_indices are within bounds
#         if torch.any(self.feet_indices >= self.num_bodies) or torch.any(self.feet_indices < 0):
#              print(f"❌ G1CurriculumBase ERROR: feet_indices ({self.feet_indices}) are out of bounds for num_bodies ({self.num_bodies}).")
#              # Handle error: perhaps set feet_num to 0 or raise an exception
#              self.feet_num = 0
#              self.feet_state = None
#              self.feet_pos = None
#              self.feet_vel = None
#              return

#         self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
#         self.feet_pos = self.feet_state[:, :, :3] # Position (x, y, z)
#         self.feet_vel = self.feet_state[:, :, 7:10] # Linear velocity (vx, vy, vz)


#     def _init_buffers(self):
#         """Initializes buffers, including foot states."""
#         super()._init_buffers() # Call parent's buffer initialization first
#         self._init_foot()       # Then initialize foot-specific buffers

#     def update_feet_state(self):
#         """Refreshes and updates foot position and velocity from simulation."""
#         # Only update if foot tracking is initialized
#         if self.feet_num > 0 and self.feet_state is not None:
#             # Refresh the main rigid body state tensor
#             # No need to refresh if parent's post_physics_step already does it
#             # self.gym.refresh_rigid_body_state_tensor(self.sim) # Usually done before this callback

#             # Re-slice the tensor to get the latest states
#             # Important: Ensure rigid_body_states_view is referencing the *latest* tensor data
#             self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
#             self.feet_pos = self.feet_state[:, :, :3]
#             self.feet_vel = self.feet_state[:, :, 7:10]

#     # _post_physics_step_callback: 重写以计算 G1 特有相位
#     def _post_physics_step_callback(self):
#         """ Callback for G1 specific updates like phase calculation,
#             after common logic in LeggedRobot's callback is executed.
#         """
#         # --- 1. 调用父类回调 (执行命令重采样、朝向计算等) ---
#         super()._post_physics_step_callback()
#         # ---------------------------------------------------

#         # --- 2. 更新足部状态 (如果父类没做) ---
#         # LeggedRobot 的回调现在不包含 update_feet_state，所以在这里调用
#         if hasattr(self, 'update_feet_state'):
#             self.update_feet_state()

#         # --- 3. 计算 G1 特有的腿部相位 ---
#         period = getattr(self.cfg.control, 'gait_period', 0.8)
#         offset = getattr(self.cfg.control, 'gait_offset', 0.5)
#         elapsed_time_in_episode = self.episode_length_buf * self.dt
#         self.phase = (elapsed_time_in_episode % period) / period
#         self.phase_left = self.phase
#         self.phase_right = (self.phase + offset) % 1.0

#         if self.feet_num >= 2:
#              if not hasattr(self, 'leg_phase') or self.leg_phase.shape != (self.num_envs, 2):
#                   self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
#              self.leg_phase[:, 0] = self.phase_left
#              self.leg_phase[:, 1] = self.phase_right

# # compute_observations: 重写以构建 G1 的基础观测 (例如 140 维)
#     def compute_observations(self):
#         """ Computes G1 specific base observations (e.g., 140 dims including phase).
#             Child classes will call this via super() and append their specific observations.
#         """
#         # --- 1. 计算通用组件 ---
#         # Phase (ensure it's calculated in _post_physics_step_callback)
#         if not hasattr(self, 'phase'): self.phase = torch.zeros(self.num_envs, device=self.device)
#         sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
#         cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)

#         # Scaled DoF states (ensure dof_pos/vel are correct shape [num_envs, num_dof])
#         if self.dof_pos.shape != (self.num_envs, self.num_dof) or self.dof_vel.shape != (self.num_envs, self.num_dof):
#              print(f"❌ ERROR compute_observations: DOF state shape mismatch!")
#              # Fallback to zeros with correct shape for buffer concatenation
#              dof_pos_scaled = torch.zeros((self.num_envs, self.num_dof), device=self.device)
#              dof_vel_scaled = torch.zeros((self.num_envs, self.num_dof), device=self.device)
#         else:
#              dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
#              dof_vel_scaled = self.dof_vel * self.obs_scales.dof_vel

#         # Scaled base velocities and commands
#         base_ang_vel_scaled = self.base_ang_vel * self.obs_scales.ang_vel
#         commands_scaled = self.commands[:, :3] * self.commands_scale # Assuming commands_scale is set

#         # Previous actions (ensure shape is correct)
#         actions_to_include = self.actions
#         if actions_to_include.shape != (self.num_envs, self.num_actions):
#              print(f"⚠️ WARNING compute_observations: self.actions shape {actions_to_include.shape} unexpected. Using zeros.")
#              actions_to_include = torch.zeros((self.num_envs, self.num_actions), device=self.device)


#         # --- 2. 组装 G1 基础观测列表 (140 维) ---
#         # 顺序必须与 _get_noise_scale_vec 中的假设一致
#         # 3(ang_vel) + 3(gravity) + 3(commands) + 43(dof_pos) + 43(dof_vel) + 43(actions) + 2(phase) = 140
#         obs_list = [
#             base_ang_vel_scaled,    # 3
#             self.projected_gravity, # 3
#             commands_scaled,        # 3
#             dof_pos_scaled,         # 43 (num_actions)
#             dof_vel_scaled,         # 43 (num_actions)
#             actions_to_include,     # 43 (num_actions) - previous actions
#             sin_phase,              # 1
#             cos_phase               # 1
#         ]

#         # --- 3. 拼接成最终的 obs_buf ---
#         try:
#             self.obs_buf = torch.cat(obs_list, dim=-1)
#             # 验证维度
#             if self.obs_buf.shape[1] != self.num_observations:
#                  print(f"❌ ERROR: G1CurriculumBase computed obs dim ({self.obs_buf.shape[1]}) != configured num_observations ({self.num_observations}). Check obs_list!")
#                  # Attempt to fix shape for safety, though it indicates a logic error
#                  if self.obs_buf.shape[1] > self.num_observations:
#                      self.obs_buf = self.obs_buf[:, :self.num_observations]
#                  else:
#                      padding = torch.zeros((self.num_envs, self.num_observations - self.obs_buf.shape[1]), device=self.device)
#                      self.obs_buf = torch.cat([self.obs_buf, padding], dim=-1)

#         except Exception as e:
#             print("❌ G1CurriculumBase ERROR concatenating observation buffer:")
#             for i, item in enumerate(obs_list): print(f"  Item {i}: shape={item.shape if hasattr(item,'shape') else 'N/A'}")
#             print(f"  Error: {e}")
#             # Fallback to zeros
#             self.obs_buf = torch.zeros(self.num_envs, self.num_observations, device=self.device, dtype=torch.float)

#         # --- 4. 组装特权观测 ---
#         if self.privileged_obs_buf is not None:
#             # 特权观测 = 基础线速度(3) + 普通观测(140) = 143 (这是基础特权观测)
#             priv_obs_list = [
#                 self.base_lin_vel * self.obs_scales.lin_vel, # 3
#                 self.obs_buf                                 # 140 (G1 base obs)
#             ]
#             try:
#                 self.privileged_obs_buf = torch.cat(priv_obs_list, dim=-1)
#                 # 验证维度
#                 if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
#                      print(f"❌ ERROR: G1CurriculumBase computed priv_obs dim ({self.privileged_obs_buf.shape[1]}) != configured num_privileged_obs ({self.num_privileged_obs}). Check priv_obs_list!")
#                      # Attempt to fix shape
#                      if self.privileged_obs_buf.shape[1] > self.num_privileged_obs:
#                           self.privileged_obs_buf = self.privileged_obs_buf[:, :self.num_privileged_obs]
#                      else:
#                           padding = torch.zeros((self.num_envs, self.num_privileged_obs - self.privileged_obs_buf.shape[1]), device=self.device)
#                           self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, padding], dim=-1)

#             except Exception as e:
#                  print("❌ G1CurriculumBase ERROR concatenating privileged observation buffer:")
#                  for i, item in enumerate(priv_obs_list): print(f"  Item {i}: shape={item.shape if hasattr(item,'shape') else 'N/A'}")
#                  print(f"  Error: {e}")
#                  # Fallback to zeros
#                  self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)


#         # --- 5. 添加噪声 (在观测计算完成后) ---
#         if self.add_noise:
#             # Check noise vector compatibility
#             if self.noise_scale_vec is not None and self.noise_scale_vec.shape[0] == self.num_observations:
#                 self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
#             else:
#                  # Warning already printed in _get_noise_scale_vec if size mismatch
#                  pass


#     # --- 基础奖励函数 (来自 G1 参考和 LeggedRobot) ---
#     # 确保这些函数使用正确的实例变量 (self.dof_pos, self.base_*, etc.)
#     # _reward_contact, _reward_feet_swing_height, _reward_alive,
#     # _reward_contact_no_vel, _reward_hip_pos
#     # 这些看起来是正确的，因为它们使用了 G1CurriculumBase 中定义的属性。
#     def _reward_contact(self):
#         # Check if leg_phase is available and correctly shaped
#         if not hasattr(self, 'leg_phase') or self.leg_phase is None or self.leg_phase.shape[1] < 2 or self.feet_num < 2:
#              return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

#         res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#         for i in range(2): # Only check first two legs for phase correlation
#             is_stance = self.leg_phase[:, i] < 0.55
#             contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
#             res += torch.eq(is_stance, contact)
#         return res / 2.0 # Normalize by 2 legs

#     def _reward_feet_swing_height(self):
#         if self.feet_pos is None or self.feet_num == 0:
#              return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#         target_swing_height = getattr(self.cfg.rewards, 'target_swing_height', 0.08)
#         contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
#         is_swing = ~contact
#         height_error_sq = torch.square(self.feet_pos[:, :, 2] - target_swing_height)
#         swing_height_error = height_error_sq * is_swing
#         return torch.sum(swing_height_error, dim=1)

#     def _reward_alive(self):
#         return torch.ones(self.num_envs, device=self.device)

#     def _reward_contact_no_vel(self):
#         if self.feet_vel is None or self.feet_num == 0:
#              return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#         contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
#         contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
#         penalize_sq_vel = torch.square(contact_feet_vel)
#         return torch.sum(penalize_sq_vel, dim=(1, 2))

#     def _reward_hip_pos(self):
#         # 使用 self.dof_names 查找索引，更健壮
#         try:
#              # 确保 dof_names 已被初始化
#              if not hasattr(self, 'dof_names') or not self.dof_names: return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#              left_roll_idx = self.dof_names.index('left_hip_roll_joint')
#              left_pitch_idx = self.dof_names.index('left_hip_pitch_joint')
#              right_roll_idx = self.dof_names.index('right_hip_roll_joint')
#              right_pitch_idx = self.dof_names.index('right_hip_pitch_joint')
#              indices = [left_roll_idx, left_pitch_idx, right_roll_idx, right_pitch_idx]
#              hip_pos_error_sq = torch.square(self.dof_pos[:, indices])
#              return torch.sum(hip_pos_error_sq, dim=1)
#         except ValueError: # If a joint name isn't found
#              print(f"⚠️ WARNING: Could not find all required hip joints in dof_names for _reward_hip_pos. Skipping.")
#              return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
#         except AttributeError: # If dof_pos is not yet initialized correctly
#             print(f"⚠️ WARNING: self.dof_pos not available for _reward_hip_pos. Skipping.")
#             return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


# curriculum_base.py
from g1.envs.base.legged_robot import LeggedRobot # Import the modified LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
from g1.utils.helpers import class_to_dict, get_load_path, set_seed, parse_sim_params, DotDict

class G1CurriculumBase(LeggedRobot):
    """G1 机器人所有课程学习环境的基类.
    继承自修改后的 LeggedRobot，应自动继承正确的初始化流程。
    主要负责实现 G1 特有的逻辑，如相位计算和相关奖励。
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        # --- 课程学习相关属性 (从 cfg 读取) ---
        # These are set by train_curriculum.py before calling make_env
        self.curriculum_stage = getattr(cfg.curriculum, 'stage', 1)
        self.curriculum_sub_stage = getattr(cfg.curriculum, 'sub_stage', 1)
        print(f"--- G1CurriculumBase.__init__: Initializing for Stage {self.curriculum_stage}.{self.curriculum_sub_stage} ---")

        # --- 调用父类 LeggedRobot 初始化 ---
        # It handles gym, sim, device, basic buffers (based on final cfg dims),
        # _create_envs (which sets self.num_dof=43), prepare_sim, _init_buffers, _prepare_reward_function
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

        # --- G1CurriculumBase 特定的初始化 (在父类初始化后) ---
        # 例如，确保相位变量存在
        self.phase = torch.zeros(self.num_envs, device=self.device)
        self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device) # Assuming 2 legs for phase

        # 验证最终的 self.num_dof 是否符合预期 (父类 _create_envs 应该已设为 43)
        if self.num_dof != 43:
             print(f"⚠️ G1CurriculumBase Warning: Expected self.num_dof to be 43 (from asset), but found {self.num_dof}.")
             # This might indicate an issue in LeggedRobot._create_envs

        print(f"--- G1CurriculumBase.__init__ Done ---")


    # --- Override _init_buffers to call noise vec creation at the end ---
    def _init_buffers(self):
        """ Initialize torch tensors. Call parent first, then create noise vector. """
        super()._init_buffers() # Call LeggedRobot._init_buffers

        # --- Create noise vector AFTER obs_buf is allocated and dimensions are final ---
        # BaseTask allocates obs_buf using self.num_observations from the final cfg.
        # LeggedRobot._init_buffers calls self._get_noise_scale_vec. Let's override it here
        # to ensure it's called with the *potentially overridden* num_observations.
        # Update: LeggedRobot now calls _get_noise_scale_vec inside its _init_buffers.
        # We need to override _get_noise_scale_vec instead to use the right logic.
        # print("--- G1CurriculumBase._init_buffers: Calling _get_noise_scale_vec ---")
        # self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    # --- Override _get_noise_scale_vec to match G1's observation structure ---
    def _get_noise_scale_vec(self, cfg):
        """ 设置用于缩放添加到观测值的噪声的向量 (G1 Version).
            *** MUST match the observation structure in compute_observations ***
        Args:
            cfg (DotDict): Environment configuration object.
        Returns:
            torch.Tensor: Noise scale vector.
        """
        # Use the actual num_observations of the environment instance
        # This value might have been overridden by G1BasicLocomotion for nested curriculum
        expected_obs_dim = self.num_observations
        noise_vec = torch.zeros(expected_obs_dim, device=self.device)

        self.add_noise = getattr(cfg.noise, 'add_noise', True)
        if not self.add_noise:
            # print("  Noise disabled in G1CurriculumBase.")
            return noise_vec # Return zeros

        # Safely access config sections
        noise_scales = getattr(cfg, 'noise', DotDict()).noise_scales if hasattr(cfg, 'noise') else DotDict()
        noise_level = getattr(cfg.noise, 'noise_level', 1.0) if hasattr(cfg, 'noise') else 1.0
        obs_scales = getattr(cfg.normalization, 'obs_scales', None)
        if obs_scales is None:
            print("⚠️ _get_noise_scale_vec (G1): cfg.normalization.obs_scales not found. Using scale 1.0 for noise.")
            # Create a dummy based on noise_scales keys if possible
            obs_scales = DotDict({k: 1.0 for k in dir(noise_scales) if not k.startswith('_')})

        print(f"  G1CurriculumBase: Generating noise vector for obs_dim={expected_obs_dim}")

        # Match the order in G1CurriculumBase.compute_observations
        # 3(ang_vel) + 3(gravity) + 3(commands) + N(dof_pos) + N(dof_vel) + N(actions) + 2(phase) = 11 + 3*N
        # Where N is the *current* self.num_actions
        num_act = self.num_actions # Use the current number of active actions
        current_idx = 0

        # Ang Vel (3)
        scale = getattr(obs_scales, 'ang_vel', 1.0)
        noise = getattr(noise_scales, 'ang_vel', 0.0)
        if current_idx + 3 <= expected_obs_dim:
            noise_vec[current_idx:current_idx+3] = noise * noise_level * scale
        current_idx += 3

        # Gravity (3)
        scale = 1.0 # No obs scale for gravity
        noise = getattr(noise_scales, 'gravity', 0.0)
        if current_idx + 3 <= expected_obs_dim:
             noise_vec[current_idx:current_idx+3] = noise * noise_level * scale
        current_idx += 3

        # Commands (3) - No noise usually
        if current_idx + 3 <= expected_obs_dim:
             noise_vec[current_idx:current_idx+3] = 0.0
        current_idx += 3

        # DoF Pos (num_act)
        scale = getattr(obs_scales, 'dof_pos', 1.0)
        noise = getattr(noise_scales, 'dof_pos', 0.0)
        if current_idx + num_act <= expected_obs_dim:
            noise_vec[current_idx:current_idx+num_act] = noise * noise_level * scale
        current_idx += num_act

        # DoF Vel (num_act)
        scale = getattr(obs_scales, 'dof_vel', 1.0)
        noise = getattr(noise_scales, 'dof_vel', 0.0)
        if current_idx + num_act <= expected_obs_dim:
            noise_vec[current_idx:current_idx+num_act] = noise * noise_level * scale
        current_idx += num_act

        # Actions (num_act) - No noise usually
        if current_idx + num_act <= expected_obs_dim:
            noise_vec[current_idx:current_idx+num_act] = 0.0
        current_idx += num_act

        # Phase (2) - No noise usually
        if current_idx + 2 <= expected_obs_dim:
            noise_vec[current_idx:current_idx+2] = 0.0
        current_idx += 2

        # --- Check if all dimensions were covered ---
        if current_idx != expected_obs_dim:
            print(f"❌ G1CurriculumBase ERROR: Final noise vector index ({current_idx}) != expected obs dim ({expected_obs_dim}).")
            print(f"   Check compute_observations structure and sub-stage num_observations config.")
            # Attempt to fix shape for safety
            if current_idx < expected_obs_dim:
                 print(f"   Padding noise vector with zeros.")
                 noise_vec = torch.cat((noise_vec[:current_idx], torch.zeros(expected_obs_dim - current_idx, device=self.device)))
            else:
                 print(f"   Truncating noise vector.")
                 noise_vec = noise_vec[:expected_obs_dim]

        return noise_vec


    # _init_foot: Inherited from LeggedRobot, should work if indices are correct
    # update_feet_state: Inherited from LeggedRobot

    # _post_physics_step_callback: 重写以计算 G1 特有相位
    def _post_physics_step_callback(self):
        """ Callback for G1 specific updates like phase calculation,
            after common logic in LeggedRobot's callback is executed.
        """
        # --- 1. 调用父类回调 (执行命令重采样、朝向计算等) ---
        super()._post_physics_step_callback()
        # ---------------------------------------------------

        # --- 2. 更新足部状态 (如果父类没做) ---
        # LeggedRobot 的回调现在不包含 update_feet_state，所以在这里调用
        if hasattr(self, 'update_feet_state'):
            self.update_feet_state()
        # else: print("⚠️ G1CurriculumBase: update_feet_state not found.")


        # --- 3. 计算 G1 特有的腿部相位 ---
        # Ensure config attributes exist before using getattr
        period = 0.8; offset = 0.5 # Defaults
        if hasattr(self.cfg, 'control'):
             period = getattr(self.cfg.control, 'gait_period', 0.8)
             offset = getattr(self.cfg.control, 'gait_offset', 0.5)

        # Ensure episode_length_buf and dt exist
        if hasattr(self, 'episode_length_buf') and hasattr(self, 'dt') and self.dt > 0 and period > 0:
             elapsed_time_in_episode = self.episode_length_buf * self.dt
             self.phase = (elapsed_time_in_episode % period) / period
             self.phase_left = self.phase
             self.phase_right = (self.phase + offset) % 1.0

             # Update leg_phase buffer if feet exist
             if hasattr(self, 'feet_num') and self.feet_num >= 2:
                  if not hasattr(self, 'leg_phase') or self.leg_phase.shape != (self.num_envs, 2):
                       self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)
                  self.leg_phase[:, 0] = self.phase_left
                  self.leg_phase[:, 1] = self.phase_right
        # else: # Avoid verbose warnings
             # print("⚠️ G1CurriculumBase: Cannot calculate phase due to missing attributes or zero dt/period.")
             # Ensure phase tensors exist even if not calculated
             if not hasattr(self, 'phase'): self.phase = torch.zeros(self.num_envs, device=self.device)
             if not hasattr(self, 'leg_phase'): self.leg_phase = torch.zeros(self.num_envs, 2, device=self.device)


    # compute_observations: 重写以构建 G1 的基础观测
    def compute_observations(self):
        """ Computes G1 specific base observations based on the *current* self.num_actions.
            Includes phase information.
        """
        # --- 1. 计算通用组件 ---
        # Phase (ensure it's calculated in _post_physics_step_callback)
        if not hasattr(self, 'phase'): self.phase = torch.zeros(self.num_envs, device=self.device)
        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)

        # --- Get ACTIVE DoF states and actions ---
        # Use self.num_actions which reflects the current sub-stage's active count
        num_act = self.num_actions
        # Ensure dof_pos/vel have expected full shape [num_envs, 43] before slicing
        if self.dof_pos.shape[1] != self.num_dof or self.dof_vel.shape[1] != self.num_dof:
             print(f"❌ ERROR compute_observations: Full DOF state shape mismatch! Expected {self.num_dof}, Got {self.dof_pos.shape[1]}")
             # Fallback: Use zeros with expected shape for active part
             active_dof_pos = torch.zeros((self.num_envs, num_act), device=self.device)
             active_dof_vel = torch.zeros((self.num_envs, num_act), device=self.device)
             # Also need default pos for the active part
             active_default_dof_pos = torch.zeros((1, num_act), device=self.device) # Assuming default_dof_pos is [1, 43]
             if hasattr(self, 'active_joint_indices') and self.active_joint_indices is not None and self.default_dof_pos.shape[1] == self.num_dof:
                 active_default_dof_pos = self.default_dof_pos[:, self.active_joint_indices]

        else:
             # Slice the full DoF state/default pos using active indices if available
             if hasattr(self, 'active_joint_indices') and self.active_joint_indices is not None:
                  # Check if indices length matches num_actions
                  if len(self.active_joint_indices) == num_act:
                       active_dof_pos = self.dof_pos[:, self.active_joint_indices]
                       active_dof_vel = self.dof_vel[:, self.active_joint_indices]
                       active_default_dof_pos = self.default_dof_pos[:, self.active_joint_indices]
                  else: # Mismatch between indices and num_actions!
                       print(f"❌ ERROR compute_observations: Mismatch between active_joint_indices ({len(self.active_joint_indices)}) and num_actions ({num_act})!")
                       # Fallback: Take first num_act DoFs (likely incorrect)
                       active_dof_pos = self.dof_pos[:, :num_act]
                       active_dof_vel = self.dof_vel[:, :num_act]
                       active_default_dof_pos = self.default_dof_pos[:, :num_act]
             else: # If no active indices, assume first num_act are active
                  print(f"⚠️ compute_observations: active_joint_indices not available. Assuming first {num_act} DoFs are active.")
                  active_dof_pos = self.dof_pos[:, :num_act]
                  active_dof_vel = self.dof_vel[:, :num_act]
                  active_default_dof_pos = self.default_dof_pos[:, :num_act]


        # Scale active DoF states
        # Ensure obs_scales exists
        if not hasattr(self, 'obs_scales'): self.obs_scales = DotDict({'dof_pos': 1.0, 'dof_vel': 1.0}) # Dummy if missing
        dof_pos_scaled = (active_dof_pos - active_default_dof_pos) * getattr(self.obs_scales, 'dof_pos', 1.0)
        dof_vel_scaled = active_dof_vel * getattr(self.obs_scales, 'dof_vel', 1.0)

        # Scaled base velocities and commands
        base_ang_vel_scaled = self.base_ang_vel * getattr(self.obs_scales, 'ang_vel', 1.0)
        # Ensure commands_scale exists
        if not hasattr(self, 'commands_scale'): self.commands_scale = torch.tensor([1.0, 1.0, 1.0], device=self.device)
        commands_scaled = self.commands[:, :3] * self.commands_scale

        # Previous actions (should have shape [num_envs, num_actions])
        actions_to_include = self.actions
        if actions_to_include.shape[1] != num_act:
             print(f"⚠️ WARNING compute_observations: self.actions shape {actions_to_include.shape} != num_actions {num_act}. Using zeros.")
             actions_to_include = torch.zeros((self.num_envs, num_act), device=self.device)

        # --- 2. 组装 G1 基础观测列表 (维度 = 11 + 3 * num_actions) ---
        # 顺序必须与 _get_noise_scale_vec 中的假设一致
        obs_list = [
            base_ang_vel_scaled,    # 3
            self.projected_gravity, # 3
            commands_scaled,        # 3
            dof_pos_scaled,         # num_actions
            dof_vel_scaled,         # num_actions
            actions_to_include,     # num_actions - previous actions
            sin_phase,              # 1
            cos_phase               # 1
        ]

        # --- 3. 拼接成最终的 obs_buf ---
        try:
            self.obs_buf = torch.cat(obs_list, dim=-1)
            # 验证维度
            if self.obs_buf.shape[1] != self.num_observations:
                 print(f"❌ ERROR: G1CurriculumBase computed obs dim ({self.obs_buf.shape[1]}) != configured num_observations ({self.num_observations}). Check obs_list calculation!")
                 # Attempt to fix shape for safety
                 if self.obs_buf.shape[1] > self.num_observations: self.obs_buf = self.obs_buf[:, :self.num_observations]
                 else: padding = torch.zeros((self.num_envs, self.num_observations - self.obs_buf.shape[1]), device=self.device); self.obs_buf = torch.cat([self.obs_buf, padding], dim=-1)

        except Exception as e:
            print(f"❌ G1CurriculumBase ERROR concatenating observation buffer:")
            for i, item in enumerate(obs_list): print(f"  Item {i}: shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
            print(f"  Error: {e}")
            # Fallback to zeros with the expected dimension
            self.obs_buf = torch.zeros(self.num_envs, self.num_observations, device=self.device, dtype=torch.float)

        # --- 4. 组装特权观测 ---
        if self.privileged_obs_buf is not None:
            # 特权观测 = 基础线速度(3) + 普通观测(self.num_observations)
            expected_priv_dim = self.num_privileged_obs
            priv_obs_list = [
                self.base_lin_vel * getattr(self.obs_scales, 'lin_vel', 1.0), # 3
                self.obs_buf                                                  # self.num_observations
            ]
            try:
                self.privileged_obs_buf = torch.cat(priv_obs_list, dim=-1)
                # 验证维度
                if self.privileged_obs_buf.shape[1] != expected_priv_dim:
                     print(f"❌ ERROR: G1CurriculumBase computed priv_obs dim ({self.privileged_obs_buf.shape[1]}) != configured num_privileged_obs ({expected_priv_dim}). Check priv_obs_list!")
                     # Attempt to fix shape
                     if self.privileged_obs_buf.shape[1] > expected_priv_dim: self.privileged_obs_buf = self.privileged_obs_buf[:, :expected_priv_dim]
                     else: padding = torch.zeros((self.num_envs, expected_priv_dim - self.privileged_obs_buf.shape[1]), device=self.device); self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, padding], dim=-1)

            except Exception as e:
                 print(f"❌ G1CurriculumBase ERROR concatenating privileged observation buffer:")
                 for i, item in enumerate(priv_obs_list): print(f"  Item {i}: shape={item.shape if hasattr(item,'shape') else 'N/A'}")
                 print(f"  Error: {e}")
                 # Fallback to zeros
                 self.privileged_obs_buf = torch.zeros(self.num_envs, expected_priv_dim, device=self.device, dtype=torch.float)

        # --- 5. 添加噪声 (在观测计算完成后) ---
        # Noise vector should be created in _init_buffers based on final self.num_observations
        if self.add_noise and hasattr(self, 'noise_scale_vec'):
            if self.noise_scale_vec is not None and self.noise_scale_vec.shape[0] == self.num_observations:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            elif self.noise_scale_vec is not None: # Shape mismatch
                 print(f"⚠️ G1CurriculumBase Warning: noise_scale_vec dim ({self.noise_scale_vec.shape[0]}) != obs_buf dim ({self.num_observations}). Noise not applied correctly.")
            # else: # noise_scale_vec is None (already warned if add_noise is True)
            #      pass


    # --- 基础奖励函数 (复制自 LeggedRobotCfg 列表) ---
    # These functions should now work correctly as they use instance variables like
    # self.base_lin_vel, self.dof_pos, self.contact_forces etc. which are updated.

    def _reward_tracking_lin_vel(self):
        # lin_vel tracking reward.
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # tracking_sigma defined in cfg.rewards
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # ang_vel tracking reward from heading command.
        if self.cfg.commands.heading_command:
            # Target ang vel computed in callback based on heading error
            target_ang_vel = self.commands[:, 2]
        else:
            # Target ang vel is the sampled command
            target_ang_vel = self.commands[:, 2]
        ang_vel_error = torch.square(target_ang_vel - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        # Use projected gravity vector component along z-axis in base frame
        # Expected value is -1 (gravity pointing down z-axis). Deviation penalizes.
        orientation_error = torch.square(self.projected_gravity[:, :2]).sum(dim=1)
        return orientation_error
        # Alternative: Penalize roll/pitch directly
        # return torch.sum(torch.square(self.rpy[:, :2]), dim=1)


    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self.root_states[:, 2] # Use full root states buffer for height
        height_target = getattr(self.cfg.rewards, 'base_height_target', 0.78) # Default target
        return torch.square(base_height - height_target)

    def _reward_torques(self):
        # Penalize torques applied by the policy (use the full torque buffer)
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities (use full dof_vel buffer)
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations (use full dof state buffers)
        # Use last_dof_vel which stores the full previous state
        dt = self.dt
        if dt <= 0:
            dt = 1e-5
        dof_acc = (self.dof_vel - self.last_dof_vel) / dt
        return torch.sum(torch.square(dof_acc), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions (use actions buffer which has current active dim)
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        # Uses self.penalised_contact_indices which are correct handles
        if self.penalised_contact_indices.numel() == 0: return torch.zeros_like(self.rew_buf)
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps in the air
        # Requires cfg.rewards.feet_air_time scale > 0.0
        if self.feet_air_time is None or self.last_contacts is None or self.feet_indices.numel() == 0:
            # print("Debug: feet_air_time prerequisites not met.")
            return torch.zeros_like(self.rew_buf) # Use self.rew_buf shape for consistency

        # Ensure contact force shape matches feet indices shape
        if self.contact_forces is None or self.contact_forces.shape[1] < self.feet_indices.max()+1:
            # print("Debug: feet_air_time contact_forces shape mismatch.")
            return torch.zeros_like(self.rew_buf)

        try:
            contact = self.contact_forces[:, self.feet_indices, 2] > 1. # Shape [N, num_feet]
            # Ensure last_contacts has the same shape
            if self.last_contacts.shape != contact.shape:
                 print(f"⚠️ feet_air_time: Reshaping last_contacts {self.last_contacts.shape} to match contact {contact.shape}")
                 self.last_contacts = torch.zeros_like(contact) # Reset if shape is wrong

            contact_filt = torch.logical_or(contact, self.last_contacts) # Shape [N, num_feet]
            self.last_contacts = contact # Update last_contacts

            # Ensure feet_air_time has the correct shape
            if self.feet_air_time.shape != contact.shape:
                 print(f"⚠️ feet_air_time: Reshaping feet_air_time {self.feet_air_time.shape} to match contact {contact.shape}")
                 self.feet_air_time = torch.zeros_like(contact) # Reset if shape is wrong

            first_contact = (self.feet_air_time > 0.) * contact_filt # Shape [N, num_feet]
            self.feet_air_time += self.dt # Shape [N, num_feet]

            max_air_time = getattr(self.cfg.rewards, 'max_air_time', 1.0)

            # --- Revised Logic ---
            # Calculate reward contribution for each foot where first_contact is True
            reward_contribution = (self.feet_air_time - 0.5) * first_contact # Shape [N, num_feet]

            # Create a mask for air time limit applied *only* where first_contact is True
            air_time_valid_mask = torch.ones_like(first_contact) # Initialize as all True

            # Avoid indexing with empty tensor if no first_contact
            if first_contact.any():
                 air_time_at_first_contact = self.feet_air_time[first_contact]
                 air_time_valid_mask[first_contact] = (air_time_at_first_contact < max_air_time)

            # Apply the validity mask to the reward contribution
            valid_reward_contribution = reward_contribution * air_time_valid_mask

            # Sum the valid contributions per environment
            rew_airTime = torch.sum(valid_reward_contribution, dim=1) # Shape [N]
            # --- End Revised Logic ---

            # Reset air time for feet that just touched down or were already down
            self.feet_air_time *= (~contact_filt) # Shape [N, num_feet]

            # Ensure final reward has correct shape
            if rew_airTime.shape != self.rew_buf.shape:
                 print(f"❌ ERROR feet_air_time: Final reward shape {rew_airTime.shape} != expected {self.rew_buf.shape}")
                 return torch.zeros_like(self.rew_buf)

            return rew_airTime

        except IndexError as e:
             print(f"❌ ERROR feet_air_time: IndexError during calculation: {e}")
             # Print shapes for debugging
             print(f"  Shapes: contact_forces={self.contact_forces.shape}, feet_indices={self.feet_indices}, feet_air_time={self.feet_air_time.shape}, last_contacts={self.last_contacts.shape}")
             return torch.zeros_like(self.rew_buf)
        except Exception as e_gen:
             print(f"❌ ERROR feet_air_time: Generic error: {e_gen}")
             import traceback
             traceback.print_exc()
             return torch.zeros_like(self.rew_buf)


    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        if self.feet_indices.numel() == 0: return torch.zeros_like(self.rew_buf)
        # Use contact force component along x/y directions
        vertical_contact_force = torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2)
        return torch.sum(1.*(vertical_contact_force > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        # Use the full dof_pos buffer
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # Lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.) # Upper limit
        return torch.sum(out_of_limits, dim=1)

    # --- Rewards copied from G1 reference implementation ---
    def _reward_contact(self): # Phase-contact correlation
        if not hasattr(self, 'leg_phase') or self.leg_phase is None or self.leg_phase.shape[1] < 2 or self.feet_num < 2:
            return torch.zeros_like(self.rew_buf)
        res = torch.zeros_like(self.rew_buf)
        for i in range(2): # Only check first two legs (assuming left/right feet are first)
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1.0
            res += torch.eq(is_stance, contact)
        return res / 2.0 # Normalize by 2 legs

    def _reward_feet_swing_height(self): # Penalize swing foot height deviation
        if self.feet_pos is None or self.feet_num == 0:
             return torch.zeros_like(self.rew_buf)
        target_swing_height = getattr(self.cfg.rewards, 'target_swing_height', 0.08)
        # Ensure contact force shape matches feet indices shape
        if self.contact_forces.shape[1] < self.feet_indices.max()+1: return torch.zeros_like(self.rew_buf)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        is_swing = ~contact
        # Ensure feet_pos shape matches is_swing shape
        if self.feet_pos.shape[1] != is_swing.shape[1]: return torch.zeros_like(self.rew_buf)
        height_error_sq = torch.square(self.feet_pos[:, :, 2] - target_swing_height)
        swing_height_error = height_error_sq * is_swing
        return torch.sum(swing_height_error, dim=1)

    def _reward_contact_no_vel(self): # Penalize foot velocity during contact
        if self.feet_vel is None or self.feet_num == 0:
             return torch.zeros_like(self.rew_buf)
        # Ensure contact force shape matches feet indices shape
        if self.contact_forces.shape[1] < self.feet_indices.max()+1: return torch.zeros_like(self.rew_buf)
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.0
        # Ensure feet_vel shape matches contact shape
        if self.feet_vel.shape[1] != contact.shape[1]: return torch.zeros_like(self.rew_buf)
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize_sq_vel = torch.sum(torch.square(contact_feet_vel), dim=2) # Sum over x,y,z
        return torch.sum(penalize_sq_vel, dim=1) # Sum over feet


    def _reward_hip_pos(self): # Penalize hip roll/pitch deviation from default
        try:
             if not hasattr(self, 'dof_names') or not self.dof_names: return torch.zeros_like(self.rew_buf)
             # Use full dof_pos buffer
             left_roll_idx = self.dof_names.index('left_hip_roll_joint')
             left_pitch_idx = self.dof_names.index('left_hip_pitch_joint')
             right_roll_idx = self.dof_names.index('right_hip_roll_joint')
             right_pitch_idx = self.dof_names.index('right_hip_pitch_joint')
             indices = [left_roll_idx, left_pitch_idx, right_roll_idx, right_pitch_idx]
             # Get default positions for these specific joints
             default_hip_pos = self.default_dof_pos[:, indices] # Shape [1, 4]
             current_hip_pos = self.dof_pos[:, indices] # Shape [num_envs, 4]
             hip_pos_error_sq = torch.square(current_hip_pos - default_hip_pos)
             return torch.sum(hip_pos_error_sq, dim=1)
        except ValueError: # If a joint name isn't found
             print(f"⚠️ WARNING: Could not find all required hip joints in dof_names for _reward_hip_pos. Skipping.")
             return torch.zeros_like(self.rew_buf)
        except AttributeError: # If dof_pos or default_dof_pos is not yet initialized correctly
            print(f"⚠️ WARNING: Buffers not available for _reward_hip_pos. Skipping.")
            return torch.zeros_like(self.rew_buf)