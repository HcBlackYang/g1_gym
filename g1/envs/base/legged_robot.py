
# # legged_robot.py
# import time
# from warnings import WarningMessage
# import numpy as np
# import os

# from isaacgym.torch_utils import *
# from isaacgym import gymtorch, gymapi, gymutil

# import torch
# from torch import Tensor
# from typing import Tuple, Dict

# # G1_ROOT_DIR should be defined where this script is imported, or use absolute paths
# # G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from g1.envs.base.base_task import BaseTask
# from g1.utils.math import wrap_to_pi
# from g1.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
# from g1.utils.helpers import class_to_dict, get_load_path, set_seed # Removed parse_sim_params import
# from g1.envs.base.legged_robot_config import LeggedRobotCfg

# class LeggedRobot(BaseTask):
#     # !!! 修改构造函数签名以匹配 BaseTask !!!
#     def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
#         """ Parses the provided config file,
#             calls self._create_envs() (which creates actors within the provided sim),
#             initializes pytorch buffers used during training.

#         Args:
#             cfg (LeggedRobotCfg): Environment config object.
#             sim_params (gymapi.SimParams): simulation parameters.
#             physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX).
#             sim_device (string): 'cuda' or 'cpu'.
#             headless (bool): Run without rendering if True.
#             gym_handle (gymapi.Gym): The Gym API instance (created externally).
#             sim_handle (gymapi.Sim): The Simulation instance (created externally).
#         """
#         print(f"--- LeggedRobot.__init__ (Child of BaseTask) ---")
#         self.height_samples = None
#         self.debug_viz = False
#         self.init_done = False

#         self.up_axis_idx = 2

#         # --- 调用 BaseTask 的 __init__ ---
#         # 它会处理 gym, sim, device, buffers (obs, rew, reset, etc.), viewer 等的设置
#         super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

#         # --- LeggedRobot 特定的初始化 ---
#         self._parse_cfg() # 解析 LeggedRobot 特定的配置

#         # --- 创建环境中的 Actors (在传入的 sim 中) ---
#         # This method should be defined in subclasses if they need specific terrain/ground setup
#         if hasattr(self, '_create_terrain'): # Check if terrain method exists
#              self._create_terrain()
#         else: # Fallback to ground plane
#              self._create_ground_plane()

#         self._create_envs() # This loads assets and creates actors

#         # --- !!! 在所有 Actors 创建后准备 Sim !!! ---
#         print(f"--- LeggedRobot: Preparing simulation after creating envs...")
#         self.gym.prepare_sim(self.sim)
#         # Get actual counts AFTER prepare_sim
#         self.num_total_bodies = self.gym.get_sim_rigid_body_count(self.sim)
#         self.num_total_dofs = self.gym.get_sim_dof_count(self.sim)
#         print(f"  Simulation Prepared. Total Bodies: {self.num_total_bodies}, Total DoFs: {self.num_total_dofs}")
#         # -------------------------------------------

#         # --- 初始化缓冲区和奖励函数 ---
#         # _init_buffers 需要在 prepare_sim 之后，因为它 acquire tensors
#         self._init_buffers()
#         self._prepare_reward_function()
#         # -----------------------------

#         # --- 设置相机 (如果需要) ---
#         if not self.headless and self.viewer is not None:
#              # Ensure viewer config exists
#              if hasattr(self.cfg, 'viewer') and self.cfg.viewer is not None:
#                  self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
#              else:
#                  print("⚠️ Viewer enabled but cfg.viewer settings not found. Using default camera.")
#         # -------------------------

#         self.init_done = True
#         print(f"--- LeggedRobot.__init__ Done ---")


#     def _parse_cfg(self): # Removed cfg argument, use self.cfg
#         """ Parses the legged robot specific configuration sections."""
#         self.dt = self.cfg.control.decimation * self.sim_params.dt
#         self.obs_scales = self.cfg.normalization.obs_scales
#         # Use class_to_dict to handle potential nested scales object
#         self.reward_scales = class_to_dict(self.cfg.rewards.scales)
#         self.command_ranges = class_to_dict(self.cfg.commands.ranges)

#         # Terrain curriculum check (ensure terrain attribute exists)
#         if hasattr(self.cfg, 'terrain') and self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
#              self.cfg.terrain.curriculum = False
#         self.max_episode_length_s = self.cfg.env.episode_length_s
#         self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

#         # Domain rand push interval (ensure domain_rand attribute exists)
#         if hasattr(self.cfg, 'domain_rand'):
#              self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


#     def _create_ground_plane(self):
#         """ Adds a ground plane to the simulation. """
#         print("  Creating ground plane...")
#         plane_params = gymapi.PlaneParams()
#         plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
#         # Ensure terrain attribute exists before accessing friction/restitution
#         if hasattr(self.cfg, 'terrain'):
#             plane_params.static_friction = self.cfg.terrain.static_friction
#             plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
#             plane_params.restitution = self.cfg.terrain.restitution
#         else:
#              # Default values if terrain config is missing
#              plane_params.static_friction = 1.0
#              plane_params.dynamic_friction = 1.0
#              plane_params.restitution = 0.0
#         self.gym.add_ground(self.sim, plane_params)

#     def _get_env_origins(self):
#         """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
#             Otherwise create a grid.
#         """

#         self.custom_origins = False
#         self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
#         # create a grid of robots
#         num_cols = np.floor(np.sqrt(self.num_envs))
#         num_rows = np.ceil(self.num_envs / num_cols)
#         xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
#         spacing = self.cfg.env.env_spacing
#         self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
#         self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
#         self.env_origins[:, 2] = 0.

#     def _create_envs(self):
#         """ Creates environments: loads robot asset, creates actors.
#             Called by __init__ *before* prepare_sim.
#         """
#         print("--- _create_envs (LeggedRobot) ---")
#         # Define G1_ROOT_DIR or ensure it's passed/globally available
#         try:
#              G1_ROOT_DIR_LOCAL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#         except NameError:
#              print("⚠️ WARNING: __file__ not defined, cannot automatically determine G1_ROOT_DIR. Using empty string.")
#              G1_ROOT_DIR_LOCAL = "" # Fallback

#         try:
#              asset_path = self.cfg.asset.file.format(G1_ROOT_DIR=G1_ROOT_DIR_LOCAL)
#              asset_root = os.path.dirname(asset_path)
#              asset_file = os.path.basename(asset_path)
#         except AttributeError as e:
#              raise AttributeError(f"Missing asset configuration in cfg.asset: {e}") from e
#         except KeyError as e:
#             raise KeyError(f"Missing placeholder in cfg.asset.file (e.g., {{G1_ROOT_DIR}}): {e}") from e


#         asset_options = gymapi.AssetOptions()
#         asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
#         asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
#         asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
#         asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
#         asset_options.fix_base_link = self.cfg.asset.fix_base_link
#         asset_options.density = self.cfg.asset.density
#         asset_options.angular_damping = self.cfg.asset.angular_damping
#         asset_options.linear_damping = self.cfg.asset.linear_damping
#         asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
#         asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
#         asset_options.armature = self.cfg.asset.armature
#         asset_options.thickness = self.cfg.asset.thickness
#         asset_options.disable_gravity = self.cfg.asset.disable_gravity

#         print(f"  Loading robot asset: {asset_file} from {asset_root}")
#         if not os.path.exists(os.path.join(asset_root, asset_file)):
#              print(f"❌❌❌ ERROR: Robot asset file not found at: {os.path.join(asset_root, asset_file)}")
#              raise FileNotFoundError(f"Robot asset file not found: {os.path.join(asset_root, asset_file)}")

#         self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
#         if self.robot_asset is None: raise RuntimeError(f"Failed to load robot asset: {asset_path}")

#         # Get DoF/Body counts *from the asset*
#         self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
#         self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
#         # Ensure cfg matches asset
#         if self.cfg.env.num_actions != self.num_dof:
#              print(f"⚠️ LeggedRobot Warning: cfg.env.num_actions ({self.cfg.env.num_actions}) != asset num_dof ({self.num_dof}). Overriding cfg and instance num_actions.")
#              self.cfg.env.num_actions = self.num_dof
#              self.num_actions = self.num_dof

#         print(f"  Asset Loaded. Num DoF={self.num_dof}, Num Bodies={self.num_bodies}")

#         dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
#         rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

#         # save body names from the asset
#         body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
#         self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
#         self.num_dofs = len(self.dof_names) # Should match self.num_dof
#         feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
#         penalized_contact_names = []
#         for name in self.cfg.asset.penalize_contacts_on:
#             penalized_contact_names.extend([s for s in body_names if name in s])
#         termination_contact_names = []
#         for name in self.cfg.asset.terminate_after_contacts_on:
#             termination_contact_names.extend([s for s in body_names if name in s])

#         # Base init state
#         base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
#         self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
#         start_pose = gymapi.Transform() # Will be set per env

#         self._get_env_origins()
#         env_lower = gymapi.Vec3(0., 0., 0.)
#         env_upper = gymapi.Vec3(0., 0., 0.)
#         self.actor_handles = []
#         self.envs = []
#         print(f"  Creating {self.num_envs} environments...")
#         for i in range(self.num_envs):
#             env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
#             if env_handle == gymapi.INVALID_HANDLE:
#                  raise RuntimeError(f"Failed to create env {i}")
#             self.envs.append(env_handle)

#             # Set pose for this env's robot
#             pos = self.env_origins[i].clone()
#             pos += to_torch(self.cfg.init_state.pos, device=self.device) # Add configured initial offset
#             random_offset_xy = torch_rand_float(-0.1, 0.1, (1, 2), device=self.device)
#             pos[:2] += random_offset_xy.squeeze(0)
#             start_pose.p = gymapi.Vec3(*pos)
#             start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)

#             # Process props for this actor instance
#             rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
#             # Apply props to the asset (might affect subsequent actors if not careful)
#             # Consider using set_actor_rigid_shape_properties if props need to differ significantly per env beyond friction
#             self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

#             actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name,
#                                                  i, # Collision group = env index
#                                                  self.cfg.asset.self_collisions, 0) # Collision filter mask
#             if actor_handle == gymapi.INVALID_HANDLE:
#                  raise RuntimeError(f"Failed to create actor in env {i}")
#             self.actor_handles.append(actor_handle)

#             # Set DoF and Body properties for this actor instance
#             # Process DoF props (gets limits first time, reuses after)
#             dof_props = self._process_dof_props(dof_props_asset, i)
#             self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

#             body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
#             # Check if props list is valid before processing
#             if body_props:
#                  body_props = self._process_rigid_body_props(body_props, i) # Randomize mass etc.
#                  self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
#             else:
#                  print(f"⚠️ Warning: Env {i} - got empty body_props list.")


#         # --- Store body indices (after all actors are created) ---
#         if self.actor_handles: # If actors were created
#              first_env_handle = self.envs[0]
#              first_actor_handle = self.actor_handles[0]

#              self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
#              for i, name in enumerate(feet_names):
#                  handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
#                  if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Foot body '{name}' not found.")
#                  self.feet_indices[i] = handle

#              self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
#              for i, name in enumerate(penalized_contact_names):
#                  handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
#                  if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Penalised body '{name}' not found.")
#                  self.penalised_contact_indices[i] = handle

#              self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
#              for i, name in enumerate(termination_contact_names):
#                  handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
#                  if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Termination body '{name}' not found.")
#                  self.termination_contact_indices[i] = handle
#         else:
#              print("❌ ERROR: No actors were created successfully.")
#              # Initialize indices as empty tensors to prevent errors later
#              self.feet_indices = torch.tensor([], dtype=torch.long, device=self.device)
#              self.penalised_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)
#              self.termination_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)

#         print(f"--- _create_envs (LeggedRobot) Done ---")

#     def _init_buffers(self):
#         """ Initialize torch tensors which will contain simulation states and processed quantities.
#             Called *after* prepare_sim. Handles single-actor-per-env case.
#             Needs override in multi-actor envs like G1KitchenNavigation.
#         """
#         print("--- _init_buffers (LeggedRobot) ---")
#         # get gym GPU state tensors
#         actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
#         dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
#         net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
#         rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

#         # Check if tensors were acquired successfully
#         tensor_names = ["actor_root_state", "dof_state", "net_contact_forces", "rigid_body_state"]
#         tensors = [actor_root_state, dof_state_tensor, net_contact_forces, rigid_body_state_tensor]
#         for name, tensor in zip(tensor_names, tensors):
#              if tensor is None:
#                   print(f"❌ CRITICAL ERROR: Failed to acquire {name}_tensor from Isaac Gym.")
#                   # Provide more context if possible
#                   print(f"   - Was gym.prepare_sim(sim) called after all actors were created?")
#                   print(f"   - Check for errors during gym.create_sim or actor creation.")
#                   raise RuntimeError(f"Failed to acquire simulation state tensor: {name}")

#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         self.gym.refresh_dof_state_tensor(self.sim)
#         self.gym.refresh_net_contact_force_tensor(self.sim)
#         self.gym.refresh_rigid_body_state_tensor(self.sim)

#         # Wrap tensors
#         self.root_states = gymtorch.wrap_tensor(actor_root_state)       # Shape [num_envs, 13]
#         self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)         # Shape [num_envs * num_dof, 2]
#         self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor) # Shape [num_envs * num_bodies, 13]
#         self.contact_forces_raw = gymtorch.wrap_tensor(net_contact_forces) # Raw shape, needs reshape

#         print(f"  Raw Tensor Shapes: root={self.root_states.shape}, dof={self.dof_state.shape}, rigid_body={self.rigid_body_states.shape}, contact={self.contact_forces_raw.shape}")

#         # Create Views (handle potential shape mismatches)
#         # DoF state views
#         try:
#             dof_state_reshaped = self.dof_state.view(self.num_envs, self.num_dof, 2)
#             self.dof_pos = dof_state_reshaped[..., 0]
#             self.dof_vel = dof_state_reshaped[..., 1]
#         except RuntimeError as e:
#             print(f"❌ ERROR reshaping dof_state {self.dof_state.shape} to ({self.num_envs}, {self.num_dof}, 2): {e}")
#             raise # Re-raise as this is critical

#         # Rigid body state view
#         try:
#             self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
#         except RuntimeError as e:
#             print(f"❌ ERROR reshaping rigid_body_states {self.rigid_body_states.shape} to ({self.num_envs}, {self.num_bodies}, 13): {e}")
#             raise

#         # Contact forces view
#         try:
#             # Expected size: num_envs * num_bodies * 3
#             expected_contact_elements = self.num_envs * self.num_bodies * 3
#             if self.contact_forces_raw.numel() == expected_contact_elements:
#                  self.contact_forces = self.contact_forces_raw.view(self.num_envs, self.num_bodies, 3)
#                  print(f"  Reshaped contact_forces to: {self.contact_forces.shape}")
#             else:
#                  # Fallback for safety, but indicates underlying issue
#                  print(f"⚠️ WARNING: Unexpected contact_forces_raw shape {self.contact_forces_raw.shape}. Num elements {self.contact_forces_raw.numel()} != expected {expected_contact_elements}. Using view(num_envs, -1, 3).")
#                  self.contact_forces = self.contact_forces_raw.view(self.num_envs, -1, 3)
#         except RuntimeError as e:
#             print(f"❌ ERROR reshaping contact_forces_raw {self.contact_forces_raw.shape}: {e}")
#             raise

#         # Robot state views
#         self.base_quat = self.root_states[:, 3:7]
#         self.base_pos = self.root_states[:, 0:3]
#         self.rpy = torch.zeros_like(self.base_pos)

#         # Initialize other buffers
#         self.common_step_counter = 0
#         self.extras = {}
#         # Initialize noise_scale_vec after obs_buf is correctly sized
#         # self.obs_buf initialized in BaseTask using self.num_observations
#         self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # Call method to create it

#         self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
#         self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
#         # Use self.num_actions (which should == self.num_dof)
#         self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
#         self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
#         self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
#         self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
#         self.last_actions = torch.zeros_like(self.actions)
#         self.last_dof_vel = torch.zeros_like(self.dof_vel)
#         self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
#         self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
#         self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)

#         # Feet buffers
#         if self.feet_indices.numel() > 0:
#             self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
#             self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
#         else:
#             self.feet_air_time = None
#             self.last_contacts = None

#         # Calculated velocities
#         self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
#         self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
#         self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

#         # PD gains, default pos, torque limits
#         self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
#         self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
#         dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset) # Use asset loaded in _create_envs

#         for i in range(self.num_dof):
#             name = self.dof_names[i]
#             angle = self.cfg.init_state.default_joint_angles.get(name, 0.0)
#             self.default_dof_pos[i] = angle
#             self.torque_limits[i] = dof_props_asset["effort"][i].item()

#             found = False
#             for dof_name_key in self.cfg.control.stiffness.keys():
#                 if dof_name_key in name:
#                     self.p_gains[i] = self.cfg.control.stiffness[dof_name_key]
#                     self.d_gains[i] = self.cfg.control.damping.get(dof_name_key, 0.0)
#                     found = True
#                     break
#             if not found:
#                 self.p_gains[i] = 0.; self.d_gains[i] = 0.
#                 if self.cfg.control.control_type in ["P", "V"]: print(f"  PD gain '{name}' not defined.")
#         self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

#         # Initialize foot state tracking
#         self._init_foot()
#         print("--- _init_buffers (LeggedRobot) Done ---")


#     def _prepare_reward_function(self):
#         """ Prepares a list of reward functions based on non-zero scales. """
#         print("  Preparing reward functions...")
#         # Use the scales potentially modified by the scheduler (stored in self.reward_scales)
#         current_reward_scales = self.reward_scales

#         # Prepare list of functions based on non-zero scales
#         self.reward_functions = []
#         self.reward_names = []
#         skipped_rewards = []
#         for name, scale in current_reward_scales.items():
#             if name == "termination": continue
#             if scale != 0.0:
#                  func_name = '_reward_' + name
#                  if hasattr(self, func_name) and callable(getattr(self, func_name)):
#                       self.reward_names.append(name)
#                       self.reward_functions.append(getattr(self, func_name))
#                       # Multiply scale by dt here for efficiency
#                       current_reward_scales[name] *= self.dt
#                  else:
#                       skipped_rewards.append(f"{name} (missing function {func_name})")
#             # else: # Keep track of zero-scaled rewards if needed
#             #     skipped_rewards.append(f"{name} (scale is 0)")


#         # Reward episode sums (only for active rewards)
#         self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
#                              for name in self.reward_names}
#         # Also include termination if used
#         if "termination" in current_reward_scales and current_reward_scales["termination"] != 0.0:
#             self.reward_names.append("termination") # Add termination to names for sum tracking
#             self.episode_sums["termination"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
#             # Multiply termination scale by 1 (it's applied once per episode end)
#             # current_reward_scales["termination"] *= 1.0 # Or just use the value directly

#         print(f"    Active reward functions: {self.reward_names}")
#         if skipped_rewards: print(f"    Skipped/Inactive rewards: {skipped_rewards}")


#     # --- Step, Post Physics Step, Check Termination, Reset Idx ---
#     # These methods generally remain the same as your provided version,
#     # assuming they correctly use the instance variables initialized in _init_buffers
#     # (e.g., self.root_states, self.dof_pos, self.contact_forces, etc.)

#     def step(self, actions):
#         """ Apply actions, simulate, call self.post_physics_step() """
#         if actions.shape[1] != self.num_actions:
#              print(f"❌ ERROR in step: Received actions shape {actions.shape} does not match num_actions {self.num_actions}")
#              # Handle error, e.g., by taking a subset or padding, or raising error
#              actions = actions[:, :self.num_actions] # Simple truncation

#         clip_actions = self.cfg.normalization.clip_actions
#         self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

#         # step physics and render each frame
#         self.render()
#         for _ in range(self.cfg.control.decimation):
#             # Compute torques for the robot
#             computed_torques = self._compute_torques(self.actions) # Shape [num_envs, num_actions]
#             # Set torques using the global tensor if needed (handled by specific envs)
#             # Here, assume torques buffer matches set_dof_actuation_force_tensor input
#             if computed_torques.shape == self.torques.shape:
#                  self.torques[:] = computed_torques
#                  self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques.flatten())) # Flatten if needed by API
#             else:
#                  # Handle shape mismatch - G1KitchenNav overrides _compute_torques
#                  print(f"⚠️ WARNING step: Shape mismatch computed_torques {computed_torques.shape} vs self.torques {self.torques.shape}. Using LeggedRobot logic.")
#                  self.torques = computed_torques.view(self.torques.shape) # Try to reshape
#                  self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))


#             self.gym.simulate(self.sim)
#             if self.device == 'cpu': self.gym.fetch_results(self.sim, True) # Fetch results after simulate
#             # Refresh states needed for next loop iteration or post_physics_step
#             self.gym.refresh_dof_state_tensor(self.sim)
#             # if hasattr(self, 'manual_dof_update') and self.manual_dof_update: # If using fallback
#             #      # Manually update self.dof_pos/vel from self.dof_state
#             #      print("Manual DOF update needed - NOT IMPLEMENTED YET") # TODO
#             #      pass
#             # Else: dof_pos/vel views are automatically updated

#         self.post_physics_step()

#         # return clipped obs, clipped states (None), rewards, dones and infos
#         clip_obs = self.cfg.normalization.clip_observations
#         self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
#         if self.privileged_obs_buf is not None:
#             self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
#         return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

#     def _post_physics_step_callback(self):
#         """ Callback called before computing terminations, rewards, and observations.
#             Handles common logic like command resampling and heading update.
#             Subclasses can override this and call super().
#         """
#         # Resample commands based on time
#         # Ensure necessary attributes exist before using them
#         if hasattr(self, 'episode_length_buf') and hasattr(self.cfg, 'commands') and hasattr(self.cfg.commands,
#                                                                                              'resampling_time') and self.dt > 0:
#             resampling_interval_steps = getattr(self.cfg.commands, 'resampling_interval_steps', None)
#             if resampling_interval_steps is None:  # Calculate if not pre-calculated
#                 resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt)
#                 self.cfg.commands.resampling_interval_steps = resampling_interval_steps  # Cache it

#             if resampling_interval_steps > 0:  # Avoid modulo by zero
#                 env_ids = (self.episode_length_buf % resampling_interval_steps == 0).nonzero(as_tuple=False).flatten()
#                 if len(env_ids) > 0:
#                     self._resample_commands(env_ids)
#             else:
#                 print("⚠️ Warning: resampling_interval_steps is zero or negative. Skipping command resampling.")

#         # Compute heading command if enabled
#         if getattr(self.cfg.commands, 'heading_command', False):
#             if hasattr(self, 'base_quat') and hasattr(self, 'forward_vec') and hasattr(self, 'commands') and \
#                     self.commands.shape[1] >= 4:
#                 try:
#                     forward = quat_apply(self.base_quat, self.forward_vec)
#                     heading = torch.atan2(forward[:, 1], forward[:, 0])
#                     heading_error = wrap_to_pi(self.commands[:, 3] - heading)
#                     # Apply gain (e.g., 0.5) and clip
#                     yaw_command = torch.clip(0.5 * heading_error, -1., 1.)
#                     self.commands[:, 2] = yaw_command
#                 except Exception as e:
#                     print(f"❌ Error computing heading command in LeggedRobot callback: {e}")
#             else:
#                 print("⚠️ Cannot compute heading command: required attributes missing or commands shape incorrect.")


#     def post_physics_step(self):
#         """ check terminations, compute observations and rewards """
#         self.gym.refresh_actor_root_state_tensor(self.sim)
#         self.gym.refresh_net_contact_force_tensor(self.sim)
#         # Refresh DoF state if not done in the inner loop (it is done there)
#         # self.gym.refresh_dof_state_tensor(self.sim)
#         # Refresh rigid body state for feet update
#         # self.gym.refresh_rigid_body_state_tensor(self.sim) # Done in update_feet_state if needed

#         self.episode_length_buf += 1
#         self.common_step_counter += 1

#         # prepare quantities (using robot-specific states where applicable)
#         # BaseTask uses self.root_states, LeggedRobot assumes it's [num_envs, 13]
#         # If multi-actor envs override _init_buffers, they need to set up self.robot_root_states view correctly
#         if hasattr(self, 'robot_root_states'): # Use robot-specific view if exists (like in G1KitchenNav)
#              root_states_to_use = self.robot_root_states
#         else: # Fallback to global root_states (assuming single actor per env)
#              root_states_to_use = self.root_states

#         self.base_pos[:] = root_states_to_use[:, 0:3]
#         self.base_quat[:] = root_states_to_use[:, 3:7]
#         self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:]) # Use updated base_quat
#         self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, root_states_to_use[:, 7:10])
#         self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, root_states_to_use[:, 10:13])
#         self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
#         # Update DoF pos/vel if using manual update fallback
#         if hasattr(self, 'manual_dof_update') and self.manual_dof_update:
#              # TODO: Implement manual update logic here if needed
#              pass

#         # Callbacks (phase calculation, command resampling)
#         self._post_physics_step_callback()

#         # compute observations, rewards, resets, ...
#         self.check_termination()
#         self.compute_reward()
#         env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
#         if len(env_ids) > 0: # Only call reset if needed
#              self.reset_idx(env_ids)

#         # Push robots (uses self.root_states directly, needs override for multi-actor)
#         if self.cfg.domain_rand.push_robots:
#             self._push_robots()

#         # Compute observations *after* potential resets and state updates
#         self.compute_observations()

#         # Store last states
#         self.last_actions[:] = self.actions[:]
#         self.last_dof_vel[:] = self.dof_vel[:] # Uses robot dof_vel
#         self.last_root_vel[:] = root_states_to_use[:, 7:13] # Uses robot root vel


#     def check_termination(self):
#         """ Check if environments need to be reset """
#         # Use robot's termination indices relative to robot's contact forces
#         # Need to get contact forces specific to the robot bodies
#         if self.contact_forces is None: # Check if contact forces are valid
#              print("⚠️ check_termination: Contact forces buffer is None. Cannot check termination.")
#              self.reset_buf.zero_() # Assume no termination if contacts unavailable
#              self.time_out_buf = self.episode_length_buf > self.max_episode_length
#              self.reset_buf |= self.time_out_buf
#              return

#         # Assuming self.contact_forces is [num_envs, num_total_bodies_per_env, 3]
#         # And termination_contact_indices are local indices for the robot (0 to num_bodies-1)
#         # We need contact forces only for the robot bodies
#         # If only one actor/env, num_total_bodies_per_env == self.num_bodies
#         robot_contact_forces = self.contact_forces[:, :self.num_bodies, :] # Assume robot bodies are first

#         try:
#              # Use local termination indices with the robot's contact forces view
#              self.reset_buf = torch.any(torch.norm(robot_contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
#         except IndexError as e:
#              print(f"❌ ERROR in check_termination: IndexError accessing contact_forces. Indices={self.termination_contact_indices}, Shape={robot_contact_forces.shape}. {e}")
#              self.reset_buf.zero_() # Fail safe

#         # Orientation check uses self.rpy which is based on robot_root_states
#         self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1]) > 1.0, torch.abs(self.rpy[:,0]) > 0.8)
#         # Timeout check
#         self.time_out_buf = self.episode_length_buf > self.max_episode_length
#         self.reset_buf |= self.time_out_buf


#     def reset_idx(self, env_ids):
#         """ Reset some environments. """
#         if len(env_ids) == 0: return

#         # Reset robot states (calls methods that use self.dof_pos, self.root_states etc.)
#         self._reset_dofs(env_ids) # Resets robot DoFs
#         self._reset_root_states(env_ids) # Resets robot root state

#         # Resample commands for reset envs
#         self._resample_commands(env_ids)

#         # Reset buffers
#         self.last_actions[env_ids] = 0.
#         # Ensure last_dof_vel has same shape as dof_vel before indexing
#         if self.last_dof_vel.shape[0] == self.num_envs: self.last_dof_vel[env_ids] = 0.
#         if self.feet_air_time is not None: self.feet_air_time[env_ids] = 0.
#         self.episode_length_buf[env_ids] = 0
#         self.reset_buf[env_ids] = 1 # Mark for reset in the next step logic if using delayed reset

#         # Fill extras and log episode sums
#         self.extras["episode"] = {}
#         # Ensure episode_sums keys match active rewards
#         active_reward_names = list(self.episode_sums.keys())
#         for key in active_reward_names:
#              # Safely access episode sums
#              sum_tensor = self.episode_sums.get(key)
#              if sum_tensor is not None and sum_tensor.shape[0] == self.num_envs:
#                  self.extras["episode"]['rew_' + key] = torch.mean(sum_tensor[env_ids]).item() / self.max_episode_length_s
#                  self.episode_sums[key][env_ids] = 0.
#              else:
#                  print(f"⚠️ reset_idx: Skipping episode sum logging for '{key}' due to missing or mismatched tensor.")

#         # Command curriculum logging (ensure commands attribute exists)
#         if hasattr(self.cfg, 'commands') and self.cfg.commands.curriculum:
#             self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
#         # Send timeout info
#         if self.cfg.env.send_timeouts:
#             self.extras["time_outs"] = self.time_out_buf.clone() # Send a copy


#     def compute_reward(self):
#         """ Compute rewards based on active reward functions and scales. """
#         self.rew_buf[:] = 0.
#         # episode_sums should be initialized/updated based on active rewards in _prepare_reward_function
#         active_reward_names = list(self.episode_sums.keys())
#         if "termination" in active_reward_names: active_reward_names.remove("termination") # Handle termination separately

#         for name in active_reward_names:
#              try:
#                  reward_func = getattr(self, '_reward_' + name)
#                  rew = reward_func() * self.reward_scales[name] # reward_scales holds dt adjusted value
#                  # Shape check
#                  if rew.shape == (self.num_envs,):
#                       self.rew_buf += rew
#                       self.episode_sums[name] += rew # Accumulate sum for active reward
#                  else:
#                       print(f"⚠️ compute_reward: Shape mismatch for reward '{name}'. Expected ({self.num_envs},), got {rew.shape}. Skipping.")
#              except Exception as e:
#                   print(f"❌ Error computing reward '{name}': {e}")

#         # Apply clipping
#         if self.cfg.rewards.only_positive_rewards:
#             self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

#         # Add termination reward after clipping
#         if "termination" in self.reward_scales and self.reward_scales["termination"] != 0:
#             # Check if function exists before calling
#             if hasattr(self, '_reward_termination'):
#                  rew = self._reward_termination() * self.reward_scales["termination"]
#                  if rew.shape == (self.num_envs,):
#                      self.rew_buf += rew
#                      # Check if "termination" key exists before accessing
#                      if "termination" in self.episode_sums:
#                          self.episode_sums["termination"] += rew
#                  else:
#                       print(f"⚠️ compute_reward: Shape mismatch for termination reward. Expected ({self.num_envs},), got {rew.shape}.")
#             else:
#                  print("⚠️ compute_reward: Termination reward enabled but _reward_termination function missing.")


#     def set_camera(self, position, lookat):
#         """ Set camera position and direction. """
#         if self.viewer is None: return # Do nothing if viewer doesn't exist
#         cam_pos = gymapi.Vec3(*position)
#         cam_target = gymapi.Vec3(*lookat)
#         self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


#     # --- Callbacks (_process_*) remain the same ---
#     def _process_rigid_shape_props(self, props, env_id):
#         if self.cfg.domain_rand.randomize_friction:
#             if env_id==0:
#                 friction_range = self.cfg.domain_rand.friction_range
#                 num_buckets = 64
#                 bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
#                 friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
#                 self.friction_coeffs = friction_buckets[bucket_ids]
#             if props: # Check if props is not empty
#                  for s in range(len(props)):
#                      props[s].friction = self.friction_coeffs[env_id]
#         return props

#     def _process_dof_props(self, props, env_id):
#         if env_id==0:
#             # Initialize only if not already initialized (in case called multiple times)
#             if not hasattr(self, 'dof_pos_limits') or self.dof_pos_limits.shape[0] != self.num_dof:
#                  self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
#                  self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
#                  # Torque limits initialized in _init_buffers now
#                  # self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

#             for i in range(len(props)): # Should match self.num_dof
#                 # Ensure index is within bounds
#                 if i < self.num_dof:
#                      self.dof_pos_limits[i, 0] = props["lower"][i].item()
#                      self.dof_pos_limits[i, 1] = props["upper"][i].item()
#                      self.dof_vel_limits[i] = props["velocity"][i].item()
#                      # self.torque_limits[i] = props["effort"][i].item() # Set in _init_buffers
#                      # soft limits calculation
#                      m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
#                      r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
#                      self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
#                      self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
#         return props

#     def _process_rigid_body_props(self, props, env_id):
#         if not props: # Check for empty list
#             print("⚠️ _process_rigid_body_props: Received empty props list.")
#             return props
#         if self.cfg.domain_rand.randomize_base_mass:
#             rng = self.cfg.domain_rand.added_mass_range
#             # Ensure props[0] exists before accessing mass
#             if len(props) > 0:
#                  props[0].mass += np.random.uniform(rng[0], rng[1])
#             else:
#                  print("⚠️ _process_rigid_body_props: Cannot apply base mass randomization, props list is empty.")
#         return props

#     # --- _resample_commands, _compute_torques ---
#     # These seem okay, ensure they use instance variables correctly (e.g., self.num_actions)
#     def _resample_commands(self, env_ids):
#         """ Randommly select commands of some environments """
#         if len(env_ids) == 0: return
#         # Use self.command_ranges directly
#         self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
#         self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
#         if self.cfg.commands.heading_command:
#             # Ensure self.commands has column 3
#             if self.commands.shape[1] >= 4:
#                  self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
#             else: print("⚠️ _resample_commands: Heading command enabled but self.commands has < 4 columns.")
#         else:
#             # Ensure self.commands has column 2
#              if self.commands.shape[1] >= 3:
#                  self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
#              else: print("⚠️ _resample_commands: Yaw command enabled but self.commands has < 3 columns.")


#         # set small commands to zero
#         self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

#     def _compute_torques(self, actions):
#         """ Compute torques for the robot based on its DoFs. """
#         # actions shape: [num_envs, num_actions] (e.g., [*, 43])
#         if actions.shape[1] != self.num_actions:
#              print(f"❌ ERROR _compute_torques: actions shape {actions.shape} mismatch num_actions {self.num_actions}")
#              return torch.zeros_like(self.torques) # Return zero torques

#         actions_scaled = actions * self.cfg.control.action_scale
#         control_type = self.cfg.control.control_type

#         # Use robot-specific DoF states (self.dof_pos, self.dof_vel)
#         # Ensure shapes match num_actions
#         if self.dof_pos.shape[1] != self.num_actions or self.dof_vel.shape[1] != self.num_actions:
#              print(f"❌ ERROR _compute_torques: DOF state shapes mismatch num_actions")
#              return torch.zeros_like(self.torques)

#         if control_type=="P":
#             # Ensure default_dof_pos shape is [1, num_actions] or [num_actions]
#             if self.default_dof_pos.shape[-1] != self.num_actions:
#                  print(f"❌ ERROR _compute_torques: default_dof_pos shape mismatch")
#                  return torch.zeros_like(self.torques)
#             torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
#         elif control_type=="V":
#             # Ensure last_dof_vel shape matches dof_vel
#             if self.last_dof_vel.shape != self.dof_vel.shape: self.last_dof_vel = torch.zeros_like(self.dof_vel)
#             dt = self.dt # Use calculated dt
#             if dt <= 0: dt = 1e-5 # Avoid division by zero
#             torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/ dt
#         elif control_type=="T":
#             torques = actions_scaled
#         else:
#             raise NameError(f"Unknown controller type: {control_type}")

#         # Clip using robot torque limits
#         return torch.clip(torques, -self.torque_limits, self.torque_limits) # Shape [num_envs, num_actions]

#     # --- _reset_dofs, _reset_root_states ---
#     # Need modification for multi-actor environments in subclasses.
#     # Base implementation assumes tensors map directly to the single robot.
#     def _reset_dofs(self, env_ids):
#         """ Resets DOF position and velocities of selected robots. """
#         if len(env_ids) == 0: return
#         # Generate random positions for the robot DoFs
#         # Ensure self.dof_pos buffer is used here (shape [num_envs, num_dof])
#         new_dof_pos = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
#         new_dof_vel = torch.zeros((len(env_ids), self.num_dof), device=self.device)

#         # --- Update the global dof_state tensor ---
#         # Use the indices calculated in _init_buffers (assuming fallback logic)
#         # This needs careful implementation in G1KitchenNavigation
#         # For base LeggedRobot, the simple view update might work IF dof_state is contiguous
#         try:
#              dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
#              dof_state_view[env_ids, :, 0] = new_dof_pos
#              dof_state_view[env_ids, :, 1] = new_dof_vel
#         except RuntimeError:
#              # Fallback: Use set_dof_state_tensor_indexed (less efficient for sparse updates)
#              print("⚠️ _reset_dofs: View failed, using indexed update (potentially slow).")
#              temp_dof_state = self.dof_state.clone()
#              # Assuming self.all_robot_dof_indices holds global indices [env0_dof0, ..., envN_dofM]
#              # This mapping is complex, needs careful setup in _init_buffers.
#              # --- Simplified approach for base class ---
#              # Assume dof_state is [num_envs * num_dof, 2]
#              indices_pos = env_ids * self.num_dof # This indexing is likely WRONG for sparse env_ids
#              indices_vel = indices_pos + 1
#              # This part needs a robust way to map env_ids and dof_idx to global index
#              print("TODO: Implement robust indexed DOF reset for sparse env_ids")

#         # Update internal buffers as well
#         self.dof_pos[env_ids] = new_dof_pos
#         self.dof_vel[env_ids] = new_dof_vel

#         # Call gym function to apply changes (using the full tensor)
#         self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))


#     def _reset_root_states(self, env_ids):
#         """ Resets ROOT states position and velocities of selected robots. """
#         if len(env_ids) == 0: return
#         # Base implementation assumes self.root_states is [num_envs, 13]
#         # Generate random velocities
#         random_vels = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

#         # Apply to root_states buffer
#         self.root_states[env_ids] = self.base_init_state.clone() # Start with default state
#         self.root_states[env_ids, :3] += self.env_origins[env_ids] # Add env origin
#         # Optional: Add random XY offset from origin
#         self.root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)
#         self.root_states[env_ids, 7:13] = random_vels # Set random velocities

#         # Set tensor in sim
#         env_ids_int32 = env_ids.to(dtype=torch.int32)
#         # Use set_actor_root_state_tensor_indexed - needs global actor indices
#         # If actor handles correspond directly to env_ids (single actor per env)
#         actor_indices = env_ids_int32 # Assuming actor index == env index
#         self.gym.set_actor_root_state_tensor_indexed(
#             self.sim,
#             gymtorch.unwrap_tensor(self.root_states), # Pass the full tensor
#             gymtorch.unwrap_tensor(actor_indices), # Pass the indices of actors to reset
#             len(env_ids_int32)
#         )

#     # --- _push_robots needs override for multi-actor ---
#     def _push_robots(self):
#         """ Random pushes the robots. Base implementation assumes single actor per env. """
#         env_ids = torch.arange(self.num_envs, device=self.device)
#         # Use push_interval from cfg
#         push_interval_steps = getattr(self.cfg.domain_rand, 'push_interval', 300) # Default ~1.5s
#         push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(push_interval_steps) == 0]

#         if len(push_env_ids) == 0: return

#         max_vel = self.cfg.domain_rand.max_push_vel_xy
#         random_vel_xy = torch_rand_float(-max_vel, max_vel, (len(push_env_ids), 2), device=self.device)

#         # Apply to root_states buffer for the selected envs
#         self.root_states[push_env_ids, 7:9] = random_vel_xy # Set lin vel x/y

#         # Set tensor in sim using indexed update
#         actor_indices = push_env_ids.to(dtype=torch.int32) # Assuming actor index == env index
#         self.gym.set_actor_root_state_tensor_indexed(
#             self.sim,
#             gymtorch.unwrap_tensor(self.root_states),
#             gymtorch.unwrap_tensor(actor_indices),
#             len(actor_indices)
#         )

#     # --- update_command_curriculum, _get_noise_scale_vec (base), reward functions ---
#     # These seem generally OK, relying on instance variables. Ensure reward functions use correct states.
#     def update_command_curriculum(self, env_ids):
#         """ Implements a curriculum of increasing commands """
#         if len(env_ids) == 0: return
#         # Ensure reward scales and episode sums are properly initialized
#         if "tracking_lin_vel" not in self.reward_scales or "tracking_lin_vel" not in self.episode_sums:
#              return

#         # Check for division by zero
#         if self.max_episode_length_s <= 0: return

#         # Calculate average reward, handle potential NaN
#         mean_rew = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s)
#         if torch.isnan(mean_rew): return

#         # Compare with target threshold (e.g., 80% of max possible scale * dt)
#         # Note: self.reward_scales already includes dt multiplication
#         target_rew_threshold = 0.8 * self.reward_scales["tracking_lin_vel"] / self.dt # Divide by dt to get original scale threshold

#         if mean_rew / self.dt > target_rew_threshold: # Compare original scales
#             max_curr = self.cfg.commands.max_curriculum
#             self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -max_curr, 0.) # Smaller increments
#             self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., max_curr)
#             # Optionally update other command ranges (y, yaw) as well


#     # Reward functions (_reward_*) - seem OK, ensure they use self.base_lin_vel etc.




# legged_robot.py
import time
from warnings import WarningMessage
import numpy as np
import os
import inspect # Import inspect

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from g1.envs.base.base_task import BaseTask
from g1.utils.math import wrap_to_pi
from g1.utils.helpers import class_to_dict, get_load_path, set_seed, parse_sim_params, DotDict # Import DotDict

# Local implementation of get_euler_xyz_in_tensor if not available elsewhere
# Ensure this matches the implementation in g1/utils/isaacgym_utils.py if it exists
def get_euler_xyz_in_tensor(quat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    r, p, y = get_euler_xyz(quat)
    return r, p, y

# Function from isaacgym.torch_utils (ensure consistent behavior)
@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = 1.0 - 2.0 * (q[:, qx] * q[:, qx] + q[:, qy] * q[:, qy])
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, torch.copysign(torch.tensor(np.pi / 2.0), sinp), torch.asin(sinp)) # Use torch.copysign and torch.tensor

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = 1.0 - 2.0 * (q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz])
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


from g1.envs.base.legged_robot_config import LeggedRobotCfg
import g1.envs # Import g1.envs to ensure module path is recognized

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        print(f"--- LeggedRobot.__init__ (Child of BaseTask) ---")
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        # self._parse_cfg(cfg) # <-- MOVE THIS CALL

        # --- 调用 BaseTask 的 __init__ ---
        # Sets self.cfg, self.sim_params, self.device, basic buffers, etc.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

        # --- LeggedRobot 特定的初始化 (AFTER BaseTask init) ---
        # Now self.sim_params exists
        self._parse_cfg(self.cfg) # <-- CALL PARSE CFG HERE
        self.up_axis_idx = 2 # Assuming Z-up, consistent with isaacgym default

        # --- 创建环境中的 Actors ---
        if hasattr(self, '_create_terrain'):
             self._create_terrain()
        else:
             self._create_ground_plane()
        self._create_envs()

        # --- 准备 Sim ---
        print(f"--- LeggedRobot: Preparing simulation after creating envs...")
        self.gym.prepare_sim(self.sim)
        self.num_total_bodies = self.gym.get_sim_rigid_body_count(self.sim)
        self.num_total_dofs = self.gym.get_sim_dof_count(self.sim)
        print(f"  Simulation Prepared. Total Bodies: {self.num_total_bodies}, Total DoFs: {self.num_total_dofs}")

        # --- 初始化缓冲区和奖励函数 ---
        self._init_buffers()
        self._prepare_reward_function()

        # --- 设置相机 ---
        if not self.headless and self.viewer is not None:
             if hasattr(self.cfg, 'viewer') and self.cfg.viewer is not None:
                 self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
             else:
                 print("⚠️ Viewer enabled but cfg.viewer settings not found. Using default camera.")

        self.init_done = True
        print(f"--- LeggedRobot.__init__ Done ---")


    def _parse_cfg(self, cfg): # Pass cfg explicitly
        """ Parses the legged robot specific configuration sections."""
        # --- FIX: Check if sim_params exists before accessing dt ---
        if not hasattr(self, 'sim_params') or self.sim_params is None:
             raise AttributeError("LeggedRobot._parse_cfg called before self.sim_params was set by BaseTask.__init__")
        # --- End Fix ---

        self.dt = cfg.control.decimation * self.sim_params.dt
        self.obs_scales = cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(cfg.rewards.scales)
        self.command_ranges = class_to_dict(cfg.commands.ranges)

        if hasattr(cfg, 'terrain') and cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
             cfg.terrain.curriculum = False
        self.max_episode_length_s = cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt) if self.dt > 0 else 0

        if hasattr(cfg, 'domain_rand'):
             push_interval_s = getattr(cfg.domain_rand, 'push_interval_s', 15)
             cfg.domain_rand.push_interval = np.ceil(push_interval_s / self.dt) if self.dt > 0 else 0


    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation. """
        print("  Creating ground plane...")
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        # Ensure terrain attribute exists before accessing friction/restitution
        if hasattr(self.cfg, 'terrain'):
            plane_params.static_friction = self.cfg.terrain.static_friction
            plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
            plane_params.restitution = self.cfg.terrain.restitution
        else:
             # Default values if terrain config is missing
             plane_params.static_friction = 1.0
             plane_params.dynamic_friction = 1.0
             plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, plane_params)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        # Check if terrain curriculum is active and uses custom origins
        if hasattr(self.cfg, 'terrain') and hasattr(self.cfg.terrain, 'curriculum') and self.cfg.terrain.curriculum:
             # This part depends on the specific terrain generation logic (e.g., from legged_gym)
             # Assuming a method _create_terrain exists and sets self.env_origins
             # If not using legged_gym's terrain, implement your logic here or use grid
             if hasattr(self, 'terrain_origins') and self.terrain_origins is not None:
                  self.env_origins = torch.tensor(self.terrain_origins, device=self.device, dtype=torch.float)
                  self.custom_origins = True
                  print("  Using custom terrain origins.")
                  return
             else:
                  print("  Terrain curriculum enabled but no custom origins found. Falling back to grid.")

        # Default grid creation
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows, device=self.device), torch.arange(num_cols, device=self.device)) # Create on correct device
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0. # Assuming flat ground if not terrain curriculum


    def _create_envs(self):
        """ Creates environments: loads robot asset, creates actors.
            Called by __init__ *before* prepare_sim.
        """
        print("--- _create_envs (LeggedRobot) ---")
        # Define G1_ROOT_DIR or ensure it's passed/globally available
        try:
             # More robust way to find root assuming standard project structure
             g1_module_path = os.path.dirname(inspect.getfile(g1.envs)) # Get g1/envs path
             G1_ROOT_DIR_LOCAL = os.path.dirname(g1_module_path) # Go up one level
             # print(f"  Determined G1_ROOT_DIR: {G1_ROOT_DIR_LOCAL}")
        except NameError:
             print("⚠️ WARNING: Cannot determine G1_ROOT_DIR automatically. Using empty string.")
             G1_ROOT_DIR_LOCAL = "" # Fallback

        try:
             asset_path = self.cfg.asset.file.format(G1_ROOT_DIR=G1_ROOT_DIR_LOCAL)
             asset_root = os.path.dirname(asset_path)
             asset_file = os.path.basename(asset_path)
        except AttributeError as e:
             raise AttributeError(f"Missing asset configuration in cfg.asset: {e}") from e
        except KeyError as e:
            raise KeyError(f"Missing placeholder in cfg.asset.file (e.g., {{G1_ROOT_DIR}}): {e}") from e


        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        print(f"  Loading robot asset: {asset_file} from {asset_root}")
        full_asset_path = os.path.join(asset_root, asset_file)
        if not os.path.exists(full_asset_path):
             print(f"❌❌❌ ERROR: Robot asset file not found at: {full_asset_path}")
             # Attempt fallback relative to current file? Unsafe.
             raise FileNotFoundError(f"Robot asset file not found: {full_asset_path}")

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if self.robot_asset is None: raise RuntimeError(f"Failed to load robot asset: {asset_path}")

        # Get DoF/Body counts *from the asset*
        asset_num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_dof = asset_num_dof # Store the asset's DoF count
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)

        # !!! Action Dimension Handling !!!
        # Check if the config's num_actions (potentially set by subclass) differs from asset's DoF count
        if self.cfg.env.num_actions != asset_num_dof:
             print(f"⚠️ LeggedRobot Info: cfg.env.num_actions ({self.cfg.env.num_actions}) differs from asset num_dof ({asset_num_dof}).")
             print(f"   >>> Using cfg.env.num_actions ({self.cfg.env.num_actions}) for self.num_actions. <<<")
             # Do NOT override cfg or self.num_actions here. Trust the value set previously.
             # self.cfg.env.num_actions = asset_num_dof # REMOVED
             # self.num_actions = asset_num_dof        # REMOVED
        else:
             # If they match, ensure self.num_actions is set correctly
             self.num_actions = self.cfg.env.num_actions
             print(f"  cfg.env.num_actions ({self.cfg.env.num_actions}) matches asset num_dof ({asset_num_dof}).")

        print(f"  Asset Info: Num DoF={self.num_dof}, Num Bodies={self.num_bodies}")
        print(f"  Environment using: Num Actions={self.num_actions}") # Print the final value being used

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        # Verify dof_names length matches asset_num_dof
        if len(self.dof_names) != asset_num_dof:
             print(f"❌ CRITICAL ERROR: Length of dof_names ({len(self.dof_names)}) does not match asset_num_dof ({asset_num_dof})!")
        self.num_dofs = len(self.dof_names) # Should match asset_num_dof

        # Use self.dof_names which has the full list (43) to find indices
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        # Base init state
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform() # Will be set per env

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        print(f"  Creating {self.num_envs} environments...")
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            if env_handle == gymapi.INVALID_HANDLE:
                 raise RuntimeError(f"Failed to create env {i}")
            self.envs.append(env_handle)

            # Set pose for this env's robot
            pos = self.env_origins[i].clone()
            # Add configured initial offset from self.base_init_state
            pos += self.base_init_state[:3] # Add position offset
            # Add random XY offset if not custom origins (like terrain)
            if not self.custom_origins:
                random_offset_xy = torch_rand_float(-0.1, 0.1, (1, 2), device=self.device)
                pos[:2] += random_offset_xy.squeeze(0)
            start_pose.p = gymapi.Vec3(*pos)
            start_pose.r = gymapi.Quat(*self.base_init_state[3:7]) # Use rot from base_init_state

            # Process props for this actor instance
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # Apply props to the asset (might affect subsequent actors if not careful)
            # Consider using set_actor_rigid_shape_properties if props need to differ significantly per env beyond friction
            # self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props) # Let's avoid modifying the asset globally

            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name,
                                                 i, # Collision group = env index
                                                 self.cfg.asset.self_collisions, 0) # Collision filter mask
            if actor_handle == gymapi.INVALID_HANDLE:
                 raise RuntimeError(f"Failed to create actor in env {i}")
            self.actor_handles.append(actor_handle)

            # Set Actor-specific shape properties AFTER creating the actor
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

            # Set DoF and Body properties for this actor instance
            # Process DoF props (gets limits first time, reuses after)
            # IMPORTANT: _process_dof_props might be overridden by subclasses (like G1BasicLocomotion)
            # to implement joint locking. It should operate on the full asset dof props.
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)

            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            # Check if props list is valid before processing
            if body_props:
                 body_props = self._process_rigid_body_props(body_props, i) # Randomize mass etc.
                 self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            else:
                 print(f"⚠️ Warning: Env {i} - got empty body_props list.")


        # --- Store body indices (after all actors are created) ---
        if self.actor_handles: # If actors were created
             first_env_handle = self.envs[0]
             first_actor_handle = self.actor_handles[0]

             # Use the full list of body names (self.body_names) from the asset
             # Ensure body_names is stored in self if needed later, or re-acquire
             # body_names = self.gym.get_asset_rigid_body_names(self.robot_asset) # Re-acquire if not stored

             self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
             for i, name in enumerate(feet_names):
                 handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
                 if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Foot body '{name}' not found.")
                 # Store the handle (which is the index)
                 self.feet_indices[i] = handle

             self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
             for i, name in enumerate(penalized_contact_names):
                 handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
                 if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Penalised body '{name}' not found.")
                 self.penalised_contact_indices[i] = handle

             self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
             for i, name in enumerate(termination_contact_names):
                 handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
                 if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Termination body '{name}' not found.")
                 self.termination_contact_indices[i] = handle
        else:
             print("❌ ERROR: No actors were created successfully.")
             # Initialize indices as empty tensors to prevent errors later
             self.feet_indices = torch.tensor([], dtype=torch.long, device=self.device)
             self.penalised_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)
             self.termination_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)

        print(f"--- _create_envs (LeggedRobot) Done ---")


    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities.
            Called *after* prepare_sim. Handles single-actor-per-env case.
            Needs override in multi-actor envs like G1KitchenNavigation.
        """
        print("--- _init_buffers (LeggedRobot) ---")
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        # Check if tensors were acquired successfully
        tensor_names = ["actor_root_state", "dof_state", "net_contact_forces", "rigid_body_state"]
        tensors = [actor_root_state, dof_state_tensor, net_contact_forces, rigid_body_state_tensor]
        for name, tensor in zip(tensor_names, tensors):
             if tensor is None:
                  print(f"❌ CRITICAL ERROR: Failed to acquire {name}_tensor from Isaac Gym.")
                  raise RuntimeError(f"Failed to acquire simulation state tensor: {name}")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Wrap tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)       # Shape [num_envs, 13]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)         # Shape [num_envs * asset_num_dof, 2]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor) # Shape [num_envs * num_bodies, 13]
        self.contact_forces_raw = gymtorch.wrap_tensor(net_contact_forces) # Shape [num_envs * num_bodies, 3]

        print(f"  Raw Tensor Shapes: root={self.root_states.shape}, dof={self.dof_state.shape}, rigid_body={self.rigid_body_states.shape}, contact={self.contact_forces_raw.shape}")

        # --- Create Views (using self.num_dof = asset_num_dof) ---
        # DoF state views (for the *full* state tensor)
        try:
            # Use self.num_dof (which is the asset's full DoF count) for reshaping the raw tensor
            dof_state_reshaped = self.dof_state.view(self.num_envs, self.num_dof, 2)
            self.dof_pos = dof_state_reshaped[..., 0] # Full DoF positions [num_envs, 43]
            self.dof_vel = dof_state_reshaped[..., 1] # Full DoF velocities [num_envs, 43]
        except RuntimeError as e:
            print(f"❌ ERROR reshaping dof_state {self.dof_state.shape} to ({self.num_envs}, {self.num_dof}, 2): {e}")
            raise

        # Rigid body state view
        try:
            self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        except RuntimeError as e:
            print(f"❌ ERROR reshaping rigid_body_states {self.rigid_body_states.shape} to ({self.num_envs}, {self.num_bodies}, 13): {e}")
            raise

        # Contact forces view
        try:
            self.contact_forces = self.contact_forces_raw.view(self.num_envs, self.num_bodies, 3)
            print(f"  Reshaped contact_forces to: {self.contact_forces.shape}")
        except RuntimeError as e:
            print(f"❌ ERROR reshaping contact_forces_raw {self.contact_forces_raw.shape} to ({self.num_envs}, {self.num_bodies}, 3): {e}")
            raise

        # Robot state views
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, 0:3]
        self.rpy = torch.zeros_like(self.base_pos)

        # Initialize other buffers
        self.common_step_counter = 0
        self.extras = {}

        # --- Initialize noise_scale_vec AFTER obs_buf is allocated by BaseTask ---
        # BaseTask allocates obs_buf based on self.cfg.env.num_observations
        # which might have been overridden by subclasses for nested curriculum.
        # We call _get_noise_scale_vec here to ensure it uses the *final* dimension.
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        # ---------------------------------------------------------------------

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # --- Action related buffers (sized based on self.num_actions, which reflects the *current* active DoFs) ---
        print(f"  Initializing action-related buffers with num_actions = {self.num_actions}")
        # IMPORTANT: self.torques buffer needs to be the size for the *full* DOF set (43)
        # because set_dof_actuation_force_tensor expects a tensor for all DOFs in the sim envs.
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # Size [num_envs, 43]
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # Size [num_envs, current_active]
        self.last_actions = torch.zeros_like(self.actions) # Size [num_envs, current_active]
        # -------------------------------------------------------------------------------------------------------

        # --- PD gains, default pos, torque limits (sized based on the full asset DoF count self.num_dof=43) ---
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # Size [43]
        self.d_gains = torch.zeros_like(self.p_gains) # Size [43]
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # Size [43]
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) # Size [43]

        # Get asset properties again if needed, or use stored self.robot_asset
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)

        for i in range(self.num_dof): # Iterate through all 43 DoFs
            name = self.dof_names[i] # Use the full name list
            angle = self.cfg.init_state.default_joint_angles.get(name, 0.0)
            self.default_dof_pos[i] = angle
            # Ensure effort value is correctly extracted
            effort_val = dof_props_asset["effort"][i]
            self.torque_limits[i] = float(effort_val.item()) if hasattr(effort_val, 'item') else float(effort_val) # Handle potential scalar tensor

            found = False
            # Use self.cfg.control which should contain stiffness/damping for all controllable joints
            for dof_name_key in self.cfg.control.stiffness.keys():
                if dof_name_key in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name_key]
                    self.d_gains[i] = self.cfg.control.damping.get(dof_name_key, 0.0)
                    found = True
                    break
            if not found:
                self.p_gains[i] = 0.; self.d_gains[i] = 0.
                # Don't warn if nested curriculum, as locked joints intentionally have 0 gains here
                # Only warn if not nested and gains are missing for potentially active joints
                is_nested = getattr(self.cfg, 'nested_locomotion_curriculum', False)
                if not is_nested and self.cfg.control.control_type in ["P", "V"]:
                     print(f"  PD gain '{name}' not defined in config.")

        # Unsqueeze default_dof_pos for broadcasting: [1, 43]
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        # -----------------------------------------------------------------------------------------

        # Other buffers
        # Use self.num_dof for last_dof_vel as it stores the full state
        self.last_dof_vel = torch.zeros_like(self.dof_vel) # Size [num_envs, 43]
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)

        # Feet buffers
        if self.feet_indices.numel() > 0:
            self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
            self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        else:
            self.feet_air_time = None
            self.last_contacts = None

        # Calculated velocities
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Initialize foot state tracking (uses feet_indices which are correct)
        self._init_foot()
        print("--- _init_buffers (LeggedRobot) Done ---")

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions based on non-zero scales from self.reward_scales. """
        print("  Preparing reward functions...")
        # Ensure self.reward_scales is populated (should be done in _parse_cfg)
        if not hasattr(self, 'reward_scales') or not self.reward_scales:
            print("⚠️ _prepare_reward_function: self.reward_scales is empty or missing. No rewards will be computed.")
            self.reward_functions = []
            self.reward_names = []
            self.episode_sums = {}
            return

        # Get scales from the instance attribute (potentially modified by scheduler)
        current_reward_scales = self.reward_scales

        # Prepare list of functions based on non-zero scales
        self.reward_functions = []
        self.reward_names = []
        skipped_rewards = []
        print(f"    Available scales in self.reward_scales: {list(current_reward_scales.keys())}")

        # --- Iterate through ALL possible reward functions defined in the class ---
        # --- This makes it independent of the scales dict containing all names ---
        for func_name in dir(self):
            if func_name.startswith("_reward_"):
                 reward_name = func_name[len("_reward_"):]
                 scale = current_reward_scales.get(reward_name, 0.0) # Get scale, default to 0 if not in dict

                 if reward_name == "termination": continue # Handled separately

                 if scale != 0.0:
                     func = getattr(self, func_name)
                     if callable(func):
                         self.reward_names.append(reward_name)
                         self.reward_functions.append(func)
                         # Multiply scale by dt here for efficiency
                         current_reward_scales[reward_name] = scale * self.dt # Update the dict value
                     else:
                         skipped_rewards.append(f"{reward_name} (attribute exists but not callable)")
                 else:
                     skipped_rewards.append(f"{reward_name} (scale is 0 or missing)")

        # Reward episode sums (only for active rewards)
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_names}

        # Special handling for termination reward sum if used
        termination_scale = current_reward_scales.get("termination", 0.0)
        if termination_scale != 0.0:
            # Check if _reward_termination exists
            if not hasattr(self, '_reward_termination') or not callable(getattr(self, '_reward_termination')):
                 print("⚠️ Warning: Termination reward scale is non-zero, but _reward_termination function is missing or not callable.")
            else:
                 # Add termination to names for sum tracking if function exists
                 if "termination" not in self.reward_names: self.reward_names.append("termination")
                 self.episode_sums["termination"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                 # Termination reward scale is applied once, no dt multiplication needed in current_reward_scales

        print(f"    Active reward functions prepared: {self.reward_names}")
        if skipped_rewards: print(f"    Skipped/Inactive/Missing rewards: {skipped_rewards}")


    # --- Step, Post Physics Step, Check Termination, Reset Idx ---
    # (Keep existing implementations, they should use instance vars correctly now)
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step() """
        # Action dimension check: actions should match self.num_actions (current active DoFs)
        if actions.shape[1] != self.num_actions:
             print(f"❌ ERROR in step: Received actions shape {actions.shape} does not match current env num_actions {self.num_actions}")
             # Option 1: Pad with zeros if actions are fewer than expected (less likely)
             if actions.shape[1] < self.num_actions:
                  print(f"   Padding actions with zeros.")
                  padded_actions = torch.zeros((actions.shape[0], self.num_actions), device=self.device)
                  padded_actions[:, :actions.shape[1]] = actions
                  actions = padded_actions
             # Option 2: Truncate if actions are more than expected (could happen if policy output is fixed)
             elif actions.shape[1] > self.num_actions:
                  print(f"   Truncating actions to {self.num_actions} dimensions.")
                  actions = actions[:, :self.num_actions]
             # Option 3: Raise error (safest)
             # raise ValueError(f"Action shape mismatch: received {actions.shape}, expected [*, {self.num_actions}]")


        clip_actions = self.cfg.normalization.clip_actions
        # self.actions buffer has shape [num_envs, self.num_actions]
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # Compute torques for *all* robot DoFs (size 43)
            # _compute_torques might be overridden (e.g., G1BasicLocomotion)
            computed_torques = self._compute_torques(self.actions) # Pass active actions

            # Ensure computed_torques has the shape [num_envs, self.num_dof] (e.g., [*, 43])
            if computed_torques.shape != (self.num_envs, self.num_dof):
                  print(f"❌ ERROR in step: _compute_torques returned incorrect shape {computed_torques.shape}. Expected ({self.num_envs}, {self.num_dof}).")
                  # Fallback: Set zero torques to avoid crash
                  self.torques.zero_()
            else:
                  self.torques[:] = computed_torques # Fill the full torque buffer

            # Set actuation forces using the full buffer
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques)) # Pass [num_envs, 43] tensor

            self.gym.simulate(self.sim)
            if self.device == 'cpu': self.gym.fetch_results(self.sim, True) # Fetch results after simulate

            # Refresh states needed for next loop iteration or post_physics_step
            self.gym.refresh_dof_state_tensor(self.sim)
            # Views self.dof_pos/vel are automatically updated if using the reshaped tensor

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        # obs_buf has shape [num_envs, self.num_observations] (current active obs dim)
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            # privileged_obs_buf has shape [num_envs, self.num_privileged_obs] (current active priv obs dim)
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # Ensure extras dict exists
        if not hasattr(self, 'extras'): self.extras = {}

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras


    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations.
            Handles common logic like command resampling and heading update.
            Subclasses can override this and call super().
        """
        # Resample commands based on time
        if hasattr(self, 'episode_length_buf') and hasattr(self.cfg, 'commands') and hasattr(self.cfg.commands, 'resampling_time') and self.dt > 0:
            resampling_interval_steps = getattr(self.cfg.commands, 'resampling_interval_steps', None)
            if resampling_interval_steps is None:  # Calculate if not pre-calculated
                resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt)
                self.cfg.commands.resampling_interval_steps = resampling_interval_steps # Cache it

            if resampling_interval_steps > 0: # Avoid modulo by zero
                env_ids = (self.episode_length_buf % resampling_interval_steps == 0).nonzero(as_tuple=False).flatten()
                if len(env_ids) > 0:
                    self._resample_commands(env_ids)
            # else: # Avoid warning if resampling time is intentionally 0 or less
            #     if self.cfg.commands.resampling_time > 0:
            #          print("⚠️ Warning: resampling_interval_steps is zero or negative. Skipping command resampling.")

        # Compute heading command if enabled
        if getattr(self.cfg.commands, 'heading_command', False):
            if hasattr(self, 'base_quat') and hasattr(self, 'forward_vec') and hasattr(self, 'commands') and \
               self.commands.shape[1] >= 4: # Need column 3 for heading target
                try:
                    forward = quat_apply(self.base_quat, self.forward_vec)
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    # Ensure heading target (self.commands[:, 3]) is valid
                    heading_target = self.commands[:, 3]
                    heading_error = wrap_to_pi(heading_target - heading)
                    # Apply gain (e.g., 0.5) and clip
                    yaw_command = torch.clip(0.5 * heading_error, -1., 1.)
                    # Update the yaw command in column 2
                    self.commands[:, 2] = yaw_command
                except Exception as e:
                    print(f"❌ Error computing heading command in LeggedRobot callback: {e}")
                    # Fallback: set yaw command to 0
                    self.commands[:, 2] = 0.0
            # else: # Avoid excessive warnings
            #      pass
            #      # print("⚠️ Cannot compute heading command: required attributes missing or commands shape incorrect.")


    def post_physics_step(self):
        """ check terminations, compute observations and rewards """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim) # Refreshed in step loop
        # self.gym.refresh_rigid_body_state_tensor(self.sim) # Refreshed in update_feet_state if needed

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities (using robot-specific states where applicable)
        if hasattr(self, 'robot_root_states'): # Use robot-specific view if exists (like in G1KitchenNav)
             root_states_to_use = self.robot_root_states
        else: # Fallback to global root_states (assuming single actor per env)
             root_states_to_use = self.root_states

        # Ensure buffers exist before writing to them
        if not hasattr(self, 'base_pos'): self.base_pos = torch.zeros_like(root_states_to_use[:, 0:3])
        if not hasattr(self, 'base_quat'): self.base_quat = torch.zeros_like(root_states_to_use[:, 3:7])
        if not hasattr(self, 'rpy'): self.rpy = torch.zeros_like(root_states_to_use[:, 0:3])
        if not hasattr(self, 'base_lin_vel'): self.base_lin_vel = torch.zeros_like(root_states_to_use[:, 7:10])
        if not hasattr(self, 'base_ang_vel'): self.base_ang_vel = torch.zeros_like(root_states_to_use[:, 10:13])
        if not hasattr(self, 'projected_gravity'): self.projected_gravity = torch.zeros_like(self.gravity_vec)


        self.base_pos[:] = root_states_to_use[:, 0:3]
        self.base_quat[:] = root_states_to_use[:, 3:7]
        roll, pitch, yaw = get_euler_xyz(self.base_quat)
        self.rpy[:, 0] = roll
        self.rpy[:, 1] = pitch
        self.rpy[:, 2] = yaw
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, root_states_to_use[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, root_states_to_use[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Update DoF pos/vel if using manual update fallback (less relevant now)
        # if hasattr(self, 'manual_dof_update') and self.manual_dof_update: pass

        # Callbacks (phase calculation, command resampling, heading calc)
        # Needs to be implemented or overridden in subclasses like G1CurriculumBase
        # Call explicitly here or ensure subclasses call super()._post_physics_step_callback()
        self._post_physics_step_callback() # Call the base callback

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0: # Only call reset if needed
             self.reset_idx(env_ids)

        # Push robots
        if hasattr(self.cfg, 'domain_rand') and self.cfg.domain_rand.push_robots:
             self._push_robots()

        # Compute observations *after* potential resets and state updates
        self.compute_observations() # Should be implemented by subclasses

        # Store last states
        if hasattr(self, 'last_actions'): self.last_actions[:] = self.actions[:]
        if hasattr(self, 'last_dof_vel'): self.last_dof_vel[:] = self.dof_vel[:] # Store full DoF velocity
        if hasattr(self, 'last_root_vel'): self.last_root_vel[:] = root_states_to_use[:, 7:13]


    def check_termination(self):
        """ Check if environments need to be reset """
        if self.contact_forces is None:
             print("⚠️ check_termination: Contact forces buffer is None. Cannot check termination.")
             self.reset_buf.zero_()
             self.time_out_buf = self.episode_length_buf > self.max_episode_length
             self.reset_buf |= self.time_out_buf
             return

        # termination_contact_indices are indices within the robot's bodies (0 to num_bodies-1)
        # contact_forces view is [num_envs, num_bodies, 3]
        try:
             # Ensure indices are valid
             if torch.any(self.termination_contact_indices >= self.num_bodies):
                  print(f"❌ ERROR check_termination: termination_contact_indices ({self.termination_contact_indices.max().item()}) out of bounds for num_bodies ({self.num_bodies})")
                  contact_termination = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
             else:
                  contact_termination = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

             self.reset_buf = contact_termination
        except IndexError as e:
             print(f"❌ ERROR in check_termination accessing contact_forces: {e}. Indices={self.termination_contact_indices}, Shape={self.contact_forces.shape}.")
             self.reset_buf.zero_() # Fail safe

        # Orientation check
        orientation_limit_roll = getattr(self.cfg.termination, 'orientation_limit_roll', 0.8)
        orientation_limit_pitch = getattr(self.cfg.termination, 'orientation_limit_pitch', 1.0)
        # Ensure rpy buffer exists
        if hasattr(self, 'rpy'):
             self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:, 0]) > orientation_limit_roll, torch.abs(self.rpy[:, 1]) > orientation_limit_pitch)

        # Timeout check
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf


    def reset_idx(self, env_ids):
        """ Reset some environments. """
        if len(env_ids) == 0: return

        # Reset robot states (calls methods that use self.dof_pos, self.root_states etc.)
        self._reset_dofs(env_ids) # Resets DoFs for the *asset*
        self._reset_root_states(env_ids) # Resets robot root state

        # Resample commands for reset envs
        self._resample_commands(env_ids)

        # Reset buffers (use self.num_actions for action buffers, self.num_dof for dof buffers)
        if hasattr(self, 'last_actions'): self.last_actions[env_ids] = 0.
        if hasattr(self, 'last_dof_vel'): self.last_dof_vel[env_ids] = 0. # Reset full DoF vel buffer
        if self.feet_air_time is not None: self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1 # Mark for reset in the next step logic if using delayed reset

        # Fill extras and log episode sums
        if not hasattr(self, 'extras'): self.extras = {} # Ensure extras exists
        self.extras["episode"] = {}
        # Use self.reward_names which contains only active rewards
        for key in self.reward_names:
             sum_tensor = self.episode_sums.get(key)
             if sum_tensor is not None and sum_tensor.shape[0] == self.num_envs:
                 # Avoid division by zero for episode length
                 ep_len_s = self.episode_length_buf[env_ids].float() * self.dt
                 mean_ep_len_s = torch.mean(ep_len_s)
                 if mean_ep_len_s > 1e-6: # Only divide if episode length is meaningful
                     self.extras["episode"]['rew_' + key] = torch.mean(sum_tensor[env_ids] / mean_ep_len_s).item()
                 else: # Handle zero length episodes (e.g., immediate termination)
                      self.extras["episode"]['rew_' + key] = 0.0
                 # Reset sum for this env
                 self.episode_sums[key][env_ids] = 0.
             # else: # Avoid verbose warnings if sum tensor is missing (already warned in prep)
             #      # print(f"⚠️ reset_idx: Skipping episode sum logging for '{key}' due to missing or mismatched tensor.")
             #      pass

        # Command curriculum logging (ensure commands attribute exists)
        if hasattr(self.cfg, 'commands') and getattr(self.cfg.commands, 'curriculum', False):
             # Ensure command_ranges exists and has the expected key
             if hasattr(self, 'command_ranges') and "lin_vel_x" in self.command_ranges:
                  self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
             # else: print("⚠️ reset_idx: Cannot log max_command_x, command_ranges missing or key invalid.")

        # Send timeout info
        if self.cfg.env.send_timeouts:
            # Ensure time_out_buf exists and has correct shape
            if hasattr(self, 'time_out_buf') and self.time_out_buf.shape == (self.num_envs,):
                 self.extras["time_outs"] = self.time_out_buf.clone() # Send a copy
            # else: print("⚠️ reset_idx: Cannot send time_outs, buffer missing or shape invalid.")


    def compute_reward(self):
        """ Compute rewards based on active reward functions and scales. """
        self.rew_buf[:] = 0.
        # Use self.reward_names generated in _prepare_reward_function
        active_reward_names = self.reward_names

        # Create a temporary dict to store individual reward values for debugging
        # current_rewards_debug = {}

        for name in active_reward_names:
             if name == "termination": continue # Skip termination here
             try:
                 # Find the reward function (already checked for existence in prepare)
                 reward_func = getattr(self, '_reward_' + name)
                 rew = reward_func() # Calculate raw reward
                 scale = self.reward_scales.get(name, 0.0) # Get the dt-adjusted scale

                 # Shape check before applying scale
                 if rew.shape == (self.num_envs,):
                      scaled_rew = rew * scale
                      self.rew_buf += scaled_rew
                      # Ensure episode_sums exists before adding
                      if name in self.episode_sums: self.episode_sums[name] += scaled_rew
                      # current_rewards_debug[name] = scaled_rew.mean().item() # Store mean for debug
                 else:
                      print(f"⚠️ compute_reward: Shape mismatch for reward '{name}'. Expected ({self.num_envs},), got {rew.shape}. Skipping.")
             except Exception as e:
                  print(f"❌ Error computing reward '{name}': {e}")
                  # import traceback; traceback.print_exc() # Uncomment for detailed trace

        # Apply clipping
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # Optional: Clip based on reward range if defined
        if hasattr(self.cfg.rewards, 'clip_reward'):
             clip_val = self.cfg.rewards.clip_reward
             self.rew_buf[:] = torch.clip(self.rew_buf[:], -clip_val, clip_val)

        # Add termination reward AFTER clipping other rewards
        termination_scale = self.reward_scales.get("termination", 0.0)
        if termination_scale != 0.0 and hasattr(self, '_reward_termination'):
             try:
                 rew = self._reward_termination() # Raw termination penalty/reward
                 if rew.shape == (self.num_envs,):
                     scaled_rew = rew * termination_scale # Apply scale
                     self.rew_buf += scaled_rew
                     if "termination" in self.episode_sums: self.episode_sums["termination"] += scaled_rew
                     # current_rewards_debug["termination"] = scaled_rew.mean().item() # Store mean for debug
                 else:
                      print(f"⚠️ compute_reward: Shape mismatch for termination reward. Expected ({self.num_envs},), got {rew.shape}.")
             except Exception as e:
                  print(f"❌ Error computing termination reward: {e}")

        # Print mean rewards per component for debugging (optional)
        # if self.common_step_counter % 1000 == 0: # Print every 1000 steps
        #      print(f"--- Step {self.common_step_counter} Rewards ---")
        #      for name, mean_val in current_rewards_debug.items():
        #           print(f"  {name}: {mean_val:.4f}")
        #      print(f"  TotalMean: {self.rew_buf.mean().item():.4f}")


    def compute_observations(self):
        """ Computes observations. Needs override in subclasses. """
        # Base implementation just zeros the buffer
        # Subclasses like G1CurriculumBase should implement the actual observation calculation
        self.obs_buf.zero_()
        if self.privileged_obs_buf is not None:
             self.privileged_obs_buf.zero_()
        # print("⚠️ LeggedRobot.compute_observations called - Subclass should override this.")


    def set_camera(self, position, lookat):
        """ Set camera position and direction. """
        if self.viewer is None: return # Do nothing if viewer doesn't exist
        try:
            cam_pos = gymapi.Vec3(*position)
            cam_target = gymapi.Vec3(*lookat)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        except Exception as e:
             print(f"Error setting camera: {e}")


    def _process_rigid_shape_props(self, props, env_id):
        # Randomize friction
        if self.cfg.domain_rand.randomize_friction:
            if env_id == 0: # Calculate friction coeffs only once
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64 # Or read from config if needed
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs,), device=self.device)

                # --- FIX: Adjust shape argument for torch_rand_float ---
                # Create a 2D tensor [num_buckets, 1] first
                friction_buckets_2d = torch_rand_float(friction_range[0], friction_range[1], (num_buckets, 1), device=self.device)
                # Squeeze the second dimension to get a 1D tensor [num_buckets]
                friction_buckets = friction_buckets_2d.squeeze(1)
                # --------------------------------------------------------

                self.friction_coeffs = friction_buckets[bucket_ids] # Shape [num_envs]

            current_friction = self.friction_coeffs[env_id].item()
            if props:
                 for p in props:
                     p.friction = current_friction
                     # Also set restitution if configured
                     if hasattr(self.cfg, 'terrain'):
                          p.restitution = self.cfg.terrain.restitution
            # else: print(f"⚠️ Env {env_id}: _process_rigid_shape_props received None or empty props.") # Optional warning
        return props

    def _process_dof_props(self, props, env_id):
        # (Keep implementation as previously corrected)
        if env_id == 0:
            if not hasattr(self, 'dof_pos_limits') or self.dof_pos_limits.shape[0] != self.num_dof:
                 self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
                 self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                 for i in range(self.num_dof):
                     self.dof_pos_limits[i, 0] = props["lower"][i].item()
                     self.dof_pos_limits[i, 1] = props["upper"][i].item()
                     self.dof_vel_limits[i] = props["velocity"][i].item()
                     m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                     r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                     if abs(r) < 1e-6: r = 1.0
                     soft_limit_ratio = getattr(self.cfg.rewards, 'soft_dof_pos_limit', 1.0)
                     self.dof_pos_limits[i, 0] = m - 0.5 * r * soft_limit_ratio
                     self.dof_pos_limits[i, 1] = m + 0.5 * r * soft_limit_ratio

        for i in range(self.num_dof):
             name = self.dof_names[i]
             found = False
             for gain_key, stiffness_val in self.cfg.control.stiffness.items():
                 if gain_key in name:
                     props["stiffness"][i] = stiffness_val
                     props["damping"][i] = self.cfg.control.damping.get(gain_key, 0.0)
                     found = True; break
             if not found: props["stiffness"][i] = 0.0; props["damping"][i] = 0.0
        return props

    def _process_rigid_body_props(self, props, env_id):
        # (Keep implementation as previously corrected)
        if not props: return props
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            if len(props) > 0:
                 added_mass = np.random.uniform(rng[0], rng[1])
                 props[0].mass += added_mass
            # else: print("⚠️ _process_rigid_body_props: Cannot apply base mass randomization, props list is empty.")
        return props


    # --- _resample_commands, _compute_torques ---
    # _resample_commands should be fine.
    # _compute_torques needs careful checking in base vs subclass.
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments """
        if len(env_ids) == 0: return
        # Use self.command_ranges directly
        cmd_ranges = self.command_ranges # Local ref
        commands_buf = self.commands # Local ref

        commands_buf[env_ids, 0] = torch_rand_float(cmd_ranges["lin_vel_x"][0], cmd_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        commands_buf[env_ids, 1] = torch_rand_float(cmd_ranges["lin_vel_y"][0], cmd_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # Check if heading command is enabled and buffer has enough columns
        if self.cfg.commands.heading_command and commands_buf.shape[1] >= 4:
             commands_buf[env_ids, 3] = torch_rand_float(cmd_ranges["heading"][0], cmd_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
             # Yaw command (col 2) will be computed later
        elif commands_buf.shape[1] >= 3: # If not heading or not enough columns, sample yaw directly
             commands_buf[env_ids, 2] = torch_rand_float(cmd_ranges["ang_vel_yaw"][0], cmd_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # else: print("⚠️ _resample_commands: Command buffer has < 3 columns.")

        # set small commands to zero
        commands_buf[env_ids, :2] *= (torch.norm(commands_buf[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques for the robot based on *active* DoFs.
            This base implementation assumes all DoFs are active (self.num_actions == self.num_dof).
            Subclasses like G1BasicLocomotion override this for nested curriculum.
        """
        # actions shape: [num_envs, self.num_actions]
        # Base case: self.num_actions == self.num_dof (43)
        if actions.shape[1] != self.num_dof:
             print(f"❌ ERROR _compute_torques (Base): actions shape {actions.shape} mismatch num_dof {self.num_dof}")
             return torch.zeros(self.num_envs, self.num_dof, device=self.device) # Return zero torques

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        # Use full DoF state buffers
        dof_pos = self.dof_pos # [num_envs, 43]
        dof_vel = self.dof_vel # [num_envs, 43]

        if control_type=="P":
            # default_dof_pos is [1, 43]
            if self.default_dof_pos.shape[1] != self.num_dof:
                 print(f"❌ ERROR _compute_torques (Base): default_dof_pos shape mismatch")
                 return torch.zeros(self.num_envs, self.num_dof, device=self.device)
            # p_gains/d_gains are [43]
            torques = self.p_gains * (actions_scaled + self.default_dof_pos - dof_pos) - self.d_gains * dof_vel
        elif control_type=="V":
            if self.last_dof_vel.shape != dof_vel.shape: self.last_dof_vel = torch.zeros_like(dof_vel)
            dt = self.dt
            if dt <= 0:
                dt = 1e-5
            torques = self.p_gains * (actions_scaled - dof_vel) - self.d_gains * (dof_vel - self.last_dof_vel) / dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # Clip using full torque limits [43]
        return torch.clip(torques, -self.torque_limits, self.torque_limits) # Shape [num_envs, 43]


    # --- _reset_dofs, _reset_root_states ---
    # Need modification for multi-actor environments in subclasses.
    # Base implementation assumes tensors map directly to the single robot per env.
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected robots.
            Operates on the *full* DOF state tensor.
        """
        if len(env_ids) == 0: return
        # Generate random positions/velocities for all asset DoFs (43)
        num_reset_envs = len(env_ids)
        new_dof_pos = self.default_dof_pos.repeat(num_reset_envs, 1) * torch_rand_float(0.5, 1.5, (num_reset_envs, self.num_dof), device=self.device)
        new_dof_vel = torch.zeros((num_reset_envs, self.num_dof), device=self.device)

        # --- Update the global dof_state tensor ---
        # Use indexed assignment which is safer and works for sparse updates
        # self.dof_state has shape [num_envs * num_dof, 2]
        # Need to map env_ids to the flat tensor indices
        # global_dof_indices = (env_ids.unsqueeze(1) * self.num_dof + torch.arange(self.num_dof, device=self.device).unsqueeze(0)).view(-1) # Calculate flat indices

        # Directly update the view if possible (more efficient)
        try:
             dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
             dof_state_view[env_ids, :, 0] = new_dof_pos
             dof_state_view[env_ids, :, 1] = new_dof_vel
        except RuntimeError as e:
             print(f"⚠️ _reset_dofs: View update failed ({e}). Using less efficient indexed update.")
             # Fallback: Indexed update (careful with index calculation)
             indices_pos = (env_ids * self.num_dof).unsqueeze(1) + torch.arange(self.num_dof, device=self.device).unsqueeze(0)
             indices_vel = indices_pos # Same indices, different column in dof_state
             flat_indices_pos = indices_pos.view(-1)
             flat_indices_vel = indices_vel.view(-1) # Should be the same

             # Ensure indices are within bounds
             if flat_indices_pos.max() >= self.dof_state.shape[0] or flat_indices_vel.max() >= self.dof_state.shape[0]:
                  print(f"❌ ERROR _reset_dofs: Calculated indices out of bounds for dof_state.")
             else:
                  self.dof_state[flat_indices_pos, 0] = new_dof_pos.view(-1)
                  self.dof_state[flat_indices_vel, 1] = new_dof_vel.view(-1)


        # Update internal buffers (full DoF state)
        self.dof_pos[env_ids] = new_dof_pos
        self.dof_vel[env_ids] = new_dof_vel

        # Call gym function to apply changes
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # set_dof_state_tensor_indexed requires global *DOF* indices, not env indices
        # It's usually better to just set the whole tensor if many envs are reset
        # Or use set_actor_dof_states per environment (slower)
        # Simplest for base class: update the full tensor view and set the full tensor
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))


    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected robots. """
        if len(env_ids) == 0: return
        num_reset_envs = len(env_ids)
        # Generate random velocities
        random_vels = torch_rand_float(-0.5, 0.5, (num_reset_envs, 6), device=self.device)

        # Apply to root_states buffer
        # Start with default init state and add env origin
        new_root_states = self.base_init_state.repeat(num_reset_envs, 1)
        new_root_states[:, :3] += self.env_origins[env_ids]
        # Add random XY offset if not custom origins
        if not self.custom_origins:
             new_root_states[:, :2] += torch_rand_float(-0.1, 0.1, (num_reset_envs, 2), device=self.device)
        new_root_states[:, 7:13] = random_vels # Set random velocities

        # Update the main root_states buffer
        self.root_states[env_ids] = new_root_states

        # Set tensor in sim using indexed update (needs actor indices)
        # Assuming actor index == env index for base LeggedRobot
        actor_indices = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states), # Pass the full tensor
            gymtorch.unwrap_tensor(actor_indices), # Pass the indices of actors to reset
            len(actor_indices)
        )

    # --- _push_robots ---
    def _push_robots(self):
        """ Random pushes the robots. Base implementation assumes single actor per env. """
        # Calculate push interval steps safely
        push_interval_steps = getattr(self.cfg.domain_rand, 'push_interval', 0)
        if push_interval_steps <= 0: return # Skip if interval is not positive

        # Determine which envs to push
        push_env_ids = (self.episode_length_buf % int(push_interval_steps) == 0).nonzero(as_tuple=False).flatten()

        if len(push_env_ids) == 0: return

        max_vel = self.cfg.domain_rand.max_push_vel_xy
        random_vel_xy = torch_rand_float(-max_vel, max_vel, (len(push_env_ids), 2), device=self.device)

        # Apply push to root_states buffer
        self.root_states[push_env_ids, 7:9] += random_vel_xy # Add to current lin vel x/y

        # Set tensor in sim using indexed update
        actor_indices = push_env_ids.to(dtype=torch.int32) # Assuming actor index == env index
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices)
        )

    # --- update_command_curriculum, _get_noise_scale_vec, reward functions ---
    # These are generally okay but _get_noise_scale_vec needs attention.

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands """
        # Check prerequisites
        if len(env_ids) == 0: return
        if "tracking_lin_vel" not in self.reward_scales or "tracking_lin_vel" not in self.episode_sums: return
        if self.max_episode_length_s <= 0: return
        if not hasattr(self.cfg.commands, 'curriculum') or not self.cfg.commands.curriculum: return

        # Calculate mean reward safely
        sum_tensor = self.episode_sums["tracking_lin_vel"][env_ids]
        ep_len_s = self.episode_length_buf[env_ids].float() * self.dt
        # Avoid division by zero/small numbers
        mean_rew = torch.mean(sum_tensor[ep_len_s > 1e-6] / ep_len_s[ep_len_s > 1e-6])
        if torch.isnan(mean_rew) or len(ep_len_s[ep_len_s > 1e-6]) == 0: return # No valid episodes

        # Get original scale threshold (before dt multiplication)
        original_scale = self.reward_scales["tracking_lin_vel"] / self.dt if self.dt > 0 else 0
        target_rew_threshold = 0.8 * original_scale

        if mean_rew > target_rew_threshold:
            max_curr = self.cfg.commands.max_curriculum
            step = 0.1 # Increment step
            # Update lin_vel_x range
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - step, -max_curr, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + step, 0., max_curr)
            # Optionally update other ranges (y, yaw)
            # self.command_ranges["lin_vel_y"][0] = np.clip(...)
            # self.command_ranges["lin_vel_y"][1] = np.clip(...)
            # self.command_ranges["ang_vel_yaw"][0] = np.clip(...)
            # self.command_ranges["ang_vel_yaw"][1] = np.clip(...)


    def _get_noise_scale_vec(self, cfg):
        """ Sets the noise scale vector using the *current* self.num_observations.
            Called from _init_buffers *after* BaseTask has initialized obs_buf.
        """
        # Use the actual num_observations of the environment instance
        expected_obs_dim = self.num_observations
        noise_vec = torch.zeros(expected_obs_dim, device=self.device)

        self.add_noise = getattr(cfg.noise, 'add_noise', True) # Default to True
        if not self.add_noise:
             print("  Noise disabled.")
             return noise_vec

        noise_scales = cfg.noise.noise_scales
        noise_level = cfg.noise.noise_level
        # Ensure obs_scales attribute exists
        obs_scales = getattr(cfg.normalization, 'obs_scales', None)
        if obs_scales is None:
             print("⚠️ _get_noise_scale_vec: cfg.normalization.obs_scales not found. Using scale 1.0 for noise.")
             # Create a dummy obs_scales with default 1.0 to avoid errors
             obs_scales = DotDict({k: 1.0 for k in dir(noise_scales) if not k.startswith('_')})

        print(f"  Generating noise vector for obs_dim={expected_obs_dim}")

        # --- This part MUST match the structure of compute_observations in the *final* subclass ---
        # --- It's better implemented in the subclass (G1CurriculumBase) ---
        # --- Base implementation provides a placeholder or minimal version ---
        current_idx = 0
        # Example for a minimal base observation (adjust based on LeggedRobot obs structure if needed)
        num_act = self.num_actions # Use current action dim

        # Ang Vel (3)
        noise_vec[current_idx:current_idx+3] = getattr(noise_scales, 'ang_vel', 0.0) * noise_level * getattr(obs_scales, 'ang_vel', 1.0); current_idx += 3
        # Gravity (3)
        noise_vec[current_idx:current_idx+3] = getattr(noise_scales, 'gravity', 0.0) * noise_level * 1.0; current_idx += 3 # No obs scale for gravity
        # Commands (3) - Usually no noise
        noise_vec[current_idx:current_idx+3] = 0.0; current_idx += 3
        # DoF Pos (num_act)
        if current_idx + num_act <= expected_obs_dim:
            noise_vec[current_idx:current_idx+num_act] = getattr(noise_scales, 'dof_pos', 0.0) * noise_level * getattr(obs_scales, 'dof_pos', 1.0); current_idx += num_act
        # DoF Vel (num_act)
        if current_idx + num_act <= expected_obs_dim:
            noise_vec[current_idx:current_idx+num_act] = getattr(noise_scales, 'dof_vel', 0.0) * noise_level * getattr(obs_scales, 'dof_vel', 1.0); current_idx += num_act
        # Actions (num_act) - Usually no noise
        if current_idx + num_act <= expected_obs_dim:
            noise_vec[current_idx:current_idx+num_act] = 0.0; current_idx += num_act
        # Phase (2) - Usually no noise
        if current_idx + 2 <= expected_obs_dim:
            noise_vec[current_idx:current_idx+2] = 0.0; current_idx += 2

        # Final check
        if current_idx != expected_obs_dim:
            print(f"❌ LeggedRobot ERROR: Final noise vector index ({current_idx}) != expected obs dim ({expected_obs_dim}). Check _get_noise_scale_vec logic!")
            # Pad or truncate noise_vec for safety, though it indicates a bug
            if current_idx < expected_obs_dim: noise_vec = torch.cat((noise_vec[:current_idx], torch.zeros(expected_obs_dim - current_idx, device=self.device)))
            else: noise_vec = noise_vec[:expected_obs_dim]

        return noise_vec

    # --- Reward functions ---
    # Minimal set, subclasses should add more specific ones.

    def _reward_alive(self):
        # Base survival reward
        return torch.ones(self.num_envs, device=self.device)

    def _reward_termination(self):
        # Base termination penalty
        # reset_buf includes timeout, contact, orientation failures
        # Only penalize non-timeout terminations
        return self.reset_buf * (~self.time_out_buf) * -1.0 # Penalize non-timeout resets

    # --- Foot state tracking ---
    def _init_foot(self):
        """Initializes buffers for foot states using rigid body states."""
        if self.feet_indices is None or len(self.feet_indices) == 0:
             self.feet_num = 0; self.feet_state = None; self.feet_pos = None; self.feet_vel = None; return

        self.feet_num = len(self.feet_indices)
        # Ensure rigid_body_states_view is valid (should be initialized in _init_buffers)
        if not hasattr(self, 'rigid_body_states_view') or self.rigid_body_states_view is None:
             print("⚠️ _init_foot: rigid_body_states_view not available. Foot state tracking disabled.")
             self.feet_num = 0; self.feet_state = None; self.feet_pos = None; self.feet_vel = None; return

        # Ensure feet_indices are within bounds for the view [num_envs, num_bodies, 13]
        if torch.any(self.feet_indices >= self.num_bodies) or torch.any(self.feet_indices < 0):
             print(f"❌ ERROR _init_foot: feet_indices ({self.feet_indices}) out of bounds for num_bodies ({self.num_bodies}).")
             self.feet_num = 0; self.feet_state = None; self.feet_pos = None; self.feet_vel = None; return

        # Slice the view using the indices
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3] # Position (x, y, z)
        self.feet_vel = self.feet_state[:, :, 7:10] # Linear velocity (vx, vy, vz)
        # print("  ✅ Initialized foot state views.")


    def update_feet_state(self):
        """Refreshes and updates foot position and velocity from simulation."""
        # Only update if foot tracking is initialized and view exists
        if self.feet_num > 0 and self.feet_state is not None and hasattr(self, 'rigid_body_states_view') and self.rigid_body_states_view is not None:
            # Refresh the main rigid body state tensor (usually done before this)
            # self.gym.refresh_rigid_body_state_tensor(self.sim)

            # Re-slice the tensor to get the latest states
            # Check bounds again just in case
            if torch.any(self.feet_indices >= self.num_bodies): return # Already warned
            self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
            self.feet_pos = self.feet_state[:, :, :3]
            self.feet_vel = self.feet_state[:, :, 7:10]