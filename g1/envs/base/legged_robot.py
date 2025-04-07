
# legged_robot.py
import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

# G1_ROOT_DIR should be defined where this script is imported, or use absolute paths
# G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1.envs.base.base_task import BaseTask
from g1.utils.math import wrap_to_pi
from g1.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from g1.utils.helpers import class_to_dict, get_load_path, set_seed # Removed parse_sim_params import
from g1.envs.base.legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    # !!! 修改构造函数签名以匹配 BaseTask !!!
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        """ Parses the provided config file,
            calls self._create_envs() (which creates actors within the provided sim),
            initializes pytorch buffers used during training.

        Args:
            cfg (LeggedRobotCfg): Environment config object.
            sim_params (gymapi.SimParams): simulation parameters.
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX).
            sim_device (string): 'cuda' or 'cpu'.
            headless (bool): Run without rendering if True.
            gym_handle (gymapi.Gym): The Gym API instance (created externally).
            sim_handle (gymapi.Sim): The Simulation instance (created externally).
        """
        print(f"--- LeggedRobot.__init__ (Child of BaseTask) ---")
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False

        self.up_axis_idx = 2

        # --- 调用 BaseTask 的 __init__ ---
        # 它会处理 gym, sim, device, buffers (obs, rew, reset, etc.), viewer 等的设置
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

        # --- LeggedRobot 特定的初始化 ---
        self._parse_cfg() # 解析 LeggedRobot 特定的配置

        # --- 创建环境中的 Actors (在传入的 sim 中) ---
        # This method should be defined in subclasses if they need specific terrain/ground setup
        if hasattr(self, '_create_terrain'): # Check if terrain method exists
             self._create_terrain()
        else: # Fallback to ground plane
             self._create_ground_plane()

        self._create_envs() # This loads assets and creates actors

        # --- !!! 在所有 Actors 创建后准备 Sim !!! ---
        print(f"--- LeggedRobot: Preparing simulation after creating envs...")
        self.gym.prepare_sim(self.sim)
        # Get actual counts AFTER prepare_sim
        self.num_total_bodies = self.gym.get_sim_rigid_body_count(self.sim)
        self.num_total_dofs = self.gym.get_sim_dof_count(self.sim)
        print(f"  Simulation Prepared. Total Bodies: {self.num_total_bodies}, Total DoFs: {self.num_total_dofs}")
        # -------------------------------------------

        # --- 初始化缓冲区和奖励函数 ---
        # _init_buffers 需要在 prepare_sim 之后，因为它 acquire tensors
        self._init_buffers()
        self._prepare_reward_function()
        # -----------------------------

        # --- 设置相机 (如果需要) ---
        if not self.headless and self.viewer is not None:
             # Ensure viewer config exists
             if hasattr(self.cfg, 'viewer') and self.cfg.viewer is not None:
                 self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
             else:
                 print("⚠️ Viewer enabled but cfg.viewer settings not found. Using default camera.")
        # -------------------------

        self.init_done = True
        print(f"--- LeggedRobot.__init__ Done ---")


    def _parse_cfg(self): # Removed cfg argument, use self.cfg
        """ Parses the legged robot specific configuration sections."""
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        # Use class_to_dict to handle potential nested scales object
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        # Terrain curriculum check (ensure terrain attribute exists)
        if hasattr(self.cfg, 'terrain') and self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
             self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        # Domain rand push interval (ensure domain_rand attribute exists)
        if hasattr(self.cfg, 'domain_rand'):
             self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)


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

        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        # create a grid of robots
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
        spacing = self.cfg.env.env_spacing
        self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
        self.env_origins[:, 2] = 0.

    def _create_envs(self):
        """ Creates environments: loads robot asset, creates actors.
            Called by __init__ *before* prepare_sim.
        """
        print("--- _create_envs (LeggedRobot) ---")
        # Define G1_ROOT_DIR or ensure it's passed/globally available
        try:
             G1_ROOT_DIR_LOCAL = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except NameError:
             print("⚠️ WARNING: __file__ not defined, cannot automatically determine G1_ROOT_DIR. Using empty string.")
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
        if not os.path.exists(os.path.join(asset_root, asset_file)):
             print(f"❌❌❌ ERROR: Robot asset file not found at: {os.path.join(asset_root, asset_file)}")
             raise FileNotFoundError(f"Robot asset file not found: {os.path.join(asset_root, asset_file)}")

        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if self.robot_asset is None: raise RuntimeError(f"Failed to load robot asset: {asset_path}")

        # Get DoF/Body counts *from the asset*
        self.num_dof = self.gym.get_asset_dof_count(self.robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.robot_asset)
        # Ensure cfg matches asset
        if self.cfg.env.num_actions != self.num_dof:
             print(f"⚠️ LeggedRobot Warning: cfg.env.num_actions ({self.cfg.env.num_actions}) != asset num_dof ({self.num_dof}). Overriding cfg and instance num_actions.")
             self.cfg.env.num_actions = self.num_dof
             self.num_actions = self.num_dof

        print(f"  Asset Loaded. Num DoF={self.num_dof}, Num Bodies={self.num_bodies}")

        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(self.robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(self.robot_asset)
        self.num_dofs = len(self.dof_names) # Should match self.num_dof
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
            pos += to_torch(self.cfg.init_state.pos, device=self.device) # Add configured initial offset
            random_offset_xy = torch_rand_float(-0.1, 0.1, (1, 2), device=self.device)
            pos[:2] += random_offset_xy.squeeze(0)
            start_pose.p = gymapi.Vec3(*pos)
            start_pose.r = gymapi.Quat(*self.cfg.init_state.rot)

            # Process props for this actor instance
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            # Apply props to the asset (might affect subsequent actors if not careful)
            # Consider using set_actor_rigid_shape_properties if props need to differ significantly per env beyond friction
            self.gym.set_asset_rigid_shape_properties(self.robot_asset, rigid_shape_props)

            actor_handle = self.gym.create_actor(env_handle, self.robot_asset, start_pose, self.cfg.asset.name,
                                                 i, # Collision group = env index
                                                 self.cfg.asset.self_collisions, 0) # Collision filter mask
            if actor_handle == gymapi.INVALID_HANDLE:
                 raise RuntimeError(f"Failed to create actor in env {i}")
            self.actor_handles.append(actor_handle)

            # Set DoF and Body properties for this actor instance
            # Process DoF props (gets limits first time, reuses after)
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

             self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
             for i, name in enumerate(feet_names):
                 handle = self.gym.find_actor_rigid_body_handle(first_env_handle, first_actor_handle, name)
                 if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Foot body '{name}' not found.")
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
                  # Provide more context if possible
                  print(f"   - Was gym.prepare_sim(sim) called after all actors were created?")
                  print(f"   - Check for errors during gym.create_sim or actor creation.")
                  raise RuntimeError(f"Failed to acquire simulation state tensor: {name}")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Wrap tensors
        self.root_states = gymtorch.wrap_tensor(actor_root_state)       # Shape [num_envs, 13]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)         # Shape [num_envs * num_dof, 2]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor) # Shape [num_envs * num_bodies, 13]
        self.contact_forces_raw = gymtorch.wrap_tensor(net_contact_forces) # Raw shape, needs reshape

        print(f"  Raw Tensor Shapes: root={self.root_states.shape}, dof={self.dof_state.shape}, rigid_body={self.rigid_body_states.shape}, contact={self.contact_forces_raw.shape}")

        # Create Views (handle potential shape mismatches)
        # DoF state views
        try:
            dof_state_reshaped = self.dof_state.view(self.num_envs, self.num_dof, 2)
            self.dof_pos = dof_state_reshaped[..., 0]
            self.dof_vel = dof_state_reshaped[..., 1]
        except RuntimeError as e:
            print(f"❌ ERROR reshaping dof_state {self.dof_state.shape} to ({self.num_envs}, {self.num_dof}, 2): {e}")
            raise # Re-raise as this is critical

        # Rigid body state view
        try:
            self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        except RuntimeError as e:
            print(f"❌ ERROR reshaping rigid_body_states {self.rigid_body_states.shape} to ({self.num_envs}, {self.num_bodies}, 13): {e}")
            raise

        # Contact forces view
        try:
            # Expected size: num_envs * num_bodies * 3
            expected_contact_elements = self.num_envs * self.num_bodies * 3
            if self.contact_forces_raw.numel() == expected_contact_elements:
                 self.contact_forces = self.contact_forces_raw.view(self.num_envs, self.num_bodies, 3)
                 print(f"  Reshaped contact_forces to: {self.contact_forces.shape}")
            else:
                 # Fallback for safety, but indicates underlying issue
                 print(f"⚠️ WARNING: Unexpected contact_forces_raw shape {self.contact_forces_raw.shape}. Num elements {self.contact_forces_raw.numel()} != expected {expected_contact_elements}. Using view(num_envs, -1, 3).")
                 self.contact_forces = self.contact_forces_raw.view(self.num_envs, -1, 3)
        except RuntimeError as e:
            print(f"❌ ERROR reshaping contact_forces_raw {self.contact_forces_raw.shape}: {e}")
            raise

        # Robot state views
        self.base_quat = self.root_states[:, 3:7]
        self.base_pos = self.root_states[:, 0:3]
        self.rpy = torch.zeros_like(self.base_pos)

        # Initialize other buffers
        self.common_step_counter = 0
        self.extras = {}
        # Initialize noise_scale_vec after obs_buf is correctly sized
        # self.obs_buf initialized in BaseTask using self.num_observations
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # Call method to create it

        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        # Use self.num_actions (which should == self.num_dof)
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
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

        # PD gains, default pos, torque limits
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        dof_props_asset = self.gym.get_asset_dof_properties(self.robot_asset) # Use asset loaded in _create_envs

        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles.get(name, 0.0)
            self.default_dof_pos[i] = angle
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

            found = False
            for dof_name_key in self.cfg.control.stiffness.keys():
                if dof_name_key in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name_key]
                    self.d_gains[i] = self.cfg.control.damping.get(dof_name_key, 0.0)
                    found = True
                    break
            if not found:
                self.p_gains[i] = 0.; self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]: print(f"  PD gain '{name}' not defined.")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # Initialize foot state tracking
        self._init_foot()
        print("--- _init_buffers (LeggedRobot) Done ---")


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions based on non-zero scales. """
        print("  Preparing reward functions...")
        # Use the scales potentially modified by the scheduler (stored in self.reward_scales)
        current_reward_scales = self.reward_scales

        # Prepare list of functions based on non-zero scales
        self.reward_functions = []
        self.reward_names = []
        skipped_rewards = []
        for name, scale in current_reward_scales.items():
            if name == "termination": continue
            if scale != 0.0:
                 func_name = '_reward_' + name
                 if hasattr(self, func_name) and callable(getattr(self, func_name)):
                      self.reward_names.append(name)
                      self.reward_functions.append(getattr(self, func_name))
                      # Multiply scale by dt here for efficiency
                      current_reward_scales[name] *= self.dt
                 else:
                      skipped_rewards.append(f"{name} (missing function {func_name})")
            # else: # Keep track of zero-scaled rewards if needed
            #     skipped_rewards.append(f"{name} (scale is 0)")


        # Reward episode sums (only for active rewards)
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_names}
        # Also include termination if used
        if "termination" in current_reward_scales and current_reward_scales["termination"] != 0.0:
            self.reward_names.append("termination") # Add termination to names for sum tracking
            self.episode_sums["termination"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
            # Multiply termination scale by 1 (it's applied once per episode end)
            # current_reward_scales["termination"] *= 1.0 # Or just use the value directly

        print(f"    Active reward functions: {self.reward_names}")
        if skipped_rewards: print(f"    Skipped/Inactive rewards: {skipped_rewards}")


    # --- Step, Post Physics Step, Check Termination, Reset Idx ---
    # These methods generally remain the same as your provided version,
    # assuming they correctly use the instance variables initialized in _init_buffers
    # (e.g., self.root_states, self.dof_pos, self.contact_forces, etc.)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step() """
        if actions.shape[1] != self.num_actions:
             print(f"❌ ERROR in step: Received actions shape {actions.shape} does not match num_actions {self.num_actions}")
             # Handle error, e.g., by taking a subset or padding, or raising error
             actions = actions[:, :self.num_actions] # Simple truncation

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            # Compute torques for the robot
            computed_torques = self._compute_torques(self.actions) # Shape [num_envs, num_actions]
            # Set torques using the global tensor if needed (handled by specific envs)
            # Here, assume torques buffer matches set_dof_actuation_force_tensor input
            if computed_torques.shape == self.torques.shape:
                 self.torques[:] = computed_torques
                 self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques.flatten())) # Flatten if needed by API
            else:
                 # Handle shape mismatch - G1KitchenNav overrides _compute_torques
                 print(f"⚠️ WARNING step: Shape mismatch computed_torques {computed_torques.shape} vs self.torques {self.torques.shape}. Using LeggedRobot logic.")
                 self.torques = computed_torques.view(self.torques.shape) # Try to reshape
                 self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))


            self.gym.simulate(self.sim)
            if self.device == 'cpu': self.gym.fetch_results(self.sim, True) # Fetch results after simulate
            # Refresh states needed for next loop iteration or post_physics_step
            self.gym.refresh_dof_state_tensor(self.sim)
            # if hasattr(self, 'manual_dof_update') and self.manual_dof_update: # If using fallback
            #      # Manually update self.dof_pos/vel from self.dof_state
            #      print("Manual DOF update needed - NOT IMPLEMENTED YET") # TODO
            #      pass
            # Else: dof_pos/vel views are automatically updated

        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations.
            Handles common logic like command resampling and heading update.
            Subclasses can override this and call super().
        """
        # Resample commands based on time
        # Ensure necessary attributes exist before using them
        if hasattr(self, 'episode_length_buf') and hasattr(self.cfg, 'commands') and hasattr(self.cfg.commands,
                                                                                             'resampling_time') and self.dt > 0:
            resampling_interval_steps = getattr(self.cfg.commands, 'resampling_interval_steps', None)
            if resampling_interval_steps is None:  # Calculate if not pre-calculated
                resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt)
                self.cfg.commands.resampling_interval_steps = resampling_interval_steps  # Cache it

            if resampling_interval_steps > 0:  # Avoid modulo by zero
                env_ids = (self.episode_length_buf % resampling_interval_steps == 0).nonzero(as_tuple=False).flatten()
                if len(env_ids) > 0:
                    self._resample_commands(env_ids)
            else:
                print("⚠️ Warning: resampling_interval_steps is zero or negative. Skipping command resampling.")

        # Compute heading command if enabled
        if getattr(self.cfg.commands, 'heading_command', False):
            if hasattr(self, 'base_quat') and hasattr(self, 'forward_vec') and hasattr(self, 'commands') and \
                    self.commands.shape[1] >= 4:
                try:
                    forward = quat_apply(self.base_quat, self.forward_vec)
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    heading_error = wrap_to_pi(self.commands[:, 3] - heading)
                    # Apply gain (e.g., 0.5) and clip
                    yaw_command = torch.clip(0.5 * heading_error, -1., 1.)
                    self.commands[:, 2] = yaw_command
                except Exception as e:
                    print(f"❌ Error computing heading command in LeggedRobot callback: {e}")
            else:
                print("⚠️ Cannot compute heading command: required attributes missing or commands shape incorrect.")


    def post_physics_step(self):
        """ check terminations, compute observations and rewards """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # Refresh DoF state if not done in the inner loop (it is done there)
        # self.gym.refresh_dof_state_tensor(self.sim)
        # Refresh rigid body state for feet update
        # self.gym.refresh_rigid_body_state_tensor(self.sim) # Done in update_feet_state if needed

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities (using robot-specific states where applicable)
        # BaseTask uses self.root_states, LeggedRobot assumes it's [num_envs, 13]
        # If multi-actor envs override _init_buffers, they need to set up self.robot_root_states view correctly
        if hasattr(self, 'robot_root_states'): # Use robot-specific view if exists (like in G1KitchenNav)
             root_states_to_use = self.robot_root_states
        else: # Fallback to global root_states (assuming single actor per env)
             root_states_to_use = self.root_states

        self.base_pos[:] = root_states_to_use[:, 0:3]
        self.base_quat[:] = root_states_to_use[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:]) # Use updated base_quat
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, root_states_to_use[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, root_states_to_use[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # Update DoF pos/vel if using manual update fallback
        if hasattr(self, 'manual_dof_update') and self.manual_dof_update:
             # TODO: Implement manual update logic here if needed
             pass

        # Callbacks (phase calculation, command resampling)
        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0: # Only call reset if needed
             self.reset_idx(env_ids)

        # Push robots (uses self.root_states directly, needs override for multi-actor)
        if self.cfg.domain_rand.push_robots:
            self._push_robots()

        # Compute observations *after* potential resets and state updates
        self.compute_observations()

        # Store last states
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:] # Uses robot dof_vel
        self.last_root_vel[:] = root_states_to_use[:, 7:13] # Uses robot root vel


    def check_termination(self):
        """ Check if environments need to be reset """
        # Use robot's termination indices relative to robot's contact forces
        # Need to get contact forces specific to the robot bodies
        if self.contact_forces is None: # Check if contact forces are valid
             print("⚠️ check_termination: Contact forces buffer is None. Cannot check termination.")
             self.reset_buf.zero_() # Assume no termination if contacts unavailable
             self.time_out_buf = self.episode_length_buf > self.max_episode_length
             self.reset_buf |= self.time_out_buf
             return

        # Assuming self.contact_forces is [num_envs, num_total_bodies_per_env, 3]
        # And termination_contact_indices are local indices for the robot (0 to num_bodies-1)
        # We need contact forces only for the robot bodies
        # If only one actor/env, num_total_bodies_per_env == self.num_bodies
        robot_contact_forces = self.contact_forces[:, :self.num_bodies, :] # Assume robot bodies are first

        try:
             # Use local termination indices with the robot's contact forces view
             self.reset_buf = torch.any(torch.norm(robot_contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        except IndexError as e:
             print(f"❌ ERROR in check_termination: IndexError accessing contact_forces. Indices={self.termination_contact_indices}, Shape={robot_contact_forces.shape}. {e}")
             self.reset_buf.zero_() # Fail safe

        # Orientation check uses self.rpy which is based on robot_root_states
        self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1]) > 1.0, torch.abs(self.rpy[:,0]) > 0.8)
        # Timeout check
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf


    def reset_idx(self, env_ids):
        """ Reset some environments. """
        if len(env_ids) == 0: return

        # Reset robot states (calls methods that use self.dof_pos, self.root_states etc.)
        self._reset_dofs(env_ids) # Resets robot DoFs
        self._reset_root_states(env_ids) # Resets robot root state

        # Resample commands for reset envs
        self._resample_commands(env_ids)

        # Reset buffers
        self.last_actions[env_ids] = 0.
        # Ensure last_dof_vel has same shape as dof_vel before indexing
        if self.last_dof_vel.shape[0] == self.num_envs: self.last_dof_vel[env_ids] = 0.
        if self.feet_air_time is not None: self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1 # Mark for reset in the next step logic if using delayed reset

        # Fill extras and log episode sums
        self.extras["episode"] = {}
        # Ensure episode_sums keys match active rewards
        active_reward_names = list(self.episode_sums.keys())
        for key in active_reward_names:
             # Safely access episode sums
             sum_tensor = self.episode_sums.get(key)
             if sum_tensor is not None and sum_tensor.shape[0] == self.num_envs:
                 self.extras["episode"]['rew_' + key] = torch.mean(sum_tensor[env_ids]).item() / self.max_episode_length_s
                 self.episode_sums[key][env_ids] = 0.
             else:
                 print(f"⚠️ reset_idx: Skipping episode sum logging for '{key}' due to missing or mismatched tensor.")

        # Command curriculum logging (ensure commands attribute exists)
        if hasattr(self.cfg, 'commands') and self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # Send timeout info
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf.clone() # Send a copy


    def compute_reward(self):
        """ Compute rewards based on active reward functions and scales. """
        self.rew_buf[:] = 0.
        # episode_sums should be initialized/updated based on active rewards in _prepare_reward_function
        active_reward_names = list(self.episode_sums.keys())
        if "termination" in active_reward_names: active_reward_names.remove("termination") # Handle termination separately

        for name in active_reward_names:
             try:
                 reward_func = getattr(self, '_reward_' + name)
                 rew = reward_func() * self.reward_scales[name] # reward_scales holds dt adjusted value
                 # Shape check
                 if rew.shape == (self.num_envs,):
                      self.rew_buf += rew
                      self.episode_sums[name] += rew # Accumulate sum for active reward
                 else:
                      print(f"⚠️ compute_reward: Shape mismatch for reward '{name}'. Expected ({self.num_envs},), got {rew.shape}. Skipping.")
             except Exception as e:
                  print(f"❌ Error computing reward '{name}': {e}")

        # Apply clipping
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        # Add termination reward after clipping
        if "termination" in self.reward_scales and self.reward_scales["termination"] != 0:
            # Check if function exists before calling
            if hasattr(self, '_reward_termination'):
                 rew = self._reward_termination() * self.reward_scales["termination"]
                 if rew.shape == (self.num_envs,):
                     self.rew_buf += rew
                     # Check if "termination" key exists before accessing
                     if "termination" in self.episode_sums:
                         self.episode_sums["termination"] += rew
                 else:
                      print(f"⚠️ compute_reward: Shape mismatch for termination reward. Expected ({self.num_envs},), got {rew.shape}.")
            else:
                 print("⚠️ compute_reward: Termination reward enabled but _reward_termination function missing.")


    def set_camera(self, position, lookat):
        """ Set camera position and direction. """
        if self.viewer is None: return # Do nothing if viewer doesn't exist
        cam_pos = gymapi.Vec3(*position)
        cam_target = gymapi.Vec3(*lookat)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


    # --- Callbacks (_process_*) remain the same ---
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]
            if props: # Check if props is not empty
                 for s in range(len(props)):
                     props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        if env_id==0:
            # Initialize only if not already initialized (in case called multiple times)
            if not hasattr(self, 'dof_pos_limits') or self.dof_pos_limits.shape[0] != self.num_dof:
                 self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
                 self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
                 # Torque limits initialized in _init_buffers now
                 # self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

            for i in range(len(props)): # Should match self.num_dof
                # Ensure index is within bounds
                if i < self.num_dof:
                     self.dof_pos_limits[i, 0] = props["lower"][i].item()
                     self.dof_pos_limits[i, 1] = props["upper"][i].item()
                     self.dof_vel_limits[i] = props["velocity"][i].item()
                     # self.torque_limits[i] = props["effort"][i].item() # Set in _init_buffers
                     # soft limits calculation
                     m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                     r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                     self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                     self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if not props: # Check for empty list
            print("⚠️ _process_rigid_body_props: Received empty props list.")
            return props
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            # Ensure props[0] exists before accessing mass
            if len(props) > 0:
                 props[0].mass += np.random.uniform(rng[0], rng[1])
            else:
                 print("⚠️ _process_rigid_body_props: Cannot apply base mass randomization, props list is empty.")
        return props

    # --- _resample_commands, _compute_torques ---
    # These seem okay, ensure they use instance variables correctly (e.g., self.num_actions)
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments """
        if len(env_ids) == 0: return
        # Use self.command_ranges directly
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            # Ensure self.commands has column 3
            if self.commands.shape[1] >= 4:
                 self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            else: print("⚠️ _resample_commands: Heading command enabled but self.commands has < 4 columns.")
        else:
            # Ensure self.commands has column 2
             if self.commands.shape[1] >= 3:
                 self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
             else: print("⚠️ _resample_commands: Yaw command enabled but self.commands has < 3 columns.")


        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques for the robot based on its DoFs. """
        # actions shape: [num_envs, num_actions] (e.g., [*, 43])
        if actions.shape[1] != self.num_actions:
             print(f"❌ ERROR _compute_torques: actions shape {actions.shape} mismatch num_actions {self.num_actions}")
             return torch.zeros_like(self.torques) # Return zero torques

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        # Use robot-specific DoF states (self.dof_pos, self.dof_vel)
        # Ensure shapes match num_actions
        if self.dof_pos.shape[1] != self.num_actions or self.dof_vel.shape[1] != self.num_actions:
             print(f"❌ ERROR _compute_torques: DOF state shapes mismatch num_actions")
             return torch.zeros_like(self.torques)

        if control_type=="P":
            # Ensure default_dof_pos shape is [1, num_actions] or [num_actions]
            if self.default_dof_pos.shape[-1] != self.num_actions:
                 print(f"❌ ERROR _compute_torques: default_dof_pos shape mismatch")
                 return torch.zeros_like(self.torques)
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            # Ensure last_dof_vel shape matches dof_vel
            if self.last_dof_vel.shape != self.dof_vel.shape: self.last_dof_vel = torch.zeros_like(self.dof_vel)
            dt = self.dt # Use calculated dt
            if dt <= 0: dt = 1e-5 # Avoid division by zero
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/ dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")

        # Clip using robot torque limits
        return torch.clip(torques, -self.torque_limits, self.torque_limits) # Shape [num_envs, num_actions]

    # --- _reset_dofs, _reset_root_states ---
    # Need modification for multi-actor environments in subclasses.
    # Base implementation assumes tensors map directly to the single robot.
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected robots. """
        if len(env_ids) == 0: return
        # Generate random positions for the robot DoFs
        # Ensure self.dof_pos buffer is used here (shape [num_envs, num_dof])
        new_dof_pos = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        new_dof_vel = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        # --- Update the global dof_state tensor ---
        # Use the indices calculated in _init_buffers (assuming fallback logic)
        # This needs careful implementation in G1KitchenNavigation
        # For base LeggedRobot, the simple view update might work IF dof_state is contiguous
        try:
             dof_state_view = self.dof_state.view(self.num_envs, self.num_dof, 2)
             dof_state_view[env_ids, :, 0] = new_dof_pos
             dof_state_view[env_ids, :, 1] = new_dof_vel
        except RuntimeError:
             # Fallback: Use set_dof_state_tensor_indexed (less efficient for sparse updates)
             print("⚠️ _reset_dofs: View failed, using indexed update (potentially slow).")
             temp_dof_state = self.dof_state.clone()
             # Assuming self.all_robot_dof_indices holds global indices [env0_dof0, ..., envN_dofM]
             # This mapping is complex, needs careful setup in _init_buffers.
             # --- Simplified approach for base class ---
             # Assume dof_state is [num_envs * num_dof, 2]
             indices_pos = env_ids * self.num_dof # This indexing is likely WRONG for sparse env_ids
             indices_vel = indices_pos + 1
             # This part needs a robust way to map env_ids and dof_idx to global index
             print("TODO: Implement robust indexed DOF reset for sparse env_ids")

        # Update internal buffers as well
        self.dof_pos[env_ids] = new_dof_pos
        self.dof_vel[env_ids] = new_dof_vel

        # Call gym function to apply changes (using the full tensor)
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))


    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected robots. """
        if len(env_ids) == 0: return
        # Base implementation assumes self.root_states is [num_envs, 13]
        # Generate random velocities
        random_vels = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)

        # Apply to root_states buffer
        self.root_states[env_ids] = self.base_init_state.clone() # Start with default state
        self.root_states[env_ids, :3] += self.env_origins[env_ids] # Add env origin
        # Optional: Add random XY offset from origin
        self.root_states[env_ids, :2] += torch_rand_float(-0.1, 0.1, (len(env_ids), 2), device=self.device)
        self.root_states[env_ids, 7:13] = random_vels # Set random velocities

        # Set tensor in sim
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        # Use set_actor_root_state_tensor_indexed - needs global actor indices
        # If actor handles correspond directly to env_ids (single actor per env)
        actor_indices = env_ids_int32 # Assuming actor index == env index
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states), # Pass the full tensor
            gymtorch.unwrap_tensor(actor_indices), # Pass the indices of actors to reset
            len(env_ids_int32)
        )

    # --- _push_robots needs override for multi-actor ---
    def _push_robots(self):
        """ Random pushes the robots. Base implementation assumes single actor per env. """
        env_ids = torch.arange(self.num_envs, device=self.device)
        # Use push_interval from cfg
        push_interval_steps = getattr(self.cfg.domain_rand, 'push_interval', 300) # Default ~1.5s
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(push_interval_steps) == 0]

        if len(push_env_ids) == 0: return

        max_vel = self.cfg.domain_rand.max_push_vel_xy
        random_vel_xy = torch_rand_float(-max_vel, max_vel, (len(push_env_ids), 2), device=self.device)

        # Apply to root_states buffer for the selected envs
        self.root_states[push_env_ids, 7:9] = random_vel_xy # Set lin vel x/y

        # Set tensor in sim using indexed update
        actor_indices = push_env_ids.to(dtype=torch.int32) # Assuming actor index == env index
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices)
        )

    # --- update_command_curriculum, _get_noise_scale_vec (base), reward functions ---
    # These seem generally OK, relying on instance variables. Ensure reward functions use correct states.
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands """
        if len(env_ids) == 0: return
        # Ensure reward scales and episode sums are properly initialized
        if "tracking_lin_vel" not in self.reward_scales or "tracking_lin_vel" not in self.episode_sums:
             return

        # Check for division by zero
        if self.max_episode_length_s <= 0: return

        # Calculate average reward, handle potential NaN
        mean_rew = torch.mean(self.episode_sums["tracking_lin_vel"][env_ids] / self.max_episode_length_s)
        if torch.isnan(mean_rew): return

        # Compare with target threshold (e.g., 80% of max possible scale * dt)
        # Note: self.reward_scales already includes dt multiplication
        target_rew_threshold = 0.8 * self.reward_scales["tracking_lin_vel"] / self.dt # Divide by dt to get original scale threshold

        if mean_rew / self.dt > target_rew_threshold: # Compare original scales
            max_curr = self.cfg.commands.max_curriculum
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1, -max_curr, 0.) # Smaller increments
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0., max_curr)
            # Optionally update other command ranges (y, yaw) as well


    # Reward functions (_reward_*) - seem OK, ensure they use self.base_lin_vel etc.