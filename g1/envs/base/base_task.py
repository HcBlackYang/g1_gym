#
# import sys
# from isaacgym import gymapi
# from isaacgym import gymutil
# import numpy as np
# import torch
#
# # Base class for RL tasks
# class BaseTask():
#
#     def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
#         print(f"--- DEBUG BaseTask.__init__: Received cfg type: {type(cfg)}")
#         self.gym = gymapi.acquire_gym()
#
#         self.sim_params = sim_params
#         self.physics_engine = physics_engine
#         self.sim_device = sim_device
#         sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
#         self.headless = headless
#
#         # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
#         if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
#             self.device = self.sim_device
#         else:
#             self.device = 'cpu'
#
#         # graphics device for rendering, -1 for no rendering
#         self.graphics_device_id = self.sim_device_id
#         if self.headless == True:
#             self.graphics_device_id = -1
#
#         self.num_envs = cfg.env.num_envs
#         self.num_obs = cfg.env.num_observations
#         self.num_privileged_obs = cfg.env.num_privileged_obs
#         self.num_actions = cfg.env.num_actions
#
#         # optimization flags for pytorch JIT
#         torch._C._jit_set_profiling_mode(False)
#         torch._C._jit_set_profiling_executor(False)
#
#         # allocate buffers
#         self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
#         self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
#         self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
#         self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
#         self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
#         if self.num_privileged_obs is not None:
#             self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
#         else:
#             self.privileged_obs_buf = None
#             # self.num_privileged_obs = self.num_obs
#
#         self.extras = {}
#
#         # create envs, sim and viewer
#         self.create_sim()
#         self.gym.prepare_sim(self.sim)
#
#         # todo: read from config
#         self.enable_viewer_sync = True
#         self.viewer = None
#
#         # if running with a viewer, set up keyboard shortcuts and camera
#         if self.headless == False:
#             # subscribe to keyboard shortcuts
#             self.viewer = self.gym.create_viewer(
#                 self.sim, gymapi.CameraProperties())
#             self.gym.subscribe_viewer_keyboard_event(
#                 self.viewer, gymapi.KEY_ESCAPE, "QUIT")
#             self.gym.subscribe_viewer_keyboard_event(
#                 self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
#
#         self.num_observations = self.num_obs
#
#     def close(self):
#         """释放环境资源，特别是关闭查看器。"""
#         print("关闭环境中...")
#         if self.headless == False and self.viewer is not None:
#             try:
#                 self.gym.destroy_viewer(self.viewer)
#                 print("  - Isaac Gym 查看器已关闭。")
#             except Exception as e:
#                 print(f"  - 关闭 Isaac Gym 查看器时出错: {e}")
#
#     def get_observations(self):
#         return self.obs_buf
#
#     def get_privileged_observations(self):
#         return self.privileged_obs_buf
#
#     def reset_idx(self, env_ids):
#         """Reset selected robots"""
#         raise NotImplementedError
#
#     def reset(self):
#         """ Reset all robots"""
#         self.reset_idx(torch.arange(self.num_envs, device=self.device))
#         obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
#         return obs, privileged_obs
#
#     def step(self, actions):
#         raise NotImplementedError
#
#     def render(self, sync_frame_time=True):
#         if self.viewer:
#             # check for window closed
#             if self.gym.query_viewer_has_closed(self.viewer):
#                 sys.exit()
#
#             # check for keyboard events
#             for evt in self.gym.query_viewer_action_events(self.viewer):
#                 if evt.action == "QUIT" and evt.value > 0:
#                     sys.exit()
#                 elif evt.action == "toggle_viewer_sync" and evt.value > 0:
#                     self.enable_viewer_sync = not self.enable_viewer_sync
#
#             # fetch results
#             if self.device != 'cpu':
#                 self.gym.fetch_results(self.sim, True)
#
#             # step graphics
#             if self.enable_viewer_sync:
#                 self.gym.step_graphics(self.sim)
#                 self.gym.draw_viewer(self.viewer, self.sim, True)
#                 if sync_frame_time:
#                     self.gym.sync_frame_time(self.sim)
#             else:
#                 self.gym.poll_viewer_events(self.viewer)

# base_task.py
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

class BaseTask():
    # !!! 修改签名以接收 gym_handle 和 sim_handle !!!
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        print(f"--- DEBUG BaseTask.__init__: Received cfg type: {type(cfg)}")

        # --- 使用传入的 gym 和 sim ---
        if gym_handle is None:
             # 如果在外部创建并传入，这里通常不应该发生
             print("⚠️ BaseTask Warning: No Gym handle provided, acquiring new one.")
             self.gym = gymapi.acquire_gym()
        else:
             self.gym = gym_handle
             # print("  - BaseTask: Using provided Gym API handle.") # 可以取消注释用于调试

        if sim_handle is None:
             # BaseTask 不再负责创建 Sim
             raise ValueError("BaseTask requires a valid 'sim_handle' to be provided during initialization.")
        else:
             self.sim = sim_handle
             # print(f"  - BaseTask: Using provided Sim handle: {self.sim}") # 可以取消注释用于调试
        # ----------------------------

        self.cfg = cfg # Store config
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        # If not headless, graphics device id is taken from sim_device
        if self.headless:
             self.graphics_device_id = -1


        # --- 统一使用 num_observations ---
        self.num_envs = cfg.env.num_envs
        self.num_observations = cfg.env.num_observations
        self.num_obs = self.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        # ---------------------------------

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers (使用 self.num_observations)
        self.obs_buf = torch.zeros(self.num_envs, self.num_observations, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        self.extras = {}

        # --- !!! 环境和 Sim 的创建/准备移到外部 !!! ---
        # self.create_sim() # Removed
        # self.gym.prepare_sim(self.sim) # Removed (should be called after _create_envs in child)
        # -------------------------------------------

        # Viewer setup remains here, as it depends on the sim handle
        self.enable_viewer_sync = True
        self.viewer = None
        if self.graphics_device_id != -1: # Check if rendering is enabled
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            if self.viewer is None:
                print("⚠️ Failed to create viewer")
            else:
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, gymapi.KEY_ESCAPE, "QUIT")
                self.gym.subscribe_viewer_keyboard_event(
                    self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

    def create_sim(self):
        """ 创建仿真环境、地形和环境实例。
            这个方法现在**主要由子类 (如 LeggedRobot 或 G1KitchenNavigation) 实现**，
            因为它涉及到加载特定资产和创建 Actor。
            BaseTask 不再负责创建 self.sim。
        """
        # Typically, this involves:
        # 1. self._create_ground_plane() or self._create_terrain()
        # 2. self._create_envs() (which loads assets and creates actors)
        # 3. self.gym.prepare_sim(self.sim) *after* all actors are added.
        print("--- BaseTask.create_sim called (should be overridden by child class) ---")
        pass # Base implementation does nothing now


    def close(self):
        """释放环境资源，特别是关闭查看器。Sim 的销毁在外部处理。"""
        print(f"Closing environment ({self.__class__.__name__})...")
        if self.viewer is not None:
            try:
                self.gym.destroy_viewer(self.viewer)
                print("  - Isaac Gym viewer closed.")
                self.viewer = None # Avoid double closing
            except Exception as e:
                print(f"  - Error closing Isaac Gym viewer: {e}")
        # Note: self.gym.destroy_sim(self.sim) is NOT called here.
        # It's managed by the top-level script (train_curriculum.py)

    # ... (get_observations, get_privileged_observations, reset_idx, reset, step, render 保持不变) ...

    def get_observations(self):
        return self.obs_buf

    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        # Ensure step uses self.num_actions for the zero tensor
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def render(self, sync_frame_time=True):
        # Make sure viewer exists before using it
        if self.viewer is None and self.graphics_device_id != -1:
             print("⚠️ Trying to render but viewer is None. Was it created successfully?")
             # Attempt to create viewer again? Or maybe just skip rendering.
             # Let's try creating it again, maybe it failed initially silently.
             self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
             if self.viewer is None:
                 print("⚠️ Failed to create viewer again. Rendering disabled.")
                 self.graphics_device_id = -1 # Disable future attempts
                 return # Skip rendering
             else:
                 # Subscribe events again
                 self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
                 self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                print("Viewer closed by user. Exiting.")
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    print("QUIT event received. Exiting.")
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)