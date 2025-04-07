#
# # g1_basic_locomotion.py
# from g1.envs.curriculum.curriculum_base import G1CurriculumBase # Use the updated base class
#
# from isaacgym.torch_utils import *
# import torch
# import numpy as np # Import numpy for phase calculations if needed later
#
#
# # from g1.utils.task_registry import task_registry
# #
# # @task_registry.register("G1BasicLocomotion")
#
# class G1BasicLocomotion(G1CurriculumBase):
#     """第一阶段：基础运动技能训练.
#     继承自 G1CurriculumBase，实现阶段特定逻辑。
#     """
#
#     def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
#         # 调用父类初始化 (父类会处理基础 G1 设置和课程阶段属性)
#         super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
#
#         # --- Stage 1 Specific Parameters ---
#         # 从合并后的 cfg 中读取本阶段的参数 (train_curriculum.py 负责注入)
#         # 使用 getattr 安全访问，并提供默认值以防配置缺失
#         stage_params_attr = f'stage{self.curriculum_stage}_params'
#         if hasattr(cfg.curriculum, stage_params_attr):
#             stage_params = getattr(cfg.curriculum, stage_params_attr, {})
#             print(f"  G1BasicLocomotion: Loaded stage {self.curriculum_stage} params: {list(stage_params.keys())}")
#         else:
#              print(f"⚠️ G1BasicLocomotion: Warning - Could not find '{stage_params_attr}' in cfg.curriculum. Using defaults.")
#              stage_params = {}
#
#         # 安全地获取参数，提供默认值
#         # 这些值将用于 _resample_commands
#         self.base_lin_vel_range = stage_params.get('base_lin_vel_range', 1.0) # Example default
#         self.base_ang_vel_range = stage_params.get('base_ang_vel_range', 0.5) # Example default
#         print(f"  G1BasicLocomotion: lin_vel_range={self.base_lin_vel_range}, ang_vel_range={self.base_ang_vel_range}")
#
#         # --- Stage 1 Specific State ---
#         # 用于计算 success_flags 的状态变量
#         # command_duration 在父类 LeggedRobot 中以 episode_length_buf 的形式存在
#         # resampling_time_steps = int(self.cfg.commands.resampling_time / self.dt)
#         self.successful_command_tracking_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
#         # 成功所需的最小连续跟踪步数 (例如，命令周期的 80%)
#         self.min_tracking_steps_for_success = int((self.cfg.commands.resampling_time / self.dt) * 0.8)
#
#
#     def _resample_commands(self, env_ids):
#         """重写命令采样方法以应用课程子阶段进度"""
#         if len(env_ids) == 0:
#             return
#
#         # 根据课程子阶段动态调整命令范围
#         # cfg.curriculum.sub_stage 由 train_curriculum.py 更新
#         sub_stage = getattr(self.cfg.curriculum, 'sub_stage', 1) # Default to 1 if not set
#         # 假设总共有 5 个子阶段来调整难度
#         # progress_factor = min(1.0, sub_stage / 5.0) # Linear scaling
#         progress_factor = np.clip(sub_stage / 5.0, 0.1, 1.0) # Clip to avoid zero range at start
#
#         current_vel_range = self.base_lin_vel_range * progress_factor
#         current_ang_range = self.base_ang_vel_range * progress_factor
#
#         # 重采样命令 (使用基类 LeggedRobot 的方法或者直接采样)
#         # super()._resample_commands(env_ids) # 如果想用基类的采样逻辑（可能包含heading等）
#         # --- 或者直接在这里采样 Stage 1 需要的命令 ---
#         self.commands[env_ids, 0] = torch_rand_float(
#             -current_vel_range, current_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
#         self.commands[env_ids, 1] = torch_rand_float(
#             -current_vel_range, current_vel_range, (len(env_ids), 1), device=self.device).squeeze(1)
#         self.commands[env_ids, 2] = torch_rand_float(
#             -current_ang_range, current_ang_range, (len(env_ids), 1), device=self.device).squeeze(1)
#
#         # set small commands to zero (copied from base LeggedRobot)
#         self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
#
#         # 重置这些环境的成功跟踪计数器
#         self.successful_command_tracking_streak[env_ids] = 0
#         # print(f"Resampled commands for {len(env_ids)} envs. Sub-stage: {sub_stage}, Factor: {progress_factor:.2f}")
#
#     # --- Stage 1 Specific Reward Overrides ---
#     def _reward_tracking_lin_vel(self):
#         """(Stage 1 Override) 线速度跟踪奖励 - 更关注精确跟踪"""
#         # 使用更严格的 sigma (0.15)
#         lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
#         return torch.exp(-lin_vel_error / 0.15) # Smaller sigma means higher penalty for error
#
#     def _reward_tracking_ang_vel(self):
#         """(Stage 1 Override) 角速度跟踪奖励 - 更关注精确跟踪"""
#         # 使用更严格的 sigma (0.15)
#         ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
#         return torch.exp(-ang_vel_error / 0.15) # Smaller sigma
#
#     # --- Stage 1 Specific Success Criteria ---
#     def compute_success_criteria(self):
#         """计算每个环境在当前步骤是否满足成功跟踪命令的条件"""
#         # 判断是否成功跟踪命令 (使用较宽松的阈值判断单步成功)
#         lin_vel_threshold = 0.3 # m/s error tolerance
#         ang_vel_threshold = 0.2 # rad/s error tolerance
#
#         lin_vel_close_enough = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
#         ang_vel_close_enough = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]) < ang_vel_threshold
#
#         current_step_success = lin_vel_close_enough & ang_vel_close_enough
#
#         # 更新连续成功计数
#         # 如果当前成功，计数器加1；否则重置为0
#         self.successful_command_tracking_streak = (self.successful_command_tracking_streak + 1) * current_step_success
#
#         # 判断是否达到了连续成功的阈值
#         episode_success_flags = self.successful_command_tracking_streak >= self.min_tracking_steps_for_success
#
#         # --- 重要: 返回 success_flags 给 Runner ---
#         # Runner (e.g., PPO runner) 需要在 step() 返回的 infos 字典中获取这些标志
#         # 以便计算整个批次的平均成功率 (runner.current_statistics['success_rate'])
#         # 这里我们只计算标志，实际传递在 post_physics_step 中完成
#         return episode_success_flags
#
#     def post_physics_step(self):
#         """重写后处理步骤以计算并将成功标志放入 extras"""
#         # 调用父类的 post_physics_step (处理终止、重置、计算通用奖励和观测等)
#         super().post_physics_step()
#
#         # 计算本阶段的成功标准
#         # 注意: compute_success_criteria 更新了 self.successful_command_tracking_streak
#         # 并返回了哪些环境 *在本命令周期内* 达到了连续成功的条件
#         success_flags = self.compute_success_criteria()
#
#         # --- 将成功标志放入 extras 字典 ---
#         # Runner 通常会检查 self.extras 来获取需要在训练循环中记录或使用的信息
#         if "success_flags" not in self.extras:
#             self.extras["success_flags"] = torch.zeros_like(success_flags)
#
#         # 更新 extras 中的标志 (只记录那些刚刚达到成功阈值的回合)
#         # Runner 可以通过聚合这些标志来计算成功率
#         # 注意：Runner 可能需要知道哪些回合是新达到成功的，哪些是已经成功的
#         # 一个简单的策略是只在回合结束时记录最终状态，或者让 Runner 处理历史记录。
#         # 这里我们传递每一步的判断结果，Runner负责聚合。
#         self.extras["success_flags"] = success_flags
#
#         # 当环境被重置时 (由父类的 post_physics_step 处理 reset_buf)
#         # 相应的成功跟踪计数器也需要重置
#         env_ids_to_reset = self.reset_buf.nonzero(as_tuple=False).flatten()
#         if len(env_ids_to_reset) > 0:
#             self.successful_command_tracking_streak[env_ids_to_reset] = 0

# g1_basic_locomotion.py
from g1.envs.curriculum.curriculum_base import G1CurriculumBase # Use the updated base class

from isaacgym.torch_utils import *
import torch
import numpy as np

class G1BasicLocomotion(G1CurriculumBase):
    """第一阶段：基础运动技能训练.
    继承自 G1CurriculumBase，实现阶段特定逻辑。
    """

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        # --- Stage 1 Specific Config Processing ---
        # Stage params should be accessed *after* super().__init__ if they modify base cfg behavior
        # But we need base_lin_vel_range *before* command resampling might happen in reset.
        # Let's read them here, but ensure super init happens correctly.
        stage_params_attr = f'stage{cfg.curriculum.stage}_params' # Access cfg directly before super init
        stage_params = {}
        if hasattr(cfg.curriculum, stage_params_attr):
            stage_params = getattr(cfg.curriculum, stage_params_attr, {})
        else:
             print(f"⚠️ G1BasicLocomotion: Warning - Could not find '{stage_params_attr}' in cfg.curriculum during pre-init.")

        self.base_lin_vel_range = stage_params.get('base_lin_vel_range', 1.0)
        self.base_ang_vel_range = stage_params.get('base_ang_vel_range', 0.5)
        print(f"--- G1BasicLocomotion Pre-init: lin_vel_range={self.base_lin_vel_range}, ang_vel_range={self.base_ang_vel_range}")

        # --- 调用父类初始化 ---
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)
        print(f"--- G1BasicLocomotion Post-super().__init__ ---")


        # --- Stage 1 Specific State Initialization (after buffers are ready) ---
        # 使用 episode_length_buf 来判断命令周期
        # self.command_duration = 0 # No longer needed

        # Resetting streak counter logic moved to _resample_commands and reset_idx
        self.successful_command_tracking_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # Calculate steps needed based on final dt
        self.resampling_interval_steps = int(self.cfg.commands.resampling_time / self.dt)
        self.min_tracking_steps_for_success = int(self.resampling_interval_steps * 0.8) # 80% of interval
        print(f"  Resampling interval: {self.resampling_interval_steps} steps")
        print(f"  Min tracking steps for success: {self.min_tracking_steps_for_success}")


    def _resample_commands(self, env_ids):
        """重写命令采样方法以应用课程子阶段进度"""
        if len(env_ids) == 0: return

        sub_stage = getattr(self.cfg.curriculum, 'sub_stage', 1)
        # progress_factor = min(1.0, sub_stage / 5.0) # Linear
        progress_factor = np.clip(sub_stage / 5.0, 0.1, 1.0) # Clipped

        current_vel_range = self.base_lin_vel_range * progress_factor
        current_ang_range = self.base_ang_vel_range * progress_factor

        # --- 调用父类 LeggedRobot 的 _resample_commands ---
        # 它处理基础的速度和朝向命令采样
        super()._resample_commands(env_ids)

        # --- 如果需要覆盖父类的采样逻辑或进行特定调整，在此处进行 ---
        # 例如，强制 Stage 1 不使用侧向速度 (如果父类采样了 lin_vel_y)
        # self.commands[env_ids, 1] = 0.0

        # --- 重置这些环境的成功跟踪计数器 ---
        if hasattr(self, 'successful_command_tracking_streak'): # Ensure buffer exists
            self.successful_command_tracking_streak[env_ids] = 0


    # --- Stage 1 Specific Reward Overrides ---
    # 这些函数会覆盖 G1CurriculumBase 中同名的函数
    def _reward_tracking_lin_vel(self):
        """(Stage 1 Override) 线速度跟踪奖励 - 更关注精确跟踪"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        # 使用更严格的 sigma (例如 0.15 或 0.1)
        sigma = getattr(self.cfg.rewards, 'tracking_sigma_stage1', 0.15) # Allow config override
        return torch.exp(-lin_vel_error / sigma)

    def _reward_tracking_ang_vel(self):
        """(Stage 1 Override) 角速度跟踪奖励 - 更关注精确跟踪"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        sigma = getattr(self.cfg.rewards, 'tracking_sigma_stage1', 0.15) # Allow config override
        return torch.exp(-ang_vel_error / sigma)


    # --- Stage 1 Specific Success Criteria ---
    def compute_success_criteria(self):
        """计算每个环境是否在当前命令周期内达到了连续成功跟踪的条件"""
        # 1. 判断当前步骤是否在跟踪容差内
        # lin_vel_threshold = getattr(self.cfg.curriculum, 'stage1_success_lin_tol', 0.3)
        # ang_vel_threshold = getattr(self.cfg.curriculum, 'stage1_success_ang_tol', 0.2)
        #
        # lin_vel_close = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
        # ang_vel_close = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]) < ang_vel_threshold
        lin_vel_threshold = 0.3 # m/s error tolerance
        ang_vel_threshold = 0.2 # rad/s error tolerance

        lin_vel_close = torch.norm(self.commands[:, :2] - self.base_lin_vel[:, :2], dim=1) < lin_vel_threshold
        ang_vel_close = torch.abs(self.commands[:, 2] - self.base_ang_vel[:, 2]) < ang_vel_threshold

        current_step_tracking = lin_vel_close & ang_vel_close

        # 2. 更新连续跟踪计数器
        # 仅当当前步骤在跟踪时才增加计数，否则重置
        self.successful_command_tracking_streak = (self.successful_command_tracking_streak + 1) * current_step_tracking

        # 3. 判断是否达到连续成功的步数阈值
        reached_threshold = self.successful_command_tracking_streak >= self.min_tracking_steps_for_success

        # 4. 检查是否是命令周期的最后一步 (近似)
        # 如果 episode_length_buf 能被 resampling_interval_steps 整除 (或余数很小)，认为是周期结束
        is_end_of_command_cycle = (self.episode_length_buf % self.resampling_interval_steps == (self.resampling_interval_steps - 1))

        # 5. 成功标志：当达到阈值 *并且* 处于命令周期结束时，才标记为成功
        #    这样可以确保在一个完整的命令周期内评估跟踪性能
        # success_flags = reached_threshold & is_end_of_command_cycle
        # --- 或者更简单的：只要达到阈值就认为是成功（允许在一个周期内多次成功？）---
        success_flags = reached_threshold

        # 如果一个环境刚刚成功，重置其计数器，以便它可以为下一个命令周期重新计数
        # self.successful_command_tracking_streak[success_flags] = 0 # Reset counter immediately on success?

        return success_flags


    def post_physics_step(self):
        """Stage 1 后处理: 计算成功标志并放入 extras"""
        # --- 1. 调用父类 (G1CurriculumBase) 的后处理 ---
        # 这会处理相位计算、命令重采样、调用 LeggedRobot 的后处理等
        super().post_physics_step()

        # --- 2. 计算本阶段的成功标准 ---
        # compute_success_criteria 更新了 streak 并返回了哪些环境达到了连续成功阈值
        success_flags = self.compute_success_criteria()

        # --- 3. 将成功标志放入 extras 字典供 Runner 使用 ---

        self.extras["success_flags"] = success_flags.clone() # Send a copy

        # --- 4. 重置成功计数器 (可选，取决于你的成功定义) ---
        # 如果成功是一次性事件（达到N次就算成功），则需要重置计数器
        # 如果 Runner 期望每一步都报告是否处于“成功状态”，则不需要在这里重置
        # 假设 Runner 会统计 True 的比例，我们在成功时重置计数器，以便开始下一个周期的计数
        if success_flags.any():
            self.successful_command_tracking_streak[success_flags] = 0

        # --- 5. 重置因环境 reset 而中断的计数器 ---
        # (父类的 post_physics_step 调用了 reset_idx，reset_idx 会重置 streak)
        # 所以这里不需要再次处理 reset_buf


    def reset_idx(self, env_ids):
         """ 重写 reset_idx 以重置 Stage 1 特定状态 """
         if len(env_ids) == 0: return

         # 调用父类 reset (重置机器人状态、命令、基础缓冲区等)
         super().reset_idx(env_ids)

         # 重置本阶段特定的状态
         if hasattr(self, 'successful_command_tracking_streak'): # Ensure buffer exists
              self.successful_command_tracking_streak[env_ids] = 0
         # print(f"  G1BasicLocomotion: Resetting success streak for {len(env_ids)} environments.")
