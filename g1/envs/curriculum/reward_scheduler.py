

# # reward_scheduler.py
# import numpy as np

# class RewardScheduler:
#     """奖励权重调度器，根据课程阶段动态调整奖励权重"""

#     def __init__(self, cfg):
#         self.cfg = cfg # Store the main config if needed later

#         # 定义所有可能用到的奖励项及其在不同阶段的基础权重
#         self.stage_rewards = {
#             # ------------------------------------------------------------------
#             # 阶段 1: 基础运动 (G1BasicLocomotion)
#             # 基于 unitree_rl_gym/g1_config.py 的成功行走奖励配置
#             # ------------------------------------------------------------------
#             1: {
#                 # --- 来自 g1_config.py 的主要奖励项 ---
#                 "tracking_lin_vel": 1.0,    # 线性速度跟踪 (来自 g1_config)
#                 "tracking_ang_vel": 0.5,    # 角速度跟踪 (来自 g1_config)
#                 "lin_vel_z": -2.0,       # 惩罚 Z 轴线速度 (来自 g1_config & base)
#                 "ang_vel_xy": -0.05,     # 惩罚 XY 轴角速度 (来自 g1_config & base)
#                 "orientation": -1.0,     # 惩罚非水平姿态 (来自 g1_config, base 为 -0.0)
#                 "base_height": -10.0,    # 惩罚偏离目标高度 (来自 g1_config, base 为 -0.0)
#                 "dof_acc": -2.5e-7,    # 惩罚关节加速度 (来自 g1_config & base)
#                 "dof_vel": -1e-3,      # 惩罚关节速度 (来自 g1_config, base 为 -0.0)
#                 "action_rate": -0.01,    # 惩罚动作变化率 (来自 g1_config & base)
#                 "dof_pos_limits": -5.0,    # 惩罚接近关节极限 (来自 g1_config)
#                 "alive": 0.15,       # 存活奖励 (来自 g1_config)
#                 "hip_pos": -1.0,       # 惩罚髋关节偏离 (来自 g1_config)
#                 "contact_no_vel": -0.2,    # 惩罚接触时有速度 (来自 g1_config)
#                 "feet_swing_height": -20.0, # 惩罚摆动腿高度偏差 (来自 g1_config)
#                 "contact": 0.18,       # 奖励符合步态相位的接触 (来自 g1_config)

#                 # --- 来自 legged_robot_config.py (基础配置) 中, g1_config 未覆盖但可能仍有用的项 ---
#                 "torques": -1e-5,      # 轻微惩罚力矩 (来自 base, g1_config 未指定但通常保留)
#                 "collision": 0.0,        # 碰撞惩罚 (g1_config 设为 0.0, base 为 -1.0) -> 采用 g1_config 的值
#                 "feet_air_time": 0.0,    # 腾空时间奖励 (g1_config 设为 0.0, base 为 1.0) -> 采用 g1_config 的值

#                 # --- 在 g1_config 和 base 中都为 0 或不常用的项 ---
#                 "termination": 0.0,      # 终止奖励/惩罚 (通常由环境逻辑处理或设为0)
#                 "feet_stumble": 0.0,     # 绊倒惩罚 (来自 base, 为 0.0)
#                 "stand_still": 0.0,      # 静止惩罚 (来自 base, 为 0.0)
#                 "feet_contact_forces": 0.0,# 接触力惩罚 (base 中提及但无默认值，g1_config 未用)
#             },

#             # ------------------------------------------------------------------
#             # 阶段 2: 厨房导航 (G1KitchenNavigation) - 示例权重
#             # ------------------------------------------------------------------
#             2: {
#                 # --- 基础运动奖励 (权重可能降低) ---
#                 "tracking_lin_vel": 0.5,
#                 "tracking_ang_vel": 0.3,
#                 "orientation": -1.0,
#                 "base_height": -20.0,
#                 "alive": 0.1,
#                 "action_rate": -0.01,
#                 "torques": -2e-5,
#                 "dof_pos_limits": -5.0,
#                 "collision": -2.0, # 增加对一般碰撞的惩罚
#                 # --- 导航特定奖励 ---
#                 "waypoint_distance": 1.5, # 奖励接近导航点
#                 "waypoint_reached": 10.0, # 到达导航点的大奖励
#                 "kitchen_collision": -8.0,# 惩罚与厨房环境碰撞 (需要新 reward func)
#                 "target_facing": 0.5,    # 奖励朝向目标 (需要新 reward func)
#                 # --- 可能不再需要的 Stage 1 奖励 ---
#                 "lin_vel_z": -1.0, # 可能减少
#                 "ang_vel_xy": -0.02,
#                 "dof_acc": 0.0,
#                 "dof_vel": 0.0,
#                 "hip_pos": -0.5,
#                 "contact_no_vel": -0.1,
#                 "feet_swing_height": -10.0,
#                 "contact": 0.0,
#             },

#             # ------------------------------------------------------------------
#             # 阶段 3: 厨房交互 (G1KitchenInteraction) - 示例权重
#             # ------------------------------------------------------------------
#             3: {
#                 "alive": 0.1,
#                 "base_height": -10.0,
#                 "action_rate": -0.01,
#                 "torques": -1e-5,
#                 "collision": -1.0,
#                  # --- 导航奖励 (进一步降低) ---
#                 "waypoint_distance": 0.5,
#                 "waypoint_reached": 5.0,
#                 "kitchen_collision": -5.0,
#                 # --- 交互特定奖励 ---
#                 "interaction_progress": 2.0, # 奖励交互任务进展 (需要新 reward func)
#                 "interaction_success": 15.0, # 成功交互的大奖励 (需要新 reward func)
#                 "arm_pose": -0.5,          # 惩罚手臂姿势偏差 (需要新 reward func)
#                 "end_effector_force": -0.1,# 惩罚末端执行器力过大 (需要新 reward func)
#                 # --- 可能不再需要的奖励 ---
#                  "tracking_lin_vel": 0.0,
#                  "tracking_ang_vel": 0.0,
#                  "orientation": -0.5,
#                  "dof_pos_limits": -2.0,
#             },

#             # ------------------------------------------------------------------
#             # 阶段 4: 完整任务 (G1KitchenFullTask) - 示例权重
#             # ------------------------------------------------------------------
#             4: {
#                 "alive": 0.1,
#                 "base_height": -10.0,
#                 "action_rate": -0.01,
#                 "torques": -1e-5,
#                 "collision": -1.0,
#                 # --- 导航奖励 (可能很低) ---
#                 "waypoint_distance": 0.2,
#                 "waypoint_reached": 3.0,
#                 "kitchen_collision": -3.0,
#                 # --- 交互奖励 ---
#                 "interaction_progress": 1.0,
#                 "interaction_success": 8.0,
#                  # --- 任务序列奖励 ---
#                 "step_progress": 2.0,       # 奖励任务步骤进展 (需要新 reward func)
#                 "grab_success": 5.0,        # 成功抓取的奖励 (需要新 reward func)
#                 "sequence_completion": 20.0 # 完成整个任务序列的大奖励 (需要新 reward func)
#             }
#         }

#         # 子阶段调整因子 (一个字典，键是阶段号，值是另一个字典 {reward_name: factor_func})
#         self.sub_stage_factors = {}
#         # 设置默认的调整函数
#         self.setup_default_adjustments() # Keep this to allow sub-stage tuning later

#     def get_reward_scales(self, stage, sub_stage):
#         """获取特定课程阶段和子阶段的最终奖励权重字典"""
#         # 1. 获取主阶段的基础奖励配置
#         if stage in self.stage_rewards:
#             reward_scales = self.stage_rewards[stage].copy()
#         else:
#             print(f"⚠️ RewardScheduler Warning: No specific rewards found for stage {stage}. Using stage 1 as default.")
#             # Fallback to Stage 1 rewards if current stage is not defined
#             reward_scales = self.stage_rewards.get(1, {}).copy()

#         # 2. 应用子阶段调整因子
#         if stage in self.sub_stage_factors:
#             stage_factors = self.sub_stage_factors[stage]
#             for reward_name, factor_func in stage_factors.items():
#                 if reward_name in reward_scales:
#                     try:
#                         factor = factor_func(sub_stage)
#                         reward_scales[reward_name] *= factor
#                     except Exception as e:
#                         print(f"⚠️ RewardScheduler Error applying factor function for '{reward_name}' in stage {stage}: {e}")

#         # 3. 返回最终的奖励权重字典
#         # Filter out rewards with zero scale before returning? Optional.
#         # final_scales = {k: v for k, v in reward_scales.items() if v != 0.0}
#         # return final_scales
#         return reward_scales # Return all defined scales for clarity

#     def add_sub_stage_adjustment(self, stage, reward_name, factor_func):
#         """为特定阶段的某个奖励项添加子阶段调整函数。"""
#         if stage not in self.sub_stage_factors:
#             self.sub_stage_factors[stage] = {}
#         self.sub_stage_factors[stage][reward_name] = factor_func

#     def setup_default_adjustments(self):
#         """设置一些默认的子阶段调整示例 (可以保持用于后续阶段)"""
#         # 阶段 1: 可以暂时不加调整，或轻微调整
#         self.add_sub_stage_adjustment(1, "tracking_lin_vel", lambda s: 1.0 + 0.05 * (s - 1)) # 稍微增加跟踪权重
#         self.add_sub_stage_adjustment(1, "tracking_ang_vel", lambda s: 1.0 + 0.05 * (s - 1))

#         # 阶段 2 调整 (保持示例)
#         self.add_sub_stage_adjustment(2, "kitchen_collision", lambda s: 1.0 + 0.3 * (s - 1))
#         self.add_sub_stage_adjustment(2, "waypoint_distance", lambda s: max(0.1, 1.0 - 0.15 * (s - 1)))

#         # 阶段 3 调整 (保持示例)
#         self.add_sub_stage_adjustment(3, "interaction_success", lambda s: 1.0 + 0.3 * (s - 1))
#         self.add_sub_stage_adjustment(3, "interaction_progress", lambda s: 1.0 + 0.2 * (s-1))

#         # 阶段 4 调整 (保持示例)
#         self.add_sub_stage_adjustment(4, "sequence_completion", lambda s: 1.0 + 0.4 * (s - 1))

#         print("  RewardScheduler: Default sub-stage adjustments configured.")



# import numpy as np

# class RewardScheduler:
#     """奖励权重调度器，根据课程阶段动态调整奖励权重"""

#     def __init__(self, cfg):
#         self.cfg = cfg # Store the main config if needed later

#         # 定义所有可能用到的奖励项及其在不同 *主阶段* 的 *基础* 权重
#         # 注意：Stage 1 的权重应该反映 *最终子阶段* (全自由度) 的目标
#         self.stage_rewards = {
#             # ------------------------------------------------------------------
#             # 阶段 1: 基础运动 (G1BasicLocomotion) - 最终子阶段 (1.5) 的目标权重
#             # ------------------------------------------------------------------
#             1: {
#                 # --- 主要运动奖励 ---
#                 "tracking_lin_vel": 1.5,    # 最终阶段提高跟踪权重
#                 "tracking_ang_vel": 1.0,    # 最终阶段提高跟踪权重
#                 "orientation": -1.5,        # 对姿态要求更高
#                 "base_height": -15.0,       # 对高度要求更高
#                 "alive": 0.2,               # 存活奖励可以稍微提高

#                 # --- 运动平稳性与效率 ---
#                 "lin_vel_z": -2.0,
#                 "ang_vel_xy": -0.1,
#                 "dof_acc": -2.0e-7, # 稍微降低加速度惩罚，允许更动态的动作
#                 "dof_vel": -5e-4,   # 稍微降低速度惩罚
#                 "action_rate": -0.02, # 可以稍微增加动作平滑性惩罚
#                 "torques": -5e-6,   # 轻微惩罚力矩

#                 # --- 关节与接触 ---
#                 "dof_pos_limits": -8.0,    # 增加极限惩罚
#                 "collision": -2.0,       # 基础碰撞惩罚 (KitchenNav 会覆盖)
#                 "hip_pos": -1.0,       # 保持髋关节惩罚
#                 "contact_no_vel": -0.3,    # 增加接触速度惩罚
#                 "feet_swing_height": -15.0, # 保持摆动腿高度惩罚
#                 "contact": 0.15,       # 保持接触相位奖励

#                 # --- 手臂/手部相关 (最终阶段需要考虑) ---
#                 "arm_pose_penalty": -0.5, # 惩罚手臂姿势异常 (需要新 reward func)
#                 "hand_pose_penalty": -0.2, # 惩罚手部姿势异常 (需要新 reward func)

#                 # --- 其他可能为 0 的项 ---
#                 "termination": 0.0, "feet_air_time": 0.0, "feet_stumble": 0.0,
#                 "stand_still": 0.0, "feet_contact_forces": 0.0,
#             },

#             # ------------------------------------------------------------------
#             # 阶段 2: 厨房导航 (G1KitchenNavigation)
#             # ------------------------------------------------------------------
#             2: {
#                 # --- 基础运动奖励 (权重降低) ---
#                 "tracking_lin_vel": 0.5, "tracking_ang_vel": 0.3,
#                 "orientation": -1.0, "base_height": -10.0, "alive": 0.1,
#                 "action_rate": -0.01, "torques": -2e-5, "dof_pos_limits": -5.0,
#                 "collision": -3.0, # 轻微增加一般碰撞惩罚

#                 # --- 导航特定奖励 ---
#                 "waypoint_distance": 2.0, # 奖励接近导航点
#                 "waypoint_reached": 15.0, # 到达导航点的大奖励
#                 "kitchen_collision": -10.0,# 严厉惩罚与厨房环境碰撞
#                 "target_facing": 0.8,    # 奖励朝向目标

#                 # --- 降低或移除的 Stage 1 奖励 ---
#                 "lin_vel_z": -0.5, "ang_vel_xy": -0.01, "dof_acc": 0.0,
#                 "dof_vel": 0.0, "hip_pos": -0.2, "contact_no_vel": 0.0,
#                 "feet_swing_height": 0.0, "contact": 0.0,
#                 "arm_pose_penalty": -0.2, "hand_pose_penalty": 0.0, # 导航时不太关注手
#             },

#             # ------------------------------------------------------------------
#             # 阶段 3: 厨房交互 (G1KitchenInteraction)
#             # ------------------------------------------------------------------
#             3: {
#                 "alive": 0.1, "base_height": -5.0, "action_rate": -0.01,
#                 "torques": -1e-5, "collision": -2.0,

#                 # --- 导航奖励 (进一步降低) ---
#                 "waypoint_distance": 0.8, "waypoint_reached": 8.0,
#                 "kitchen_collision": -6.0,

#                 # --- 交互特定奖励 ---
#                 "interaction_progress": 3.0, # 奖励交互任务进展
#                 "interaction_success": 20.0, # 成功交互的大奖励
#                 "arm_pose": -1.0,          # 惩罚手臂姿势偏差以精确交互
#                 "end_effector_force": -0.5,# 惩罚末端执行器力过大

#                 # --- 移除或设为 0 的奖励 ---
#                 "tracking_lin_vel": 0.0, "tracking_ang_vel": 0.0, "orientation": -0.2,
#                  "dof_pos_limits": -1.0, "target_facing": 0.0,
#             },

#             # ------------------------------------------------------------------
#             # 阶段 4: 完整任务 (G1KitchenFullTask)
#             # ------------------------------------------------------------------
#             4: {
#                 "alive": 0.1, "base_height": -5.0, "action_rate": -0.01,
#                 "torques": -1e-5, "collision": -1.0,

#                 # --- 导航奖励 (较低) ---
#                 "waypoint_distance": 0.5, "waypoint_reached": 5.0,
#                 "kitchen_collision": -4.0,

#                 # --- 交互奖励 (权重适中) ---
#                 "interaction_progress": 1.5, "interaction_success": 10.0,
#                 "arm_pose": -0.5, "end_effector_force": -0.2,

#                 # --- 任务序列奖励 ---
#                 "step_progress": 2.5,       # 奖励任务步骤进展
#                 "grab_success": 8.0,        # 成功抓取的奖励
#                 "sequence_completion": 25.0 # 完成整个任务序列的大奖励
#             }
#         }

#         # 子阶段调整因子 (一个字典，键是阶段号，值是另一个字典 {reward_name: factor_func})
#         self.sub_stage_factors = {}
#         # 为 Stage 1 设置嵌套课程的奖励调整
#         self.setup_stage1_adjustments()
#         # 设置其他阶段的默认调整 (可以保持用于后续阶段)
#         self.setup_default_adjustments()

#     def get_reward_scales(self, stage, sub_stage):
#         """获取特定课程阶段和子阶段的最终奖励权重字典"""
#         # 1. 获取主阶段的基础奖励配置
#         if stage in self.stage_rewards:
#             reward_scales = self.stage_rewards[stage].copy()
#         else:
#             print(f"⚠️ RewardScheduler Warning: No specific rewards found for stage {stage}. Using stage 1 as default.")
#             reward_scales = self.stage_rewards.get(1, {}).copy() # Fallback to Stage 1

#         # 2. 应用子阶段调整因子
#         if stage in self.sub_stage_factors:
#             stage_factors = self.sub_stage_factors[stage]
#             for reward_name, factor_func in stage_factors.items():
#                 if reward_name in reward_scales:
#                     try:
#                         # 使用 lambda s, max_s: ... 的形式，传入当前和最大子阶段数
#                         max_s = self.cfg.get_max_sub_stages_for_stage(stage) if hasattr(self.cfg, 'get_max_sub_stages_for_stage') else 5
#                         factor = factor_func(sub_stage, max_s) # 传入子阶段号和最大子阶段数
#                         # 应用因子，确保奖励不为负（除非有意为之）
#                         # reward_scales[reward_name] = max(0.0, reward_scales[reward_name] * factor) if reward_scales[reward_name] > 0 else reward_scales[reward_name] * factor
#                         reward_scales[reward_name] *= factor # 直接乘以因子，允许负奖励被调整
#                     except Exception as e:
#                         print(f"⚠️ RewardScheduler Error applying factor function for '{reward_name}' in stage {stage}: {e}")

#         # 3. 返回最终的奖励权重字典
#         return reward_scales

#     def add_sub_stage_adjustment(self, stage, reward_name, factor_func):
#         """为特定阶段的某个奖励项添加子阶段调整函数。
#            factor_func 应该接受两个参数: (current_sub_stage, max_sub_stages)
#         """
#         if stage not in self.sub_stage_factors:
#             self.sub_stage_factors[stage] = {}
#         self.sub_stage_factors[stage][reward_name] = factor_func

#     def setup_stage1_adjustments(self):
#         """为 Stage 1 的嵌套课程设置奖励调整"""
#         print("  RewardScheduler: Setting up adjustments for Stage 1 nested curriculum...")

#         # 示例：线性插值函数 (从 value_start 线性过渡到 1.0)
#         def linear_scale(s, max_s, value_start=0.1):
#             if max_s <= 1: return 1.0
#             return np.clip(value_start + (1.0 - value_start) * (s - 1) / (max_s - 1), value_start, 1.0)

#         # 示例：反向线性插值函数 (从 1.0 线性过渡到 value_end)
#         def inv_linear_scale(s, max_s, value_end=0.1):
#              if max_s <= 1: return 1.0
#              return np.clip(1.0 - (1.0 - value_end) * (s - 1) / (max_s - 1), value_end, 1.0)

#         # --- 调整 Stage 1 的奖励 ---
#         # 跟踪奖励：从较低权重开始，逐步增加到基础权重 (1.0 对应 stage_rewards[1] 的值)
#         self.add_sub_stage_adjustment(1, "tracking_lin_vel", lambda s, max_s: linear_scale(s, max_s, 0.2))
#         self.add_sub_stage_adjustment(1, "tracking_ang_vel", lambda s, max_s: linear_scale(s, max_s, 0.2))

#         # 姿态/高度惩罚：可以从较低惩罚开始，逐步增加
#         self.add_sub_stage_adjustment(1, "orientation", lambda s, max_s: linear_scale(s, max_s, 0.3))
#         self.add_sub_stage_adjustment(1, "base_height", lambda s, max_s: linear_scale(s, max_s, 0.2))

#         # 运动平滑性惩罚：可以从较低惩罚开始
#         self.add_sub_stage_adjustment(1, "lin_vel_z", lambda s, max_s: linear_scale(s, max_s, 0.4))
#         self.add_sub_stage_adjustment(1, "ang_vel_xy", lambda s, max_s: linear_scale(s, max_s, 0.4))
#         self.add_sub_stage_adjustment(1, "dof_acc", lambda s, max_s: linear_scale(s, max_s, 0.3))
#         self.add_sub_stage_adjustment(1, "dof_vel", lambda s, max_s: linear_scale(s, max_s, 0.3))
#         self.add_sub_stage_adjustment(1, "action_rate", lambda s, max_s: linear_scale(s, max_s, 0.5))
#         self.add_sub_stage_adjustment(1, "torques", lambda s, max_s: linear_scale(s, max_s, 0.5))

#         # 关节极限惩罚：逐步增加
#         self.add_sub_stage_adjustment(1, "dof_pos_limits", lambda s, max_s: linear_scale(s, max_s, 0.2))

#         # 接触相关奖励/惩罚：逐步引入或加强
#         self.add_sub_stage_adjustment(1, "collision", lambda s, max_s: linear_scale(s, max_s, 0.1))
#         self.add_sub_stage_adjustment(1, "hip_pos", lambda s, max_s: linear_scale(s, max_s, 0.3))
#         self.add_sub_stage_adjustment(1, "contact_no_vel", lambda s, max_s: linear_scale(s, max_s, 0.2))
#         self.add_sub_stage_adjustment(1, "feet_swing_height", lambda s, max_s: linear_scale(s, max_s, 0.1))
#         self.add_sub_stage_adjustment(1, "contact", lambda s, max_s: linear_scale(s, max_s, 0.5)) # 相位奖励可以晚点引入

#         # 手臂/手部惩罚：只在激活后引入并逐步加强
#         self.add_sub_stage_adjustment(1, "arm_pose_penalty", lambda s, max_s: linear_scale(s, max_s, 0.0) if s >= 3 else 0.0) # 从子阶段3开始
#         self.add_sub_stage_adjustment(1, "hand_pose_penalty", lambda s, max_s: linear_scale(s, max_s, 0.0) if s >= 4 else 0.0) # 从子阶段4开始

#         # 存活奖励：保持稳定
#         self.add_sub_stage_adjustment(1, "alive", lambda s, max_s: 1.0)


#     def setup_default_adjustments(self):
#         """设置其他阶段的默认子阶段调整示例"""
#         print("  RewardScheduler: Setting up default adjustments for Stages 2, 3, 4...")

#         # 示例：线性插值函数
#         def linear_scale(s, max_s, value_start=0.1):
#             if max_s <= 1: return 1.0
#             return np.clip(value_start + (1.0 - value_start) * (s - 1) / (max_s - 1), value_start, 1.0)

#         # 阶段 2 (导航) 调整
#         self.add_sub_stage_adjustment(2, "kitchen_collision", lambda s, max_s: linear_scale(s, max_s, 0.5)) # 逐步增加厨房碰撞惩罚
#         self.add_sub_stage_adjustment(2, "waypoint_distance", lambda s, max_s: linear_scale(s, max_s, 0.7)) # 逐步增加距离奖励权重
#         self.add_sub_stage_adjustment(2, "waypoint_reached", lambda s, max_s: linear_scale(s, max_s, 0.5)) # 逐步增加到达奖励

#         # 阶段 3 (交互) 调整
#         self.add_sub_stage_adjustment(3, "interaction_success", lambda s, max_s: linear_scale(s, max_s, 0.4)) # 逐步增加成功奖励
#         self.add_sub_stage_adjustment(3, "interaction_progress", lambda s, max_s: linear_scale(s, max_s, 0.6))
#         self.add_sub_stage_adjustment(3, "arm_pose", lambda s, max_s: linear_scale(s, max_s, 0.5)) # 逐步增加手臂姿态惩罚

#         # 阶段 4 (完整任务) 调整
#         self.add_sub_stage_adjustment(4, "sequence_completion", lambda s, max_s: linear_scale(s, max_s, 0.3)) # 逐步增加序列完成奖励
#         self.add_sub_stage_adjustment(4, "step_progress", lambda s, max_s: linear_scale(s, max_s, 0.7))

#         print("  RewardScheduler: Default sub-stage adjustments configured.")

import numpy as np
from g1.utils.helpers import DotDict # Import DotDict if needed

class RewardScheduler:
    """奖励权重调度器，根据课程阶段动态调整奖励权重"""

    def __init__(self, cfg):
        # cfg should be the main curriculum config (DotDict) containing stages and methods
        self.cfg = cfg
        print(f"  RewardScheduler initialized with cfg type: {type(cfg)}")
        # Verify the passed cfg has the expected method
        if not hasattr(cfg, 'get_max_sub_stages_for_stage'):
             print("⚠️ RewardScheduler Warning: Provided cfg object does not have 'get_max_sub_stages_for_stage' method. Sub-stage scaling might use default max.")


        # 定义所有可能用到的奖励项及其在不同 *主阶段* 的 *基础* 权重
        # (Stage 1 weights reflect the *final* sub-stage goals)
        self.stage_rewards = {
            # ... (Keep the detailed reward definitions from the previous response) ...
            # ------------------------------------------------------------------
            # 阶段 1: 基础运动 (G1BasicLocomotion) - 最终子阶段 (1.5) 的目标权重
            # ------------------------------------------------------------------
            1: {
                "tracking_lin_vel": 1.5, "tracking_ang_vel": 1.0, "orientation": -1.5,
                "base_height": -15.0, "alive": 0.2, "lin_vel_z": -2.0, "ang_vel_xy": -0.1,
                "dof_acc": -2.0e-7, "dof_vel": -5e-4, "action_rate": -0.02, "torques": -5e-6,
                "dof_pos_limits": -8.0, "collision": -2.0, "hip_pos": -1.0,
                "contact_no_vel": -0.3, "feet_swing_height": -15.0, "contact": 0.15,
                "arm_pose_penalty": -0.5, "hand_pose_penalty": -0.2, # Added for final stage
                "termination": 0.0, "feet_air_time": 0.0, "feet_stumble": 0.0,
                "stand_still": 0.0, "feet_contact_forces": 0.0,
            },
             # Stages 2, 3, 4 definitions... (keep as before)
            2: {
                "tracking_lin_vel": 0.5, "tracking_ang_vel": 0.3, "orientation": -1.0,
                "base_height": -10.0, "alive": 0.1, "action_rate": -0.01, "torques": -2e-5,
                "dof_pos_limits": -5.0, "collision": -3.0, "waypoint_distance": 2.0,
                "waypoint_reached": 15.0, "kitchen_collision": -10.0, "target_facing": 0.8,
                "lin_vel_z": -0.5, "ang_vel_xy": -0.01, "dof_acc": 0.0, "dof_vel": 0.0,
                "hip_pos": -0.2, "contact_no_vel": 0.0, "feet_swing_height": 0.0, "contact": 0.0,
                "arm_pose_penalty": -0.2, "hand_pose_penalty": 0.0,
            },
            3: {
                "alive": 0.1, "base_height": -5.0, "action_rate": -0.01, "torques": -1e-5,
                "collision": -2.0, "waypoint_distance": 0.8, "waypoint_reached": 8.0,
                "kitchen_collision": -6.0, "interaction_progress": 3.0, "interaction_success": 20.0,
                "arm_pose": -1.0, "end_effector_force": -0.5, "tracking_lin_vel": 0.0,
                "tracking_ang_vel": 0.0, "orientation": -0.2, "dof_pos_limits": -1.0, "target_facing": 0.0,
            },
            4: {
                "alive": 0.1, "base_height": -5.0, "action_rate": -0.01, "torques": -1e-5,
                "collision": -1.0, "waypoint_distance": 0.5, "waypoint_reached": 5.0,
                "kitchen_collision": -4.0, "interaction_progress": 1.5, "interaction_success": 10.0,
                "arm_pose": -0.5, "end_effector_force": -0.2, "step_progress": 2.5,
                "grab_success": 8.0, "sequence_completion": 25.0
            }
        }

        # Sub-stage adjustment factors dictionary
        self.sub_stage_factors = {}
        # Setup adjustments (ensure these are called)
        self.setup_stage1_adjustments()
        self.setup_default_adjustments() # For stages 2, 3, 4

    def get_reward_scales(self, stage, sub_stage):
        """获取特定课程阶段和子阶段的最终奖励权重字典"""
        # 1. 获取主阶段的基础奖励配置
        if stage in self.stage_rewards:
            reward_scales = self.stage_rewards[stage].copy()
        else:
            print(f"⚠️ RewardScheduler Warning: No specific rewards found for stage {stage}. Using stage 1 as default.")
            reward_scales = self.stage_rewards.get(1, {}).copy()

        # 2. 应用子阶段调整因子
        if stage in self.sub_stage_factors:
            stage_factors = self.sub_stage_factors[stage]
            # --- Get max_s for this stage ONCE ---
            max_s = 5 # Default max sub-stages
            if hasattr(self.cfg, 'get_max_sub_stages_for_stage'):
                 try:
                     max_s = self.cfg.get_max_sub_stages_for_stage(stage)
                 except Exception as e_cfg:
                      print(f"⚠️ Error calling get_max_sub_stages_for_stage for stage {stage}: {e_cfg}")
            # else: print(f"⚠️ self.cfg missing get_max_sub_stages_for_stage method.")

            for reward_name, factor_func in stage_factors.items():
                if reward_name in reward_scales:
                    try:
                        # --- Pass max_s explicitly to the lambda function ---
                        factor = factor_func(sub_stage, max_s)
                        reward_scales[reward_name] *= factor
                    except TypeError as e_type:
                         # Catch if the lambda wasn't defined correctly or factor_func is None
                         print(f"❌ RewardScheduler TypeError applying factor for '{reward_name}' (Stage {stage}.{sub_stage}): {e_type}. Is factor_func defined correctly?")
                         # traceback.print_exc() # Uncomment for full trace
                    except Exception as e:
                        print(f"⚠️ RewardScheduler Error applying factor function for '{reward_name}' in stage {stage}: {e}")

        # 3. 返回最终的奖励权重字典
        return reward_scales

    def add_sub_stage_adjustment(self, stage, reward_name, factor_func):
        """为特定阶段的某个奖励项添加子阶段调整函数。
           factor_func 应该接受两个参数: (current_sub_stage, max_sub_stages)
        """
        if stage not in self.sub_stage_factors:
            self.sub_stage_factors[stage] = {}
        # Store the lambda function directly
        self.sub_stage_factors[stage][reward_name] = factor_func

    def setup_stage1_adjustments(self):
        """为 Stage 1 的嵌套课程设置奖励调整"""
        print("  RewardScheduler: Setting up adjustments for Stage 1 nested curriculum...")
        stage = 1 # Define stage number for clarity

        # Define scaling functions (accept s and max_s)
        def linear_scale(s, max_s, value_start=0.1):
            if max_s <= 1: return 1.0
            # Ensure progress calculation doesn't divide by zero
            progress = (s - 1) / max(1, max_s - 1)
            return np.clip(value_start + (1.0 - value_start) * progress, value_start, 1.0)

        def inv_linear_scale(s, max_s, value_end=0.1):
             if max_s <= 1: return 1.0
             progress = (s - 1) / max(1, max_s - 1)
             return np.clip(1.0 - (1.0 - value_end) * progress, value_end, 1.0)

        # Helper to add adjustments for stage 1
        def add_adj(reward_name, func):
            self.add_sub_stage_adjustment(stage, reward_name, func)

        # Apply adjustments using the helper and scaling functions
        add_adj("tracking_lin_vel", lambda s, max_s: linear_scale(s, max_s, 0.2))
        add_adj("tracking_ang_vel", lambda s, max_s: linear_scale(s, max_s, 0.2))
        add_adj("orientation", lambda s, max_s: linear_scale(s, max_s, 0.3))
        add_adj("base_height", lambda s, max_s: linear_scale(s, max_s, 0.2))
        add_adj("lin_vel_z", lambda s, max_s: linear_scale(s, max_s, 0.4))
        add_adj("ang_vel_xy", lambda s, max_s: linear_scale(s, max_s, 0.4))
        add_adj("dof_acc", lambda s, max_s: linear_scale(s, max_s, 0.3))
        add_adj("dof_vel", lambda s, max_s: linear_scale(s, max_s, 0.3))
        add_adj("action_rate", lambda s, max_s: linear_scale(s, max_s, 0.5))
        add_adj("torques", lambda s, max_s: linear_scale(s, max_s, 0.5))
        add_adj("dof_pos_limits", lambda s, max_s: linear_scale(s, max_s, 0.2))
        add_adj("collision", lambda s, max_s: linear_scale(s, max_s, 0.1))
        add_adj("hip_pos", lambda s, max_s: linear_scale(s, max_s, 0.3))
        add_adj("contact_no_vel", lambda s, max_s: linear_scale(s, max_s, 0.2))
        add_adj("feet_swing_height", lambda s, max_s: linear_scale(s, max_s, 0.1))
        add_adj("contact", lambda s, max_s: linear_scale(s, max_s, 0.5))
        # Arm/Hand penalties activate later
        add_adj("arm_pose_penalty", lambda s, max_s: linear_scale(s, max_s, 0.0) if s >= 3 else 0.0)
        add_adj("hand_pose_penalty", lambda s, max_s: linear_scale(s, max_s, 0.0) if s >= 4 else 0.0)
        add_adj("alive", lambda s, max_s: 1.0) # Keep alive constant


    def setup_default_adjustments(self):
        """设置其他阶段的默认子阶段调整示例"""
        print("  RewardScheduler: Setting up default adjustments for Stages 2, 3, 4...")
        # Define scaling functions again or ensure they are accessible
        def linear_scale(s, max_s, value_start=0.1):
            if max_s <= 1: return 1.0
            progress = (s - 1) / max(1, max_s - 1)
            return np.clip(value_start + (1.0 - value_start) * progress, value_start, 1.0)

        # Stage 2 (Nav) Adjustments
        self.add_sub_stage_adjustment(2, "kitchen_collision", lambda s, max_s: linear_scale(s, max_s, 0.5))
        self.add_sub_stage_adjustment(2, "waypoint_distance", lambda s, max_s: linear_scale(s, max_s, 0.7))
        self.add_sub_stage_adjustment(2, "waypoint_reached", lambda s, max_s: linear_scale(s, max_s, 0.5))

        # Stage 3 (Interact) Adjustments
        self.add_sub_stage_adjustment(3, "interaction_success", lambda s, max_s: linear_scale(s, max_s, 0.4))
        self.add_sub_stage_adjustment(3, "interaction_progress", lambda s, max_s: linear_scale(s, max_s, 0.6))
        self.add_sub_stage_adjustment(3, "arm_pose", lambda s, max_s: linear_scale(s, max_s, 0.5))

        # Stage 4 (Full Task) Adjustments
        self.add_sub_stage_adjustment(4, "sequence_completion", lambda s, max_s: linear_scale(s, max_s, 0.3))
        self.add_sub_stage_adjustment(4, "step_progress", lambda s, max_s: linear_scale(s, max_s, 0.7))

        print("  RewardScheduler: Default sub-stage adjustments configured.")