

# reward_scheduler.py
import numpy as np

class RewardScheduler:
    """奖励权重调度器，根据课程阶段动态调整奖励权重"""

    def __init__(self, cfg):
        self.cfg = cfg # Store the main config if needed later

        # 定义所有可能用到的奖励项及其在不同阶段的基础权重
        self.stage_rewards = {
            # ------------------------------------------------------------------
            # 阶段 1: 基础运动 (G1BasicLocomotion)
            # 基于 unitree_rl_gym/g1_config.py 的成功行走奖励配置
            # ------------------------------------------------------------------
            1: {
                # --- 来自 g1_config.py 的主要奖励项 ---
                "tracking_lin_vel": 1.0,    # 线性速度跟踪 (来自 g1_config)
                "tracking_ang_vel": 0.5,    # 角速度跟踪 (来自 g1_config)
                "lin_vel_z": -2.0,       # 惩罚 Z 轴线速度 (来自 g1_config & base)
                "ang_vel_xy": -0.05,     # 惩罚 XY 轴角速度 (来自 g1_config & base)
                "orientation": -1.0,     # 惩罚非水平姿态 (来自 g1_config, base 为 -0.0)
                "base_height": -10.0,    # 惩罚偏离目标高度 (来自 g1_config, base 为 -0.0)
                "dof_acc": -2.5e-7,    # 惩罚关节加速度 (来自 g1_config & base)
                "dof_vel": -1e-3,      # 惩罚关节速度 (来自 g1_config, base 为 -0.0)
                "action_rate": -0.01,    # 惩罚动作变化率 (来自 g1_config & base)
                "dof_pos_limits": -5.0,    # 惩罚接近关节极限 (来自 g1_config)
                "alive": 0.15,       # 存活奖励 (来自 g1_config)
                "hip_pos": -1.0,       # 惩罚髋关节偏离 (来自 g1_config)
                "contact_no_vel": -0.2,    # 惩罚接触时有速度 (来自 g1_config)
                "feet_swing_height": -20.0, # 惩罚摆动腿高度偏差 (来自 g1_config)
                "contact": 0.18,       # 奖励符合步态相位的接触 (来自 g1_config)

                # --- 来自 legged_robot_config.py (基础配置) 中, g1_config 未覆盖但可能仍有用的项 ---
                "torques": -1e-5,      # 轻微惩罚力矩 (来自 base, g1_config 未指定但通常保留)
                "collision": 0.0,        # 碰撞惩罚 (g1_config 设为 0.0, base 为 -1.0) -> 采用 g1_config 的值
                "feet_air_time": 0.0,    # 腾空时间奖励 (g1_config 设为 0.0, base 为 1.0) -> 采用 g1_config 的值

                # --- 在 g1_config 和 base 中都为 0 或不常用的项 ---
                "termination": 0.0,      # 终止奖励/惩罚 (通常由环境逻辑处理或设为0)
                "feet_stumble": 0.0,     # 绊倒惩罚 (来自 base, 为 0.0)
                "stand_still": 0.0,      # 静止惩罚 (来自 base, 为 0.0)
                "feet_contact_forces": 0.0,# 接触力惩罚 (base 中提及但无默认值，g1_config 未用)
            },

            # ------------------------------------------------------------------
            # 阶段 2: 厨房导航 (G1KitchenNavigation) - 示例权重
            # ------------------------------------------------------------------
            2: {
                # --- 基础运动奖励 (权重可能降低) ---
                "tracking_lin_vel": 0.5,
                "tracking_ang_vel": 0.3,
                "orientation": -1.0,
                "base_height": -20.0,
                "alive": 0.1,
                "action_rate": -0.01,
                "torques": -2e-5,
                "dof_pos_limits": -5.0,
                "collision": -2.0, # 增加对一般碰撞的惩罚
                # --- 导航特定奖励 ---
                "waypoint_distance": 1.5, # 奖励接近导航点
                "waypoint_reached": 10.0, # 到达导航点的大奖励
                "kitchen_collision": -8.0,# 惩罚与厨房环境碰撞 (需要新 reward func)
                "target_facing": 0.5,    # 奖励朝向目标 (需要新 reward func)
                # --- 可能不再需要的 Stage 1 奖励 ---
                "lin_vel_z": -1.0, # 可能减少
                "ang_vel_xy": -0.02,
                "dof_acc": 0.0,
                "dof_vel": 0.0,
                "hip_pos": -0.5,
                "contact_no_vel": -0.1,
                "feet_swing_height": -10.0,
                "contact": 0.0,
            },

            # ------------------------------------------------------------------
            # 阶段 3: 厨房交互 (G1KitchenInteraction) - 示例权重
            # ------------------------------------------------------------------
            3: {
                "alive": 0.1,
                "base_height": -10.0,
                "action_rate": -0.01,
                "torques": -1e-5,
                "collision": -1.0,
                 # --- 导航奖励 (进一步降低) ---
                "waypoint_distance": 0.5,
                "waypoint_reached": 5.0,
                "kitchen_collision": -5.0,
                # --- 交互特定奖励 ---
                "interaction_progress": 2.0, # 奖励交互任务进展 (需要新 reward func)
                "interaction_success": 15.0, # 成功交互的大奖励 (需要新 reward func)
                "arm_pose": -0.5,          # 惩罚手臂姿势偏差 (需要新 reward func)
                "end_effector_force": -0.1,# 惩罚末端执行器力过大 (需要新 reward func)
                # --- 可能不再需要的奖励 ---
                 "tracking_lin_vel": 0.0,
                 "tracking_ang_vel": 0.0,
                 "orientation": -0.5,
                 "dof_pos_limits": -2.0,
            },

            # ------------------------------------------------------------------
            # 阶段 4: 完整任务 (G1KitchenFullTask) - 示例权重
            # ------------------------------------------------------------------
            4: {
                "alive": 0.1,
                "base_height": -10.0,
                "action_rate": -0.01,
                "torques": -1e-5,
                "collision": -1.0,
                # --- 导航奖励 (可能很低) ---
                "waypoint_distance": 0.2,
                "waypoint_reached": 3.0,
                "kitchen_collision": -3.0,
                # --- 交互奖励 ---
                "interaction_progress": 1.0,
                "interaction_success": 8.0,
                 # --- 任务序列奖励 ---
                "step_progress": 2.0,       # 奖励任务步骤进展 (需要新 reward func)
                "grab_success": 5.0,        # 成功抓取的奖励 (需要新 reward func)
                "sequence_completion": 20.0 # 完成整个任务序列的大奖励 (需要新 reward func)
            }
        }

        # 子阶段调整因子 (一个字典，键是阶段号，值是另一个字典 {reward_name: factor_func})
        self.sub_stage_factors = {}
        # 设置默认的调整函数
        self.setup_default_adjustments() # Keep this to allow sub-stage tuning later

    def get_reward_scales(self, stage, sub_stage):
        """获取特定课程阶段和子阶段的最终奖励权重字典"""
        # 1. 获取主阶段的基础奖励配置
        if stage in self.stage_rewards:
            reward_scales = self.stage_rewards[stage].copy()
        else:
            print(f"⚠️ RewardScheduler Warning: No specific rewards found for stage {stage}. Using stage 1 as default.")
            # Fallback to Stage 1 rewards if current stage is not defined
            reward_scales = self.stage_rewards.get(1, {}).copy()

        # 2. 应用子阶段调整因子
        if stage in self.sub_stage_factors:
            stage_factors = self.sub_stage_factors[stage]
            for reward_name, factor_func in stage_factors.items():
                if reward_name in reward_scales:
                    try:
                        factor = factor_func(sub_stage)
                        reward_scales[reward_name] *= factor
                    except Exception as e:
                        print(f"⚠️ RewardScheduler Error applying factor function for '{reward_name}' in stage {stage}: {e}")

        # 3. 返回最终的奖励权重字典
        # Filter out rewards with zero scale before returning? Optional.
        # final_scales = {k: v for k, v in reward_scales.items() if v != 0.0}
        # return final_scales
        return reward_scales # Return all defined scales for clarity

    def add_sub_stage_adjustment(self, stage, reward_name, factor_func):
        """为特定阶段的某个奖励项添加子阶段调整函数。"""
        if stage not in self.sub_stage_factors:
            self.sub_stage_factors[stage] = {}
        self.sub_stage_factors[stage][reward_name] = factor_func

    def setup_default_adjustments(self):
        """设置一些默认的子阶段调整示例 (可以保持用于后续阶段)"""
        # 阶段 1: 可以暂时不加调整，或轻微调整
        self.add_sub_stage_adjustment(1, "tracking_lin_vel", lambda s: 1.0 + 0.05 * (s - 1)) # 稍微增加跟踪权重
        self.add_sub_stage_adjustment(1, "tracking_ang_vel", lambda s: 1.0 + 0.05 * (s - 1))

        # 阶段 2 调整 (保持示例)
        self.add_sub_stage_adjustment(2, "kitchen_collision", lambda s: 1.0 + 0.3 * (s - 1))
        self.add_sub_stage_adjustment(2, "waypoint_distance", lambda s: max(0.1, 1.0 - 0.15 * (s - 1)))

        # 阶段 3 调整 (保持示例)
        self.add_sub_stage_adjustment(3, "interaction_success", lambda s: 1.0 + 0.3 * (s - 1))
        self.add_sub_stage_adjustment(3, "interaction_progress", lambda s: 1.0 + 0.2 * (s-1))

        # 阶段 4 调整 (保持示例)
        self.add_sub_stage_adjustment(4, "sequence_completion", lambda s: 1.0 + 0.4 * (s - 1))

        print("  RewardScheduler: Default sub-stage adjustments configured.")