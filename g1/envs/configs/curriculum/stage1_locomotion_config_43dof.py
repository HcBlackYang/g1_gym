# stage1_locomotion_config_43dof.py
import os
from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Stage1LocomotionConfig43DoF(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        print("--- Loading Stage1LocomotionConfig43DoF (Direct Full Body Training, Reduced Collision) ---")

        # --- 环境配置 ---
        self.env.num_envs = 4096
        self.env.test = False
        self.env.episode_length_s = 20
        self.env.env_spacing = 3.0
        self.env.send_timeouts = True
        # Dimensions for FULL 43 DoF robot
        self.env.num_observations = 140 # 11 base + 3 * 43 actions = 140
        self.env.num_privileged_obs = 143 # 140 + 3 base_lin_vel = 143
        self.env.num_actions = 43      # Full 43 DoF

        # --- 领域随机化 ---
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.1, 1.25]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 3.0]
        self.domain_rand.push_robots = True
        self.domain_rand.push_interval_s = 7
        self.domain_rand.max_push_vel_xy = 1.0

        # --- 初始状态配置 ---
        self.init_state.pos = [0.0, 0.0, 0.8]
        # 使用包含所有 43 个关节的默认角度
        self.init_state.default_joint_angles = {
            # Legs (12)
            "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
            # Torso (3) - 假设腰部关节现在是 revolute
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            # Left Arm (7)
            "left_shoulder_pitch_joint": 0.5, "left_shoulder_roll_joint": 0.1, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
            # Right Arm (7)
            "right_shoulder_pitch_joint": 0.5, "right_shoulder_roll_joint": -0.1, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
            # Left Hand (7) - 使用 URDF 中定义的关节名
            "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0, "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0, "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
            # Right Hand (7) - 使用 URDF 中定义的关节名
            "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0, "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0, "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0, "right_hand_index_1_joint": 0.0,
        }

        # --- 奖励配置 ---
        self.rewards.base_height_target = 0.78
        self.rewards.soft_dof_pos_limit = 0.9
        self.rewards.only_positive_rewards = False
        self.rewards.tracking_sigma = 0.25 # Sigma for tracking rewards
        self.rewards.soft_torque_limit = 0.8 # Factor for torque limit reward
        # self.rewards.max_contact_force = 100.0 # Example if using feet_contact_forces reward

        # --- 定义奖励权重 (确保名称与 G1FullLocomotionEnv 中的 _reward_* 方法对应) ---
        class scales:
            # --- Locomotion ---
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.8
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            # --- Stability & Smoothness ---
            orientation = -1.0
            base_height = -15.0
            dof_acc = -3.0e-7
            action_rate = -0.02
            # --- Constraints ---
            dof_pos_limits = -10.0
            torque_limits = -0.01      # 对应 _reward_torque_limits
            collision = -1.0           # 对应 _reward_collision
            # --- Regularization / Behavior Shaping ---
            dof_vel = -0.002
            alive = 0.2                # 对应 _reward_alive
            feet_air_time = 1.0       # 对应 _reward_feet_air_time
            # --- Arm/Hand Penalties ---
            arm_dof_vel = -0.01
            arm_dof_acc = -1.0e-6
            hand_dof_vel = -0.005
            # arm_pose_penalty = -0.5 # 取消或设为 0，因为可能不需要
            # hand_pose_penalty = -0.2 # 取消或设为 0
            # --- G1 Specific/Optional ---
            # hip_pos = -1.0             # 对应 _reward_hip_pos (如果需要)
            contact_no_vel = -0.2    # 对应 _reward_contact_no_vel
            feet_swing_height = -15.0  # 对应 _reward_feet_swing_height
            contact = 0.15             # 对应 _reward_contact
            feet_stumble = -1.0        # 对应 _reward_feet_stumble (或_reward_stumble)
            stand_still = -0.1         # 对应 _reward_stand_still
            termination = -0.0         # 对应 _reward_termination (通常设为0，由环境逻辑处理)
            torques = -1e-5            # 对应 _reward_torques

        self.rewards.scales = scales()

        # --- 控制配置 ---
        self.control.control_type = 'P'
        # 确保包含所有 43 个关节组的增益
        self.control.stiffness = {
            "hip": 100, "knee": 150, "ankle": 40, # 腿部统一设置 (简化)
            "waist": 50,                        # 腰部
            "shoulder": 40, "elbow": 25, "wrist": 15, # 手臂
            "hand": 5,                          # 手部统一设置 (简化)
        }
        self.control.damping = {
            "hip": 2, "knee": 4, "ankle": 2,
            "waist": 2,
            "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
            "hand": 0.1,
        }
        self.control.action_scale = 0.25
        self.control.decimation = 4

        # --- 资产配置 ---
        # !!! 使用你确认能加载 43DoF 的 URDF 文件 !!!
        self.asset.file = '/home/blake/g1_gym/resources/robots/g1_description/g1_29dof_no_collision.urdf' # 假设这是修复后的无碰撞版本
        self.asset.name = "g1"
        self.asset.foot_name = "ankle_roll_link" # *** 再次确认 ***
        self.asset.penalize_contacts_on = ["hip_pitch_link", "hip_roll_link", "hip_yaw_link",
                                           "knee_link", "waist_yaw_link", "torso_link",
                                           "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                                           "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"]
        self.asset.terminate_after_contacts_on = ["pelvis", "torso_link", "head_link"]
        self.asset.self_collisions = 0
        self.asset.flip_visual_attachments = False

        # --- Termination Conditions ---
        class termination:
            orientation_limit_roll = 0.8
            orientation_limit_pitch = 1.0
        self.termination = termination

        # --- 命令生成 ---
        self.commands.resampling_time = 10.0
        self.commands.num_commands = 4
        self.commands.ranges.lin_vel_x = [-1.0, 1.0]
        self.commands.ranges.lin_vel_y = [-0.5, 0.5]
        self.commands.ranges.ang_vel_yaw = [-0.8, 0.8]
        self.commands.ranges.heading = [-3.14, 3.14]
        self.commands.heading_command = True

        # --- 噪声配置 ---
        self.noise.add_noise = True
        self.noise.noise_level = 0.5
        # 确保噪声尺度名称与 _get_noise_scale_vec 中使用的匹配
        self.noise.noise_scales.lin_vel = 0.0 # Base lin_vel (通常不加噪)
        self.noise.noise_scales.ang_vel = 0.05
        self.noise.noise_scales.gravity = 0.05
        self.noise.noise_scales.dof_pos = 0.01
        self.noise.noise_scales.dof_vel = 0.05

        # --- Normalization 配置 ---
        self.normalization.obs_scales.lin_vel = 2.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.dof_pos = 1.0
        self.normalization.obs_scales.dof_vel = 0.05
        self.normalization.clip_observations = 100.
        self.normalization.clip_actions = 100.

        print("--- Stage1LocomotionConfig43DoF Loaded ---")


class Stage1LocomotionConfig43DoFPPO(LeggedRobotCfgPPO):
    # PPO 配置与之前一致
    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        num_steps_per_env = 24
        max_iterations = 20000
        run_name = 'g1_43dof_loco_direct' # 更新名称
        experiment_name = 'g1_locomotion_43dof'
        save_interval = 500 # 可以适当调整保存频率