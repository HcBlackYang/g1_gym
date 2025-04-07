
import os
from g1.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO

class Stage1LocomotionConfig(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        print("--- Loading Stage1LocomotionConfig ---")

        # --- 环境配置 ---
        self.env.num_envs = 4096
        self.env.test = False
        self.env.episode_length_s = 20
        # 关键：根据 URDF 定义，共有 43 个 actuated 关节
        self.env.num_observations = 140
        self.env.num_privileged_obs = 143
        self.env.num_actions = 43  # 12(腿) + 3(躯干) + 7(左臂) + 7(右臂) + 7(左手) + 7(右手) = 43
        self.env.env_spacing = 3.0
        self.env.send_timeouts = True

        # --- 领域随机化 ---
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.1, 1.25]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 3.0]
        self.domain_rand.push_robots = True
        self.domain_rand.push_interval_s = 5
        self.domain_rand.max_push_vel_xy = 1.5

        # --- 初始状态配置 ---
        self.init_state.pos = [0.0, 0.0, 0.8]
        # 补全所有 43 个关节的默认角度：
        self.init_state.default_joint_angles = {
            # 腿部（共 12 个关节）
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            # 躯干（3 个关节）
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            # 左臂（7 个关节）
            "left_shoulder_pitch_joint": 0.3,
            "left_shoulder_roll_joint": 0.1,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            # 右臂（7 个关节）
            "right_shoulder_pitch_joint": 0.3,
            "right_shoulder_roll_joint": -0.1,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            # 左手（7 个关节），若不控制手部，可全部设为固定值
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            # 右手（7 个关节）
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
        }

        # --- 奖励配置 ---
        self.rewards.base_height_target = 0.78
        self.rewards.soft_dof_pos_limit = 0.9
        self.rewards.only_positive_rewards = True

        # --- 控制配置 ---
        self.control.control_type = 'P'
        # 此处对各组关节设置刚度与阻尼。若您不打算对手部进行控制，可将其值设为 0
        self.control.stiffness = {
            # 腿部
            "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
            # 躯干
            "waist": 30,
            # 手臂
            "shoulder": 40, "elbow": 25, "wrist": 15,
            # 手部（如果希望固定手部，可将这些值设为 0）
            "thumb": 0, "middle": 0, "index": 0,
        }
        self.control.damping = {
            # 腿部
            "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
            # 躯干
            "waist": 1,
            # 手臂
            "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
            # 手部
            "thumb": 0, "middle": 0, "index": 0,
        }
        self.control.action_scale = 0.25
        self.control.decimation = 4

        # --- 资产配置 ---
        self.asset.file = '/home/blake/g1_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
        self.asset.name = "g1"
        # 这里假设足部接触体名称采用 URDF 中的 link 名称（请根据实际情况确认）
        self.asset.foot_name = "ankle_roll"
        self.asset.penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "waist"]
        self.asset.terminate_after_contacts_on = ["pelvis"]
        self.asset.self_collisions = 0
        self.asset.flip_visual_attachments = False

        # --- 命令生成 ---
        self.commands.resampling_time = 10.0
        self.commands.num_commands = 43  # 与 env.num_actions 保持一致
        self.commands.ranges.lin_vel_x = [-1.5, 1.5]
        self.commands.ranges.lin_vel_y = [-0.5, 0.5]
        self.commands.ranges.ang_vel_yaw = [-0.8, 0.8]
        self.commands.ranges.heading = [-3.14, 3.14]
        self.commands.heading_command = True

        # --- 噪声配置 ---
        self.noise.add_noise = True
        self.noise.noise_level = 0.5
        self.noise.noise_scales.dof_pos = 0.01
        self.noise.noise_scales.dof_vel = 0.05
        self.noise.noise_scales.ang_vel = 0.05
        self.noise.noise_scales.gravity = 0.05
        self.noise.noise_scales.lin_vel = 0.0

        # --- Normalization 配置 ---
        self.normalization.obs_scales.lin_vel = 2.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.dof_pos = 1.0
        self.normalization.obs_scales.dof_vel = 0.05

        print("--- Stage1LocomotionConfig Loaded ---")


class Stage1LocomotionConfigPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'


