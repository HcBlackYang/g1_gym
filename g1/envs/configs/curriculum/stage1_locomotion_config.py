
# import os
# from g1.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO

# class Stage1LocomotionConfig(LeggedRobotCfg):
#     def __init__(self):
#         super().__init__()
#         print("--- Loading Stage1LocomotionConfig ---")

#         # --- 环境配置 ---
#         self.env.num_envs = 4096
#         self.env.test = False
#         self.env.episode_length_s = 20
#         # 关键：根据 URDF 定义，共有 43 个 actuated 关节
#         self.env.num_observations = 140
#         self.env.num_privileged_obs = 143
#         self.env.num_actions = 43  # 12(腿) + 3(躯干) + 7(左臂) + 7(右臂) + 7(左手) + 7(右手) = 43
#         self.env.env_spacing = 3.0
#         self.env.send_timeouts = True

#         # --- 领域随机化 ---
#         self.domain_rand.randomize_friction = True
#         self.domain_rand.friction_range = [0.1, 1.25]
#         self.domain_rand.randomize_base_mass = True
#         self.domain_rand.added_mass_range = [-1.0, 3.0]
#         self.domain_rand.push_robots = True
#         self.domain_rand.push_interval_s = 5
#         self.domain_rand.max_push_vel_xy = 1.5

#         # --- 初始状态配置 ---
#         self.init_state.pos = [0.0, 0.0, 0.8]
#         # 补全所有 43 个关节的默认角度：
#         self.init_state.default_joint_angles = {
#             # 腿部（共 12 个关节）
#             "left_hip_pitch_joint": 0.0,
#             "left_hip_roll_joint": 0.0,
#             "left_hip_yaw_joint": 0.0,
#             "left_knee_joint": 0.3,
#             "left_ankle_pitch_joint": -0.2,
#             "left_ankle_roll_joint": 0.0,
#             "right_hip_pitch_joint": 0.0,
#             "right_hip_roll_joint": 0.0,
#             "right_hip_yaw_joint": 0.0,
#             "right_knee_joint": 0.3,
#             "right_ankle_pitch_joint": -0.2,
#             "right_ankle_roll_joint": 0.0,
#             # 躯干（3 个关节）
#             "waist_yaw_joint": 0.0,
#             "waist_roll_joint": 0.0,
#             "waist_pitch_joint": 0.0,
#             # 左臂（7 个关节）
#             "left_shoulder_pitch_joint": 0.3,
#             "left_shoulder_roll_joint": 0.1,
#             "left_shoulder_yaw_joint": 0.0,
#             "left_elbow_joint": -0.5,
#             "left_wrist_roll_joint": 0.0,
#             "left_wrist_pitch_joint": 0.0,
#             "left_wrist_yaw_joint": 0.0,
#             # 右臂（7 个关节）
#             "right_shoulder_pitch_joint": 0.3,
#             "right_shoulder_roll_joint": -0.1,
#             "right_shoulder_yaw_joint": 0.0,
#             "right_elbow_joint": -0.5,
#             "right_wrist_roll_joint": 0.0,
#             "right_wrist_pitch_joint": 0.0,
#             "right_wrist_yaw_joint": 0.0,
#             # 左手（7 个关节），若不控制手部，可全部设为固定值
#             "left_hand_thumb_0_joint": 0.0,
#             "left_hand_thumb_1_joint": 0.0,
#             "left_hand_thumb_2_joint": 0.0,
#             "left_hand_middle_0_joint": 0.0,
#             "left_hand_middle_1_joint": 0.0,
#             "left_hand_index_0_joint": 0.0,
#             "left_hand_index_1_joint": 0.0,
#             # 右手（7 个关节）
#             "right_hand_thumb_0_joint": 0.0,
#             "right_hand_thumb_1_joint": 0.0,
#             "right_hand_thumb_2_joint": 0.0,
#             "right_hand_middle_0_joint": 0.0,
#             "right_hand_middle_1_joint": 0.0,
#             "right_hand_index_0_joint": 0.0,
#             "right_hand_index_1_joint": 0.0,
#         }

#         # --- 奖励配置 ---
#         self.rewards.base_height_target = 0.78
#         self.rewards.soft_dof_pos_limit = 0.9
#         self.rewards.only_positive_rewards = True

#         # --- 控制配置 ---
#         self.control.control_type = 'P'
#         # 此处对各组关节设置刚度与阻尼。若您不打算对手部进行控制，可将其值设为 0
#         self.control.stiffness = {
#             # 腿部
#             "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
#             # 躯干
#             "waist": 30,
#             # 手臂
#             "shoulder": 40, "elbow": 25, "wrist": 15,
#             # 手部（如果希望固定手部，可将这些值设为 0）
#             "thumb": 0, "middle": 0, "index": 0,
#         }
#         self.control.damping = {
#             # 腿部
#             "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
#             # 躯干
#             "waist": 1,
#             # 手臂
#             "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
#             # 手部
#             "thumb": 0, "middle": 0, "index": 0,
#         }
#         self.control.action_scale = 0.25
#         self.control.decimation = 4

#         # --- 资产配置 ---
#         self.asset.file = '/home/blake/g1_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
#         self.asset.name = "g1"
#         # 这里假设足部接触体名称采用 URDF 中的 link 名称（请根据实际情况确认）
#         self.asset.foot_name = "ankle_roll"
#         self.asset.penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "waist"]
#         self.asset.terminate_after_contacts_on = ["pelvis"]
#         self.asset.self_collisions = 0
#         self.asset.flip_visual_attachments = False

#         # --- 命令生成 ---
#         self.commands.resampling_time = 10.0
#         self.commands.num_commands = 43  # 与 env.num_actions 保持一致
#         self.commands.ranges.lin_vel_x = [-1.5, 1.5]
#         self.commands.ranges.lin_vel_y = [-0.5, 0.5]
#         self.commands.ranges.ang_vel_yaw = [-0.8, 0.8]
#         self.commands.ranges.heading = [-3.14, 3.14]
#         self.commands.heading_command = True

#         # --- 噪声配置 ---
#         self.noise.add_noise = True
#         self.noise.noise_level = 0.5
#         self.noise.noise_scales.dof_pos = 0.01
#         self.noise.noise_scales.dof_vel = 0.05
#         self.noise.noise_scales.ang_vel = 0.05
#         self.noise.noise_scales.gravity = 0.05
#         self.noise.noise_scales.lin_vel = 0.0

#         # --- Normalization 配置 ---
#         self.normalization.obs_scales.lin_vel = 2.0
#         self.normalization.obs_scales.ang_vel = 0.25
#         self.normalization.obs_scales.dof_pos = 1.0
#         self.normalization.obs_scales.dof_vel = 0.05

#         print("--- Stage1LocomotionConfig Loaded ---")


# class Stage1LocomotionConfigPPO(LeggedRobotCfgPPO):
#     class policy:
#         init_noise_std = 0.8
#         actor_hidden_dims = [32]
#         critic_hidden_dims = [32]
#         # actor_hidden_dims = [256, 128]
#         # critic_hidden_dims = [256, 128]
#         activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         rnn_type = 'lstm'
#         # rnn_hidden_size = 64
#         rnn_hidden_size = 256  
#         rnn_num_layers = 1

#     class algorithm(LeggedRobotCfgPPO.algorithm):
#         entropy_coef = 0.01

#     class runner(LeggedRobotCfgPPO.runner):
#         policy_class_name = "ActorCriticRecurrent"
#         # policy_class_name = "ActorCritic"
#         max_iterations = 10000
#         run_name = ''
#         experiment_name = 'g1'


import os
from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from g1.utils.helpers import class_to_dict # 用于嵌套课程

class Stage1LocomotionConfig(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        print("--- Loading Stage1LocomotionConfig (with Nested Curriculum Support) ---")

        # --- 环境配置 ---
        self.env.num_envs = 4096  # 基础运动阶段使用较多环境
        self.env.test = False
        self.env.episode_length_s = 20
        self.env.env_spacing = 3.0
        self.env.send_timeouts = True
        # !!! 关键: num_observations/actions 将由 G1BasicLocomotion 根据子阶段动态设置 !!!
        # !!! 这里设置的值应为 *最终* (全自由度) 阶段的值，以确保 PPO 网络维度正确 !!!
        self.env.num_observations = 140 # 最终全自由度观测维度
        self.env.num_privileged_obs = 143 # 最终全自由度特权观测维度
        self.env.num_actions = 43      # 最终全自由度动作维度

        # --- 领域随机化 (保持或微调) ---
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.1, 1.25]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 3.0]
        self.domain_rand.push_robots = True
        self.domain_rand.push_interval_s = 7 # 可以稍微频繁一些
        self.domain_rand.max_push_vel_xy = 1.0 # 推动速度可以稍大

        # --- 初始状态配置 (全关节的默认角度) ---
        self.init_state.pos = [0.0, 0.0, 0.8]
        self.init_state.default_joint_angles = {
            # 腿部（12）
            "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
            # 躯干（3）
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            # 左臂（7） - 设置为自然下垂或稍微弯曲的姿态
            "left_shoulder_pitch_joint": 0.5, "left_shoulder_roll_joint": 0.1, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
            # 右臂（7）
            "right_shoulder_pitch_joint": 0.5, "right_shoulder_roll_joint": -0.1, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
            # 左手（7） - 设置为半握拳或自然放松
            "left_hand_thumb_0_joint": 0.2, "left_hand_thumb_1_joint": 0.2, "left_hand_thumb_2_joint": 0.2,
            "left_hand_middle_0_joint": 0.3, "left_hand_middle_1_joint": 0.3,
            "left_hand_index_0_joint": 0.3, "left_hand_index_1_joint": 0.3,
            # 右手（7）
            "right_hand_thumb_0_joint": 0.2, "right_hand_thumb_1_joint": 0.2, "right_hand_thumb_2_joint": 0.2,
            "right_hand_middle_0_joint": 0.3, "right_hand_middle_1_joint": 0.3,
            "right_hand_index_0_joint": 0.3, "right_hand_index_1_joint": 0.3,
        }

        # --- 奖励配置 (这些是基础参数，具体权重由 RewardScheduler 控制) ---
        self.rewards.base_height_target = 0.78
        self.rewards.soft_dof_pos_limit = 0.9
        self.rewards.only_positive_rewards = False # 允许负奖励以更好地塑造行为
        self.rewards.tracking_sigma = 0.25 # 基础跟踪 sigma， G1BasicLocomotion 会在内部覆盖
        self.rewards.soft_torque_limit = 0.8 # 可以增加对力矩的限制

        # --- 控制配置 ---
        self.control.control_type = 'P' # PD 控制
        # 定义活跃关节的基础刚度和阻尼
        self.control.stiffness = {
            "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
            "waist": 50, # 可以适当增加腰部刚度
            "shoulder": 40, "elbow": 25, "wrist": 15,
            "thumb": 5, "middle": 5, "index": 5, # 手部可以给一些低刚度
        }
        self.control.damping = {
            "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
            "waist": 2, # 增加腰部阻尼
            "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
            "thumb": 0.1, "middle": 0.1, "index": 0.1,
        }
        # 定义锁定关节的高增益
        self.control.locked_stiffness = 500.0 # 用于锁定关节的高刚度
        self.control.locked_damping = 50.0   # 用于锁定关节的高阻尼

        self.control.action_scale = 0.25
        self.control.decimation = 4

        # --- 资产配置 ---
        self.asset.file = '/home/blake/g1_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
        self.asset.name = "g1"
        self.asset.foot_name = "ankle_roll" # 确认 URDF 中的足部 link 名称
        self.asset.penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "waist"]
        self.asset.terminate_after_contacts_on = ["pelvis", "waist"] # 基座或腰部碰撞终止
        self.asset.self_collisions = 0 # 启用自碰撞检查
        self.asset.flip_visual_attachments = False
    
        class termination:
            # Specify termination conditions and parameters
            orientation_limit_roll = 0.8  # rad (~45 deg)
            orientation_limit_pitch = 1.0 # rad (~57 deg)
            # Add other termination conditions if needed, e.g.,
            # velocity_limit = 3.0 # m/s
            # height_limit = [0.3, 1.5] # min/max base height [m]
        self.termination = termination # Assign the class to the attribute

        # --- 命令生成 ---
        self.commands.resampling_time = 10.0
        self.commands.num_commands = 4 # vx, vy, vyaw, heading
        # !!! 基础速度范围，G1BasicLocomotion 会根据子阶段覆盖 !!!
        self.commands.ranges.lin_vel_x = [-1.0, 1.0] # 这是最终目标范围
        self.commands.ranges.lin_vel_y = [-0.5, 0.5]
        self.commands.ranges.ang_vel_yaw = [-0.8, 0.8]
        self.commands.ranges.heading = [-3.14, 3.14]
        self.commands.heading_command = True

        # --- 噪声配置 ---
        self.noise.add_noise = True
        self.noise.noise_level = 0.5 # 可以根据子阶段调整，但这里设为基础值
        self.noise.noise_scales.dof_pos = 0.01
        self.noise.noise_scales.dof_vel = 0.05
        self.noise.noise_scales.ang_vel = 0.05
        self.noise.noise_scales.gravity = 0.05
        self.noise.noise_scales.lin_vel = 0.0 # 不对速度观测加噪声

        # --- Normalization 配置 ---
        self.normalization.obs_scales.lin_vel = 2.0
        self.normalization.obs_scales.ang_vel = 0.25
        self.normalization.obs_scales.dof_pos = 1.0
        self.normalization.obs_scales.dof_vel = 0.05
        self.normalization.clip_observations = 100.
        self.normalization.clip_actions = 100.


        # --- !!! 嵌套课程学习配置 !!! ---
        self.nested_locomotion_curriculum = True # 启用嵌套课程

        # 定义每个子阶段的参数
        # key 是子阶段号 (1, 2, 3, ...)
        # 注意：num_observations/actions 需要仔细估算或精确计算
        # 观测基础维度 = 3(ang_vel) + 3(gravity) + 3(commands) + 2(phase) = 11
        # 总观测 = 基础维度 + N(active_dof_pos) + N(active_dof_vel) + N(active_actions)
        # N = 活跃关节数 (num_actions)
        # 总观测 = 11 + 3 * N
        self.sub_stage_params = {
            1: { # 子阶段 1.1: 仅腿部 (锁躯干、臂、手)
                "name": "Legs Only (Locked Torso/Arms/Hands)",
                "active_joints": [ # 12 个关节
                    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"
                ],
                "num_actions": 12,
                "num_observations": 11 + 3 * 12, # 11 + 36 = 47
                "num_privileged_obs": 47 + 3, # + base_lin_vel = 50
                "base_lin_vel_range": 0.4, # 初始速度范围很小
                "base_ang_vel_range": 0.2,
                "reward_focus": ["stability", "base_motion_low_penalty", "leg_control"], # 奖励重点
                "push_robots_scale": 0.3, # 较小的推动
                "success_threshold": 0.6 # 较低的成功阈值
            },
            2: { # 子阶段 1.2: 激活躯干 (锁臂、手)
                "name": "Legs + Torso (Locked Arms/Hands)",
                "active_joints": [
                    "hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll",
                    "waist_yaw", "waist_roll", "waist_pitch" # +3 躯干
                ],
                "num_actions": 15, # 12 + 3
                "num_observations": 11 + 3 * 15, # 11 + 45 = 56
                "num_privileged_obs": 56 + 3, # 59
                "base_lin_vel_range": 0.8,
                "base_ang_vel_range": 0.4,
                "reward_focus": ["stability", "tracking_moderate", "base_motion", "torso_control"],
                "push_robots_scale": 0.5,
                "success_threshold": 0.65
            },
            3: { # 子阶段 1.3: 激活手臂 (锁手)
                "name": "Legs + Torso + Arms (Locked Hands)",
                "active_joints": [
                    "hip", "knee", "ankle", "waist", # 用关键词匹配
                    "shoulder", "elbow", "wrist"     # +14 手臂
                ],
                "num_actions": 29, # 15 + 14
                "num_observations": 11 + 3 * 29, # 11 + 87 = 98
                "num_privileged_obs": 98 + 3, # 101
                "base_lin_vel_range": 1.2,
                "base_ang_vel_range": 0.6,
                "reward_focus": ["tracking", "stability", "base_motion", "arm_penalty_low"], # 开始轻微惩罚手臂乱动
                "push_robots_scale": 0.7,
                "success_threshold": 0.7
            },
            4: { # 子阶段 1.4: 激活部分手部 (例如：拇指 + 食指/中指)
                "name": "Legs + Torso + Arms + Simple Hands",
                "active_joints": [
                    "hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist",
                    "thumb", "index", "middle" # + 7*2 = 14 手部关节 (拇指3+食指2+中指2) * 2
                ],
                 # 验证一下关节数：12腿+3躯干+14臂+14手 = 43 全激活了？
                 # 这里假设只激活了部分手部关节，比如每只手激活 5 个？
                 # 示例：激活拇指(3)+食指(2)+中指(2) = 7 个/手 -> 14 个手部关节
                "num_actions": 29 + 14, # 43
                "num_observations": 11 + 3 * 43, # 11 + 129 = 140
                "num_privileged_obs": 140 + 3, # 143 -> 最终维度
                "base_lin_vel_range": 1.6,
                "base_ang_vel_range": 0.8,
                "reward_focus": ["tracking", "stability", "base_motion", "arm_penalty", "hand_penalty_low"],
                "push_robots_scale": 0.8,
                "success_threshold": 0.75
            },
            5: { # 子阶段 1.5: 全关节激活，接近最终目标
                "name": "Full Body Locomotion",
                "active_joints": ["all"], # 特殊值表示全部激活
                "num_actions": 43,
                "num_observations": 140, # 最终观测维度
                "num_privileged_obs": 143, # 最终特权观测维度
                "base_lin_vel_range": 2.0, # 最终速度范围
                "base_ang_vel_range": 1.0,
                "reward_focus": ["full_locomotion_rewards"], # 使用类似原始 Stage 1 的奖励组合
                "push_robots_scale": 1.0, # 全推动
                "success_threshold": 0.8 # 最终目标阈值
            },
        }

        print("--- Stage1LocomotionConfig (with Nested Curriculum) Loaded ---")


class Stage1LocomotionConfigPPO(LeggedRobotCfgPPO):
    # --- 确保网络维度足够大以容纳最终阶段的观测/动作 ---
    # --- 隐藏层可以根据需要调整 ---
    class policy:
        init_noise_std = 1.0 # 初始噪声可以大一些
        # 增加隐藏层维度以处理更复杂的输入
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # elu 通常效果不错
        # 使用 RNN (LSTM) 可能有助于处理时序信息和不同阶段的动态
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01 # 鼓励探索

    class runner(LeggedRobotCfgPPO.runner):
        # 必须使用 Recurrent 版本，因为 policy 中定义了 RNN
        policy_class_name = "ActorCriticRecurrent"
        # 迭代次数和 env 步数可以根据需要调整
        # num_steps_per_env = 24 # 每轮迭代每个 env 跑的步数
        # max_iterations = 15000 # 总迭代次数
        num_steps_per_env = 48 # 增加 rollout 长度可能有助于学习长期依赖
        max_iterations = 20000 # 可能需要更多迭代次数

        run_name = 'nested_loco' # 方便区分
        experiment_name = 'g1_stage1'
        save_interval = 200 # 增加保存频率
        # 其他 runner 参数保持默认或根据需要调整