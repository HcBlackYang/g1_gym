# stage2_kitchen_nav_config.py
# G1机器人厨房环境避障导航训练配置
import os
from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO # 导入基础配置
from g1.utils.helpers import class_to_dict # 如果需要用到

# 获取项目根目录 (可选，用于资源路径)
G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class Stage2KitchenNavConfig(LeggedRobotCfg):
    def __init__(self):
        # 调用父类初始化
        super().__init__()
        print("--- Loading Stage2KitchenNavConfig ---")

        # --- 环境配置 ---
        self.env.num_envs = 512 # Stage 2 使用较少环境
        self.env.test = False
        self.env.episode_length_s = 30 # 导航任务可能需要更长时间
        # 观测空间: 基础(140) + 3D相对目标位置 = 143
        self.env.num_observations = 140
        # 特权观测: 基础(143) + 导航(3) + 距离(1) + 碰撞(1) = 148 (暂定)
        # 注意：基础特权观测是 143 (基础观测140 + 基础线速度3)
        # 所以 Stage 2 特权 = 143 + 3(nav) + 1(dist) + 1(collision) = 148
        self.env.num_privileged_obs = 143
        self.env.num_actions = 43  # 保持与 Stage 1 一致
        self.env.env_spacing = 5.0 # 可能需要更大间距以容纳 Kitchen
        self.env.send_timeouts = True

        # --- 领域随机化 ---
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.2, 1.25] # 可以稍微增加最小摩擦力
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 2.0] # 可以稍微减小范围
        self.domain_rand.push_robots = True
        self.domain_rand.push_interval_s = 8 # 减少推动频率
        self.domain_rand.max_push_vel_xy = 0.5 # 显著降低推动速度

        # --- 初始状态配置 ---
        self.init_state.pos = [2.0, 2.0, 0.8] # 可根据 Kitchen 布局调整
        # 关节角度: 可以使用 Stage 1 的，或者调整手臂姿态
        self.init_state.default_joint_angles = {
            # 腿部（共 12 个关节）- 与 Stage 1 保持一致
            "left_hip_pitch_joint": 0.0, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
            # 躯干（3 个关节）- 与 Stage 1 保持一致
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            # 手臂（14 个关节）- 调整为导航姿态 (例如，稍微抬起和弯曲)
            "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.1, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.1, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
            # 手部（14 个关节）- 保持固定/半开
            "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0, "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.1, "left_hand_middle_1_joint": 0.1, # Slightly closed
            "left_hand_index_0_joint": 0.1, "left_hand_index_1_joint": 0.1,
            "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0, "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.1, "right_hand_middle_1_joint": 0.1,
            "right_hand_index_0_joint": 0.1, "right_hand_index_1_joint": 0.1,
        }

        # --- 奖励配置 ---
        # 奖励权重由 RewardScheduler 根据 Stage 2 提供
        # 只需设置基础参数
        self.rewards.base_height_target = 0.78
        self.rewards.soft_dof_pos_limit = 0.9
        self.rewards.only_positive_rewards = False # 允许负奖励以惩罚碰撞
        self.rewards.tracking_sigma = 0.5 # 可以放宽跟踪奖励的 sigma，因为不是主要目标

        # --- 控制配置 ---
        # 通常保持与 Stage 1 相同，除非需要调整
        self.control.control_type = 'P'
        self.control.stiffness = { # 与 Stage 1 相同
            "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
            "waist": 30, "shoulder": 40, "elbow": 25, "wrist": 15,
            "thumb": 0, "middle": 0, "index": 0, # 手部关节保持 P=0 D=0
        }
        self.control.damping = { # 与 Stage 1 相同
            "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
            "waist": 1, "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
            "thumb": 0, "middle": 0, "index": 0,
        }
        self.control.action_scale = 0.25
        self.control.decimation = 4

        # --- 资产配置 ---
        # 机器人 URDF
        self.asset.file = '/home/blake/g1_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
        self.asset.name = "g1"
        self.asset.foot_name = "ankle_roll"
        # 惩罚接触部位可以增加手臂等
        self.asset.penalize_contacts_on = ["hip", "knee", "shoulder", "elbow", "waist", "wrist"]
        self.asset.terminate_after_contacts_on = ["pelvis"] # 基座碰撞终止
        self.asset.self_collisions = 0
        self.asset.flip_visual_attachments = False

        # --- 新增: Kitchen 资源配置 ---
        self.asset.kitchen_asset_root = os.path.join(G1_ROOT_DIR, "resources/kitchen_assets/models")
        self.asset.kitchen_lisdf_path = os.path.join(G1_ROOT_DIR, "resources/kitchen_assets/scenes/kitchen_basics.lisdf")
        self.asset.kitchen_origin_offset = [0.0, 0.0, 0.0] # Kitchen 场景相对于 (0,0,0) 的偏移

        # --- 命令生成 ---
        # 命令仍然是控制机器人基座运动
        self.commands.resampling_time = 10.0
        self.commands.num_commands = 4 # vx, vy, vyaw, heading
        # 导航阶段降低命令速度范围
        self.commands.ranges.lin_vel_x = [-0.8, 0.8]
        self.commands.ranges.lin_vel_y = [-0.5, 0.5] # 侧向移动需求较低
        self.commands.ranges.ang_vel_yaw = [-0.6, 0.6]
        self.commands.ranges.heading = [-3.14, 3.14]
        self.commands.heading_command = True

        # --- 噪声配置 ---
        # 可以稍微降低噪声以专注于导航任务
        self.noise.add_noise = True
        self.noise.noise_level = 0.3 # 降低整体噪声水平
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
        # 新增: 导航目标观测的尺度 (相对位置，单位是米，可以不缩放或小幅缩放)
        self.normalization.obs_scales.target_pos = 1.0 # Target relative position scale

        print("--- Stage2KitchenNavConfig Loaded ---")

# --- PPO 配置 (可选，可以放在单独文件或这里) ---
class Stage2KitchenNavCfgPPO(LeggedRobotCfgPPO):
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