

# --- START OF MODIFIED stage1_locomotion_config.py (No Curriculum) ---

import os
from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
# from g1.utils.helpers import class_to_dict # Not used directly here

class Stage1LocomotionConfig(LeggedRobotCfg):
    def __init__(self):
        super().__init__()
        print("--- Loading Stage1LocomotionConfig (Direct Full Body Training, Reduced Collision) ---")

        # --- 环境配置 ---
        self.env.num_envs = 4096
        self.env.test = False
        self.env.episode_length_s = 20
        self.env.env_spacing = 3.0
        self.env.send_timeouts = True
        # Set observation/action dimensions for the FULL robot (43 DOF)
        self.env.num_observations = 140 # 11 base + 3 * 43 actions = 140
        self.env.num_privileged_obs = 143 # 140 + 3 base_lin_vel = 143
        self.env.num_actions = 43      # Full 43 DOF

        # --- 领域随机化 (Domain Randomization) ---
        self.domain_rand.randomize_friction = True
        self.domain_rand.friction_range = [0.1, 1.25]
        self.domain_rand.randomize_base_mass = True
        self.domain_rand.added_mass_range = [-1.0, 3.0]
        self.domain_rand.push_robots = True
        self.domain_rand.push_interval_s = 7 # Interval for full robot pushes
        self.domain_rand.max_push_vel_xy = 1.0 # Push strength for full robot

        # --- 初始状态配置 (Initial State) ---
        self.init_state.pos = [0.0, 0.0, 0.8] # Base height
        self.init_state.default_joint_angles = {
            # Legs (12)
            "left_hip_pitch_joint": -0.1, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3, "left_ankle_pitch_joint": -0.2, "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3, "right_ankle_pitch_joint": -0.2, "right_ankle_roll_joint": 0.0,
            # Torso (3)
            "waist_yaw_joint": 0.0, "waist_roll_joint": 0.0, "waist_pitch_joint": 0.0,
            # Left Arm (7)
            "left_shoulder_pitch_joint": 0.5, "left_shoulder_roll_joint": 0.1, "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": -0.5, "left_wrist_roll_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_yaw_joint": 0.0,
            # Right Arm (7)
            "right_shoulder_pitch_joint": 0.5, "right_shoulder_roll_joint": -0.1, "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": -0.5, "right_wrist_roll_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_yaw_joint": 0.0,
            # Left Hand (7)
            "left_hand_thumb_0_joint": 0.0, "left_hand_thumb_1_joint": 0.0, "left_hand_thumb_2_joint": 0.0,
            "left_hand_middle_0_joint": 0.0, "left_hand_middle_1_joint": 0.0,
            "left_hand_index_0_joint": 0.0, "left_hand_index_1_joint": 0.0,
            # Right Hand (7)
            "right_hand_thumb_0_joint": 0.0, "right_hand_thumb_1_joint": 0.0, "right_hand_thumb_2_joint": 0.0,
            "right_hand_middle_0_joint": 0.0, "right_hand_middle_1_joint": 0.0,
            "right_hand_index_0_joint": 0.0, "right_hand_index_1_joint": 0.0,
        }

        # --- 奖励配置 (Rewards) ---
        self.rewards.base_height_target = 0.78
        self.rewards.soft_dof_pos_limit = 0.9
        self.rewards.only_positive_rewards = False
        # tracking_sigma is not used without explicit tracking rewards defined below
        # self.rewards.tracking_sigma = 0.25
        self.rewards.soft_torque_limit = 0.8

        # --- Explicit Reward Scales for Direct Training ---
        # Inspired by G1RoughCfg and typical locomotion rewards
        class scales:
            # --- Locomotion ---
            tracking_lin_vel = 1.5     # Increased focus on linear velocity tracking
            tracking_ang_vel = 0.8     # Increased focus on angular velocity tracking
            lin_vel_z = -2.0           # Penalize vertical velocity
            ang_vel_xy = -0.05         # Penalize roll/pitch angular velocity
            # --- Stability & Smoothness ---
            orientation = -1.0         # Penalize deviation from upright orientation
            base_height = -15.0        # Strong penalty for deviation from target height
            dof_acc = -3.0e-7          # Penalize joint accelerations (smoothness)
            action_rate = -0.02        # Penalize changes in actions (smoothness)
            # --- Constraints ---
            dof_pos_limits = -10.0     # Strong penalty for hitting joint limits
            torque_limits = -0.01      # Penalize reaching torque limits (using soft_torque_limit)
            collision = -1.0           # Penalize collisions (will only trigger for feet/hands now)
            # --- Regularization / Behavior Shaping ---
            dof_vel = -0.002           # Slight penalty on high joint velocities
            alive = 0.2                # Small reward for staying alive
            feet_air_time = 1.0       # Reward feet being in the air (encourage dynamic gaits) - Tune carefully!
            # --- Arm/Hand Penalties (to keep them relatively stable during walking) ---
            arm_dof_vel = -0.01        # Penalize arm joint velocities more
            arm_dof_acc = -1.0e-6        # Penalize arm joint accelerations more
            hand_dof_vel = -0.005      # Penalize hand joint velocities
            # Maybe add penalty for arm position deviation from default?
            # arm_pos_deviation = -0.5

            # --- Remove or adjust unused/less relevant rewards from G1RoughCfg ---
            # hip_pos = -1.0             # Less relevant for full body
            # contact_no_vel = -0.2    # Might conflict with feet_air_time
            # feet_swing_height = -20.0  # Can be complex to implement/tune
            # contact = 0.18             # Might conflict with feet_air_time or collision

        self.rewards.scales = scales() # Assign the scales class

        # --- 控制配置 (Control) ---
        self.control.control_type = 'P'
        self.control.stiffness = {
            "hip_yaw": 100, "hip_roll": 100, "hip_pitch": 100, "knee": 150, "ankle": 40,
            "waist": 50,
            "shoulder": 40, "elbow": 25, "wrist": 15,
            "thumb": 5, "middle": 5, "index": 5,
        }
        self.control.damping = {
            "hip_yaw": 2, "hip_roll": 2, "hip_pitch": 2, "knee": 4, "ankle": 2,
            "waist": 2,
            "shoulder": 1.5, "elbow": 1, "wrist": 0.5,
            "thumb": 0.1, "middle": 0.1, "index": 0.1,
        }
        # locked gains are not needed without curriculum
        # self.control.locked_stiffness = 500.0
        # self.control.locked_damping = 50.0
        self.control.action_scale = 0.25
        self.control.decimation = 4

        # --- 资产配置 (Asset) ---
        self.asset.file = '/root/autodl-tmp/g1/g1_gym/resources/robots/g1_description/g1_29dof_no_collision.urdf'
        # self.asset.file = '/root/autodl-tmp/g1/g1_gym/resources/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf'
        self.asset.name = "g1"
        self.asset.foot_name = "ankle_roll_link" # *** IMPORTANT: Double-check this link name in your URDF ***
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

        # --- 命令生成 (Commands) ---
        self.commands.resampling_time = 10.0
        self.commands.num_commands = 4 # vx, vy, vyaw, heading
        # Set final command ranges directly
        self.commands.ranges.lin_vel_x = [-1.0, 1.0]
        self.commands.ranges.lin_vel_y = [-0.5, 0.5]
        self.commands.ranges.ang_vel_yaw = [-0.8, 0.8]
        self.commands.ranges.heading = [-3.14, 3.14]
        self.commands.heading_command = True

        # --- 噪声配置 (Noise) ---
        self.noise.add_noise = True
        self.noise.noise_level = 0.5 # Noise level for full complexity
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
        self.normalization.clip_observations = 100.
        self.normalization.clip_actions = 100.

        # --- !!! No Nested Curriculum !!! ---
        # self.nested_locomotion_curriculum = False # Or just remove the attribute
        # Remove sub_stage_params dictionary entirely

        print("--- Stage1LocomotionConfig (Direct Full Body Training) Loaded ---")


# class Stage1LocomotionConfigPPO(LeggedRobotCfgPPO):
#     # Keep PPO settings suitable for the complex full-body task
#     class policy:
#         init_noise_std = 1.0
#         actor_hidden_dims = [512, 256, 128]
#         critic_hidden_dims = [512, 256, 128]
#         activation = 'elu'
#         # RNN is still likely beneficial for complex full-body dynamics
#         rnn_type = 'lstm'
#         rnn_hidden_size = 512
#         rnn_num_layers = 1

#     class algorithm(LeggedRobotCfgPPO.algorithm):
#         entropy_coef = 0.01

#     class runner(LeggedRobotCfgPPO.runner):
#         policy_class_name = "ActorCriticRecurrent" # Use RNN policy
#         num_steps_per_env = 24 # Rollout length
#         max_iterations = 20000 # Total training iterations (adjust as needed)

#         run_name = 'direct_loco_reduced_coll' # Naming convention for direct training
#         experiment_name = 'g1_stage1_direct'
#         save_interval = 200 # Save frequency

# # --- END OF MODIFIED stage1_locomotion_config.py (No Curriculum) ---

class Stage1LocomotionConfigPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

# import os
# from g1.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
# # from g1.utils.helpers import class_to_dict # Not used directly here

# class Stage1LocomotionConfig(LeggedRobotCfg):
#     class init_state( LeggedRobotCfg.init_state ):
#         pos = [0.0, 0.0, 0.8] # x,y,z [m]
#         default_joint_angles = { # = target angles [rad] when action = 0.0
#            'left_hip_yaw_joint' : 0. ,   
#            'left_hip_roll_joint' : 0,               
#            'left_hip_pitch_joint' : -0.1,         
#            'left_knee_joint' : 0.3,       
#            'left_ankle_pitch_joint' : -0.2,     
#            'left_ankle_roll_joint' : 0,     
#            'right_hip_yaw_joint' : 0., 
#            'right_hip_roll_joint' : 0, 
#            'right_hip_pitch_joint' : -0.1,                                       
#            'right_knee_joint' : 0.3,                                             
#            'right_ankle_pitch_joint': -0.2,                              
#            'right_ankle_roll_joint' : 0,       
#            'torso_joint' : 0.
#         }
    
#     class env(LeggedRobotCfg.env):
#         num_observations = 47
#         num_privileged_obs = 50
#         num_actions = 12


#     class domain_rand(LeggedRobotCfg.domain_rand):
#         randomize_friction = True
#         friction_range = [0.1, 1.25]
#         randomize_base_mass = True
#         added_mass_range = [-1., 3.]
#         push_robots = True
#         push_interval_s = 5
#         max_push_vel_xy = 1.5
      

#     class control( LeggedRobotCfg.control ):
#         # PD Drive parameters:
#         control_type = 'P'
#           # PD Drive parameters:
#         stiffness = {'hip_yaw': 100,
#                      'hip_roll': 100,
#                      'hip_pitch': 100,
#                      'knee': 150,
#                      'ankle': 40,
#                      }  # [N*m/rad]
#         damping = {  'hip_yaw': 2,
#                      'hip_roll': 2,
#                      'hip_pitch': 2,
#                      'knee': 4,
#                      'ankle': 2,
#                      }  # [N*m/rad]  # [N*m*s/rad]
#         # action scale: target angle = actionScale * action + defaultAngle
#         action_scale = 0.25
#         # decimation: Number of control action updates @ sim DT per policy DT
#         decimation = 4

#     class asset( LeggedRobotCfg.asset ):
#         file = '/root/autodl-tmp/g1/g1_gym/resources/robots/g1_description/g1_12dof.urdf'
#         name = "g1"
#         foot_name = "ankle_roll"
#         penalize_contacts_on = ["hip", "knee"]
#         terminate_after_contacts_on = ["pelvis"]
#         self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
#         flip_visual_attachments = False
  
#     class rewards( LeggedRobotCfg.rewards ):
#         soft_dof_pos_limit = 0.9
#         base_height_target = 0.78
        
#         class scales( LeggedRobotCfg.rewards.scales ):
#             tracking_lin_vel = 1.0
#             tracking_ang_vel = 0.5
#             lin_vel_z = -2.0
#             ang_vel_xy = -0.05
#             orientation = -1.0
#             base_height = -10.0
#             dof_acc = -2.5e-7
#             dof_vel = -1e-3
#             feet_air_time = 0.0
#             collision = 0.0
#             action_rate = -0.01
#             dof_pos_limits = -5.0
#             alive = 0.15
#             hip_pos = -1.0
#             contact_no_vel = -0.2
#             feet_swing_height = -20.0
#             contact = 0.18

# class Stage1LocomotionConfigPPO(LeggedRobotCfgPPO):
#     class policy:
#         init_noise_std = 0.8
#         actor_hidden_dims = [32]
#         critic_hidden_dims = [32]
#         activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
#         # only for 'ActorCriticRecurrent':
#         rnn_type = 'lstm'
#         rnn_hidden_size = 64
#         rnn_num_layers = 1
        
#     class algorithm( LeggedRobotCfgPPO.algorithm ):
#         entropy_coef = 0.01
#     class runner( LeggedRobotCfgPPO.runner ):
#         policy_class_name = "ActorCriticRecurrent"
#         max_iterations = 10000
#         run_name = ''
#         experiment_name = 'g1'

  
