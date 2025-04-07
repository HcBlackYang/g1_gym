
# g1_kitchen_navigation.py
import os
import numpy as np
from isaacgym import gymapi, gymtorch
# from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, to_torch
from isaacgym.torch_utils import quat_rotate_inverse, quat_apply, to_torch, get_axis_params  # 添加get_axis_params
from isaacgym.torch_utils import torch_rand_float


import torch

# 从课程基础类继承
from g1.envs.curriculum.curriculum_base import G1CurriculumBase
# 导入 kitchen 解析工具
try:
    from g1.utils.kitchen_utils import parse_lisdf # 确保路径正确
except ImportError:
    print("❌ 错误: 无法导入 kitchen_utils. 请确保 g1/utils/kitchen_utils.py 文件存在。")
    raise

# 导入基础机器人配置类 (可选)
from g1.envs.base.legged_robot_config import LeggedRobotCfg

# 定义项目根目录 (如果需要加载资源)
try:
    G1_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    G1_ROOT_DIR = "" # Fallback

class G1KitchenNavigation(G1CurriculumBase):
    """第二阶段：厨房环境导航避障训练.
    继承自 G1CurriculumBase，加载厨房环境，处理多 Actor 状态，
    并实现导航和避障逻辑。
    """

    # !!! 修改构造函数签名以匹配 BaseTask !!!
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, gym_handle=None, sim_handle=None):
        print("--- Initializing G1KitchenNavigation (Stage 2) ---")


        # --- 记录传入的gym和sim句柄 ---
        self.external_gym_handle = gym_handle
        self.external_sim_handle = sim_handle

        # --- Kitchen Assets and Actors (初始化为空) ---
        self.kitchen_assets = {}
        self.kitchen_poses = {}
        self.kitchen_scales = {}
        self.kitchen_actors_by_env = []
        self.num_kitchen_bodies = 0
        self.num_kitchen_dofs = 0

        # --- 重要：添加调试标记 ---
        self.create_sim_called = False
        self.create_envs_called = False

        # --- Navigation State (初始化为空) ---
        self.waypoints = {}
        self.waypoint_names = []
        self.waypoint_positions = None
        self.active_waypoint_indices = []
        self.current_waypoint_idx = None
        self.last_waypoint_idx = None
        self.target_positions = None

        # --- State Tracking (初始化为空) ---
        self.dist_to_target = None
        self.reached_waypoint = None
        self.consecutive_successes = None
        self.collision_detected = None
        self.robot_non_foot_indices = None

        # --- 调用父类 G1CurriculumBase 初始化 ---
        # 这会设置好 gym, sim, device, 基础缓冲区（基于cfg）, viewer
        # *但是* 不会调用 prepare_sim 或 init_buffers
        # 它会调用被我们重写的 create_sim -> _load_kitchen_assets -> _create_envs
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless, gym_handle=gym_handle, sim_handle=sim_handle)

        # --- 检查create_sim和create_envs是否被调用 ---
        print(f"父类__init__执行完成.")
        print(f"create_sim是否被调用: {self.create_sim_called}")
        print(f"create_envs是否被调用: {self.create_envs_called}")
        print(f"环境是否已创建: {'是' if hasattr(self, 'envs') and len(self.envs) > 0 else '否'}")


        # --- 后续初始化 (在环境和 Actors 创建后, 且 prepare_sim 调用后) ---
        # 注意: _init_buffers 和 _prepare_reward_function 会在父类的 __init__ 末尾被调用
        # 因此导航点和碰撞索引的初始化需要在这之后，或者在 _init_buffers 中处理

        # 现在 _init_buffers 已被调用 (因为它在父类 __init__ 末尾)
        # 我们可以安全地初始化导航状态了
        self._initialize_kitchen_waypoints()
        if self.waypoint_positions is not None and len(self.active_waypoint_indices) > 0:
             num_active_waypoints = len(self.active_waypoint_indices)
             self.current_waypoint_idx = torch.randint(0, num_active_waypoints, (self.num_envs,), device=self.device, dtype=torch.long)
             self.current_waypoint_idx = torch.tensor(self.active_waypoint_indices, device=self.device)[self.current_waypoint_idx]
             self.last_waypoint_idx = self.current_waypoint_idx.clone()
             self.target_positions = self.waypoint_positions[self.current_waypoint_idx]
             self.dist_to_target = torch.full((self.num_envs,), float('inf'), device=self.device)
             self.reached_waypoint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
             self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
             self.collision_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
             print(f"  导航初始化完成。激活导航点数: {num_active_waypoints}")
        else:
             print("⚠️ 导航点初始化失败或无激活导航点。导航功能将受限。")
             # Set defaults
             self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
             self.last_waypoint_idx = self.current_waypoint_idx.clone()
             self.target_positions = torch.zeros((self.num_envs, 3), device=self.device)
             self.dist_to_target = torch.full((self.num_envs,), float('inf'), device=self.device)
             self.reached_waypoint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
             self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
             self.collision_detected = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 获取用于碰撞检测的 body indices (应在 _init_buffers 后)
        self._get_collision_body_indices()

        # --- 验证观测/动作维度 (在 _init_buffers 之后进行) ---
        if self.num_actions != 43: print(f"❌ ERROR: G1KitchenNavigation - num_actions ({self.num_actions}) != 43")
        expected_obs_dim = getattr(self.cfg.env, 'num_observations', 143) # Read from config
        if self.obs_buf.shape[1] != expected_obs_dim: print(f"❌ ERROR: G1KitchenNavigation - obs_buf dim ({self.obs_buf.shape[1]}) != configured ({expected_obs_dim})")
        expected_priv_obs_dim = getattr(self.cfg.env, 'num_privileged_obs', 148)
        if self.privileged_obs_buf is not None and self.privileged_obs_buf.shape[1] != expected_priv_obs_dim: print(f"❌ ERROR: G1KitchenNavigation - priv_obs_buf dim ({self.privileged_obs_buf.shape[1]}) != configured ({expected_priv_obs_dim})")

        print("--- G1KitchenNavigation Initialized Successfully ---")



    def _load_kitchen_assets(self):
        """加载所有Kitchen资产，在创建环境之前"""
        print("🔍 开始加载Kitchen资产...")
        asset_root = "/home/blake/kitchen-worlds/assets/models/"
        lisdf_path = "/home/blake/kitchen-worlds/assets/scenes/kitchen_basics.lisdf"
        pose_data = parse_lisdf(lisdf_path)

        # 加载所有URDF资产
        for urdf_path, data in pose_data.items():
            urdf_relative_path = os.path.relpath(urdf_path, asset_root)
            if not os.path.exists(urdf_path):
                print(f"⚠️ Warning: URDF 文件不存在: {urdf_path}")
                continue

            pose = data["pose"]
            scale = data["scale"]

            asset_options = gymapi.AssetOptions()
            asset_options.fix_base_link = True
            asset_options.disable_gravity = False
            asset_options.use_mesh_materials = True
            asset_options.override_com = True
            asset_options.override_inertia = True

            # 修改碰撞生成选项，避免凸包分解错误
            asset_options.convex_decomposition_from_submeshes = False  # 禁用从子网格创建凸包
            asset_options.vhacd_enabled = False  # 禁用VHACD
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.01  # 增加厚度可能有助于稳定性

            try:
                object_asset = self.gym.load_asset(self.sim, asset_root, urdf_relative_path, asset_options)
                if object_asset is None:
                    print(f"❌ ERROR: 无法加载 URDF: {urdf_relative_path}")
                    continue

                self.kitchen_assets[urdf_relative_path] = object_asset
                self.kitchen_poses[urdf_relative_path] = pose
                # 存储scale信息
                self.kitchen_scales = getattr(self, 'kitchen_scales', {})
                self.kitchen_scales[urdf_relative_path] = scale
                print(f"✅ 成功加载: {urdf_relative_path} (Scale: {scale})")
            except Exception as e:
                print(f"❌ 加载'{urdf_relative_path}'时发生错误: {e}")
                # 如果加载失败，尝试使用更简单的碰撞设置
                try:
                    print(f"🔄 尝试使用简化选项重新加载'{urdf_relative_path}'")
                    asset_options.convex_decomposition_from_submeshes = False
                    asset_options.vhacd_enabled = False
                    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
                    asset_options.use_mesh_materials = False
                    # 使用非常简单的碰撞模型
                    asset_options.create_convex_meshes = False
                    asset_options.replace_cylinder_with_capsule = True

                    object_asset = self.gym.load_asset(self.sim, asset_root, urdf_relative_path, asset_options)
                    if object_asset is not None:
                        self.kitchen_assets[urdf_relative_path] = object_asset
                        self.kitchen_poses[urdf_relative_path] = pose
                        # 存储scale信息
                        self.kitchen_scales = getattr(self, 'kitchen_scales', {})
                        self.kitchen_scales[urdf_relative_path] = scale
                        print(f"✅ 成功使用简化选项加载: {urdf_relative_path} (Scale: {scale})")
                except Exception as e2:
                    print(f"❌ 简化加载仍然失败: {e2}")
                    continue

    def _add_kitchen_actors_to_envs(self):
        """向现有环境添加厨房Actor"""
        if not self.kitchen_assets:
            print("❌ 没有厨房资产可添加！")
            return

        print(f"准备向 {self.num_envs} 个环境添加厨房Actor...")

        # 厨房基准偏移（与您的kitchen_env.py中相似）
        kitchen_base_offset = torch.tensor([2.3, 4.7, -0.002], device=self.device)

        # 环境间距放大因子（从kitchen_env.py借鉴）
        spacing_factor = 5.0

        # 为每个环境添加厨房Actor
        total_added = 0
        for i, env_handle in enumerate(self.envs):
            # 这里应用了spacing_factor，与kitchen_env.py保持一致
            env_origin = self.env_origins[i].clone() * spacing_factor

            successful_kitchen_actors = []
            for asset_key, asset in self.kitchen_assets.items():
                # 获取资产姿态和缩放
                pose_info = self.kitchen_poses[asset_key]
                scale = self.kitchen_scales.get(asset_key, 1.0)

                # 创建变换
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(
                    float(pose_info.pos[0]) - kitchen_base_offset[0] + env_origin[0],
                    float(pose_info.pos[1]) - kitchen_base_offset[1] + env_origin[1],
                    float(pose_info.pos[2]) - kitchen_base_offset[2] + env_origin[2]
                )
                transform.r = gymapi.Quat(
                    float(pose_info.quat_wxyz[1]),
                    float(pose_info.quat_wxyz[2]),
                    float(pose_info.quat_wxyz[3]),
                    float(pose_info.quat_wxyz[0])
                )

                # 创建Actor
                actor_name = f"kitchen_{i}_{os.path.basename(asset_key)}"
                kitchen_actor_handle = self.gym.create_actor(
                    env_handle,
                    asset,
                    transform,
                    actor_name,
                    2,  # 碰撞组（与kitchen_env.py一致）
                    1  # 碰撞过滤器
                )

                if kitchen_actor_handle != gymapi.INVALID_HANDLE:
                    # 设置缩放
                    try:
                        if isinstance(scale, (list, tuple)) or hasattr(scale, 'tolist'):
                            scale_value = float(scale[0] if isinstance(scale, (list, tuple)) else scale.tolist()[0])
                        else:
                            scale_value = float(scale)
                        self.gym.set_actor_scale(env_handle, kitchen_actor_handle, scale_value)
                    except Exception as e:
                        print(f"⚠️ 设置缩放失败 ({asset_key}): {e}")

                    successful_kitchen_actors.append(kitchen_actor_handle)
                    total_added += 1

            self.kitchen_actors_by_env[i] = successful_kitchen_actors

        print(f"✅ 成功添加了 {total_added} 个厨房Actor到 {self.num_envs} 个环境中")



    # --- 重写 Simulation 和 Environment 创建 ---

    def create_sim(self):
        """ 重写 create_sim 以按正确顺序加载 Kitchen.
            这个方法实际上由父类 (LeggedRobot) 的 __init__ 调用，
            这里只是为了明确逻辑顺序。
            实际的 Sim 创建由外部 (train_curriculum.py) 完成。
            这个方法现在负责创建地面和调用 _create_envs。
        """
        print("--- create_sim (G1KitchenNavigation) ---")
        self.create_sim_called = True
        # Ground plane/Terrain should be created *before* actors

        if not hasattr(self.sim_params, "up_axis"):
            self.sim_params.up_axis = 2  # Z-up
        if not hasattr(self.sim_params, "gravity"):
            self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)  # 正常重力

        # 确保碰撞参数合理
        if hasattr(self.sim_params, "physx"):
            self.sim_params.physx.contact_offset = 0.01  # 增加碰撞偏移量
            self.sim_params.physx.solver_type = 1  # 使用PGS求解器可能更稳定

        if hasattr(self, '_create_terrain'): self._create_terrain()
        else: self._create_ground_plane()
        print("  ✅ Ground/Terrain created.")
        print("  即将调用 _load_kitchen_assets")
        self._load_kitchen_assets()  # 加载 Kitchen 资产
        print("  _load_kitchen_assets 调用完成")
        self._create_envs() # 创建环境（包括机器人和 Kitchen actors）
        # prepare_sim is called AFTER _create_envs in the parent __init__

        return self.sim

    def _create_envs(self):
        """ 重写环境创建，添加 Kitchen Actors """
        print("--- _create_envs (G1KitchenNavigation) ---")
        self.create_envs_called = True

        # --- 确保厨房资产已加载 ---
        if not self.kitchen_assets:
            print("  厨房资产尚未加载，先加载厨房资产...")
            self._load_kitchen_assets()

        # --- 1. 加载机器人资源 (与父类 LeggedRobot 一致) ---
        # Define G1_ROOT_DIR locally if needed
        G1_ROOT_DIR_LOCAL = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) if '__file__' in globals() else ""
        asset_path = self.cfg.asset.file.format(G1_ROOT_DIR=G1_ROOT_DIR_LOCAL)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)
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

        print(f"  加载机器人资源: {asset_file}")
        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if robot_asset is None: raise RuntimeError("Failed to load robot asset.")
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)  # Robot DoFs (43)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)  # Robot Bodies
        if self.cfg.env.num_actions != self.num_dof:
            print(
                f"⚠️ _create_envs Warning: cfg.env.num_actions ({self.cfg.env.num_actions}) != asset num_dof ({self.num_dof}). Overriding.")
            self.cfg.env.num_actions = self.num_dof
        self.num_actions = self.num_dof
        print(f"  机器人资源加载: Num DoF={self.num_dof}, Num Bodies={self.num_bodies}")

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on: penalized_contact_names.extend(
            [s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on: termination_contact_names.extend(
            [s for s in body_names if name in s])

        # --- 2. 初始化列表 ---
        self.envs = []
        self.env_actor_handles = []  # 嵌套列表，每个元素存储一个环境的所有actors
        self.kitchen_actors_by_env = [[] for _ in range(self.num_envs)]

        # --- 3. 获取原点和初始状态 ---
        self._get_env_origins()
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        robot_start_pose = gymapi.Transform()

        # --- 4. 循环创建环境 ---
        env_lower = gymapi.Vec3(0., 0., 0.);
        env_upper = gymapi.Vec3(0., 0., 0.)
        kitchen_base_offset = torch.tensor(getattr(self.cfg.asset, 'kitchen_origin_offset', [0.0, 0.0, 0.0]),
                                           device=self.device)
        print(f"  Creating {self.num_envs} environments with Kitchen...")

        for i in range(self.num_envs):
            # 创建单个环境
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            if env_handle == gymapi.INVALID_HANDLE: raise RuntimeError(f"Failed to create env {i}")
            self.envs.append(env_handle)
            env_actor_handles = []  # 当前环境的所有actor句柄

            # 设置机器人姿态
            env_origin = self.env_origins[i]
            robot_start_pose.p = gymapi.Vec3(env_origin[0] + self.base_init_state[0],
                                             env_origin[1] + self.base_init_state[1],
                                             env_origin[2] + self.base_init_state[2])
            robot_start_pose.r = gymapi.Quat(*self.base_init_state[3:7])

            # 创建机器人actor
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            robot_actor_handle = self.gym.create_actor(env_handle, robot_asset, robot_start_pose, self.cfg.asset.name,
                                                       0,  # 碰撞组0用于机器人
                                                       self.cfg.asset.self_collisions, 0)  # 过滤器0
            if robot_actor_handle == gymapi.INVALID_HANDLE: raise RuntimeError(
                f"Failed to create robot actor in env {i}")
            env_actor_handles.append(robot_actor_handle)

            # 设置机器人属性
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, robot_actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, robot_actor_handle)
            if body_props:
                processed_body_props = self._process_rigid_body_props(body_props, i)
                self.gym.set_actor_rigid_body_properties(env_handle, robot_actor_handle, processed_body_props,
                                                         recomputeInertia=True)

            # 立即在同一环境中创建所有厨房actor
            kitchen_actors_in_env = []
            for asset_key, asset in self.kitchen_assets.items():
                kitchen_handle = self._add_kitchen_actor(i, env_handle, asset_key, asset, env_origin,
                                                         kitchen_base_offset)
                if kitchen_handle != gymapi.INVALID_HANDLE:
                    kitchen_actors_in_env.append(kitchen_handle)
                    env_actor_handles.append(kitchen_handle)

            # 为当前环境存储厨房actor句柄
            self.kitchen_actors_by_env[i] = kitchen_actors_in_env

            # 将当前环境的所有actor句柄存入嵌套列表
            self.env_actor_handles.append(env_actor_handles)

        # --- 5. 构建扁平化的actor_handles列表 ---
        flat_actor_handles = []
        for env_actors in self.env_actor_handles:
            flat_actor_handles.extend(env_actors)
        self.actor_handles = flat_actor_handles

        print(f"  创建了 {len(self.envs)} 个环境, 总共 {len(self.actor_handles)} 个actors")

        # --- 6. 设置索引 (在所有 Actors 创建后) ---
        if self.actor_handles:
            # 获取第一个环境的机器人actor句柄
            first_robot_handle = self.env_actor_handles[0][0]  # 第一个环境的第一个actor是机器人
            self.dof_props_storage = self.gym.get_actor_dof_properties(self.envs[0], first_robot_handle)

            # 设置feet_indices
            self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
            for idx, name in enumerate(feet_names):
                handle = self.gym.find_actor_rigid_body_handle(self.envs[0], first_robot_handle, name)
                if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Foot body '{name}' not found.")
                self.feet_indices[idx] = handle  # Store local body index

            # Penalised/Termination indices (local to robot)
            self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long,
                                                         device=self.device, requires_grad=False)
            for idx, name in enumerate(penalized_contact_names):
                handle = self.gym.find_actor_rigid_body_handle(self.envs[0], first_robot_handle, name)
                if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Penalised body '{name}' not found.")
                self.penalised_contact_indices[idx] = handle

            self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long,
                                                           device=self.device, requires_grad=False)
            for idx, name in enumerate(termination_contact_names):
                handle = self.gym.find_actor_rigid_body_handle(self.envs[0], first_robot_handle, name)
                if handle == gymapi.INVALID_HANDLE: print(f"⚠️ Warning: Termination body '{name}' not found.")
                self.termination_contact_indices[idx] = handle

            # Store robot non-foot indices (local)
            self.robot_non_foot_indices = [j for j in range(self.num_bodies) if
                                           j not in self.feet_indices.cpu().tolist()]
            print(f"  Feet indices (local): {self.feet_indices.cpu().tolist()}")
        else:
            print("❌ ERROR: No actors created.")
            # Init empty tensors
            self.feet_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.penalised_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.termination_contact_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self.robot_non_foot_indices = []

        print(f"--- _create_envs (G1KitchenNavigation) Done ---")

    def _add_kitchen_actor(self, env_idx, env_handle, asset_key, asset, env_origin, kitchen_base_offset):
        """ Helper to add a single kitchen actor, returns handle """
        pose_info = self.kitchen_poses[asset_key]
        scale = self.kitchen_scales.get(asset_key, 1.0)

        # 计算位置
        transform = gymapi.Transform()
        transform.p = gymapi.Vec3(env_origin[0] + (pose_info.pos[0] - kitchen_base_offset[0]),
                                  env_origin[1] + (pose_info.pos[1] - kitchen_base_offset[1]),
                                  env_origin[2] + (pose_info.pos[2] - kitchen_base_offset[2]))
        transform.r = gymapi.Quat(pose_info.quat_wxyz[1], pose_info.quat_wxyz[2], pose_info.quat_wxyz[3],
                                  pose_info.quat_wxyz[0])

        # 创建Actor
        kitchen_actor_handle = self.gym.create_actor(env_handle, asset, transform,
                                                     f"kitchen_{env_idx}_{os.path.basename(asset_key)}",
                                                     1,  # Collision group
                                                     0)  # Collision filter

        if kitchen_actor_handle == gymapi.INVALID_HANDLE:
            print(f"    ❌ ERROR: Failed to create kitchen actor '{asset_key}'")
            return gymapi.INVALID_HANDLE

        # 适当应用缩放 - 使用第一个值作为统一缩放
        try:
            # 确保scale是一个浮点数
            if isinstance(scale, (list, tuple, np.ndarray)) or hasattr(scale, 'tolist'):
                scale_value = float(scale[0] if isinstance(scale, (list, tuple)) else scale.tolist()[0])
            else:
                scale_value = float(scale)

            self.gym.set_actor_scale(env_handle, kitchen_actor_handle, scale_value)
        except Exception as e:
            print(f"    ⚠️ Warning: Failed to apply scale {scale} to '{asset_key}': {e}")

        return kitchen_actor_handle

    def _reset_dofs(self, env_ids):
        """ 在G1KitchenNavigation中重写DOF重置以适应复杂环境 """
        if len(env_ids) == 0:
            return

        # 生成随机位置和零速度，与基类相同
        new_dof_pos = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof),
                                                              device=self.device)
        new_dof_vel = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        # 更新内部缓冲区
        self.dof_pos[env_ids] = new_dof_pos
        self.dof_vel[env_ids] = new_dof_vel

        # 对于厨房环境，直接使用索引更新方法，不尝试视图操作
        # 因为我们知道在多Actor环境中视图操作会失败
        dof_state_cpu = self.dof_state.cpu().numpy()

        # 对每个环境单独更新
        for i, env_id in enumerate(env_ids.cpu().numpy()):
            # 获取此环境的DOF起始索引
            if env_id < len(self.robot_dof_start_indices):
                start_idx = self.robot_dof_start_indices[env_id]

                # 更新位置和速度
                for j in range(self.num_dof):
                    dof_state_cpu[start_idx + j, 0] = new_dof_pos[i, j].cpu().numpy()
                    dof_state_cpu[start_idx + j, 1] = new_dof_vel[i, j].cpu().numpy()

        # 将更新后的状态复制回GPU
        self.dof_state.copy_(torch.tensor(dof_state_cpu, device=self.device))

        # 使用全局张量更新
        try:
            self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_state))
            return  # 成功则直接返回
        except Exception as e:
            print(f"⚠️ set_dof_state_tensor失败，尝试备用方法: {e}")

        # 备用方法：使用单个环境API逐个更新
        for i, env_id in enumerate(env_ids.cpu().numpy()):
            if env_id >= len(self.envs):
                continue

            # 找到此环境的机器人actor句柄
            robot_handle = self.env_actor_handles[env_id][0]  # 第一个actor是机器人

            # 创建numpy数组以传递给gym API
            dof_states = np.zeros((self.num_dof, 2), dtype=np.float32)
            dof_states[:, 0] = new_dof_pos[i].cpu().numpy()
            dof_states[:, 1] = new_dof_vel[i].cpu().numpy()

            # 使用单个环境API设置DOF状态
            try:
                self.gym.set_actor_dof_states(
                    self.envs[env_id],
                    robot_handle,
                    dof_states,
                    gymapi.STATE_ALL
                )
            except Exception as e3:
                print(f"❌ 环境 {env_id} 的单个环境API也失败: {e3}")

    def _manual_dof_update(self):
        """实现手动DOF状态更新，确保没有NaN或Inf"""
        try:
            # 刷新DOF状态张量
            self.gym.refresh_dof_state_tensor(self.sim)

            # 对应每个环境
            for i in range(self.num_envs):
                if i >= len(self.robot_dof_start_indices):
                    continue

                start_idx = self.robot_dof_start_indices[i]
                end_idx = start_idx + self.num_dof

                if end_idx > self.dof_state.shape[0]:
                    continue

                # 提取DOF位置和速度
                pos = self.dof_state[start_idx:end_idx, 0].clone()
                vel = self.dof_state[start_idx:end_idx, 1].clone()

                # 检查并处理非法值
                pos = torch.where(torch.isnan(pos) | torch.isinf(pos),
                                  torch.zeros_like(pos), pos)
                vel = torch.where(torch.isnan(vel) | torch.isinf(vel),
                                  torch.zeros_like(vel), vel)

                # 更新位置和速度
                self.dof_pos[i] = pos
                self.dof_vel[i] = vel

            return True
        except Exception as e:
            print(f"手动DOF更新失败: {e}")
            import traceback
            traceback.print_exc()
            return False



    # --- 重写 Buffer 初始化以适应多个 Actor ---
    def _init_buffers(self):
        """ 重写 _init_buffers 以处理包含 Kitchen Actors 的环境 """
        print("--- _init_buffers (G1KitchenNavigation) ---")

        # --- 1. 获取 Gym 张量 (在 prepare_sim 后调用) ---
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        for name, tensor in zip(["actor_root_state", "dof_state", "net_contact_forces", "rigid_body_state"],
                                [actor_root_state, dof_state_tensor, net_contact_forces, rigid_body_state_tensor]):
             if tensor is None: raise RuntimeError(f"Failed to acquire {name}_tensor.")

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # --- 2. 包装基础张量 ---
        self.root_states = gymtorch.wrap_tensor(actor_root_state)         # Shape: [num_envs * actors_per_env, 13]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)           # Shape: [total_sim_dofs, 2]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor) # Shape: [total_sim_bodies, 13]
        self.contact_forces_raw = gymtorch.wrap_tensor(net_contact_forces) # Raw shape

        # --- 3. 确定 Actor/DoF/Body 结构 ---
        self.actors_per_env = self.gym.get_actor_count(self.envs[0]) # Actors per env
        self.sim_total_dofs = self.gym.get_sim_dof_count(self.sim)
        self.sim_total_bodies = self.gym.get_sim_rigid_body_count(self.sim)
        self.dofs_per_env = self.sim_total_dofs // self.num_envs
        self.bodies_per_env = self.sim_total_bodies // self.num_envs

        print(f"  Buffer Info: num_envs={self.num_envs}, actors_per_env={self.actors_per_env}")
        print(f"               dofs_per_env={self.dofs_per_env}, bodies_per_env={self.bodies_per_env}")
        print(f"  Raw Shapes: root={self.root_states.shape}, dof={self.dof_state.shape}, rigid={self.rigid_body_states.shape}, contact={self.contact_forces_raw.shape}")

        # --- 4. 创建机器人特定的视图 ---
        self.robot_root_indices = torch.arange(0, self.root_states.shape[0], self.actors_per_env, device=self.device, dtype=torch.long)
        self.robot_root_states = self.root_states[self.robot_root_indices]
        self.base_pos = self.robot_root_states[:, 0:3]
        self.base_quat = self.robot_root_states[:, 3:7]

        # --- 5. 创建机器人 DoF 状态的缓冲区和索引 ---
        # (self.num_dof is the robot's DoF count, e.g., 43)
        self.dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        self.dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device, dtype=torch.float)
        self.manual_dof_update = True # Always update manually in multi-actor case

        # Calculate global indices for robot DoFs
        self.robot_dof_start_indices = []
        current_dof_start = 0
        for i in range(self.num_envs):
            self.robot_dof_start_indices.append(current_dof_start)
            # In this env, robot is actor 0, assume its DoFs come first
            current_dof_start += self.dofs_per_env # Increment by total DoFs in the env

        self.robot_dof_start_indices = []
        current_dof_start = 0
        for i in range(self.num_envs):
            self.robot_dof_start_indices.append(current_dof_start)
            # In this env, robot is actor 0, assume its DoFs come first
            current_dof_start += self.dofs_per_env  # Increment by total DoFs in the env

        print(f"  初始化 robot_dof_start_indices (num_envs={self.num_envs}, dofs_per_env={self.dofs_per_env})")
        print(
            f"  前5个索引: {self.robot_dof_start_indices[:5] if len(self.robot_dof_start_indices) >= 5 else self.robot_dof_start_indices}")

        # 在g1_kitchen_navigation.py的_init_buffers方法中，确保这段代码被执行
        all_robot_dof_indices = []
        for i in range(self.num_envs):
            start = self.robot_dof_start_indices[i]
            all_robot_dof_indices.extend(range(start, start + self.num_dof))
        self.all_robot_dof_indices = torch.tensor(all_robot_dof_indices, dtype=torch.long, device=self.device)
        print(f"  已初始化 all_robot_dof_indices (形状: {self.all_robot_dof_indices.shape})")

        # --- 6. Reshape global tensors for easier access ---
        try:
            self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, self.bodies_per_env, 13)
            print(f"  Reshaped rigid_body_states to: {self.rigid_body_states_view.shape}")
        except RuntimeError as e:
            print(f"⚠️ WARNING: Failed to reshape rigid_body_states: {e}. Feet tracking might fail.")
            self.rigid_body_states_view = None

        try:
            self.contact_forces = self.contact_forces_raw.view(self.num_envs, self.bodies_per_env, 3)
            print(f"  Reshaped contact_forces to: {self.contact_forces.shape}")
        except RuntimeError as e:
            print(f"⚠️ WARNING: Failed to reshape contact_forces_raw: {e}. Using raw tensor.")
            self.contact_forces = self.contact_forces_raw # Use raw

        # --- 7. 初始化其他缓冲区 (与 LeggedRobot 类似, 使用机器人维度) ---
        self.rpy = torch.zeros_like(self.base_pos)
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg) # Noise vec for robot obs
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))

        # Action related (size num_actions = robot's num_dof)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros_like(self.actions)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros_like(self.p_gains)
        self.torque_limits = torch.zeros_like(self.p_gains)

        # Global torque buffer (size total_sim_dofs)
        self.torques = torch.zeros(self.sim_total_dofs, dtype=torch.float, device=self.device, requires_grad=False)
        print(f"  Torques buffer shape (KitchenNav): {self.torques.shape}")

        # Last velocities (robot specific shapes)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.robot_root_states[:, 7:13])

        # Commands
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,)

        # Feet contact buffers
        if self.feet_indices.numel() > 0:
            self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
            self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        else: self.feet_air_time = None; self.last_contacts = None

        # Calculated velocities (using robot states)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # --- 8. 设置默认 DoF 位置和 PD 增益 (针对机器人 DoF) ---
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        # Need robot asset's DoF props
        first_robot_handle = self.actor_handles[0]  # Use first robot actor handle
        dof_props_asset = self.gym.get_actor_dof_properties(self.envs[0], first_robot_handle)  # Get props from actor

        for i in range(self.num_dof):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles.get(name, 0.0)
            self.default_dof_pos[i] = angle

            # 安全地转换类型
            effort_value = dof_props_asset["effort"][i]
            self.torque_limits[i] = float(effort_value.item()) if hasattr(effort_value, 'item') else float(effort_value)

            found = False
            for gain_key, stiffness_val in self.cfg.control.stiffness.items():
                if gain_key in name:
                    self.p_gains[i] = stiffness_val
                    self.d_gains[i] = self.cfg.control.damping.get(gain_key, 0.0)
                    found = True;
                    break
            if not found: self.p_gains[i] = 0.; self.d_gains[i] = 0.
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        # --- 9. 初始化足部追踪 (使用 view) ---
        self._init_foot() # Uses self.rigid_body_states_view and self.feet_indices

        print("--- _init_buffers (G1KitchenNavigation) Done ---")


    def _init_foot(self):
        """ 重写以使用正确的视图和索引 """
        if self.feet_indices is None or self.feet_indices.numel() == 0:
            self.feet_num = 0; self.feet_state = None; self.feet_pos = None; self.feet_vel = None; return

        self.feet_num = len(self.feet_indices)

        if self.rigid_body_states_view is not None:
            # feet_indices are *local* robot body indices
            # rigid_body_states_view is [num_envs, bodies_per_env, 13]
            # We need the view for the *robot* bodies within each env slice
            # Assuming robot bodies are the first self.num_bodies in each env slice
            robot_rigid_body_view = self.rigid_body_states_view[:, :self.num_bodies, :]
            if torch.any(self.feet_indices >= robot_rigid_body_view.shape[1]):
                 print(f"❌ ERROR _init_foot: feet_indices ({self.feet_indices.max().item()}) out of bounds for robot body view dim ({robot_rigid_body_view.shape[1]})")
                 self.feet_num = 0; self.feet_state=None; self.feet_pos=None; self.feet_vel=None; return

            self.feet_state = robot_rigid_body_view[:, self.feet_indices, :]
            self.feet_pos = self.feet_state[:, :, :3]
            self.feet_vel = self.feet_state[:, :, 7:10]
            print("  ✅ Initialized foot state views.")
        else:
            print("⚠️ _init_foot: rigid_body_states_view not available. Foot state tracking disabled.")
            self.feet_state = None; self.feet_pos = None; self.feet_vel = None


    def update_feet_state(self):
        """ 更新足部状态 (Kitchen Nav version) """
        if self.feet_num > 0 and self.rigid_body_states_view is not None:
            # Refresh global tensor
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            # Re-slice from the robot's part of the view
            robot_rigid_body_view = self.rigid_body_states_view[:, :self.num_bodies, :]
            if torch.any(self.feet_indices >= robot_rigid_body_view.shape[1]): return # Already warned
            self.feet_state = robot_rigid_body_view[:, self.feet_indices, :]
            self.feet_pos = self.feet_state[:, :, :3]
            self.feet_vel = self.feet_state[:, :, 7:10]


    def _compute_torques(self, actions):
        """ 计算扭矩，填充到全局扭矩张量 (Kitchen Nav version) """
        # --- 1. 计算机器人的扭矩 ---
        if actions.shape[1] != self.num_actions: raise ValueError("Action shape mismatch")
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        # Use robot's DoF states from manually updated buffers
        if not self.manual_dof_update: print("⚠️ _compute_torques expects manual_dof_update=True")
        if self.dof_pos.shape[1] != self.num_actions: raise ValueError("dof_pos shape mismatch")
        if self.dof_vel.shape[1] != self.num_actions: raise ValueError("dof_vel shape mismatch")

        if control_type == "P":
            robot_torques = self.p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains * self.dof_vel
        elif control_type == "V":
            if self.last_dof_vel.shape != self.dof_vel.shape: self.last_dof_vel = torch.zeros_like(self.dof_vel)
            dt = self.dt
            if dt <= 0: dt = 1e-5
            robot_torques = self.p_gains * (actions_scaled - self.dof_vel) - self.d_gains * ((self.dof_vel - self.last_dof_vel) / dt)
        elif control_type == "T": robot_torques = actions_scaled
        else: raise NameError(f"未知控制器类型: {control_type}")

        robot_torques = torch.clip(robot_torques, -self.torque_limits, self.torque_limits) # Shape [num_envs, 43]

        # --- 2. 将机器人扭矩填充到全局扭矩张量 ---
        self.torques.zero_() # Zero out global tensor
        flat_robot_torques = robot_torques.view(-1) # Shape [num_envs * 43]
        if flat_robot_torques.shape == self.all_robot_dof_indices.shape:
             self.torques.index_copy_(0, self.all_robot_dof_indices, flat_robot_torques)
        else: print(f"❌ ERROR _compute_torques: Mismatch flat torques vs indices")

        return self.torques # Return the full torque tensor [sim_total_dofs]


    # --- 导航和特定阶段逻辑 ---
    def _initialize_kitchen_waypoints(self):
        """ 初始化厨房导航点 """
        print("--- _initialize_kitchen_waypoints (G1KitchenNavigation) ---")
        # 从配置或默认值获取
        # 确保坐标相对于 Kitchen 场景原点或世界原点一致
        # !!! 这些坐标需要根据你的实际场景精确设置 !!!
        default_waypoints = {
            "entrance": [0.0, -1.0, 0.0],       # Env origin entrance area
            "fridge_front": [1.5, 1.0, 0.0],    # In front of fridge
            "counter_middle": [0.0, 2.5, 0.0],   # Middle of counter
            "table_center": [-1.5, 1.0, 0.0],   # Center of table area
        }
        wp_dict_cfg = getattr(self.cfg, 'waypoints', default_waypoints)
        self.waypoints = {name: torch.tensor(pos, device=self.device) for name, pos in wp_dict_cfg.items()}

        self.waypoint_names = list(self.waypoints.keys())
        if not self.waypoint_names: print("⚠️ 未定义任何导航点。"); return
        self.waypoint_positions = torch.stack(list(self.waypoints.values()))
        print(f"  Defined {len(self.waypoint_names)} waypoints: {self.waypoint_names}")

        sub_stage = self.curriculum_sub_stage # Use instance attribute
        # Example sub-stage logic (adjust as needed)
        if sub_stage <= 2:   self.active_waypoints = ["entrance", "fridge_front", "counter_middle"]
        else:                self.active_waypoints = self.waypoint_names
        try:
            self.active_waypoint_indices = [self.waypoint_names.index(name) for name in self.active_waypoints if name in self.waypoint_names] # Filter only existing names
            print(f"  Sub-stage {sub_stage}: Active waypoints = {self.active_waypoints} (Indices: {self.active_waypoint_indices})")
        except ValueError as e:
             print(f"❌ Error finding waypoint index: {e}")
             self.active_waypoint_indices = list(range(len(self.waypoint_names))) # Fallback


    def _sample_navigation_targets(self, env_ids=None):
        """ 为指定环境采样新的、不同于上一个目标的导航点 """
        if env_ids is None: env_ids = torch.arange(self.num_envs, device=self.device)
        if len(env_ids) == 0 or not self.active_waypoint_indices: return

        num_active = len(self.active_waypoint_indices)
        if num_active <= 1:
            new_indices_in_active_list = torch.zeros_like(env_ids)
        else:
            new_indices_in_active_list = torch.randint(0, num_active, (len(env_ids),), device=self.device)
            current_target_full_idx = self.current_waypoint_idx[env_ids]
            active_indices_tensor = torch.tensor(self.active_waypoint_indices, device=self.device)

            for i, env_id in enumerate(env_ids):
                 current_wp_idx = current_target_full_idx[i].item()
                 current_active_idx = -1
                 if current_wp_idx in self.active_waypoint_indices:
                      current_active_idx = self.active_waypoint_indices.index(current_wp_idx)

                 num_tries = 0
                 while new_indices_in_active_list[i].item() == current_active_idx and num_tries < 5:
                       new_indices_in_active_list[i] = torch.randint(0, num_active, (1,), device=self.device)[0]
                       num_tries += 1

        new_waypoint_indices = active_indices_tensor[new_indices_in_active_list]
        self.last_waypoint_idx[env_ids] = self.current_waypoint_idx[env_ids].clone()
        self.current_waypoint_idx[env_ids] = new_waypoint_indices
        self.target_positions[env_ids] = self.waypoint_positions[new_waypoint_indices]
        self.reached_waypoint[env_ids] = False
        self.collision_detected[env_ids] = False


    def _get_collision_body_indices(self):
        """ 获取用于碰撞检测的 Body Indices """
        # Robot non-foot bodies (local indices 0 to num_bodies-1)
        if self.feet_indices is not None:
             self.robot_non_foot_indices = [j for j in range(self.num_bodies) if j not in self.feet_indices.cpu().tolist()]
             print(f"  Robot non-foot indices (local): {self.robot_non_foot_indices}")
        else:
             self.robot_non_foot_indices = list(range(self.num_bodies)) # Assume all bodies if feet unknown
             print("  ⚠️ Feet indices not found, using all robot bodies for collision check.")


    def _check_collision(self):
        """ 检查机器人非足部与环境的碰撞 """
        if self.contact_forces is None or not self.robot_non_foot_indices:
             return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Assuming self.contact_forces is shape [num_envs, bodies_per_env, 3]
        # Robot bodies are the first self.num_bodies slice
        robot_contact_forces = self.contact_forces[:, :self.num_bodies, :]
        non_foot_forces_norm = torch.norm(robot_contact_forces[:, self.robot_non_foot_indices, :], dim=2)

        collision_threshold = getattr(self.cfg.rewards, 'collision_force_threshold', 5.0)
        collision = torch.any(non_foot_forces_norm > collision_threshold, dim=1)

        # Update latched collision flag for the current target pursuit
        newly_collided = collision & (~self.collision_detected) # Detect only new collisions
        if newly_collided.any():
             self.collision_detected[newly_collided] = True # Latch collision
             self.consecutive_successes[newly_collided] = 0 # Reset success count on collision
             # print(f"Collision detected in envs: {torch.nonzero(newly_collided).squeeze(-1).cpu().tolist()}")

        return collision # Return per-step collision status


    def _check_waypoint_reached(self, distance_threshold=None):
        """ 检查是否到达当前导航点 """
        if distance_threshold is None:
             distance_threshold = getattr(self.cfg.rewards, 'waypoint_reach_threshold', 0.5)
        if self.target_positions is None: return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        robot_pos_xy = self.robot_root_states[:, :2]
        target_pos_xy = self.target_positions[:, :2]
        distances = torch.norm(robot_pos_xy - target_pos_xy, dim=1)
        self.dist_to_target = distances
        reached = distances < distance_threshold
        # Don't latch self.reached_waypoint here, do it in post_physics_step based on resample condition
        return reached # Return per-step reach status

    def compute_observations(self):
        """ Computes G1 specific base observations (e.g., 140 dims including phase).
            Child classes will call this via super() and append their specific observations.
        """
        # --- 1. 计算通用组件 ---
        # Phase (ensure it's calculated in _post_physics_step_callback)
        if not hasattr(self, 'phase'): self.phase = torch.zeros(self.num_envs, device=self.device)
        sin_phase = torch.sin(2 * torch.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * torch.pi * self.phase).unsqueeze(1)

        # Scaled DoF states (ensure dof_pos/vel are correct shape [num_envs, num_dof])
        if self.dof_pos.shape != (self.num_envs, self.num_dof) or self.dof_vel.shape != (self.num_envs, self.num_dof):
            print(f"❌ ERROR compute_observations: DOF state shape mismatch!")
            # Fallback to zeros with correct shape for buffer concatenation
            dof_pos_scaled = torch.zeros((self.num_envs, self.num_dof), device=self.device)
            dof_vel_scaled = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        else:
            dof_pos_scaled = (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos
            dof_vel_scaled = self.dof_vel * self.obs_scales.dof_vel

        # Scaled base velocities and commands
        base_ang_vel_scaled = self.base_ang_vel * self.obs_scales.ang_vel
        commands_scaled = self.commands[:, :3] * self.commands_scale  # Assuming commands_scale is set

        # Previous actions (ensure shape is correct)
        actions_to_include = self.actions
        if actions_to_include.shape != (self.num_envs, self.num_actions):
            print(
                f"⚠️ WARNING compute_observations: self.actions shape {actions_to_include.shape} unexpected. Using zeros.")
            actions_to_include = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # --- 2. 组装 G1 基础观测列表 (140 维) ---
        # 顺序必须与 _get_noise_scale_vec 中的假设一致
        # 3(ang_vel) + 3(gravity) + 3(commands) + 43(dof_pos) + 43(dof_vel) + 43(actions) + 2(phase) = 140
        obs_list = [
            base_ang_vel_scaled,  # 3
            self.projected_gravity,  # 3
            commands_scaled,  # 3
            dof_pos_scaled,  # 43 (num_actions)
            dof_vel_scaled,  # 43 (num_actions)
            actions_to_include,  # 43 (num_actions) - previous actions
            sin_phase,  # 1
            cos_phase  # 1
        ]

        # --- 3. 拼接成最终的 obs_buf ---
        try:
            self.obs_buf = torch.cat(obs_list, dim=-1)
            # 验证维度
            if self.obs_buf.shape[1] != self.num_observations:
                print(
                    f"❌ ERROR: G1CurriculumBase computed obs dim ({self.obs_buf.shape[1]}) != configured num_observations ({self.num_observations}). Check obs_list!")
                # Attempt to fix shape for safety, though it indicates a logic error
                if self.obs_buf.shape[1] > self.num_observations:
                    self.obs_buf = self.obs_buf[:, :self.num_observations]
                else:
                    padding = torch.zeros((self.num_envs, self.num_observations - self.obs_buf.shape[1]),
                                          device=self.device)
                    self.obs_buf = torch.cat([self.obs_buf, padding], dim=-1)

        except Exception as e:
            print("❌ G1CurriculumBase ERROR concatenating observation buffer:")
            for i, item in enumerate(obs_list): print(
                f"  Item {i}: shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
            print(f"  Error: {e}")
            # Fallback to zeros
            self.obs_buf = torch.zeros(self.num_envs, self.num_observations, device=self.device, dtype=torch.float)

        # --- 4. 组装特权观测 ---
        if self.privileged_obs_buf is not None:
            # 特权观测 = 基础线速度(3) + 普通观测(140) = 143 (这是基础特权观测)
            priv_obs_list = [
                self.base_lin_vel * self.obs_scales.lin_vel,  # 3
                self.obs_buf  # 140 (G1 base obs)
            ]
            try:
                self.privileged_obs_buf = torch.cat(priv_obs_list, dim=-1)
                # 验证维度
                if self.privileged_obs_buf.shape[1] != self.num_privileged_obs:
                    print(
                        f"❌ ERROR: G1CurriculumBase computed priv_obs dim ({self.privileged_obs_buf.shape[1]}) != configured num_privileged_obs ({self.num_privileged_obs}). Check priv_obs_list!")
                    # Attempt to fix shape
                    if self.privileged_obs_buf.shape[1] > self.num_privileged_obs:
                        self.privileged_obs_buf = self.privileged_obs_buf[:, :self.num_privileged_obs]
                    else:
                        padding = torch.zeros(
                            (self.num_envs, self.num_privileged_obs - self.privileged_obs_buf.shape[1]),
                            device=self.device)
                        self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, padding], dim=-1)

            except Exception as e:
                print("❌ G1CurriculumBase ERROR concatenating privileged observation buffer:")
                for i, item in enumerate(priv_obs_list): print(
                    f"  Item {i}: shape={item.shape if hasattr(item, 'shape') else 'N/A'}")
                print(f"  Error: {e}")
                # Fallback to zeros
                self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device,
                                                      dtype=torch.float)

        # --- 5. 添加噪声 (在观测计算完成后) ---
        if self.add_noise:
            # Check noise vector compatibility
            if self.noise_scale_vec is not None and self.noise_scale_vec.shape[0] == self.num_observations:
                self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            else:
                # Warning already printed in _get_noise_scale_vec if size mismatch
                pass



    # --- Stage 2 Specific Reward Functions ---
    def _reward_waypoint_distance(self):
        if self.dist_to_target is None: return torch.zeros(self.num_envs, device=self.device)
        k = getattr(self.cfg.rewards, 'distance_reward_k', 0.5)
        return torch.exp(-k * self.dist_to_target)

    def _reward_waypoint_reached(self):
        # Reward only on the step the waypoint is first reached before resampling
        return self.reached_waypoint.float() # Use the per-step flag

    def _reward_kitchen_collision(self):
        # Penalize based on the per-step collision check
        collision_this_step = self._check_collision() # This also updates the latched flag
        return collision_this_step.float() # Returns 1.0 if collision this step, 0.0 otherwise

    def _reward_target_facing(self):
        if self.target_positions is None: return torch.zeros(self.num_envs, device=self.device)
        robot_pos_xy = self.robot_root_states[:, :2]
        target_pos_xy = self.target_positions[:, :2]
        target_dir_world = target_pos_xy - robot_pos_xy
        target_dir_norm = torch.norm(target_dir_world, dim=1, keepdim=True)
        target_dir_world = target_dir_world / (target_dir_norm + 1e-8)

        robot_fwd_world = quat_apply(self.robot_root_states[:, 3:7], self.forward_vec)[:, :2]
        robot_fwd_norm = torch.norm(robot_fwd_world, dim=1, keepdim=True)
        robot_fwd_world = robot_fwd_world / (robot_fwd_norm + 1e-8)

        cos_similarity = torch.sum(robot_fwd_world * target_dir_world, dim=1)
        # Reward based on cosine similarity (e.g., scale to [0, 1])
        facing_reward = torch.clamp(cos_similarity, min=0.0) # Only reward positive alignment
        # facing_reward = (cos_similarity + 1.0) / 2.0 # Scale -1..1 to 0..1
        return facing_reward

    # compute_reward: Inherited from G1CurriculumBase/LeggedRobot,
    # it will call all _reward_* functions found based on reward_scales.

    # --- Stage 2 Success Criteria ---
    def compute_success_criteria(self):
        """ Stage 2 Success: Reached N waypoints consecutively without collision. """
        success_threshold = getattr(self.cfg.curriculum, 'stage2_success_waypoint_count', 3)
        # Check the counter *before* potentially resetting it in post_physics_step
        success_flags = self.consecutive_successes >= success_threshold
        return success_flags


    def post_physics_step(self):
        """ Stage 2 Post Physics Step """
        # --- 1. 调用父类 (G1CurriculumBase) -> LeggedRobot ---

        if hasattr(self, '_manual_dof_update'):
            self._manual_dof_update()
        else:
            print("警告: _manual_dof_update 方法不存在")
        # Handles refreshing tensors, base state updates, phase calculation, command resampling, heading command
        super().post_physics_step() # Calls G1CurriculumBase._post_physics_step_callback internally

        # --- 2. Stage 2 Specific Logic: Collision and Waypoint Reached ---
        self._check_collision() # Updates self.collision_detected, resets counter if collision
        reached = self._check_waypoint_reached() # Per-step check

        # --- 3. Re-sample Target if Reached Successfully ---
        resample_condition = reached & (~self.collision_detected) # Must reach *without* collision currently latched
        env_ids_to_resample = torch.nonzero(resample_condition).squeeze(-1)
        if len(env_ids_to_resample) > 0:
             # Increment consecutive success counter *before* resampling
             self.consecutive_successes[env_ids_to_resample] += 1
             self._sample_navigation_targets(env_ids_to_resample) # Samples new target, resets reached_waypoint flag
             # Don't set self.reached_waypoint = True here, use the per-step 'reached' for reward


        # --- 4. Compute Observations, Rewards, Termination ---
        # These are called by the parent's post_physics_step *after* the callback.
        # We might need to recompute observations if target changed? Let's test.
        # Recompute observations here AFTER potential target change
        self.compute_observations()

        # --- 5. Set extras for Runner ---
        success_flags = self.compute_success_criteria()
        self.extras["success_flags"] = success_flags.clone()
        # Reset counter only AFTER checking success for this step
        if success_flags.any():
             self.consecutive_successes[success_flags] = 0

        # --- 6. Handle Resets (Parent's post_physics_step calls reset_idx) ---
        # Our overridden reset_idx will handle navigation state resets.

    def reset_idx(self, env_ids):
         """ 重写 reset_idx 以重置导航状态 """
         if len(env_ids) == 0: return
         # print(f"Resetting envs (G1KitchenNav): {env_ids.cpu().tolist()}")
         # 调用父类 reset (重置机器人状态、命令、基础缓冲区等)
         super().reset_idx(env_ids) # Calls LeggedRobot.reset_idx -> BaseTask.reset_idx placeholder -> G1BasicLocomotion.reset_idx
         # G1BasicLocomotion reset_idx resets streak counter, we need to reset nav state here

         # 重置本阶段特定的状态
         if self.consecutive_successes is not None: self.consecutive_successes[env_ids] = 0
         if self.collision_detected is not None: self.collision_detected[env_ids] = False
         if self.reached_waypoint is not None: self.reached_waypoint[env_ids] = False
         self._sample_navigation_targets(env_ids) # Sample initial targets for reset envs
         # print(f"  G1KitchenNavigation: Reset navigation state for {len(env_ids)} environments.")