
# model_transfer.py
import os
import torch
import numpy as np
from copy import deepcopy

class ModelTransfer:
    """模型迁移工具，处理不同阶段间的策略迁移，特别是输入维度变化。"""

    def __init__(self, cfg):
        # cfg is expected to be the model_transfer sub-config, e.g., cfg.curriculum.model_transfer
        self.cfg = cfg
        # Determine device: check if cfg has device, otherwise default
        self.device = getattr(cfg, 'device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        # Get init scale for new weights, provide default
        self.init_scale = getattr(cfg, 'init_scale', 0.01)
        self.transfer_weights_flag = getattr(cfg, 'transfer_weights', True)
        print(f"  ModelTransfer: Initialized with device='{self.device}', init_scale={self.init_scale}, transfer_weights={self.transfer_weights_flag}")


    def transfer_policy(self, old_policy_state_dict, old_cfg, new_cfg, target_policy):
        """
        将旧策略的状态字典迁移到新的策略对象 (target_policy)。
        处理观察空间和动作空间维度的变化。

        Args:
            old_policy_state_dict (dict): 从检查点加载的旧策略 state_dict。
            old_cfg (object): 旧阶段的环境配置 (用于获取旧维度)。
            new_cfg (object): 新阶段的环境配置 (用于获取新维度)。
            target_policy (torch.nn.Module): 要加载状态的新策略模型实例。

        Returns:
            torch.nn.Module: 加载并可能调整了权重的目标策略模型。
        """
        if not self.transfer_weights_flag:
            print("  ModelTransfer: transfer_weights is False. Returning target policy without loading state.")
            # Optionally re-initialize target_policy here if needed
            return target_policy

        old_obs_dim = old_cfg.env.num_observations
        new_obs_dim = new_cfg.env.num_observations
        old_act_dim = old_cfg.env.num_actions
        new_act_dim = new_cfg.env.num_actions

        print(f"  ModelTransfer: Attempting transfer...")
        print(f"    Obs dim: {old_obs_dim} -> {new_obs_dim}")
        print(f"    Act dim: {old_act_dim} -> {new_act_dim}")

        # Create a deep copy of the loaded state dict to modify safely
        new_state_dict = deepcopy(old_policy_state_dict)

        # --- Handle Actor Network ---
        if hasattr(target_policy, 'actor'):
            print("    Processing Actor network...")
            actor_state_dict_prefix = 'actor.' # Common prefix for actor weights
            self._adapt_network_weights(new_state_dict, actor_state_dict_prefix,
                                        old_obs_dim, new_obs_dim, # Input dimension change
                                        old_act_dim, new_act_dim, # Output dimension change
                                        is_actor=True)
        else:
            print("    Target policy has no 'actor' attribute.")

        # --- Handle Critic Network ---
        if hasattr(target_policy, 'critic'):
            print("    Processing Critic network...")
            critic_state_dict_prefix = 'critic.' # Common prefix for critic weights
            # Critic output is typically 1 (value), so only input dim changes
            self._adapt_network_weights(new_state_dict, critic_state_dict_prefix,
                                        old_obs_dim, new_obs_dim, # Input dimension change
                                        1, 1,                     # Output dimension (value) remains 1
                                        is_actor=False)
        else:
             print("    Target policy has no 'critic' attribute.")


        # --- Load the (potentially modified) state dict ---
        try:
            # Set strict=False allows loading even if some keys don't match exactly
            # (e.g., if network structure changed beyond input/output layers)
            missing_keys, unexpected_keys = target_policy.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"    Warning: Missing keys when loading state dict: {missing_keys}")
            if unexpected_keys:
                print(f"    Warning: Unexpected keys in state dict: {unexpected_keys}")
            print("    ✅ State dictionary loaded into target policy.")

        except Exception as e:
            print(f"❌ ModelTransfer ERROR: Failed to load state dict into target policy: {e}")
            print("     Returning target policy with potentially partial or no loaded weights.")

        return target_policy


    def _adapt_network_weights(self, state_dict, prefix, old_in_dim, new_in_dim, old_out_dim, new_out_dim, is_actor):
        """
        Helper function to adapt weights within a state_dict for a specific network part (actor/critic).
        Modifies the state_dict in-place.
        """
        # Find the first linear layer (input layer)
        first_layer_weight_key = None
        first_layer_bias_key = None
        for key in state_dict.keys():
            if key.startswith(prefix) and '0.weight' in key: # Assuming '0' is the first layer
                first_layer_weight_key = key
                first_layer_bias_key = key.replace('weight', 'bias')
                break

        # Adapt input layer if dimensions changed
        if first_layer_weight_key and old_in_dim != new_in_dim:
            print(f"      Adapting input layer ({first_layer_weight_key}) for input dim change: {old_in_dim} -> {new_in_dim}")
            old_weight = state_dict[first_layer_weight_key]
            old_bias = state_dict.get(first_layer_bias_key) # Bias might not exist
            out_features = old_weight.shape[0]

            new_weight = torch.zeros((out_features, new_in_dim), device=old_weight.device) # Match device

            # Copy overlapping weights
            copy_dim = min(old_in_dim, new_in_dim)
            new_weight[:, :copy_dim] = old_weight[:, :copy_dim]

            # Initialize new weights if expanding
            if new_in_dim > old_in_dim:
                print(f"      Initializing new input weights ({new_in_dim - old_in_dim} dims) with scale {self.init_scale}")
                new_weight[:, old_in_dim:] = self.init_scale * torch.randn((out_features, new_in_dim - old_in_dim), device=new_weight.device)

            # Update state dict
            state_dict[first_layer_weight_key] = new_weight
            # Bias remains the same as it depends only on output features
            # if old_bias is not None and first_layer_bias_key in state_dict:
            #    state_dict[first_layer_bias_key] = old_bias
            print(f"      Input layer weight shape: {old_weight.shape} -> {new_weight.shape}")


        # Find the last linear layer (output layer) - this is tricky, depends on network structure
        # We need a more robust way than assuming layer index. Let's find based on dimension.
        last_layer_weight_key = None
        last_layer_bias_key = None
        candidate_keys = sorted([k for k in state_dict.keys() if k.startswith(prefix) and 'weight' in k])

        if candidate_keys:
             # Iterate backwards through layers looking for the one with matching output dimension
             for key in reversed(candidate_keys):
                  weight = state_dict[key]
                  if weight.shape[0] == old_out_dim: # Found potential output layer
                       last_layer_weight_key = key
                       last_layer_bias_key = key.replace('weight', 'bias')
                       break

        # Adapt output layer if dimensions changed (typically only for actor)
        if is_actor and last_layer_weight_key and old_out_dim != new_out_dim:
            print(f"      Adapting output layer ({last_layer_weight_key}) for output dim change: {old_out_dim} -> {new_out_dim}")
            old_weight = state_dict[last_layer_weight_key]
            old_bias = state_dict.get(last_layer_bias_key)
            in_features = old_weight.shape[1] # Input features to the last layer

            new_weight = torch.zeros((new_out_dim, in_features), device=old_weight.device)
            new_bias = torch.zeros(new_out_dim, device=old_weight.device) if old_bias is not None else None

            # Copy overlapping weights/biases
            copy_dim = min(old_out_dim, new_out_dim)
            new_weight[:copy_dim, :] = old_weight[:copy_dim, :]
            if new_bias is not None:
                new_bias[:copy_dim] = old_bias[:copy_dim]

            # Initialize new weights/biases if expanding
            if new_out_dim > old_out_dim:
                print(f"      Initializing new output weights/biases ({new_out_dim - old_out_dim} dims) with scale {self.init_scale}")
                new_weight[old_out_dim:, :] = self.init_scale * torch.randn((new_out_dim - old_out_dim, in_features), device=new_weight.device)
                if new_bias is not None:
                    new_bias[old_out_dim:] = self.init_scale * torch.randn(new_out_dim - old_out_dim, device=new_bias.device)

            # Update state dict
            state_dict[last_layer_weight_key] = new_weight
            if new_bias is not None and last_layer_bias_key in state_dict:
                state_dict[last_layer_bias_key] = new_bias
            print(f"      Output layer weight shape: {old_weight.shape} -> {new_weight.shape}")
            if new_bias is not None: print(f"      Output layer bias shape: {old_bias.shape} -> {new_bias.shape}")

        elif is_actor and not last_layer_weight_key and old_out_dim != new_out_dim:
             print(f"    Warning: Could not definitively find the actor output layer to adapt for dim change {old_out_dim} -> {new_out_dim}.")


    # --- Checkpoint saving/loading remains the same ---
    def save_checkpoint(self, policy, env_cfg, total_env_steps, curr_stage_tuple, output_dir):
        """保存策略检查点，包含环境配置信息"""
        # curr_stage_tuple should be (stage, sub_stage)
        stage, sub_stage = curr_stage_tuple

        # We need env_cfg to save dimensions, but saving the whole object might be large/complex.
        # Let's save essential dimensions instead.
        checkpoint = {
            "policy_state_dict": policy.state_dict(),
            "env_config_dims": { # Save key dimensions
                 "num_observations": env_cfg.env.num_observations,
                 "num_privileged_obs": env_cfg.env.num_privileged_obs,
                 "num_actions": env_cfg.env.num_actions,
            },
            "total_env_steps": total_env_steps,
            "curriculum_stage": curr_stage_tuple
        }

        # 创建文件名
        filename = os.path.join(output_dir, f"policy_S{stage}_{sub_stage}_T{total_env_steps}.pt")

        # 保存检查点
        try:
            torch.save(checkpoint, filename)
            # print(f"  ModelTransfer: Checkpoint saved to {filename}")
            return filename
        except Exception as e:
             print(f"❌ ModelTransfer ERROR: Failed to save checkpoint to {filename}: {e}")
             return None

    # def load_checkpoint(self, filename):
    #     """加载策略检查点，返回 state_dict 和相关信息"""
    #     if not os.path.exists(filename):
    #         print(f"❌ ModelTransfer ERROR: Checkpoint file not found: {filename}")
    #         return None, None, None, None
    #
    #     try:
    #         checkpoint = torch.load(filename, map_location=self.device)
    #
    #         # 提取信息
    #         policy_state_dict = checkpoint["policy_state_dict"]
    #         env_config_dims = checkpoint["env_config_dims"] # Load dimensions dict
    #         total_env_steps = checkpoint["total_env_steps"]
    #         curr_stage_tuple = checkpoint["curriculum_stage"]
    #
    #         # print(f"  ModelTransfer: Checkpoint loaded from {filename}")
    #         # print(f"    - Stage: {curr_stage_tuple}, Steps: {total_env_steps}")
    #         # print(f"    - Dims: Obs={env_config_dims['num_observations']}, PrivObs={env_config_dims['num_privileged_obs']}, Act={env_config_dims['num_actions']}")
    #
    #         return policy_state_dict, env_config_dims, total_env_steps, curr_stage_tuple
    #
    #     except Exception as e:
    #         print(f"❌ ModelTransfer ERROR: Failed to load checkpoint from {filename}: {e}")
    #         return None, None, None, None

    # 将这些修改添加到model_transfer.py的load_checkpoint方法中
    def load_checkpoint(self, checkpoint_path):
        """加载检查点并返回模型状态和环境维度"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 查找策略状态字典（支持多种键名）
            policy_state_dict = None
            for key in ['model_state_dict', 'actor_critic_state_dict', 'state_dict']:
                if key in checkpoint:
                    policy_state_dict = checkpoint[key]
                    print(f"  找到策略状态字典，使用键: '{key}'")
                    break

            # 环境维度信息
            env_dims = {'num_observations': 0, 'num_actions': 0, 'num_privileged_obs': 0}

            # 尝试多种可能的键结构来找环境维度
            if 'env_dims' in checkpoint:
                env_dims.update(checkpoint['env_dims'])
            elif 'env' in checkpoint and isinstance(checkpoint['env'], dict):
                env_dims.update(checkpoint['env'])
            else:
                # 从网络结构推断维度
                try:
                    if policy_state_dict:
                        # 尝试从actor网络结构推断
                        if 'actor.0.weight' in policy_state_dict:
                            actor_input_layer = policy_state_dict['actor.0.weight']
                            env_dims['num_observations'] = actor_input_layer.shape[1]

                        # 尝试从critic网络结构推断
                        if 'critic.0.weight' in policy_state_dict:
                            critic_input_layer = policy_state_dict['critic.0.weight']
                            env_dims['num_privileged_obs'] = critic_input_layer.shape[1]

                        # 尝试从最后一层推断动作维度
                        for key in policy_state_dict:
                            if '.weight' in key and 'actor' in key:
                                act_layer = policy_state_dict[key]
                                if len(act_layer.shape) == 2:  # 只考虑全连接层
                                    env_dims['num_actions'] = max(env_dims['num_actions'], act_layer.shape[0])
                except Exception as e:
                    print(f"  ⚠️ 从网络结构推断维度失败: {e}")

                print("⚠️ 无法找到环境维度信息，使用默认值")

            # 获取其他元数据
            loaded_steps = checkpoint.get('iterations', checkpoint.get('iter', 0)) * 1000
            loaded_stage = checkpoint.get('stage', 0)

            return policy_state_dict, env_dims, loaded_steps, loaded_stage
        except Exception as e:
            print(f"  ❌ 加载检查点失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0, 0
