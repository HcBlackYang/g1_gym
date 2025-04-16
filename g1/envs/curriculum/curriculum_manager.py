
# # curriculum_manager.py
# import os
# import yaml
# import torch
# import numpy as np
# from collections import deque
# import matplotlib.pyplot as plt
# from datetime import datetime
# import time # For potential timing checks


# class CurriculumManager:
#     """课程学习管理器，负责管理课程进度、状态和模型路径"""

#     def __init__(self, cfg):
#         self.cfg = cfg # Keep the main config accessible

#         # 课程设置 (从传入的 cfg 中获取)
#         curriculum_cfg = cfg.curriculum # Access the curriculum sub-config
#         self.current_stage = curriculum_cfg.initial_stage
#         self.current_sub_stage = curriculum_cfg.initial_sub_stage
#         self.max_stages = curriculum_cfg.max_stages
#         self.max_sub_stages = curriculum_cfg.max_sub_stages

#         # 进阶条件
#         self.success_threshold = curriculum_cfg.success_threshold
#         # self.min_episodes = curriculum_cfg.min_episodes # Might be better to use env steps
#         self.min_steps_between_eval = getattr(curriculum_cfg, 'min_steps_between_eval', 1000000) # Min steps before checking advancement
#         self.evaluation_window_size = curriculum_cfg.evaluation_window # Number of samples for rate calculation

#         # 训练跟踪 (使用标量成功率，由 Runner 提供)
#         # self.success_history = deque(maxlen=self.evaluation_window_size) # Not used if using scalar rate
#         self.scalar_success_rate_history = deque(maxlen=50) # Store recent scalar rates for smoothing/logging
#         self.reward_history = deque(maxlen=self.evaluation_window_size) # Still useful for logging rewards
#         self.total_env_steps_in_stage = 0 # Track steps within the current stage/sub-stage
#         self.last_eval_step = 0 # Track env step count at last advancement check

#         # 课程进展记录
#         self.stage_history = []
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         # Use output path from main config
#         self.output_dir = os.path.join(cfg.output.output_dir, f"curriculum_{self.timestamp}")
#         os.makedirs(self.output_dir, exist_ok=True)
#         self.latest_model_path = None # Track the path of the last saved model

#         # 记录日志
#         self.log_file = os.path.join(self.output_dir, "curriculum_log.txt")
#         self._log(f"课程学习开始时间: {self.timestamp}")
#         self._log(f"初始阶段: {self.current_stage}.{self.current_sub_stage}")
#         self._log(f"成功率阈值: {self.success_threshold}, 评估窗口: {self.evaluation_window_size}, 最小评估间隔步数: {self.min_steps_between_eval}")
#         self._log("-" * 30)


#     # def _log(self, message):
#     #     """Helper function to print and write to log file."""
#     #     print(message)
#     #     try:
#     #         with open(self.log_file, "a") as f:
#     #             f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
#     #     except Exception as e:
#     #         print(f"Error writing to log file: {e}")


#     def _log(self, message):
#         """Helper function to print and write to log file."""
#         print(message)
#         try:
#             # *** 指定 encoding='utf-8' ***
#             with open(self.log_file, "a", encoding='utf-8') as f:
#                 f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
#         except Exception as e:
#             print(f"Error writing to log file: {e}")


#     def update_statistics(self, success_rate_scalar, mean_reward, env_steps_this_iter):
#         """使用 Runner 提供的标量统计数据更新管理器状态"""
#         # Update step counts
#         self.total_env_steps_in_stage += env_steps_this_iter

#         # Store scalar success rate and mean reward
#         if isinstance(success_rate_scalar, (float, int, np.number)):
#              self.scalar_success_rate_history.append(float(success_rate_scalar))
#         else:
#              print(f"⚠️ CurriculumManager Warning: Received non-scalar success rate: {success_rate_scalar} (type: {type(success_rate_scalar)})")

#         if isinstance(mean_reward, (float, int, np.number)):
#              self.reward_history.append(float(mean_reward))

#         # Optional: Log every update (can be verbose)
#         # self._log(f"阶段 {self.current_stage}.{self.current_sub_stage} - Steps in stage: {self.total_env_steps_in_stage}, Current SR: {success_rate_scalar:.4f}, Mean Rew: {mean_reward:.4f}")


#     def get_smoothed_success_rate(self):
#         """获取平滑后的成功率 (例如，最近N次的平均值)"""
#         if not self.scalar_success_rate_history:
#             return 0.0
#         # Calculate moving average over the history deque
#         return np.mean(self.scalar_success_rate_history)

#     def get_average_reward(self):
#         """获取评估窗口内的平均奖励"""
#         if not self.reward_history:
#             return 0.0
#         return np.mean(self.reward_history)


#     def should_advance_curriculum(self, current_total_env_steps):
#         """判断是否应该推进课程 (基于标量成功率和平滑)"""

#         # force_advance_iterations = 10  # 设置强制推进的迭代次数阈值
#         # if self.current_stage == 1 and current_iteration is not None and current_iteration >= force_advance_iterations:
#         #     self._log(
#         #         f"*** 强制推进条件满足: Stage 1 且当前迭代次数 ({current_iteration}) >= {force_advance_iterations} ***")
#         #     self.last_eval_step = current_total_env_steps  # 更新评估步数，避免立即再次触发
#         #     return True




#         # 1. 检查自上次评估/推进以来是否经过了足够的步数
#         steps_since_last_eval = current_total_env_steps - self.last_eval_step
#         if steps_since_last_eval < self.min_steps_between_eval:
#             # print(f"  Curriculum check: Not enough steps since last eval ({steps_since_last_eval}/{self.min_steps_between_eval}).")
#             return False

#         # 2. 确保有足够的成功率数据点进行平滑
#         if len(self.scalar_success_rate_history) < self.evaluation_window_size: # Use eval window for history size check
#              # print(f"  Curriculum check: Not enough success rate data points ({len(self.scalar_success_rate_history)}/{self.evaluation_window_size}).")
#              return False

#         # 3. 计算平滑后的成功率
#         smoothed_success_rate = self.get_smoothed_success_rate()
#         # print(f"  Curriculum check: Steps since last eval: {steps_since_last_eval}. Smoothed SR ({len(self.scalar_success_rate_history)} samples): {smoothed_success_rate:.4f}")


#         # 4. 检查成功率是否达到阈值
#         if smoothed_success_rate >= self.success_threshold:
#              self._log(f"*** 条件满足: 平滑成功率 {smoothed_success_rate:.4f} >= 阈值 {self.success_threshold} ***")
#              self.last_eval_step = current_total_env_steps # Update last evaluation step count
#              return True
#         else:
#              # Threshold not met, but enough steps have passed, so update last_eval_step anyway
#              # to reset the min_steps counter for the *next* check.
#              self.last_eval_step = current_total_env_steps
#              # print(f"  Curriculum check: Threshold not met. Updated last_eval_step to {current_total_env_steps}.")
#              return False


#     def advance_curriculum(self, current_total_env_steps):
#         """推进课程阶段，返回新的阶段字符串 "stage.sub_stage" """
#         old_stage_str = f"{self.current_stage}.{self.current_sub_stage}"
#         smoothed_success_rate = self.get_smoothed_success_rate() # Get rate before clearing history
#         avg_reward = self.get_average_reward()

#         # 更新子阶段
#         self.current_sub_stage += 1

#         # 如果子阶段达到最大值，进入下一个主阶段
#         if self.current_sub_stage > self.max_sub_stages:
#             self.current_stage += 1
#             self.current_sub_stage = 1

#         # 检查是否超过最大阶段 (保持在最大阶段)
#         if self.current_stage > self.max_stages:
#             self.current_stage = self.max_stages
#             self.current_sub_stage = self.max_sub_stages
#             self._log("已达到最大课程阶段。")
#             # 返回旧阶段字符串表示不再推进? 或者返回最大阶段?
#             # Let's return the max stage string.
#             return f"{self.current_stage}.{self.current_sub_stage}"

#         new_stage_str = f"{self.current_stage}.{self.current_sub_stage}"

#         # 记录阶段变化
#         stage_entry = {
#             "old_stage": old_stage_str,
#             "new_stage": new_stage_str,
#             "total_env_steps": current_total_env_steps, # Record total steps at transition
#             "steps_in_stage": self.total_env_steps_in_stage,
#             "success_rate_at_transition": smoothed_success_rate,
#             "average_reward_at_transition": avg_reward,
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "model_checkpoint": self.latest_model_path # Record model used for transition
#         }
#         self.stage_history.append(stage_entry)

#         # 重置评估状态
#         self.reset_evaluation()

#         # 记录日志
#         self._log(f"\n======== 🎓 课程推进 🎓 ========")
#         self._log(f"从阶段 {old_stage_str} 推进到 {new_stage_str}")
#         self._log(f"总环境步数: {stage_entry['total_env_steps']:,}")
#         self._log(f"此阶段步数: {stage_entry['steps_in_stage']:,}")
#         self._log(f"过渡时成功率: {stage_entry['success_rate_at_transition']:.4f}")
#         self._log(f"过渡时平均奖励: {stage_entry['average_reward_at_transition']:.4f}")
#         self._log(f"模型检查点: {stage_entry['model_checkpoint']}")
#         self._log(f"时间: {stage_entry['timestamp']}")
#         self._log("================================\n")

#         # 绘制并保存课程进展图表
#         self._save_curriculum_progress_plot()
#         # 保存一次课程状态
#         self.save_curriculum_state(current_total_env_steps)

#         return new_stage_str

#     def reset_evaluation(self):
#         """重置用于评估阶段进展的历史记录和计数器"""
#         self.scalar_success_rate_history.clear()
#         self.reward_history.clear()
#         self.total_env_steps_in_stage = 0
#         # self.last_eval_step is updated in should_advance_curriculum or advance_curriculum
#         self._log("评估窗口和阶段步数已重置。")

#     def _save_curriculum_progress_plot(self):
#         """保存课程进展图表（成功率和奖励）"""
#         if len(self.stage_history) == 0:
#             return

#         # 准备数据
#         stages_labels = [entry["old_stage"] for entry in self.stage_history] # Label points with the stage *before* transition
#         success_rates = [entry["success_rate_at_transition"] for entry in self.stage_history]
#         rewards = [entry["average_reward_at_transition"] for entry in self.stage_history]
#         steps_at_transition = [entry["total_env_steps"] for entry in self.stage_history]

#         try:
#             # 创建图表
#             fig, ax1 = plt.subplots(figsize=(12, 6))

#             # 成功率 (左 Y 轴)
#             color = 'tab:blue'
#             ax1.set_xlabel('总环境步数 (Environment Steps)')
#             ax1.set_ylabel('平滑成功率 (Smoothed Success Rate)', color=color)
#             ax1.plot(steps_at_transition, success_rates, 'o-', color=color, label='Success Rate')
#             ax1.tick_params(axis='y', labelcolor=color)
#             ax1.set_ylim(0, 1.05) # Extend ylim slightly
#             ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

#              # 在点上标注阶段号
#             for i, label in enumerate(stages_labels):
#                  ax1.text(steps_at_transition[i], success_rates[i] + 0.02, f"->{self.stage_history[i]['new_stage']}", ha='center', va='bottom', fontsize=8, color=color)

#             # 平均奖励 (右 Y 轴)
#             ax2 = ax1.twinx() # 共享 X 轴
#             color = 'tab:orange'
#             ax2.set_ylabel('平均奖励 (Mean Reward)', color=color)
#             ax2.plot(steps_at_transition, rewards, 's--', color=color, label='Mean Reward')
#             ax2.tick_params(axis='y', labelcolor=color)

#             # 添加标题和图例
#             plt.title(f'课程学习进展 ({self.timestamp})')
#             fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2) # Combined legend below plot
#             fig.tight_layout(rect=[0, 0.05, 1, 1]) # Adjust layout to make space for legend

#             # 保存图表
#             plot_filename = os.path.join(self.output_dir, "curriculum_progress.png")
#             plt.savefig(plot_filename)
#             plt.close(fig) # Close the figure to free memory
#             self._log(f"课程进展图表已保存: {plot_filename}")

#         except Exception as e:
#             self._log(f"❌ 绘制或保存课程进展图表失败: {e}")


#     def save_curriculum_state(self, current_total_env_steps):
#         """保存当前课程状态到 YAML 文件"""
#         state = {
#             "current_stage": self.current_stage,
#             "current_sub_stage": self.current_sub_stage,
#             "total_env_steps_at_save": current_total_env_steps,
#             "last_eval_step": self.last_eval_step,
#             "stage_history": self.stage_history,
#             "latest_model_path": self.latest_model_path, # Save the model path
#             "save_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }

#         # 保存为YAML文件
#         # Include step count in filename for easier identification
#         filename = os.path.join(self.output_dir, f"curriculum_state_S{self.current_stage}_{self.current_sub_stage}_T{current_total_env_steps}.yaml")
#         try:
#             with open(filename, "w") as f:
#                 yaml.dump(state, f, default_flow_style=False)
#             self._log(f"💾 当前课程状态已保存: {filename}")
#             return filename
#         except Exception as e:
#             self._log(f"❌ 保存课程状态失败: {e}")
#             return None

#     def load_curriculum_state(self, filename):
#         """从 YAML 文件加载课程状态"""
#         if not os.path.exists(filename):
#             self._log(f"❌ 课程状态文件不存在: {filename}")
#             return False

#         try:
#             with open(filename, "r") as f:
#                 state = yaml.safe_load(f)

#             self.current_stage = state["current_stage"]
#             self.current_sub_stage = state["current_sub_stage"]
#             # total_env_steps should be obtained from the loaded runner checkpoint
#             self.last_eval_step = state.get("last_eval_step", 0) # Load last eval step
#             self.stage_history = state["stage_history"]
#             self.latest_model_path = state.get("latest_model_path") # Load model path

#             # 重置评估窗口，因为历史数据没有保存
#             self.reset_evaluation()

#             self._log(f"\n======== 🔄 加载课程状态 ========")
#             self._log(f"文件: {filename}")
#             self._log(f"恢复到阶段: {self.current_stage}.{self.current_sub_stage}")
#             self._log(f"上次评估步数: {self.last_eval_step}")
#             self._log(f"恢复的模型路径: {self.latest_model_path}")
#             self._log(f"加载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#             self._log("================================\n")

#             # 返回加载的模型路径，让 train_curriculum 使用
#             return True, self.latest_model_path

#         except Exception as e:
#             self._log(f"❌ 加载课程状态失败: {e}")
#             import traceback
#             traceback.print_exc()
#             return False, None

#     def get_current_stage_info(self):
#          """返回当前阶段和子阶段"""
#          return self.current_stage, self.current_sub_stage

#     def set_latest_model_path(self, path):
#          """设置最近保存的模型路径"""
#          self.latest_model_path = path

import os
import yaml
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import time # For potential timing checks
# No need to import DotDict here if cfg is already DotDict when passed in
import traceback # For debug printing

class CurriculumManager:
    """课程学习管理器，负责管理课程进度、状态和模型路径"""

    def __init__(self, cfg): # cfg is expected to be the DotDict created in train_curriculum
        self.cfg = cfg

        # 课程设置
        # Access the 'curriculum' sub-dict/DotDict within cfg
        curriculum_cfg = cfg.get('curriculum')
        if curriculum_cfg is None:
            raise ValueError("Curriculum configuration ('curriculum' key) missing in provided cfg.")

        self.current_stage = curriculum_cfg.initial_stage
        self.current_sub_stage = curriculum_cfg.initial_sub_stage
        self.max_stages = curriculum_cfg.max_stages

        # --- !!! FIX: Access data attribute directly, don't call method !!! ---
        max_subs_dict = getattr(curriculum_cfg, 'max_sub_stages_per_stage', None)
        global_max_subs = getattr(curriculum_cfg, 'max_sub_stages', 5) # Default global max

        if isinstance(max_subs_dict, dict):
            # Safely get value for the current stage, fallback to global max
            self.current_max_sub_stages = max_subs_dict.get(str(self.current_stage), global_max_subs) # Use str key for safety if needed
            if str(self.current_stage) not in max_subs_dict:
                 print(f"⚠️ CM __init__ Warning: Stage {self.current_stage} not found in max_sub_stages_per_stage dict. Using global max {global_max_subs}.")
        else:
            # Fallback if max_sub_stages_per_stage is missing or not a dict
            print("⚠️ CM __init__ Warning: 'max_sub_stages_per_stage' missing or invalid in config. Using global 'max_sub_stages'.")
            self.current_max_sub_stages = global_max_subs
        # --------------------------------------------------------------------

        # 进阶条件
        self.success_threshold = curriculum_cfg.success_threshold
        self.min_steps_between_eval = getattr(curriculum_cfg, 'min_steps_between_eval', 1_000_000)
        self.evaluation_window_size = curriculum_cfg.evaluation_window

        # 训练跟踪
        self.scalar_success_rate_history = deque(maxlen=self.evaluation_window_size)
        self.reward_history = deque(maxlen=self.evaluation_window_size)
        self.total_env_steps_in_stage = 0
        self.last_eval_step = 0

        # 课程进展记录
        self.stage_history = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Access output config safely
        output_cfg = cfg.get('output', {})
        output_dir_base = output_cfg.get('output_dir', "output/curriculum_default") # Default if missing
        self.output_dir = os.path.join(output_dir_base, f"curriculum_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.latest_model_path = None

        # 记录日志
        self.log_file = os.path.join(self.output_dir, "curriculum_log.txt")
        self._log(f"课程学习开始时间: {self.timestamp}")
        self._log(f"初始阶段: {self.current_stage}.{self.current_sub_stage} (Max Sub: {self.current_max_sub_stages})")
        self._log(f"成功率阈值: {self.success_threshold}, 评估窗口: {self.evaluation_window_size}, 最小评估间隔步数: {self.min_steps_between_eval}")
        self._log("-" * 30)


    def _log(self, message):
        # (Keep previous implementation)
        print(message)
        try:
            with open(self.log_file, "a", encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        except Exception as e: print(f"Error writing to log file: {e}")

    def update_statistics(self, success_rate_scalar, mean_reward, env_steps_this_iter):
        # (Keep previous implementation)
        self.total_env_steps_in_stage += env_steps_this_iter
        if success_rate_scalar is not None and isinstance(success_rate_scalar, (float, int, np.number)) and np.isfinite(success_rate_scalar):
             self.scalar_success_rate_history.append(float(success_rate_scalar))
        if mean_reward is not None and isinstance(mean_reward, (float, int, np.number)) and np.isfinite(mean_reward):
             self.reward_history.append(float(mean_reward))

    def get_smoothed_success_rate(self):
        # (Keep previous implementation)
        if not self.scalar_success_rate_history: return 0.0
        return np.mean(self.scalar_success_rate_history)

    def get_average_reward(self):
        # (Keep previous implementation)
        if not self.reward_history: return 0.0
        return np.mean(self.reward_history)

    def should_advance_curriculum(self, current_total_env_steps):
        # (Keep previous implementation)
        steps_since_last_eval = current_total_env_steps - self.last_eval_step
        if steps_since_last_eval < self.min_steps_between_eval: return False
        if len(self.scalar_success_rate_history) < self.evaluation_window_size: return False
        smoothed_success_rate = self.get_smoothed_success_rate()
        if smoothed_success_rate >= self.success_threshold:
             self._log(f"*** 条件满足: 平滑成功率 {smoothed_success_rate:.4f} >= 阈值 {self.success_threshold} ***")
             self.last_eval_step = current_total_env_steps
             return True
        else:
             self.last_eval_step = current_total_env_steps
             return False

    def advance_curriculum(self, current_total_env_steps):
        """推进课程阶段，返回新的阶段字符串 "stage.sub_stage" """
        old_stage_str = f"{self.current_stage}.{self.current_sub_stage}"
        smoothed_success_rate = self.get_smoothed_success_rate()
        avg_reward = self.get_average_reward()

        # 更新子阶段
        self.current_sub_stage += 1

        # --- !!! FIX: Access data attribute directly !!! ---
        max_subs_dict = getattr(self.cfg.curriculum, 'max_sub_stages_per_stage', None)
        global_max_subs = getattr(self.cfg.curriculum, 'max_sub_stages', 5)

        # Determine max sub-stages for the *current* stage before potentially advancing it
        max_subs_for_current_stage = self.current_max_sub_stages # Use the value stored in the instance

        if self.current_sub_stage > max_subs_for_current_stage:
            self.current_stage += 1
            self.current_sub_stage = 1

            if self.current_stage > self.max_stages:
                self.current_stage = self.max_stages
                # Get max sub for the *final* stage
                if isinstance(max_subs_dict, dict):
                    self.current_max_sub_stages = max_subs_dict.get(str(self.current_stage), global_max_subs)
                else: self.current_max_sub_stages = global_max_subs
                self.current_sub_stage = self.current_max_sub_stages # Stay at last sub-stage
                self._log("已达到最大课程阶段。")
            else:
                 # Update current_max_sub_stages for the NEW main stage
                 if isinstance(max_subs_dict, dict):
                     self.current_max_sub_stages = max_subs_dict.get(str(self.current_stage), global_max_subs)
                     if str(self.current_stage) not in max_subs_dict:
                          print(f"⚠️ CM advance Warning: Stage {self.current_stage} not found in max_sub_stages_per_stage dict. Using global max {global_max_subs}.")
                 else:
                     print("⚠️ CM advance Warning: 'max_sub_stages_per_stage' missing or invalid. Using global 'max_sub_stages'.")
                     self.current_max_sub_stages = global_max_subs
        # ----------------------------------------------------

        new_stage_str = f"{self.current_stage}.{self.current_sub_stage}"

        # 记录阶段变化 (Keep as before)
        stage_entry = {
            "old_stage": old_stage_str, "new_stage": new_stage_str,
            "total_env_steps": current_total_env_steps, "steps_in_stage": self.total_env_steps_in_stage,
            "success_rate_at_transition": smoothed_success_rate, "average_reward_at_transition": avg_reward,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model_checkpoint": self.latest_model_path
        }
        self.stage_history.append(stage_entry)

        self.reset_evaluation()

        # 记录日志 (Keep as before, added current_max_sub_stages)
        self._log(f"\n======== 🎓 课程推进 🎓 ========")
        self._log(f"从阶段 {old_stage_str} 推进到 {new_stage_str}")
        self._log(f"新阶段最大子阶段数: {self.current_max_sub_stages}") # Log the max for the new stage
        self._log(f"总环境步数: {stage_entry['total_env_steps']:,}")
        self._log(f"旧阶段步数: {stage_entry['steps_in_stage']:,}")
        self._log(f"过渡时成功率: {stage_entry['success_rate_at_transition']:.4f}")
        self._log(f"过渡时平均奖励: {stage_entry['average_reward_at_transition']:.4f}")
        self._log(f"模型检查点: {stage_entry['model_checkpoint']}")
        self._log(f"时间: {stage_entry['timestamp']}")
        self._log("================================\n")

        self._save_curriculum_progress_plot()
        self.save_curriculum_state(current_total_env_steps)

        return new_stage_str

    def reset_evaluation(self):
        # (Keep previous implementation)
        self.scalar_success_rate_history.clear()
        self.reward_history.clear()
        self.total_env_steps_in_stage = 0
        self._log("评估窗口和阶段步数已重置。")

    def _save_curriculum_progress_plot(self):
        # (Keep previous implementation)
        if len(self.stage_history) == 0: return
        stages_labels = [entry["old_stage"] for entry in self.stage_history]
        new_stages_labels = [entry["new_stage"] for entry in self.stage_history]
        success_rates = [entry["success_rate_at_transition"] for entry in self.stage_history]
        rewards = [entry["average_reward_at_transition"] for entry in self.stage_history]
        steps_at_transition = [entry["total_env_steps"] for entry in self.stage_history]
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
            fig, ax1 = plt.subplots(figsize=(15, 7))
            color = 'tab:blue'
            ax1.set_xlabel('总环境步数 (Environment Steps)')
            ax1.set_ylabel('平滑成功率 (Smoothed Success Rate)', color=color, fontsize=12)
            ax1.plot(steps_at_transition, success_rates, marker='o', linestyle='-', color=color, linewidth=2, markersize=8, label='Success Rate @ Transition')
            ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
            ax1.tick_params(axis='x', labelsize=10)
            ax1.set_ylim(0, 1.05)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
            ax1.axhline(y=self.success_threshold, color='r', linestyle=':', linewidth=1.5, label=f'Success Threshold ({self.success_threshold})')
            for i, label in enumerate(new_stages_labels):
                 offset = 0.03 if i % 2 == 0 else -0.05
                 ax1.text(steps_at_transition[i], success_rates[i] + offset, f"{label}", ha='center', va='bottom', fontsize=9, color=color, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('平均奖励 (Mean Reward)', color=color, fontsize=12)
            ax2.plot(steps_at_transition, rewards, marker='s', linestyle='--', color=color, linewidth=2, markersize=8, label='Mean Reward @ Transition')
            ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
            plt.title(f'课程学习进展 - {self.timestamp}', fontsize=16)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=10, frameon=True, shadow=True)
            fig.tight_layout(rect=[0, 0.05, 1, 0.96])
            plot_filename = os.path.join(self.output_dir, "curriculum_progress.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self._log(f"课程进展图表已保存: {plot_filename}")
        except Exception as e:
            self._log(f"❌ 绘制或保存课程进展图表失败: {e}")
            traceback.print_exc()

    def save_curriculum_state(self, current_total_env_steps):
        # (Keep previous implementation)
        state = {
            "current_stage": self.current_stage, "current_sub_stage": self.current_sub_stage,
            "total_env_steps_at_save": current_total_env_steps, "last_eval_step": self.last_eval_step,
            "stage_history": self.stage_history, "latest_model_path": self.latest_model_path,
            "save_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        filename = os.path.join(self.output_dir, f"curriculum_state_S{self.current_stage}_{self.current_sub_stage}_T{current_total_env_steps}.yaml")
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w", encoding='utf-8') as f: yaml.dump(state, f, default_flow_style=False, allow_unicode=True)
            self._log(f"💾 当前课程状态已保存: {filename}")
            latest_filename = os.path.join(self.output_dir, "curriculum_state_latest.yaml")
            with open(latest_filename, "w", encoding='utf-8') as f: yaml.dump(state, f, default_flow_style=False, allow_unicode=True)
            return filename
        except Exception as e: self._log(f"❌ 保存课程状态失败: {e}"); return None

    def load_curriculum_state(self, filename):
        # (Keep previous implementation, added update for current_max_sub_stages)
        if not os.path.exists(filename):
            latest_filename = os.path.join(os.path.dirname(filename), "curriculum_state_latest.yaml")
            if filename.endswith("_latest.yaml") or not os.path.exists(latest_filename):
                 self._log(f"❌ 课程状态文件不存在: {filename}"); return False, None
            else: print(f"指定文件 {filename} 不存在，尝试加载 latest: {latest_filename}"); filename = latest_filename
        try:
            with open(filename, "r", encoding='utf-8') as f: state = yaml.safe_load(f)
            self.current_stage = state["current_stage"]
            self.current_sub_stage = state["current_sub_stage"]
            self.last_eval_step = state.get("last_eval_step", 0)
            self.stage_history = state.get("stage_history", [])
            self.latest_model_path = state.get("latest_model_path")
            self.reset_evaluation()
            # --- FIX: Update current_max_sub_stages after loading ---
            max_subs_dict = getattr(self.cfg.curriculum, 'max_sub_stages_per_stage', None)
            global_max_subs = getattr(self.cfg.curriculum, 'max_sub_stages', 5)
            if isinstance(max_subs_dict, dict):
                self.current_max_sub_stages = max_subs_dict.get(str(self.current_stage), global_max_subs)
            else: self.current_max_sub_stages = global_max_subs
            # --- End Fix ---
            self._log(f"\n======== 🔄 加载课程状态 ========")
            self._log(f"文件: {filename}")
            self._log(f"恢复到阶段: {self.current_stage}.{self.current_sub_stage} (Max Sub: {self.current_max_sub_stages})") # Log updated max
            self._log(f"上次评估步数: {self.last_eval_step}")
            self._log(f"恢复的模型路径: {self.latest_model_path}")
            self._log(f"加载时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            self._log("================================\n")
            return True, self.latest_model_path
        except Exception as e:
            self._log(f"❌ 加载课程状态失败 ({filename}): {e}"); traceback.print_exc(); return False, None

    def get_current_stage_info(self):
         # (Keep previous implementation)
         return self.current_stage, self.current_sub_stage

    def set_latest_model_path(self, path):
         # (Keep previous implementation)
         self.latest_model_path = path
