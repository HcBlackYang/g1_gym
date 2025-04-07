import os
import re
from lisdf.parsing.sdf_j import load_sdf

def clean_path(urdf_path):
    """规范化路径，修正拼接问题"""
    return os.path.normpath(os.path.abspath(urdf_path))  # 绝对路径 & 规范路径


def parse_lisdf(lisdf_path):
    """解析 LISDF 并提取 URDF 物体及其位姿和缩放比例"""
    try:
        lisdf_results = load_sdf(lisdf_path)
    except Exception as e:
        print(f"❌ 无法加载 LISDF 文件 {lisdf_path}: {e}")
        return {}

    models = lisdf_results.worlds[0].models
    pose_data = {}

    for model in models:
        if hasattr(model, "pose") and hasattr(model, "_to_sdf"):
            try:
                sdf_content = model._to_sdf(None)
                match = re.search(r'<uri>(.*?)</uri>', sdf_content)

                if match:
                    urdf_rel_path = match.group(1).strip()

                    # 处理相对路径和绝对路径
                    if not urdf_rel_path.startswith("/"):
                        base_dir = "/home/blake/kitchen-worlds/assets/models/"
                        urdf_abs_path = clean_path(os.path.join(base_dir, urdf_rel_path))
                    else:
                        urdf_abs_path = clean_path(urdf_rel_path)

                    if not os.path.exists(urdf_abs_path):
                        print(f"❌ [ERROR] 解析到的 URDF 文件不存在: {urdf_abs_path}")
                        continue

                    print(f"✅ 解析到 URDF: {urdf_abs_path}")
                    print(f"   → Pose: {model.pose}")

                    # 提取scale信息
                    scale_info = None
                    if hasattr(model, "scale"):
                        scale_info = model.scale
                        print(f"   → Scale: {scale_info}")

                    # 存储pose和scale信息
                    pose_data[urdf_abs_path] = {
                        "pose": model.pose,
                        "scale": scale_info
                    }

            except Exception as e:
                print(f"⚠️ 解析 {model.name} 失败: {e}")

    if not pose_data:
        print("⚠️ 解析 LISDF 后，pose_data 为空，请检查 LISDF 文件是否正确。")
    else:
        print(f"🔎 最终 pose_data 内容: {pose_data}")

    return pose_data







