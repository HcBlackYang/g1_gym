import os

# 获取g1包的目录
G1_PACKAGE_DIR = os.path.dirname(os.path.realpath(__file__))

# 获取项目根目录(g1_gym)
G1_ROOT_DIR = os.path.dirname(G1_PACKAGE_DIR)

# 定义其他常用目录
G1_CONFIGS_DIR = os.path.join(G1_PACKAGE_DIR, 'configs')  # 注意这里应该指向configs目录