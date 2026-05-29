# map and planning resolution
CELL_SIZE = 0.4  # meter
NODE_RESOLUTION = 4.0  # meter
FRONTIER_CELL_SIZE = 4 * CELL_SIZE

# map representation
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# sensor and utility range
SENSOR_RANGE = 20  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE
MIN_UTILITY = 1

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION

# training parameters
MAX_EPISODE_STEP = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value

BLOCK_SIZE_IN_CELLS = 50  # the number of cells in a block
UPDATE_WINDOW_SIZE =175  # the number of cells in a window

# 置信区域模式: 'robot' | 'frontier' | 'intersect'
REGION_CONF_MODE = "intersect"

# 仅在模式包含 frontier 时生效：
# 若 frontier 距离所在“大块”的边界 ≤ 这个 cell 数，就朝该边外扩 1 个大块
FRONTIER_EDGE_MARGIN_CELLS =2    

# 过滤过远的 frontier（单位：米）；None 表示不限
MAX_FRONTIER_DIST = None
# === 预测节点与节点效用 ===
ENABLE_PREDICTED_NODES = True        # 是否把 LaMa 预测的节点加入图
PRED_NODES_AFFECT_UTILITY = True     # 预测节点是否放大节点utility
ADD_PREDICTED_TO_GRAPH = True      # 预测节点是否加入图
ADD_UNKNOW_TO_GRAPH = True
UTILITY_SCALE_MAX = 2.0              # utility 最大放大倍数（<= 这个倍率）
UTILITY_SCALE_BASE_COUNT = 10        # 预测节点数到多少时达到最大放大倍数

