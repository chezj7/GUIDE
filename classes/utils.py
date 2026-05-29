import os
import imageio
import numpy as np
from skimage.morphology import label
from scipy.ndimage import  zoom
from collections import defaultdict
from numpy import floor, ceil
from parameter import *
import imageio.v2 as imageio

def get_cell_position_from_coords(coords, map_info, check_negative=True):
    coords = np.array(coords)


    single_cell = coords.ndim == 1 and coords.shape[0] == 2


    coords = coords.reshape(-1, 2)
    coords_x = coords[:, 0]
    coords_y = coords[:, 1]


    cell_x = (coords_x - map_info.map_origin_x) / map_info.cell_size
    cell_y = (coords_y - map_info.map_origin_y) / map_info.cell_size

    cell_position = np.around(np.stack((cell_x, cell_y), axis=-1)).astype(int)


    if check_negative:
        assert np.all(cell_position >= 0), f"Invalid cell positions: {cell_position}, coords: {coords}, origin: ({map_info.map_origin_x}, {map_info.map_origin_y})"

    if single_cell:
        return cell_position[0]
    else:
        return cell_position


def get_coords_from_cell_position(cell_position, map_info):
    cell_position = np.array(cell_position)
    cell_position = cell_position.reshape(-1, 2)
    cell_x = cell_position[:, 0]
    cell_y = cell_position[:, 1]
    coords_x = cell_x * map_info.cell_size + map_info.map_origin_x
    coords_y = cell_y * map_info.cell_size + map_info.map_origin_y
    coords = np.stack((coords_x, coords_y), axis=-1)
    coords = np.around(coords, 1)
    if coords.shape[0] == 1:
        return coords[0]
    else:
        return coords


def get_free_area_coords(map_info):
    free_indices = np.where(map_info.map == FREE)
    free_cells = np.asarray([free_indices[1], free_indices[0]]).T
    free_coords = get_coords_from_cell_position(free_cells, map_info)
    return free_coords


def get_free_and_connected_map(location, map_info):
    # a binary map for free and connected areas
    free = (map_info.map == FREE).astype(float)
    labeled_free = label(free, connectivity=2)
    cell = get_cell_position_from_coords(location, map_info)
    label_number = labeled_free[cell[1], cell[0]]
    connected_free_map = (labeled_free == label_number)
    return connected_free_map


def get_updating_node_coords(location, updating_map_info, check_connectivity=True):
    x_min = updating_map_info.map_origin_x
    y_min = updating_map_info.map_origin_y
    x_max = updating_map_info.map_origin_x + (updating_map_info.map.shape[1] - 1) * CELL_SIZE
    y_max = updating_map_info.map_origin_y + (updating_map_info.map.shape[0] - 1) * CELL_SIZE

    if x_min % NODE_RESOLUTION != 0:
        x_min = (x_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if x_max % NODE_RESOLUTION != 0:
        x_max = x_max // NODE_RESOLUTION * NODE_RESOLUTION
    if y_min % NODE_RESOLUTION != 0:
        y_min = (y_min // NODE_RESOLUTION + 1) * NODE_RESOLUTION
    if y_max % NODE_RESOLUTION != 0:
        y_max = y_max // NODE_RESOLUTION * NODE_RESOLUTION

    x_coords = np.arange(x_min, x_max + 0.1, NODE_RESOLUTION)
    y_coords = np.arange(y_min, y_max + 0.1, NODE_RESOLUTION)
    t1, t2 = np.meshgrid(x_coords, y_coords) 
    nodes = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    nodes = np.around(nodes, 1)

    free_connected_map = None

    if not check_connectivity:

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < updating_map_info.map.shape[0] and 0 <= cell[0] < updating_map_info.map.shape[1]
            if updating_map_info.map[cell[1], cell[0]] == FREE:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    else:
        free_connected_map = get_free_and_connected_map(location, updating_map_info)
        free_connected_map = np.array(free_connected_map)

        indices = []
        nodes_cells = get_cell_position_from_coords(nodes, updating_map_info).reshape(-1, 2)
        for i, cell in enumerate(nodes_cells):
            assert 0 <= cell[1] < free_connected_map.shape[0] and 0 <= cell[0] < free_connected_map.shape[1]
            if free_connected_map[cell[1], cell[0]] == 1:
                indices.append(i)
        indices = np.array(indices)
        nodes = nodes[indices].reshape(-1, 2)

    return nodes, free_connected_map

def get_obstacle_node_coords(updating_map_info, free_node_set):
    map_data = updating_map_info.map
    origin_x = updating_map_info.map_origin_x
    origin_y = updating_map_info.map_origin_y
    resolution = updating_map_info.cell_size

    height, width = map_data.shape
    occupied_cells = np.argwhere(map_data == OCCUPIED)

    filtered_cells = []
    for y, x in occupied_cells:
        neighbors = [
            (y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)
        ]
        if any(0 <= ny < height and 0 <= nx < width and map_data[ny, nx] == OCCUPIED for ny, nx in neighbors):
            filtered_cells.append((y, x))
    # print(f"[Info] 原始障碍体素数量: {len(occupied_cells)}, 过滤后: {len(filtered_cells)}")

    obstacle_nodes = set()

    for y, x in filtered_cells:
        wx = origin_x + x * resolution
        wy = origin_y + y * resolution

        # floor对齐
        gx1 = floor(wx / NODE_RESOLUTION) * NODE_RESOLUTION
        gy1 = floor(wy / NODE_RESOLUTION) * NODE_RESOLUTION
        node1 = (round(gx1, 2), round(gy1, 2))

        # ceil对齐
        gx2 = ceil(wx / NODE_RESOLUTION) * NODE_RESOLUTION
        gy2 = ceil(wy / NODE_RESOLUTION) * NODE_RESOLUTION
        node2 = (round(gx2, 2), round(gy2, 2))

        if node1 not in free_node_set and node1 not in obstacle_nodes:
            obstacle_nodes.add(node1)
        elif node2 not in free_node_set and node2 not in obstacle_nodes:
            obstacle_nodes.add(node2)

    # 保留原有的节点层过滤孤立节点逻辑
    filtered_obstacle_nodes = set()
    for node in obstacle_nodes:
        x, y = node
        neighbors = [
            (x + NODE_RESOLUTION, y),    # 右
            (x - NODE_RESOLUTION, y),    # 左
            (x, y + NODE_RESOLUTION),    # 上
            (x, y - NODE_RESOLUTION),    # 下
        ]
        if any(neighbor in obstacle_nodes for neighbor in neighbors):
            filtered_obstacle_nodes.add(node)

    # print(f"[Info] 被过滤掉的孤立障碍节点数量: {len(obstacle_nodes) - len(filtered_obstacle_nodes)}")

    return np.array(list(filtered_obstacle_nodes))
def compute_frontier_allowed_blocks_edgeaware_fast(frontiers, map_info, block_size_in_cells, robot_location):
    
    H, W = map_info.map.shape
    n_rows = H // block_size_in_cells
    n_cols = W // block_size_in_cells
    allowed_mask = np.zeros((n_rows, n_cols), dtype=bool)

    if not frontiers:
        return allowed_mask

    F = np.asarray(list(frontiers), dtype=np.float32).reshape(-1, 2)

    if MAX_FRONTIER_DIST is not None and robot_location is not None:
        d = np.linalg.norm(F - np.asarray(robot_location, dtype=np.float32), axis=1)
        keep = d <= float(MAX_FRONTIER_DIST)
        if not np.any(keep):
            return allowed_mask
        F = F[keep]

    cx = np.floor((F[:, 0] - map_info.map_origin_x) / map_info.cell_size).astype(np.int32)
    cy = np.floor((F[:, 1] - map_info.map_origin_y) / map_info.cell_size).astype(np.int32)

    inb = (cx >= 0) & (cx < W) & (cy >= 0) & (cy < H)
    if not np.any(inb):
        return allowed_mask
    cx, cy = cx[inb], cy[inb]

    r = (cy // block_size_in_cells).astype(np.int32)
    c = (cx // block_size_in_cells).astype(np.int32)
    lx = (cx % block_size_in_cells).astype(np.int32)
    ly = (cy % block_size_in_cells).astype(np.int32)

    # 触发贴边条件的布尔掩码
    m = int(FRONTIER_EDGE_MARGIN_CELLS)
    near_left   = lx <= (m - 1)
    near_right  = lx >= (block_size_in_cells - m)
    near_top    = ly <= (m - 1)
    near_bottom = ly >= (block_size_in_cells - m)

    # 先标记自身块
    allowed_mask[r, c] = True

    # 然后按贴边方向外扩相邻 1 块（越界自动忽略）
    def mark(rr, cc, mask_dir):
        if not np.any(mask_dir): 
            return
        rr2 = rr[mask_dir]
        cc2 = cc[mask_dir]
        # 边界筛掉非法相邻块
        valid = (rr2 >= 0) & (rr2 < n_rows) & (cc2 >= 0) & (cc2 < n_cols)
        if np.any(valid):
            allowed_mask[rr2[valid], cc2[valid]] = True

    # 左/右/上/下相邻块
    mark(r, c - 1, near_left)
    mark(r, c + 1, near_right)
    mark(r - 1, c, near_top)
    mark(r + 1, c, near_bottom)

    return allowed_mask
from parameter import *
# from ... import compute_frontier_allowed_blocks_edgeaware_fast

def get_map_into_regions(
    map_info,
    location,
    block_size_in_cells=BLOCK_SIZE_IN_CELLS,
    update_window_in_cells=UPDATE_WINDOW_SIZE,
    frontiers=None
):
    map_array = map_info.map
    x_len, y_len = map_array.shape[1], map_array.shape[0]
    n_rows, n_cols = y_len // block_size_in_cells, x_len // block_size_in_cells

    regions, region_states, region_confidence, unknown_centers = [], [], [], []

    # 机器人周围置信
    cx, cy = get_cell_position_from_coords(location, map_info)
    half = update_window_in_cells // 2
    x_rng = (max(cx - half, 0), min(cx + half, x_len))
    y_rng = (max(cy - half, 0), min(cy + half, y_len))

    # frontier 掩码（贴边外扩版）
    if REGION_CONF_MODE in ("frontier", "intersect") and frontiers is not None:
        frontier_mask = compute_frontier_allowed_blocks_edgeaware_fast(
            frontiers, map_info, block_size_in_cells, location
        )
    else:
        frontier_mask = None

    for i in range(n_rows):
        row_regions, row_states, row_conf = [], [], []
        for j in range(n_cols):
            y0, y1 = i * block_size_in_cells, min((i + 1) * block_size_in_cells, y_len)
            x0, x1 = j * block_size_in_cells, min((j + 1) * block_size_in_cells, x_len)
            block = map_array[y0:y1, x0:x1]
            row_regions.append(block)

            in_window   = (x0 >= x_rng[0] and x1 <= x_rng[1] and y0 >= y_rng[0] and y1 <= y_rng[1])
            in_frontier = (frontier_mask is not None) and frontier_mask[i, j]

            if REGION_CONF_MODE == "robot":
                is_confident = in_window
            elif REGION_CONF_MODE == "frontier":
                is_confident = in_frontier
            else:  # "intersect"
                is_confident = in_window and in_frontier

            if np.any(block == FREE):
                state = FREE
            elif is_confident:
                state = FREE
            else:
                state = UNKNOWN

            row_states.append(state)
            row_conf.append(is_confident)

            if state == UNKNOWN:
                cx_pix = (x0 + x1) // 2
                cy_pix = (y0 + y1) // 2
                center_coord = get_coords_from_cell_position((cx_pix, cy_pix), map_info)
                unknown_centers.append(center_coord)

        regions.append(row_regions)
        region_states.append(row_states)
        region_confidence.append(row_conf)

    return regions, region_states, unknown_centers, region_confidence
def compute_node_grid_anchor(map_info, node_res):
    ax = np.ceil(map_info.map_origin_x / node_res) * node_res
    ay = np.ceil(map_info.map_origin_y / node_res) * node_res
    return float(ax), float(ay)

# def get_map_into_regions(map_info, location, block_size_in_cells=BLOCK_SIZE_IN_CELLS,update_window_in_cells =UPDATE_WINDOW_SIZE):  
#     map_array = map_info.map
#     x_len = map_array.shape[1]
#     y_len = map_array.shape[0]

#     n_rows = y_len // block_size_in_cells
#     n_cols = x_len // block_size_in_cells

#     regions = []
#     region_states = []
#     region_confidence = []
#     unknown_centers = []
#     center_idx = get_cell_position_from_coords(location, map_info)
#     cx, cy = center_idx[0], center_idx[1]

#     half_size = update_window_in_cells // 2
#     update_x_range = (max(cx - half_size, 0), min(cx + half_size, x_len))
#     update_y_range = (max(cy - half_size, 0), min(cy + half_size, y_len))

#     for i in range(n_rows):  # row blocks
#         row_regions = []
#         row_states = []
#         row_confidence = []
#         for j in range(n_cols):  # column blocks
#             y_start = i * block_size_in_cells
#             y_end = min((i + 1) * block_size_in_cells, y_len)
#             x_start = j * block_size_in_cells
#             x_end = min((j + 1) * block_size_in_cells, x_len)

#             block = map_array[y_start:y_end, x_start:x_end]
#             row_regions.append(block)
        
#             is_confident = False
#             if update_x_range and update_y_range:
#                 if(x_start>=update_x_range[0] and x_end<=update_x_range[1] \
#                    and y_start>=update_y_range[0] and y_end<=update_y_range[1]):
#                     is_confident = True
            
#             state = UNKNOWN
#             if np.any(block == FREE):
#                 state = FREE
#             elif is_confident:
#                 state = FREE
#             else:
#                 state = UNKNOWN
            
#             row_states.append(state)
#             row_confidence.append(is_confident)
            
#             if state == UNKNOWN:
#                 center_x = (x_start + x_end) // 2
#                 center_y = (y_start + y_end) // 2
#                 center_coord = get_coords_from_cell_position((center_x, center_y), map_info)
#                 unknown_centers.append((center_coord))
            
            

#         regions.append(row_regions)
#         region_states.append(row_states)
#         region_confidence.append(row_confidence)
        

#     return regions,region_states,unknown_centers,region_confidence

def build_region_to_centers_map(unknown_centers, map_info, block_size_in_cells):
    region_to_centers = defaultdict(list)
    for center in unknown_centers:
        cx, cy = get_cell_position_from_coords(center, map_info)
        r_idx = cy // block_size_in_cells
        c_idx = cx // block_size_in_cells
        region_to_centers[(r_idx, c_idx)].append(center)
    return region_to_centers

def get_region_index_from_point(map_info, point, block_size_in_cells):

    cell_pos = get_cell_position_from_coords(point, map_info)
    cx, cy = cell_pos[0], cell_pos[1]
    row_idx = cy // block_size_in_cells
    col_idx = cx // block_size_in_cells
    return row_idx, col_idx


def get_neighboring_regions(map_info, row_idx, col_idx, block_size_in_cells):
    map_array = map_info.map
    x_len = map_array.shape[1]
    y_len = map_array.shape[0]

    max_rows = y_len // block_size_in_cells
    max_cols = x_len // block_size_in_cells
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue  # 跳过自身
            nr, nc = row_idx + dr, col_idx + dc
            if 0 <= nr < max_rows and 0 <= nc < max_cols:
                neighbors.append((nr, nc))
    return neighbors

def filter_isolated_predicted_coords(predicted_coords_rounded, min_neighbors=1):

    coord_set = set(tuple(c) for c in predicted_coords_rounded)
    filtered_coords = []

    for x, y in coord_set:
        # 4-邻域方向
        neighbors = [
            (round(x + NODE_RESOLUTION, 2), y),
            (round(x - NODE_RESOLUTION, 2), y),
            (x, round(y + NODE_RESOLUTION, 2)),
            (x, round(y - NODE_RESOLUTION, 2))
        ]
        # 统计邻居中出现在预测集中的个数
        neighbor_count = sum(1 for nb in neighbors if nb in coord_set)
        if neighbor_count >= min_neighbors:
            filtered_coords.append([x, y])

    # print(f"[Filter] Predicted nodes: {len(predicted_coords_rounded)} → {len(filtered_coords)} (removed {len(predicted_coords_rounded) - len(filtered_coords)} isolated points)")
    return filtered_coords

def get_neighbor_region_centers_from_point_fast(map_info, point, regions_states, region_to_centers, block_size_in_cells=BLOCK_SIZE_IN_CELLS):
    
    row_idx, col_idx = get_region_index_from_point(map_info, point, block_size_in_cells)

    neighbor_indices = get_neighboring_regions(map_info, row_idx, col_idx, block_size_in_cells)
    neighbor_centers = []

    for nr, nc in neighbor_indices:
        if regions_states[nr][nc] == UNKNOWN:  # 只看未知区域
            neighbor_centers.extend(region_to_centers.get((nr, nc), []))  # O(1) 取出对应点

    return neighbor_centers
# def compute_allowed_blocks_from_frontiers(frontiers, map_info, block_size_in_cells, edge_margin_cells=2):
#     """
#     基于 frontier（世界坐标）生成允许的区块集合：
#     - 总是包含 frontier 所在区块；
#     - 若 frontier 在区块内靠近某条边（edge_margin_cells 内），则沿该边方向外扩 1 个相邻区块。
#     """
#     H, W = map_info.map.shape[0], map_info.map.shape[1]
#     n_rows = H // block_size_in_cells
#     n_cols = W // block_size_in_cells

#     def clamp(r, c):
#         return max(0, min(n_rows - 1, r)), max(0, min(n_cols - 1, c))

#     allowed = set()
#     if not frontiers:
#         return allowed

#     for fx, fy in frontiers:
#         cx, cy = get_cell_position_from_coords((fx, fy), map_info)
#         if not (0 <= cx < W and 0 <= cy < H):
#             continue

#         row = cy // block_size_in_cells
#         col = cx // block_size_in_cells
#         allowed.add((row, col))

#         # 区块内局部坐标（判断是否贴边）
#         lx = cx % block_size_in_cells
#         ly = cy % block_size_in_cells

#         if lx <= edge_margin_cells - 1:
#             allowed.add(clamp(row, col - 1))
#         if lx >= block_size_in_cells - edge_margin_cells:
#             allowed.add(clamp(row, col + 1))
#         if ly <= edge_margin_cells - 1:
#             allowed.add(clamp(row - 1, col))
#         if ly >= block_size_in_cells - edge_margin_cells:
#             allowed.add(clamp(row + 1, col))

#     return allowed

# def get_neighbor_region_centers_from_point(map_info, point, regions_states,unknown_centers, block_size_in_cells=BLOCK_SIZE_IN_CELLS):

#     row_idx, col_idx = get_region_index_from_point(map_info, point, block_size_in_cells)

#     # 获取周围邻居 region 的索引
#     neighbor_indices = get_neighboring_regions(map_info, row_idx, col_idx, block_size_in_cells)

#     neighbor_centers = []

#     for nr, nc in neighbor_indices:
#         if regions_states[nr][nc] == UNKNOWN:
#             # 遍历 unknown_centers，筛出属于这个格子的点
#             for center in unknown_centers:
#                 cx, cy = get_cell_position_from_coords(center, map_info)
#                 r_idx = cy // block_size_in_cells
#                 c_idx = cx // block_size_in_cells
#                 if r_idx == nr and c_idx == nc:
#                     neighbor_centers.append(center)

#     return neighbor_centers

def get_frontier_in_map(map_info, voxel_size=FRONTIER_CELL_SIZE):
    x_len = map_info.map.shape[1]
    y_len = map_info.map.shape[0]
    
    unknown = (map_info.map == UNKNOWN) * 1
    unknown = np.lib.pad(unknown, ((1, 1), (1, 1)), 'constant', constant_values=0)
    unknown_neighbor = unknown[2:][:, 1:x_len + 1] + unknown[:y_len][:, 1:x_len + 1] + unknown[1:y_len + 1][:, 2:] \
                       + unknown[1:y_len + 1][:, :x_len] + unknown[:y_len][:, 2:] + unknown[2:][:, :x_len] + \
                       unknown[2:][:, 2:] + unknown[:y_len][:, :x_len]
    free_cell_indices = np.where(map_info.map.ravel(order='F') == FREE)[0]
    frontier_cell_1 = np.where(1 < unknown_neighbor.ravel(order='F'))[0]
    frontier_cell_2 = np.where(unknown_neighbor.ravel(order='F') < 8)[0]
    frontier_cell_indices = np.intersect1d(frontier_cell_1, frontier_cell_2)
    frontier_cell_indices = np.intersect1d(free_cell_indices, frontier_cell_indices)

    x = np.linspace(0, x_len - 1, x_len)
    y = np.linspace(0, y_len - 1, y_len)
    t1, t2 = np.meshgrid(x, y)
    cells = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    frontier_cell = cells[frontier_cell_indices]

    frontier_coords = get_coords_from_cell_position(frontier_cell, map_info).reshape(-1, 2)
    if frontier_cell.shape[0] > 0 and FRONTIER_CELL_SIZE != CELL_SIZE:
        frontier_coords = frontier_coords.reshape(-1 ,2)
        frontier_coords = frontier_down_sample(frontier_coords, voxel_size)
    else:
        frontier_coords = set(map(tuple, frontier_coords))
    return frontier_coords



def frontier_down_sample(data, voxel_size=FRONTIER_CELL_SIZE):
    voxel_indices = np.array(data / voxel_size, dtype=int).reshape(-1, 2)

    voxel_dict = {}
    for i, point in enumerate(data):
        voxel_index = tuple(voxel_indices[i])

        if voxel_index not in voxel_dict:
            voxel_dict[voxel_index] = point
        else:
            current_point = voxel_dict[voxel_index]
            if np.linalg.norm(point - np.array(voxel_index) * voxel_size) < np.linalg.norm(
                    current_point - np.array(voxel_index) * voxel_size):
                voxel_dict[voxel_index] = point

    downsampled_data = set(map(tuple, voxel_dict.values()))
    return downsampled_data


def is_frontier(location, map_info):
    cell = get_cell_position_from_coords(location, map_info)
    if map_info.map[cell[1], cell[0]] != FREE:
        return False
    else:
        assert cell[1] - 1 > 0 and cell[1] - 1 > 0 and cell[1] + 2 < map_info.map.shape[1] and cell[0] + 2 < map_info.map.shape[0]
        unknwon = map_info.map[cell[1] - 1:cell[1] + 2, cell[0] - 1: cell[0] + 2] == UNKNOWN
        n = np.sum(unknwon)
        if 1 < n < 8:
            return True
        else:
            return False


def check_collision(start, end, map_info):
    # Bresenham line algorithm checking
    H, W = map_info.map.shape
    cs = map_info.cell_size
    ox, oy = map_info.map_origin_x, map_info.map_origin_y

    def _in_bounds(p):
        # p 是世界坐标
        x = (p[0] - ox) / cs
        y = (p[1] - oy) / cs
        return 0 <= x < W and 0 <= y < H

    if not (_in_bounds(start) and _in_bounds(end)):
        return True  #
    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    collision = False

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map = map_info.map

    x0 = start_cell[0]
    y0 = start_cell[1]
    x1 = end_cell[0]
    y1 = end_cell[1]
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map.shape[1] and 0 <= y < map.shape[0]:
        k = map.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if k == OCCUPIED:
            collision = True
            break
        if k == UNKNOWN:
            collision = True
            break
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return collision

def check_collision_only_occupied(start, end, map_info):

    assert start[0] >= map_info.map_origin_x
    assert start[1] >= map_info.map_origin_y
    assert end[0] >= map_info.map_origin_x
    assert end[1] >= map_info.map_origin_y
    assert start[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert start[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]
    assert end[0] <= map_info.map_origin_x + map_info.cell_size * map_info.map.shape[1]
    assert end[1] <= map_info.map_origin_y + map_info.cell_size * map_info.map.shape[0]

    start_cell = get_cell_position_from_coords(start, map_info)
    end_cell = get_cell_position_from_coords(end, map_info)
    map_ = map_info.map

    x0, y0 = start_cell[0], start_cell[1]
    x1, y1 = end_cell[0], end_cell[1]

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    while 0 <= x < map_.shape[1] and 0 <= y < map_.shape[0]:
        cell_value = map_.item(int(y), int(x))
        if x == x1 and y == y1:
            break
        if cell_value == OCCUPIED:
            return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return False

# def 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000......(path, n, frame_files, rate, delete_0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000images=True):
#     with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=1.5) as writer:
#         for frame in frame_files:
#             image = imageio.imread(frame)
#             writer.append_data(image)
#     print('gif complete\n')

#     # Remove files
#     if delete_images:
#         for filename in frame_files[:-1]:
#             os.remove(filename)
def make_gif(path, n, frame_files, rate, delete_images=True, fps=None, duration=None):

    assert (fps is None) ^ (duration is None), "fps 和 duration 只能设置一个"
    if fps is not None:
        per_frame = 1.0 / float(fps)
    else:
        per_frame = float(duration)

    os.makedirs(path, exist_ok=True)
    # 加上时间戳防止同名缓存
    out_name = f"{n}_explored_rate_{rate:.4g}.gif"
    out_path = os.path.join(path, out_name)

    # 有些后端更吃逐帧 meta 的时长
    with imageio.get_writer(out_path, mode='I', loop=0) as writer:
        for f in frame_files:
            img = imageio.imread(f)
            writer.append_data(img, {'duration': per_frame})

    print(f'gif complete: {out_path}\n')

    if delete_images:
        for filename in frame_files[:-1]:
            try:
                os.remove(filename)
            except OSError:
                pass

class MapInfo:
    def __init__(self, map, map_origin_x, map_origin_y, cell_size):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y
        self.cell_size = cell_size

    def update_map_info(self, map, map_origin_x, map_origin_y):
        self.map = map
        self.map_origin_x = map_origin_x
        self.map_origin_y = map_origin_y


