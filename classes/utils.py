import os
import imageio
import numpy as np
from skimage.morphology import label
from scipy.ndimage import  zoom
from collections import defaultdict

from parameter import *


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
    
    from numpy import floor, ceil

    map_data = updating_map_info.map
    origin_x = updating_map_info.map_origin_x
    origin_y = updating_map_info.map_origin_y
    resolution = updating_map_info.cell_size

    height, width = map_data.shape
    occupied_cells = np.argwhere(map_data == OCCUPIED)
    obstacle_nodes = set()

    for y, x in occupied_cells:
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
            
    print(f"[Info] 被过滤掉的孤立障碍节点数量: {len(obstacle_nodes) - len(filtered_obstacle_nodes)}")

            

    return np.array(list(filtered_obstacle_nodes))

def get_map_into_regions(map_info, location, block_size_in_cells=BLOCK_SIZE_IN_CELLS,update_window_in_cells =UPDATE_WINDOW_SIZE):  
    map_array = map_info.map
    x_len = map_array.shape[1]
    y_len = map_array.shape[0]

    n_rows = y_len // block_size_in_cells
    n_cols = x_len // block_size_in_cells

    regions = []
    region_states = []
    unknown_centers = []
    center_idx = get_cell_position_from_coords(location, map_info)
    cx, cy = center_idx[0], center_idx[1]

    half_size = update_window_in_cells // 2
    update_x_range = (max(cx - half_size, 0), min(cx + half_size, x_len))
    update_y_range = (max(cy - half_size, 0), min(cy + half_size, y_len))

    for i in range(n_rows):  # row blocks
        row_regions = []
        row_states = []
        for j in range(n_cols):  # column blocks
            y_start = i * block_size_in_cells
            y_end = min((i + 1) * block_size_in_cells, y_len)
            x_start = j * block_size_in_cells
            x_end = min((j + 1) * block_size_in_cells, x_len)

            block = map_array[y_start:y_end, x_start:x_end]
            row_regions.append(block)

            state = UNKNOWN
            if np.any(block == FREE):
                state = FREE
            
            if update_x_range and update_y_range:
                if(x_start>=update_x_range[0] and x_end<=update_x_range[1] \
                   and y_start>=update_y_range[0] and y_end<=update_y_range[1]):
                    if state == UNKNOWN:
                        state = FREE
            
            if state == UNKNOWN:
                center_x = (x_start + x_end) // 2
                center_y = (y_start + y_end) // 2
                center_coord = get_coords_from_cell_position((center_x, center_y), map_info)
                unknown_centers.append((center_coord))
            
            row_states.append(state)

        regions.append(row_regions)
        region_states.append(row_states)
        

    return regions,region_states,unknown_centers  

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

def get_neighbor_region_centers_from_point_fast(map_info, point, regions_states, region_to_centers, block_size_in_cells=BLOCK_SIZE_IN_CELLS):
    
    row_idx, col_idx = get_region_index_from_point(map_info, point, block_size_in_cells)

    neighbor_indices = get_neighboring_regions(map_info, row_idx, col_idx, block_size_in_cells)
    neighbor_centers = []

    for nr, nc in neighbor_indices:
        if regions_states[nr][nc] == UNKNOWN:  # 只看未知区域
            neighbor_centers.extend(region_to_centers.get((nr, nc), []))  # O(1) 取出对应点

    return neighbor_centers

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

def make_gif(path, n, frame_files, rate, delete_images=True):
    with imageio.get_writer('{}/{}_explored_rate_{:.4g}.gif'.format(path, n, rate), mode='I', duration=0.5) as writer:
        for frame in frame_files:
            image = imageio.imread(frame)
            writer.append_data(image)
    print('gif complete\n')

    # Remove files
    if delete_images:
        for filename in frame_files[:-1]:
            os.remove(filename)


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


