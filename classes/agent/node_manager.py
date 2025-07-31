import time

import numpy as np

from parameter import *
from classes.utils import *
import classes.agent.quads as quads
from classes.predictor.global_predictor import GlobalPredictor

class NodeManager:
    def __init__(self,map_info,plot=False):
        self.nodes_dict = quads.QuadTree((0, 0), 1000, 1000)
        self.plot = plot
        self.last_obstacle_coords = np.empty((0, 2))  # 保存当前帧
        self.all_obstacle_coords = set()
        self.global_predictor = GlobalPredictor(map_info)
    def check_node_exist_in_dict(self, coords):
        key = (coords[0], coords[1])
        exist = self.nodes_dict.find(key)
        return exist

    def add_node_to_dict(self, coords, local_frontiers, extended_local_map_info,new_unknown_coords=None):
        key = (coords[0], coords[1])
        node = Node(coords, local_frontiers, extended_local_map_info,new_unknown_coords)
        self.nodes_dict.insert(point=key, data=node)
        return node

    def add_node_to_dict_unknown(self, coords, local_frontiers, extended_local_map_info,new_unknown_coords=None):
        key = (coords[0], coords[1])
        node = Node(coords, local_frontiers, extended_local_map_info,new_unknown_coords)
        node.is_unknown_node = 1
        self.nodes_dict.insert(point=key, data=node)
        return node
    
    def remove_node_from_dict(self, node):
        for neighbor_coords in node.neighbor_list[1:]:
            neighbor_node = self.nodes_dict.find(neighbor_coords)
            neighbor_node.data.neighbor_list.remove(node.coords.tolist())
        self.nodes_dict.remove(node.coords)
    
    def clear_unknown_nodes(self):
        # 遍历所有节点，删除所有 is_unknown_node == True 的节点
        nodes_to_remove = []
        for node in self.nodes_dict:
            if node.data.is_unknown_node:
                nodes_to_remove.append(node.data)
        for node in nodes_to_remove:
            self.remove_node_from_dict(node)

    def filter_connected_unknown_centers(self, unknown_centers, map_info, regions_state):
        # 构建 region 映射（如已有可复用的）
        region_to_centers = build_region_to_centers_map(unknown_centers, map_info, BLOCK_SIZE_IN_CELLS)
        
        filtered_centers = set()

        for node in self.nodes_dict.__iter__():
            node_data = node.data
            if node_data.utility == 0:
                continue
            
            # 找该 utility 节点周围的未知节点候选
            neighbor_centers = get_neighbor_region_centers_from_point_fast(
                map_info, node_data.coords, regions_state, region_to_centers)
            
            for nb_center in neighbor_centers:
                collision = check_collision_only_occupied(node_data.coords, nb_center, map_info)
                if not collision:
                    filtered_centers.add(tuple(nb_center))  # 转为tuple便于set操作
        return list(filtered_centers)
    
    def update_graph(self, robot_location, frontiers, updating_map_info, map_info, regions_state,unknown_centers):
        
        # Step 1: 获取自由节点（当前帧）
        self.clear_unknown_nodes()
        node_coords, _ = get_updating_node_coords(robot_location, updating_map_info)
        free_node_set = set((round(c[0], 2), round(c[1], 2)) for c in node_coords)

        # Step 2: 累计自由节点（全局）
        if not hasattr(self, 'all_free_node_coords'):
            self.all_free_node_coords = set()
        self.all_free_node_coords.update(free_node_set)

        # Step 3: 提取障碍物节点（避免已有自由节点）
        obstacle_coords = get_obstacle_node_coords(updating_map_info, self.all_free_node_coords)

        # Step 4: 移除已变为自由空间的障碍节点
        if not hasattr(self, 'all_obstacle_coords'):
            self.all_obstacle_coords = set()
        self.all_obstacle_coords = {
            c for c in self.all_obstacle_coords if c not in self.all_free_node_coords
        }

        # Step 5: 加入新障碍物节点
        for coord in obstacle_coords:
            self.all_obstacle_coords.add(tuple(coord))

        # Step 6: 更新本帧信息
        self.node_coords = node_coords
        self.obstacle_coords = obstacle_coords
        self.last_obstacle_coords = np.array(list(self.all_obstacle_coords))

        print(f"[Debug] Generated {len(obstacle_coords)} new obs nodes. Total: {len(self.all_obstacle_coords)}")
        free_coords = np.array(list(self.all_free_node_coords))
        obs_coords = np.array(list(self.all_obstacle_coords))
        self.global_predictor.update_nodes(free_coords, obs_coords)
        inpainted_image, mask_cropped = self.global_predictor.request_lama_prediction()
        self.lama_rgb_image = inpainted_image

        # 1️⃣ 提取灰度图，并二值化
        gray_image = np.mean(inpainted_image, axis=2) / 255.0
        binary_image = np.zeros_like(gray_image, dtype=np.uint8)
        binary_image[gray_image >= 0.3] = 255
        binary_image[gray_image < 0.7] = 0

        # 2️⃣ 找出原始为未知区域的像素（mask==1）中，现在被预测为自由的区域
        mask_cropped = mask_cropped.squeeze()         # [H, W]
        predicted_free_mask = (binary_image == 255) & (mask_cropped.numpy() == 1)

        # 3️⃣ 提取这些像素的图像坐标
        predicted_indices = np.argwhere(predicted_free_mask)  # [[y1, x1], [y2, x2], ...]

        # 4️⃣ 将这些图像坐标映射回世界坐标
        def pixel_to_world_coords(indices):
            y, x = indices[:, 0], indices[:, 1]
            real_x = x * NODE_RESOLUTION + 1.5 + map_info.map_origin_x
            real_y = y * NODE_RESOLUTION + 0.5 + map_info.map_origin_y
            return np.stack([real_x, real_y], axis=1)

        predicted_coords = pixel_to_world_coords(predicted_indices)

        # 5️⃣ 四舍五入对齐到已有节点精度（保留两位小数）
        # predicted_coords_rounded = [np.round(c, 2) for c in predicted_coords]  # np.array 保持类型
        # new_unknown_coords = [
        #     c for c in predicted_coords_rounded
        #     if self.check_node_exist_in_dict(tuple(c)) is None  # 用 tuple(c) 做查找
        # ]
        predicted_coords_rounded = [np.round(c, 2) for c in predicted_coords]
        _, region_states, _ = get_map_into_regions(
            map_info=map_info,
            location=robot_location,
            block_size_in_cells=BLOCK_SIZE_IN_CELLS,
            update_window_in_cells=UPDATE_WINDOW_SIZE
        )
        # 只保留处于 FREE 大块中的预测坐标
        new_unknown_coords = []
        for c in predicted_coords_rounded:
            coord_tuple = tuple(c)
            if self.check_node_exist_in_dict(coord_tuple) is not None:
                continue
            try:
                row_idx, col_idx = get_region_index_from_point(
                    map_info, coord_tuple, block_size_in_cells=BLOCK_SIZE_IN_CELLS
                )
                if region_states[row_idx][col_idx] != FREE:
                    continue  
            except IndexError:
                continue  
            new_unknown_coords.append(coord_tuple)
        print(f"[Debug] Filtered predicted coords: {len(predicted_coords_rounded)} → {len(new_unknown_coords)}")
        
        all_node_list = []
        # 6️⃣ 去除已存在节点，只添加新的预测点
        for coords in node_coords:
            node = self.check_node_exist_in_dict(coords)
            if node is None:
                node = self.add_node_to_dict(coords, frontiers, updating_map_info,new_unknown_coords=new_unknown_coords)
            else:
                node = node.data
                if node.utility == 0 or np.linalg.norm(node.coords - robot_location) > 2 * SENSOR_RANGE:
                    pass
                else:
                    node.update_node_observable_frontiers(frontiers, updating_map_info, map_info,new_unknown_coords=new_unknown_coords)
            all_node_list.append(node)
        # for coord in new_unknown_coords:
        #         coord_tuple = tuple(coord)
        #         node = self.check_node_exist_in_dict(coord_tuple)
        #         if node is None:           
        #             node = self.add_node_to_dict_unknown(coord_tuple, frontiers, map_info,new_unknown_coords=new_unknown_coords)
        #         else:
        #             node = node.data           
        #         all_node_list.append(node)

        print(f"[Debug] Predicted new unknown nodes: {len(new_unknown_coords)}")
        if unknown_centers is not None:
            filtered_centers = self.filter_connected_unknown_centers(unknown_centers, map_info, regions_state)
            for center in filtered_centers:
                center_tuple = tuple(center)
                node = self.check_node_exist_in_dict(center_tuple)
                if node is None:           
                    node = self.add_node_to_dict_unknown(center_tuple, frontiers, map_info,robot_location)
                else:
                    node = node.data           
                all_node_list.append(node)
      

        # Step 8: 更新邻接关系
        for node in all_node_list:
            if node.is_unknown_node:
               continue  

            if node.need_update_neighbor and np.linalg.norm(node.coords - robot_location) < (
                    SENSOR_RANGE + NODE_RESOLUTION):
                node.update_neighbor_nodes(updating_map_info, self.nodes_dict)





    def get_all_node_graph(self, map_info, robot_location, robot_locations,regions_state, unknown_centers):
        all_node_coords = []
        region_to_centers = build_region_to_centers_map(unknown_centers, map_info, BLOCK_SIZE_IN_CELLS)
        for node in self.nodes_dict.__iter__():
            all_node_coords.append(node.data.coords)
        all_node_coords = np.array(all_node_coords).reshape(-1, 2)
        utility = []
        is_unknown_list =  []

        n_nodes = all_node_coords.shape[0]
        adjacent_matrix = np.ones((n_nodes, n_nodes)).astype(int)
        node_coords_to_check = all_node_coords[:, 0] + all_node_coords[:, 1] * 1j
        for i, coords in enumerate(all_node_coords):
            node = self.nodes_dict.find((coords[0], coords[1])).data
            utility.append(node.utility)
            is_unknown_list.append(node.is_unknown_node)
            for neighbor in node.neighbor_list:
                index = np.argwhere(node_coords_to_check == neighbor[0] + neighbor[1] * 1j)
                if index or index == [[0]]:
                    index = index[0][0]
                    adjacent_matrix[i, index] = 0
            if node.utility != 0:
                neighbor_centers = get_neighbor_region_centers_from_point_fast(
                map_info, coords, regions_state, region_to_centers)
                for nb_center in neighbor_centers:
                    index_nb = np.argwhere(node_coords_to_check == nb_center[0] + nb_center[1] * 1j)
                    if index_nb.size > 0:
                        index_nb = index_nb[0][0]
                        # 检查是否碰撞（只考虑占据）
                        collision = check_collision_only_occupied(coords, nb_center, map_info)
                        if not collision:
                            adjacent_matrix[i, index_nb] = 2  # 连为2表示utility有效点的邻居连边
                            adjacent_matrix[index_nb, i] = 2
        # for i in range(n_nodes):
        #     for j in range(n_nodes):
        #         if adjacent_matrix[i, j] != adjacent_matrix[j, i]:
        #             print(f"不对称边: ({i}, {j}) = {adjacent_matrix[i, j]} vs ({j}, {i}) = {adjacent_matrix[j, i]}")

        # # 简单判断整体是否对称
        # if np.array_equal(adjacent_matrix, adjacent_matrix.T):
        #     print("adjacent_matrix 是对称的 ✅")
        # else:
        #     print("adjacent_matrix 不是对称的 ❌")

            # if node.is_unknown_node:
            #     neighbor_centers = get_neighbor_region_centers_from_point_fast(
            #     map_info, coords, regions_state, region_to_centers)
            
            #     for nb_center in neighbor_centers:
            #         index_nb = np.argwhere(node_coords_to_check == nb_center[0] + nb_center[1] * 1j)
            #         if index_nb.size > 0:
            #             index_nb = index_nb[0][0]
            #             # 这里不检查碰撞，直接连边为3
            #             adjacent_matrix[i, index_nb] = 3

        utility = np.array(utility)
        is_unknown_list = np.array(is_unknown_list)

        utility = np.array(utility)

        ## guidepost 2
        indices = np.argwhere(utility > 0).reshape(-1)
        utility_node_coords = all_node_coords[indices]
        dist_dict, prev_dict = self.Dijkstra(robot_location)
        nearest_utility_coords = robot_location
        nearest_dist = 1e8
        for coords in utility_node_coords:
            dist = dist_dict[(coords[0], coords[1])]
            if 0 < dist < nearest_dist:
                nearest_dist = dist
                nearest_utility_coords = coords
                # print(nearest_dist, coords, nearest_utility_coords, robot_location)
        path_coords, dist = self.a_star(robot_location, nearest_utility_coords)
        guidepost = np.zeros_like(utility)
        for coords in path_coords:
            index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
            guidepost[index] = 1

        # ## guidepost 3
        # guidepost = np.zeros_like(utility)
        # indices = np.argwhere(utility > 0).reshape(-1)
        # utility_node_coords = all_node_coords[indices]
        # dist_dict, prev_dict = self.Dijkstra(robot_location)
        # for coords in utility_node_coords:
        #     path, _ = self.get_Dijkstra_path_and_dist(dist_dict, prev_dict, coords)
        #     for coords in path:
        #         index = np.argwhere(all_node_coords[:, 0] + all_node_coords[:, 1] * 1j == coords[0] + coords[1] * 1j)[0]
        #         guidepost[index] = 1

        robot_in_graph = self.nodes_dict.nearest_neighbors(robot_location.tolist(), 1)[0].data.coords
        current_index = np.argwhere(node_coords_to_check == robot_in_graph[0] + robot_in_graph[1] * 1j)[0][0]
        neighbor_indices = np.argwhere(adjacent_matrix[current_index] == 0).reshape(-1)

        occupancy = np.zeros((n_nodes, 1))
        for location in robot_locations:
            location_in_graph = self.nodes_dict.find((location[0], location[1])).data.coords
            index = np.argwhere(node_coords_to_check == location_in_graph[0] + location_in_graph[1] * 1j)[0][0]
            if index == current_index:
                occupancy[index] = -1
            else:
                occupancy[index] = 1
        # assert sum(occupancy) == 2, print(robot_locations)
        return all_node_coords, utility, guidepost, occupancy, adjacent_matrix, current_index, neighbor_indices, is_unknown_list

    def Dijkstra(self, start):
        q = set()
        dist_dict = {}
        prev_dict = {}

        for node in self.nodes_dict.__iter__():
            coords = node.data.coords
            key = (coords[0], coords[1])
            dist_dict[key] = 1e8
            prev_dict[key] = None
            q.add(key)

        assert (start[0], start[1]) in dist_dict.keys()
        dist_dict[(start[0], start[1])] = 0

        while len(q) > 0:
            u = None
            for coords in q:
                if u is None:
                    u = coords
                elif dist_dict[coords] < dist_dict[u]:
                    u = coords

            q.remove(u)

            if self.nodes_dict.find(u) is None:
                print(u)
                for node in self.nodes_dict.__iter__():
                    print(node.data.coords)

            node = self.nodes_dict.find(u).data
            for neighbor_node_coords in node.neighbor_list:
                v = (neighbor_node_coords[0], neighbor_node_coords[1])
                if v in q:
                    cost = ((neighbor_node_coords[0] - u[0]) ** 2 + (
                            neighbor_node_coords[1] - u[1]) ** 2) ** (1 / 2)
                    cost = np.round(cost, 2)
                    alt = dist_dict[u] + cost
                    if alt < dist_dict[v]:
                        dist_dict[v] = alt
                        prev_dict[v] = u

        return dist_dict, prev_dict

    def get_Dijkstra_path_and_dist(self, dist_dict, prev_dict, end):
        if (end[0], end[1]) not in dist_dict:
            return [], 1e8

        dist = dist_dict[(end[0], end[1])]

        path = [(end[0], end[1])]
        prev_node = prev_dict[(end[0], end[1])]
        while prev_node is not None:
            path.append(prev_node)
            temp = prev_node
            prev_node = prev_dict[temp]

        path.reverse()
        return path[1:], np.round(dist, 2)

    def h(self, coords_1, coords_2):
        # h = abs(coords_1[0] - coords_2[0]) + abs(coords_1[1] - coords_2[1])
        h = ((coords_1[0] - coords_2[0]) ** 2 + (coords_1[1] - coords_2[1]) ** 2) ** (1 / 2)
        h = np.round(h, 2)
        return h

    def a_star(self, start, destination, boundary=None, max_dist=None):
        # the path does not include the start
        if not self.check_node_exist_in_dict(start):
            print(start)
            Warning("start position is not in node dict")
            return [], 1e8
        if not self.check_node_exist_in_dict(destination):
            Warning("end position is not in node dict")
            return [], 1e8

        if start[0] == destination[0] and start[1] == destination[1]:
            return [destination], 0

        open_list = {(start[0], start[1])}
        closed_list = set()
        g = {(start[0], start[1]): 0}
        parents = {(start[0], start[1]): (start[0], start[1])}

        while len(open_list) > 0:
            n = None
            h_n = 1e8

            for v in open_list:
                h_v = self.h(v, destination)
                if n is not None:
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords
                    h_n = self.h(n_coords, destination)
                if n is None or g[v] + h_v < g[n] + h_n:
                    n = v
                    node = self.nodes_dict.find(n).data
                    n_coords = node.coords

            if max_dist is not None:
                if g[n] > max_dist:
                    return [], 1e8

            if n_coords[0] == destination[0] and n_coords[1] == destination[1]:
                path = []
                length = g[n]
                while parents[n] != n:
                    path.append(n)
                    n = parents[n]
                path.reverse()
                return path, np.round(length, 2)

            for neighbor_node_coords in node.neighbor_list:
                if self.nodes_dict.find(neighbor_node_coords.tolist()) is None:
                    continue
                if boundary is not None:
                    if not (boundary[0] < neighbor_node_coords[0] < boundary[2] and boundary[1] < neighbor_node_coords[1] < boundary[3]):
                        continue
                cost = ((neighbor_node_coords[0] - n_coords[0]) ** 2 + (
                            neighbor_node_coords[1] - n_coords[1]) ** 2) ** (1 / 2)
                cost = np.round(cost, 2)
                m = (neighbor_node_coords[0], neighbor_node_coords[1])
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + cost
                else:
                    if g[m] > g[n] + cost:
                        g[m] = g[n] + cost
                        parents[m] = n

                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)

        print('Path does not exist!')

        return [], 1e8



class Node:
    def __init__(self, coords, frontiers, updating_map_info,new_unknown_coords):
        self.coords = coords
        self.utility_range = UTILITY_RANGE
        self.utility = 0
        self.observable_frontiers = self.initialize_observable_frontiers(frontiers, updating_map_info,new_unknown_coords)
        self.visited = 0

        self.neighbor_matrix = -np.ones((5, 5))
        self.neighbor_list = []
        self.neighbor_matrix[2, 2] = 1
        self.neighbor_list.append(self.coords)
        self.need_update_neighbor = True
        self.is_unknown_node = 0 

    def initialize_observable_frontiers(self, frontiers, updating_map_info, new_unknown_coords):
        if len(frontiers) == 0:
            self.utility = 0
            return set()
        else:
            observable_frontiers = set()

            frontiers = np.array(list(frontiers)).reshape(-1, 2)
            dist_list = np.linalg.norm(frontiers - self.coords, axis=-1)
            new_frontiers_in_range = frontiers[dist_list < self.utility_range]
            for point in new_frontiers_in_range:
                collision = check_collision(self.coords, point, updating_map_info)
                if not collision:
                    observable_frontiers.add((point[0], point[1]))
            
            self.utility = len(observable_frontiers)

            # # 预测节点密度增强效用
            # pred_count = sum(
            #     1 for p in new_unknown_coords
            #     if np.linalg.norm(np.array(p) - self.coords) <= self.utility_range
            # )

            # # 可调节参数
            # max_scale = 2.0
            # scale_base = int(np.pi * self.utility_range**2 / 0.3)  # 每个节点占 0.3㎡
            # scale = min(1.0 + pred_count / max(scale_base, 1), max_scale)

            # self.utility = int(base_utility * scale)
            count = 0
            scale = 1.0
            if self.utility > 0 and new_unknown_coords is not None:
                # ==== 可调参数 ====
                max_scale = 2.0               # 最大放大倍数
                scale_base_count = 10         # 达到最大scale时所需预测节点数
                radius = self.utility_range   # 统计半径范围
                for pred_coord in new_unknown_coords:
                    if np.linalg.norm(np.array(pred_coord) - self.coords) <= radius:
                        count += 1

                scale = min(1.0 + count / scale_base_count, max_scale)
                self.utility *= scale
            if self.utility <= MIN_UTILITY:
                self.utility = 0
                observable_frontiers = set()
            print(f"[Debug][Node {self.coords}] Utility before={len(observable_frontiers)}, predicted_nodes={count}, scale={scale:.2f}, final_utility={self.utility:.2f}")
            return observable_frontiers

    def update_neighbor_nodes(self, updating_map_info, nodes_dict):
        for i in range(self.neighbor_matrix.shape[0]):
            for j in range(self.neighbor_matrix.shape[1]):
                if self.neighbor_matrix[i, j] != -1:
                    continue
                else:
                    center_index = self.neighbor_matrix.shape[0] // 2
                    if i == center_index and j == center_index:
                        self.neighbor_matrix[i, j] = 1
                        continue

                    neighbor_coords = np.around(np.array([self.coords[0] + (i - center_index) * NODE_RESOLUTION,
                                                          self.coords[1] + (j - center_index) * NODE_RESOLUTION]), 1)
                    neighbor_node = nodes_dict.find((neighbor_coords[0], neighbor_coords[1]))
                    if neighbor_node is None:
                        # cell = get_cell_position_from_coords(neighbor_coords, updating_map_info)
                        # if updating_map_info.map[cell[1], cell[0]] == 1:
                        #    self.neighbor_matrix[i, j] = 1
                        continue
                    else:
                        neighbor_node = neighbor_node.data
                        collision = check_collision(self.coords, neighbor_coords, updating_map_info)
                        neighbor_matrix_x = center_index + (center_index - i)
                        neighbor_matrix_y = center_index + (center_index - j)
                        if not collision:
                            self.neighbor_matrix[i, j] = 1
                            self.neighbor_list.append(neighbor_coords)

                            neighbor_node.neighbor_matrix[neighbor_matrix_x, neighbor_matrix_y] = 1
                            neighbor_node.neighbor_list.append(self.coords)

        if self.utility == 0:
            self.need_update_neighbor = False
        elif 0 in self.neighbor_matrix is False:
            self.need_update_neighbor = False
        # print(self.neighbor_matrix)

    def update_node_observable_frontiers(self, frontiers, updating_map_info, map_info,new_unknown_coords=None):
        # remove frontiers observed
        frontiers_observed = []
        for frontier in self.observable_frontiers:
            if not is_frontier(np.array([frontier[0], frontier[1]]), map_info):
                frontiers_observed.append(frontier)
        for frontier in frontiers_observed:
            self.observable_frontiers.remove(frontier)

        # add new frontiers in the observable frontiers
        new_frontiers = frontiers - self.observable_frontiers
        new_frontiers = np.array(list(new_frontiers)).reshape(-1, 2)
        dist_list = np.linalg.norm(new_frontiers - self.coords, axis=-1)
        new_frontiers_in_range = new_frontiers[dist_list < self.utility_range]
        for point in new_frontiers_in_range:
            collision = check_collision(self.coords, point, updating_map_info)
            if not collision:
                self.observable_frontiers.add((point[0], point[1]))

        self.utility = len(self.observable_frontiers)
        if self.utility > 0 and new_unknown_coords is not None:
            # ==== 可调参数 ====
            max_scale = 2.0               # 最大放大倍数
            scale_base_count = 10         # 达到最大scale时所需预测节点数
            radius = self.utility_range   # 统计半径范围

            count = 0
            for pred_coord in new_unknown_coords:
                if np.linalg.norm(np.array(pred_coord) - self.coords) <= radius:
                    count += 1

            scale = min(1.0 + count / scale_base_count, max_scale)
            self.utility *= scale
        if self.utility <= MIN_UTILITY:
            self.utility = 0
            self.observable_frontiers = set()
            self.need_update_neighbor = False
        # print(f"[Debug][Node {self.coords}] Utility before={len(self.observable_frontiers)}, predicted_nodes={count}, scale={scale:.2f}, final_utility={self.utility:.2f}")


    def delete_observed_frontiers(self, observed_frontiers):
        # remove observed frontiers in the observable frontiers
        self.observable_frontiers = self.observable_frontiers - observed_frontiers

    def set_visited(self):
        self.visited = 1
        self.observable_frontiers = set()
        self.utility = 0
        self.need_update_neighbor = False