import os
import time
import random
import collections

from copy import deepcopy

import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace

from test_parameter import *
from classes.utils import *
from classes.env.env import Env
from classes.agent.agent import Agent
from classes.agent.node_manager import NodeManager
from classes.planner.expert_planner import ExpertPlanner
from classes.planner.ground_truth_planner import GroundTruthPlanner
from classes.predictor.global_predictor import GlobalPredictor
# from lama.saicinpainting.training.trainers import load_checkpoint
# from lama.saicinpainting.evaluation.utils import move_to_device
import yaml
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)



def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure reproducibility in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TestWorker:
    def __init__(self, meta_agent_id, policy, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.policy = policy
        self.global_step = global_step
        self.device = device
        self.save_image = save_image
        self.env= Env(global_step, TEST_N_AGENTS, plot=save_image, test=USE_TEST_DATASET)
        self.node_manager = NodeManager(map_info=self.env.belief_info,plot=save_image)
        self.robot_list = [Agent(i, self.node_manager, self.device, save_image) for i in range(TEST_N_AGENTS)]
        self.perf_metrics = dict()
        self.global_predictor = GlobalPredictor(self.env.belief_info)
        self.obs_horizon = policy.n_obs_steps
        self.action_horizon = policy.n_action_steps if ACTION_HORIZON == None else ACTION_HORIZON
        # assert self.action_horizon == 1, "Action horizon must be 1 for this driver"
        
        self.planned_path_x = []
        self.planned_path_y = []
    # def load_lama_model(self, ckpt_path):

    #     # 加载配置
    #     config_path = os.path.join(os.path.dirname(ckpt_path), 'config.yaml')
    #     with open(config_path, 'r') as f:
    #         train_config = yaml.safe_load(f)
    #     train_config['model']['params']['predict_only_known'] = False
    #     train_config['model']['params']['side_inputs'] = False

    #     model = load_checkpoint(train_config, ckpt_path, map_location=self.device)
    #     model.freeze()  # 推理时不需要梯度
    #     model.eval()
    #     return model
    def run_episode(self):
        unique_seed = int(time.time())
        set_random_seed(unique_seed)
        self.lama_rgb_image = None
        self.lama_binary_image = None
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
            # free_coords = np.array(list(self.robot_list[0].node_manager.all_free_node_coords))
            # obs_coords = np.array(list(self.robot_list[0].node_manager.all_obstacle_coords))
            # self.global_predictor.update_nodes(free_coords, obs_coords)
            # lama_output = self.global_predictor.request_lama_prediction()
            # self.lama_rgb_image = lama_output
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        # Get the first observation
        if DATA_TYPE == 'node':
            # self.env.ground_truth_planner = GroundTruthPlanner(self.env.ground_truth_info, robot.node_manager)
            # paths = self.env.get_ground_truth_paths()
            # self.robot_list[0].global_path = paths[0]
            
            observation = self.robot_list[0].get_observation()
            node_inputs = observation[0].squeeze(0)
            node_padding_mask = observation[1].squeeze(0)
            edge_mask = observation[2].squeeze(0)
            current_index = observation[3].squeeze(0)
            current_edge = observation[4].squeeze(0)
            edge_padding_mask = observation[5].squeeze(0)

            obs = {'node_inputs': node_inputs,
                'node_padding_mask': node_padding_mask,
                'edge_mask': edge_mask,
                'current_index': current_index,
                'current_edge': current_edge,
                'edge_padding_mask': edge_padding_mask}
        elif DATA_TYPE == 'map':
            image = deepcopy(self.env.robot_belief)
            state = deepcopy(self.env.robot_locations[0])

            agent_pos = state.astype(np.float32)
            image = image.astype(np.float32)/255
            image = np.expand_dims(image, axis=0)

            obs = {'image': image,
                'agent_pos': state}
        else:
            raise ValueError('Invalid data type, check test_parameter.py')

        # keep a queue of last 2 steps of observations
        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        step = 0
        for step in range(MAX_EPISODE_STEP):
            # stack the last obs_horizon number of observations
            if DATA_TYPE == 'node':
                node_inputs = torch.stack([x['node_inputs'] for x in obs_deque])
                node_padding_mask = torch.stack([x['node_padding_mask'] for x in obs_deque])
                edge_mask = torch.stack([x['edge_mask'] for x in obs_deque])
                current_index = torch.stack([x['current_index'] for x in obs_deque])
                current_edge = torch.stack([x['current_edge'] for x in obs_deque])
                edge_padding_mask = torch.stack([x['edge_padding_mask'] for x in obs_deque])

                # device transfer
                # TODO check if need to down cast to int16/32 then upcast to int64 like dataset
                node_inputs = node_inputs.to(self.device, dtype=torch.float32) # (obs_horizon, 360, 5)
                node_padding_mask = node_padding_mask.to(self.device, dtype=torch.int16) # (obs_horizon, 1, 360)
                edge_mask = edge_mask.to(self.device, dtype=torch.int64) # (obs_horizon, 360, 360)
                current_index = current_index.to(self.device, dtype=torch.int64) # (obs_horizon, 1, 1)
                current_edge = current_edge.to(self.device, dtype=torch.int64) # (obs_horizon, 25, 1)
                edge_padding_mask = edge_padding_mask.to(self.device, dtype=torch.int16) # (obs_horizon, 1, 25)

                # observation dict
                obs_dict = {'node_inputs': node_inputs.unsqueeze(0),
                            'node_padding_mask': node_padding_mask.unsqueeze(0),
                            'edge_mask': edge_mask.unsqueeze(0),
                            'current_index': current_index.unsqueeze(0),
                            'current_edge': current_edge.unsqueeze(0),
                            'edge_padding_mask': edge_padding_mask.unsqueeze(0)}
            elif DATA_TYPE == 'map':
                # stack the last obs_horizon number of observations
                image = torch.stack([torch.tensor(x['image']) for x in obs_deque])
                agent_pos = torch.stack([torch.tensor(x['agent_pos']) for x in obs_deque])

                # device transfer
                image = image.to(self.device, dtype=torch.float32) # (obs_horizon, 512)
                agent_pos = agent_pos.to(self.device, dtype=torch.float32) # (obs_horizon, 2)

                # observation dict
                obs_dict = {'image': image.unsqueeze(0),
                            'agent_pos': agent_pos.unsqueeze(0)}
            else:
                raise ValueError('Invalid data type, check test_parameter.py')
        
            # infer action
            # time_start = time.time()
            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)
            # time_end = time.time()
            # print(f"Time taken for inference: {time_end - time_start}s")

            action_pred = action_dict['action_pred'].squeeze(0).cpu().numpy() # (pred_horizon, action_dim)

            action_pred = np.round(action_pred / NODE_RESOLUTION) * NODE_RESOLUTION  # round to nearest node resolution
            
            # only take action_horizon number of actions
            start = self.obs_horizon - 1
            end = start + self.action_horizon
            action = action_pred[start:end,:] # (action_horizon, action_dim)
            
            # execute action_horizon number of steps without replanning
            for action_step in range(self.action_horizon):
                if action_step == 0:
                    planned_location = deepcopy(self.env.robot_locations[0])
                    self.planned_path_x.append([planned_location[0]])
                    self.planned_path_y.append([planned_location[1]])
                    # get planned path for visualization
                    if USE_DELTA_POSITION:
                        for i in range(start, len(action_pred)):
                            planned_location = planned_location + action_pred[i]
                            self.planned_path_x[step].append(planned_location[0])
                            self.planned_path_y[step].append(planned_location[1])
                    else:
                        for i in range(start, len(action_pred)):
                            planned_location = action_pred[i]
                            self.planned_path_x[step].append(planned_location[0])
                            self.planned_path_y[step].append(planned_location[1])
                else:
                    self.planned_path_x.append(self.planned_path_x[step - action_step])
                    self.planned_path_y.append(self.planned_path_y[step - action_step])
                    pass
                # print(f"Step: {step}, Action Step: {action_step}")
                if USE_DELTA_POSITION:
                    selected_coord = self.env.robot_locations[0] + action[action_step]
                else:
                    selected_coord = action[action_step]

                current_node = self.robot_list[0].node_manager.nodes_dict.find(self.env.robot_locations[0].tolist()).data

                ## Collision avoidance
                # check if selected_coord is a valid neighbour of current node
                if not any(np.all(selected_coord == neighbor) for neighbor in current_node.neighbor_list):
                    # print("Collision Detected!")
                    # Vectors of 3 future positions from current position # HACK fixed number here
                    direction_vectors = np.cumsum(action_pred[start: start + 3], axis=0)
                    best_neighbor = None
                    best_average_angle = float('inf')
                    # print(f"Direction Vectors: {direction_vectors}")
                    for neighbor_coords in current_node.neighbor_list:
                        # skip current robot location
                        if np.all(neighbor_coords == self.env.robot_locations[0]):
                            continue 
                        neighbor_direction = neighbor_coords - self.env.robot_locations[0]
                        # print(f"Neighbor Direction: {neighbor_direction}")
                        angles = []
                        for direction_vector in direction_vectors:
                            direction_magnitude = np.linalg.norm(direction_vector)
                            neighbor_magnitude = np.linalg.norm(neighbor_direction)
                            if direction_magnitude == 0 or neighbor_magnitude == 0: # skip zero vectors
                                continue
                            angle = np.arctan2(np.linalg.det([direction_vector, neighbor_direction]), np.dot(direction_vector, neighbor_direction))
                            angles.append(angle)
                        weights = np.arange(len(angles), 0, -1)
                        weighted_average_angle = np.average(np.abs(angles), weights=weights)  # Use absolute values for magnitude
                        # print(f"Weighted Average Angle: {weighted_average_angle}")
                        if weighted_average_angle < best_average_angle:
                            best_average_angle = weighted_average_angle
                            best_neighbor = neighbor_coords
                    # print(f"Best Neighbor: {best_neighbor}, action: {best_neighbor - self.env.robot_locations[0]}")
                    selected_coord = best_neighbor
                else:
                    # print("Valid Action")
                    pass
                
                # step the environment
                self.env.step(selected_coord, 0)
                
                # update robot state
                self.robot_list[0].update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[0]))
                # free_coords = np.array(list(self.robot_list[0].node_manager.all_free_node_coords))
                # obs_coords = np.array(list(self.robot_list[0].node_manager.all_obstacle_coords))
                # self.global_predictor.update_nodes(free_coords, obs_coords)
                # lama_output = self.global_predictor.request_lama_prediction()  # 返回 np.uint8 的 (H, W, 3) 图像
                # self.lama_rgb_image = lama_output
                # image_tensor, mask_tensor = self.global_predictor.get_lama_input()
                # image_tensor_rgb = image_tensor.repeat(1, 3, 1, 1)
                # with torch.no_grad():
                #     result = self.lama_model({
                #         'image': image_tensor_rgb.to(self.device),
                #         'mask': mask_tensor.to(self.device)
                #     })
                #     predicted_image = result['inpainted'][0]  # shape: [3, H, W]
                self.robot_list[0].update_planning_state(self.env.robot_locations)

                if DATA_TYPE == 'node':
                    observation = self.robot_list[0].get_observation()
                    node_inputs = observation[0].squeeze(0)
                    node_padding_mask = observation[1].squeeze(0)
                    edge_mask = observation[2].squeeze(0)
                    current_index = observation[3].squeeze(0)
                    current_edge = observation[4].squeeze(0)
                    edge_padding_mask = observation[5].squeeze(0)

                    obs = {'node_inputs': node_inputs,
                        'node_padding_mask': node_padding_mask,
                        'edge_mask': edge_mask,
                        'current_index': current_index,
                        'current_edge': current_edge,
                        'edge_padding_mask': edge_padding_mask}
                elif DATA_TYPE == 'map':
                    image = deepcopy(self.env.robot_belief)
                    state = deepcopy(self.env.robot_locations[0])

                    agent_pos = state.astype(np.float32)
                    image = image.astype(np.float32)/255
                    if len(image.shape) == 2: # add channel dimension
                        image = np.expand_dims(image, axis=0)

                    obs = {'image': image,
                        'agent_pos': state}
                else:
                    raise ValueError('Invalid data type, check test_parameter.py')

                obs_deque.append(obs)

                if USE_EXPLORATION_RATE_FOR_DONE:
                    self.env.check_done()
                    done = self.env.done
                else:
                    done = self.robot_list[0].utility.sum() == 0

                if self.save_image: # save gif
                    self.plot_env(step)
                    # self.plot_real(step)

                if done: # exit action loop if done or collision
                    break

            if done: # exit episode loop if done
                break
                
        self.perf_metrics['travel_dist'] = self.robot_list[0].travel_dist
        self.perf_metrics['success_rate'] = done
        
        if self.save_image: # save gif
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, delete_images=True)

    def plot_belief(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(6, 6))
        plt.imshow(self.env.robot_belief, cmap='gray', origin='lower')
        # plt.title("robot_belief Map")
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_samples.png'.format(observe_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(observe_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()
    
    def plot_real(self, step):
        self.regions,self.regions_state,self.unknown_centers = get_map_into_regions(self.env.belief_info,self.env.robot_locations[0])
        plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=(6, 6))
        
        ax.imshow(self.env.robot_belief, cmap='gray', origin='lower')
        
        for i, row in enumerate(self.regions):
            for j, region in enumerate(row):
                y_start = i * region.shape[0]
                x_start = j * region.shape[1]
                height = region.shape[0]
                width = region.shape[1]
                state = self.regions_state[i][j]
                
                color = 'lime' if state == FREE else 'red'
                rect = patches.Rectangle((x_start, y_start), width, height,
                                        linewidth=1.5, edgecolor=color, facecolor='none')
                ax.add_patch(rect)
        # plt.title("robot_belief Map")
        color_list = ['r', 'b', 'g', 'y']
        for robot in self.robot_list:
            c = color_list[robot.id]
            if hasattr(robot, 'node_coords') and robot.node_coords is not None:
                node_cells = get_cell_position_from_coords(robot.node_coords, robot.map_info)
                ax.scatter(node_cells[:, 0], node_cells[:, 1], c=c, s=10, label=f"robot{robot.id} nodes")

                if hasattr(robot, 'adjacent_matrix') and robot.adjacent_matrix is not None:
                    adj = robot.adjacent_matrix
                    for i in range(len(node_cells)):
                        for j in range(len(node_cells)):
                            edge_type = adj[i][j]
                            if edge_type != 1:
                                x = [node_cells[i][0], node_cells[j][0]]
                                y = [node_cells[i][1], node_cells[j][1]]

                                # ✅ 根据边的类型定义样式
                                if edge_type == 0:
                                    ax.plot(x, y, color='black', linewidth=1.0, linestyle='-', alpha=0.7)
                                elif edge_type == 2:
                                    ax.plot(x, y, color='orange', linewidth=1.5, linestyle='--', alpha=0.8)
                                elif edge_type == 3:
                                    ax.plot(x, y, color='magenta', linewidth=2.0, linestyle='-.', alpha=0.9)

                if hasattr(robot, 'utility') and robot.utility is not None:
                    for node, util in zip(node_cells, robot.utility):
                        ax.text(node[0], node[1], str(round(util, 1)), fontsize=6, color='black')
                if hasattr(robot, 'is_unknown_node') and robot.is_unknown_node is not None:
                    is_unknown = robot.is_unknown_node.flatten()
                    unknown_nodes = node_cells[is_unknown.astype(bool)]
                    ax.scatter(unknown_nodes[:, 0], unknown_nodes[:, 1], c='purple', s=20, marker='x', label=f"robot{robot.id} unknown")



            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            ax.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)
        # if self.unknown_centers is not None and len(self.unknown_centers) > 0:
        #     unknown_cells = get_cell_position_from_coords(np.array(self.unknown_centers), self.env.belief_info)
        #     ax.scatter(unknown_cells[:, 0], unknown_cells[:, 1], c='purple', s=6, marker='x', label='Unknown Centers')

        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()
        
    def plot_env(self, step, planned_paths=None):
        self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)
        plt.switch_backend('agg')
        color_list = ['r', 'b', 'g', 'y']
        plt.figure(figsize=(25, 5))  # 扩展宽度以容纳 5 张子图

        # === 获取 robot 0 的信息 ===
        robot = self.robot_list[0]
        map_info = robot.map_info
        cell_size = robot.cell_size
        node_coords = robot.node_coords
        utility = robot.utility
        node_cells = get_cell_position_from_coords(node_coords, map_info)

        # 当前帧障碍物节点
        if hasattr(robot.node_manager, 'last_obstacle_coords'):
            obs_coords = robot.node_manager.last_obstacle_coords
            if isinstance(obs_coords, (set, list)):
                obs_coords = np.array(list(obs_coords))
            if len(obs_coords.shape) == 1:
                obs_coords = np.expand_dims(obs_coords, 0)
            obs_cells = get_cell_position_from_coords(obs_coords, map_info)
        else:
            obs_coords = np.empty((0, 2))
            obs_cells = np.empty((0, 2))

        # 所有累积障碍物节点（用于第三张图）
        all_obs_coords = np.array(list(robot.node_manager.all_obstacle_coords))
        all_obs_cells = get_cell_position_from_coords(all_obs_coords, map_info)

        # === subplot (1) 节点视图 ===
        plt.subplot(1, 5, 1)
        plt.imshow(map_info.map, cmap='gray')
        plt.axis('off')
        plt.scatter(node_cells[:, 0], node_cells[:, 1], c=utility, cmap='viridis', s=20, zorder=2, label='free node')
        if obs_cells.shape[0] > 0:
            plt.scatter(obs_cells[:, 0], obs_cells[:, 1], c='black', s=20, label='obstacle node', zorder=2)
        robot_cell = get_cell_position_from_coords(robot.location, map_info)
        plt.plot(robot_cell[0], robot_cell[1], 'ro', markersize=16, zorder=5)
        if len(self.env.global_frontiers) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)), self.env.belief_info)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='red', s=2, label='frontier')
        plt.title("Node View")
        plt.legend(loc='upper right')

        # === subplot (2) belief map + trajectory + path ===
        plt.subplot(1, 5, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                    (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size,
                    c, linewidth=2, zorder=1)
            plt.plot((np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size,
                    (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size,
                    'g', linewidth=1, zorder=2)
            if hasattr(robot.node_manager, 'last_obstacle_coords'):
                o_coords = robot.node_manager.last_obstacle_coords
                if isinstance(o_coords, (set, list)):
                    o_coords = np.array(list(o_coords))
                if len(o_coords.shape) == 1:
                    o_coords = np.expand_dims(o_coords, 0)
                if len(o_coords) > 0:
                    o_cells = get_cell_position_from_coords(o_coords, robot.map_info)
                    plt.scatter(o_cells[:, 0], o_cells[:, 1], c='black', marker='x', s=15, zorder=3)
        plt.title("Belief Map View")

        # === subplot (3) LaMa 输入图像（灰度编码）===
        plt.subplot(1, 5, 3)
        image_tensor, mask_tensor, mask_tensor_o,orig_shape= robot.node_manager.global_predictor.get_lama_input()
        image = image_tensor.squeeze().numpy()
        mask = mask_tensor.squeeze().numpy()
        lama_input_visual = np.full(image.shape, 0.5, dtype=np.float32)
        lama_input_visual[(mask == 0) & (image >= 0.99)] = 1.0
        lama_input_visual[(mask == 0) & (image <= 0.01)] = 0.0
        plt.imshow(lama_input_visual, cmap='gray', vmin=0, vmax=1)
        plt.title("LaMa Input")
        plt.axis("off")

        # === subplot (4) LaMa 二值输出图 ===
        plt.subplot(1, 5, 4)

        # 🔁 来自 robot.node_manager 中的 lama_rgb_image
        robot = self.robot_list[0]
        lama_rgb = robot.node_manager.lama_rgb_image

        if lama_rgb is not None:
            gray_image = np.mean(lama_rgb, axis=2) / 255.0
            binary_image = np.zeros_like(gray_image, dtype=np.uint8)
            binary_image[gray_image >= 0.3] = 255
            binary_image[gray_image < 0.7] = 0
            plt.imshow(binary_image, cmap='gray', vmin=0, vmax=255)
            plt.title("LaMa Output (Binarized)")
            plt.axis("off")
            self.lama_binary_image = binary_image
        else:
            plt.text(0.5, 0.5, "No prediction", ha='center', va='center')
            plt.axis("off")
            
        plt.subplot(1, 5, 5)
        plt.imshow(map_info.map, cmap='gray')
        plt.axis("off")

        if self.lama_binary_image is not None:
            binary_map = self.lama_binary_image
            free_indices = np.argwhere(binary_map == 255)
            obs_indices = np.argwhere(binary_map == 0)

            # ✅ 将图像坐标 → 地图 cell 坐标
            def node_pixel_to_map_cell(indices):
                y, x = indices[:, 0], indices[:, 1]
                real_x = x * 4.0 + 1 + map_info.map_origin_x  # NODE_RESOLUTION = 4.0
                real_y = y * 4.0 + 0.5 + map_info.map_origin_y
                cell_x = ((real_x - map_info.map_origin_x) / 0.4).astype(int)  # CELL_SIZE = 0.4
                cell_y = ((real_y - map_info.map_origin_y) / 0.4).astype(int)
                return np.stack([cell_x, cell_y], axis=1)

            free_cells = node_pixel_to_map_cell(free_indices)
            obs_cells = node_pixel_to_map_cell(obs_indices)

            if free_cells.shape[0] > 0:
                plt.scatter(free_cells[:, 0], free_cells[:, 1], c='blue', s=20, label='Pred Free')
            if obs_cells.shape[0] > 0:
                plt.scatter(obs_cells[:, 0], obs_cells[:, 1], c='black', s=20, label='Pred Obs')
            plt.title("Predicted Node View (Aligned)")
            plt.legend(loc='upper right')
        else:
            plt.text(0.5, 0.5, "No prediction", ha='center', va='center')


        # === 保存图像 ===
        plt.suptitle(f'Explored rate: {self.env.explored_rate:.4g}  Travel distance: {max(r.travel_dist for r in self.robot_list):.4g}')
        plt.tight_layout()
        save_path = f'{gifs_path}/{self.global_step}_{step}_samples.png'
        plt.savefig(save_path, dpi=150)
        self.env.frame_files.append(save_path)
        plt.close()
    # plt.subplot(1, 5, 5)
        # plt.axis("off")

        # if self.lama_binary_image is not None:
        #     binary_map = self.lama_binary_image  # shape: [H_pred, W_pred]
        #     H_pred, W_pred = binary_map.shape

        #     # 获取自由与障碍的像素位置
        #     free_indices = np.argwhere(binary_map == 255)
        #     obs_indices = np.argwhere(binary_map == 0)

        #     def pixel_to_world_coords(indices):
        #         i, j = indices[:, 0], indices[:, 1]
        #         real_x = (j + 0.5) * 15
        #         real_y = (i + 0.5) * 15
        #         return np.stack([real_x, real_y], axis=1)

        #     free_coords = pixel_to_world_coords(free_indices)
        #     obs_coords = pixel_to_world_coords(obs_indices)

        #     # 映射为地图 cell 位置（用于scatter显示）
        #     free_cells = get_cell_position_from_coords(free_coords, map_info)
        #     obs_cells = get_cell_position_from_coords(obs_coords, map_info)

        #     # 创建空白图层，仅显示节点
        #     blank_canvas = np.ones_like(map_info.map) * 255  # 全白背景
        #     plt.imshow(blank_canvas, cmap='gray', vmin=0, vmax=255)
        #     if free_cells.shape[0] > 0:
        #         plt.scatter(free_cells[:, 0], free_cells[:, 1], c='blue', s=20, label='Pred Free')
        #     if obs_cells.shape[0] > 0:
        #         plt.scatter(obs_cells[:, 0], obs_cells[:, 1], c='black', s=20, label='Pred Obs')

        #     plt.title("Predicted Node Only")
        #     plt.legend(loc='upper right')
        # else:
        #     plt.text(0.5, 0.5, "No prediction", ha='center', va='center')
        #     plt.axis("off")

    # def plot_env(self, step, planned_paths=None):
    #     self.env.global_frontiers = get_frontier_in_map(self.env.belief_info)
    #     plt.switch_backend('agg')
    #     color_list = ['r', 'b', 'g', 'y']
    #     plt.figure(figsize=(20, 5))  # 扩展宽度以容纳 4 张子图

    #     # === 获取 robot 0 的信息 ===
    #     robot = self.robot_list[0]
    #     map_info = robot.map_info
    #     cell_size = robot.cell_size
    #     node_coords = robot.node_coords
    #     utility = robot.utility
    #     node_cells = get_cell_position_from_coords(node_coords, map_info)

    #     # 当前帧障碍物节点
    #     if hasattr(robot.node_manager, 'last_obstacle_coords'):
    #         obs_coords = robot.node_manager.last_obstacle_coords
    #         if isinstance(obs_coords, (set, list)):
    #             obs_coords = np.array(list(obs_coords))
    #         if len(obs_coords.shape) == 1:
    #             obs_coords = np.expand_dims(obs_coords, 0)
    #         obs_cells = get_cell_position_from_coords(obs_coords, map_info)
    #     else:
    #         obs_coords = np.empty((0, 2))
    #         obs_cells = np.empty((0, 2))

    #     # 所有累积障碍物节点（用于第三张图）
    #     all_obs_coords = np.array(list(robot.node_manager.all_obstacle_coords))
    #     all_obs_cells = get_cell_position_from_coords(all_obs_coords, map_info)

    #     # === subplot (1) 显示节点（自由 + 障碍）===
    #     plt.subplot(1, 4, 1)
    #     plt.imshow(map_info.map, cmap='gray')
    #     plt.axis('off')
    #     plt.scatter(node_cells[:, 0], node_cells[:, 1], c=utility, cmap='viridis', s=20, zorder=2, label='free node')
    #     if obs_cells.shape[0] > 0:
    #         plt.scatter(obs_cells[:, 0], obs_cells[:, 1], c='black', s=20, label='obstacle node', zorder=2)
    #     robot_cell = get_cell_position_from_coords(robot.location, map_info)
    #     plt.plot(robot_cell[0], robot_cell[1], 'ro', markersize=16, zorder=5)
    #     if len(self.env.global_frontiers) > 0:
    #         frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)), self.env.belief_info)
    #         plt.scatter(frontiers[:, 0], frontiers[:, 1], c='red', s=2, label='frontier')
    #     plt.title("Node View")
    #     plt.legend(loc='upper right')

    #     # === subplot (2) belief map + trajectory + path ===
    #     plt.subplot(1, 4, 2)
    #     plt.imshow(self.env.robot_belief, cmap='gray')
    #     plt.axis('off')
    #     for robot in self.robot_list:
    #         c = color_list[robot.id]
    #         robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
    #         plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
    #         plt.plot((np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
    #                 (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size,
    #                 c, linewidth=2, zorder=1)
    #         plt.plot((np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size,
    #                 (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size,
    #                 'g', linewidth=1, zorder=2)
    #         if hasattr(robot.node_manager, 'last_obstacle_coords'):
    #             o_coords = robot.node_manager.last_obstacle_coords
    #             if isinstance(o_coords, (set, list)):
    #                 o_coords = np.array(list(o_coords))
    #             if len(o_coords.shape) == 1:
    #                 o_coords = np.expand_dims(o_coords, 0)
    #             if len(o_coords) > 0:
    #                 o_cells = get_cell_position_from_coords(o_coords, robot.map_info)
    #                 plt.scatter(o_cells[:, 0], o_cells[:, 1], c='black', marker='x', s=15, zorder=3)
    #     plt.title("Belief Map View")


    #     # === subplot (4) LaMa 输入图像和掩码 ===
    #     # 获取 LaMa 输入张量
    #     image_tensor, mask_tensor = self.global_predictor.get_lama_input()
    #     image = image_tensor.squeeze().numpy()  # shape (H, W), in [0,1]
    #     mask = mask_tensor.squeeze().numpy()    # shape (H, W), 0/1

    #     # 初始化为灰色（未知区域）= 0.5
    #     lama_input_visual = np.full(image.shape, 0.5, dtype=np.float32)

    #     # 设置已知区域的显示颜色：
    #     lama_input_visual[(mask == 0) & (image >= 0.99)] = 1.0   # 自由区域（白）
    #     lama_input_visual[(mask == 0) & (image <= 0.01)] = 0.0   # 障碍区域（黑）

    #     # 显示图像
    #     plt.subplot(1, 4, 3)
    #     plt.imshow(lama_input_visual, cmap='gray', vmin=0, vmax=1)
    #     plt.title("LaMa Input ")
    #     plt.axis("off")
    #     # === subplot (4) LaMa 预测结果（RGB） ===
    #     plt.subplot(1, 4, 4)
    #     if self.lama_rgb_image is not None:
    #         plt.imshow(self.lama_rgb_image)  # 显示 RGB 预测图
    #         plt.title("LaMa Output (RGB)")
    #         plt.axis("off")
    #     else:
    #         plt.text(0.5, 0.5, "No prediction", ha='center', va='center')
    #         plt.axis("off")

    #     # === 保存图像 ===
    #     plt.suptitle(f'Explored rate: {self.env.explored_rate:.4g}  Travel distance: {max(r.travel_dist for r in self.robot_list):.4g}')
    #     plt.tight_layout()
    #     save_path = f'{gifs_path}/{self.global_step}_{step}_samples.png'
    #     plt.savefig(save_path, dpi=150)
    #     self.env.frame_files.append(save_path)
    #     plt.close()
