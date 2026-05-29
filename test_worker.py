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
        self.env = Env(global_step, TEST_N_AGENTS, plot=save_image, test=USE_TEST_DATASET)
        self.node_manager = NodeManager(map_info=self.env.belief_info, plot=save_image)
        self.robot_list = [Agent(i, self.node_manager, self.device, save_image) for i in range(TEST_N_AGENTS)]
        self.perf_metrics = dict()
        self.global_predictor = GlobalPredictor(self.env.belief_info)
        self.obs_horizon = policy.n_obs_steps
        self.action_horizon = policy.n_action_steps if ACTION_HORIZON is None else ACTION_HORIZON
        
        self.planned_path_x = []
        self.planned_path_y = []
        self.expert_path_x = []
        self.expert_path_y = []

        # # === 分帧评估：每 5 帧取一次快照（帧从 1 开始数） ===
        # self.snapshot_steps = set(range(5, MAX_EPISODE_STEP + 1, 5))
        # self.unknown_snapshots = {}

    # ====================== 工具：宽松评估需要的基础函数 ======================
    # def _dilate_bool_mask(self, mask, r):
    #     """
    #     只用 numpy 做方形膨胀（Chebyshev 距离 r）。mask 是 bool 的 HxW。
    #     """
    #     if r <= 0:
    #         return mask
    #     H, W = mask.shape
    #     out = np.zeros_like(mask, dtype=bool)
    #     for dy in range(-r, r + 1):
    #         y0 = max(0, dy)
    #         y1 = H + min(0, dy)
    #         for dx in range(-r, r + 1):
    #             x0 = max(0, dx)
    #             x1 = W + min(0, dx)
    #             out[y0:y1, x0:x1] |= mask[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
    #     return out

    # def _cells_to_mask(self, cells, H, W):
    #     """
    #     将 (N,2) 的 cell 坐标 -> HxW 的 bool mask（越界丢弃）
    #     cells[:,0] = x（列），cells[:,1] = y（行）
    #     """
    #     m = np.zeros((H, W), dtype=bool)
    #     if cells.size == 0:
    #         return m
    #     x = cells[:, 0]
    #     y = cells[:, 1]
    #     valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    #     x = x[valid]; y = y[valid]
    #     m[y, x] = True
    #     return m

    # def _centroid_from_mask(self, mask):
    #     """ 返回 (cx, cy)，若空则返回 (np.nan, np.nan) """
    #     idx = np.argwhere(mask)
    #     if idx.shape[0] == 0:
    #         return np.nan, np.nan
    #     cy = float(idx[:, 0].mean())
    #     cx = float(idx[:, 1].mean())
    #     return cx, cy

    # def _mean_nn_distance(self, A_cells, B_cells, sample_k=2000):
    #     """
    #     近邻均距：A->B 的平均最近邻距离（像素/网格单位）。大图时随机下采样避免 O(N^2)。
    #     """
    #     if A_cells.size == 0 or B_cells.size == 0:
    #         return np.nan
    #     A = A_cells
    #     B = B_cells
    #     if A.shape[0] > sample_k:
    #         A = A[np.random.choice(A.shape[0], sample_k, replace=False)]
    #     if B.shape[0] > sample_k:
    #         B = B[np.random.choice(B.shape[0], sample_k, replace=False)]
    #     dmin = []
    #     for a in A:
    #         d = np.sqrt(np.sum((B - a) ** 2, axis=1))
    #         dmin.append(d.min())
    #     return float(np.mean(dmin)) if len(dmin) > 0 else np.nan

    # # ====================== 未知节点的采样与评估 ======================
    # def _collect_unknown_node_coords(self):
    #     nodes = []
    #     nd = self.robot_list[0].node_manager.nodes_dict
    #     for item in nd.__iter__():
    #         ndat = item.data
    #         if getattr(ndat, 'is_unknown_node', 0) == 1:
    #             nodes.append(np.asarray(ndat.coords, dtype=float))
    #     if not nodes:
    #         return np.empty((0, 2), dtype=float)
    #     pts = np.unique(np.round(np.vstack(nodes), 2), axis=0)
    #     return pts

    # def _maybe_take_snapshot(self, step):
    #     s = step + 1  # 帧从1开始数
    #     if s in self.snapshot_steps and s not in self.unknown_snapshots:
    #         self.unknown_snapshots[s] = self._collect_unknown_node_coords()

    # def _evaluate_unknown_snapshots_strict(self, tol_m=0.4):
    #     """
    #     严格-放宽版（strict-relaxed）：允许离 GT 自由格在 tol_m 米内算命中。
    #     实现：将 GT 自由区 mask 以 r_cells = ceil(tol_m/cell_size) 做方形膨胀，
    #          统计预测 cell 是否落入“膨胀后的 GT 自由”。
    #     """
    #     gt_map = self.env.ground_truth_info.map
    #     map_info = self.env.ground_truth_info
    #     H, W = gt_map.shape

    #     # 容差 -> cell 半径
    #     r_cells = int(np.ceil(tol_m / float(map_info.cell_size)))
    #     gt_free_mask = (gt_map == FREE)
    #     gt_free_dil = self._dilate_bool_mask(gt_free_mask, r_cells)

    #     results = {}      # step -> ratio
    #     per_step_cells = {}

    #     for s, coords in self.unknown_snapshots.items():
    #         if coords.shape[0] == 0:
    #             results[s] = float('nan')
    #             per_step_cells[s] = np.empty((0, 2), dtype=int)
    #             continue

    #         cells = get_cell_position_from_coords(coords, map_info, check_negative=False)
    #         cells = np.asarray(cells, dtype=int)
    #         valid = (cells[:, 0] >= 0) & (cells[:, 0] < W) & (cells[:, 1] >= 0) & (cells[:, 1] < H)
    #         cells = cells[valid]
    #         per_step_cells[s] = cells

    #         if cells.shape[0] == 0:
    #             results[s] = float('nan')
    #             continue

    #         # 命中= 落在“GT自由膨胀”里
    #         hits = int(np.sum(gt_free_dil[cells[:, 1], cells[:, 0]]))
    #         ratio = hits / max(1, cells.shape[0])
    #         results[s] = float(ratio)

    #     # 写入 perf_metrics（分帧 + 元数据）
    #     self.perf_metrics['strict'] = {str(k): {'ratio': v} for k, v in results.items()}
    #     self.perf_metrics['strict_tol_m'] = float(tol_m)
    #     self.perf_metrics['strict_r_cells'] = int(r_cells)
    #     self.perf_metrics['unknown_cells'] = {str(k): per_step_cells[k] for k in per_step_cells}

    #     # 地图均值（去 NaN）
    #     vals = [v for v in results.values() if v == v]
    #     self.perf_metrics['strict_mean'] = float(np.mean(vals)) if len(vals) > 0 else float('nan')

    # def _evaluate_unknown_snapshots_tolerant(self, r=2):
    #     """
    #     宽松评估（形状/位置大致对）：
    #     - iou_dil_gt: 预测 vs 膨胀后的 GT 自由区 的 IoU（只对 GT 宽容）
    #     - iou_sym: 预测与 GT 自由区同时膨胀后的 IoU（双向宽容）
    #     - centroid_dist: 两者质心距离（像素/网格）
    #     - nn_mean: 预测点到 GT 自由点的平均最近邻距离（越小越好）
    #     """
    #     gt_map = self.env.ground_truth_info.map
    #     H, W = gt_map.shape
    #     gt_free_mask = (gt_map == FREE)

    #     iou_dil_gt_per_frame = {}
    #     iou_sym_per_frame = {}
    #     cen_dist_per_frame = {}
    #     nn_mean_per_frame = {}

    #     for s, coords in self.unknown_snapshots.items():
    #         # 预测未知 -> cells -> mask
    #         if coords.shape[0] == 0:
    #             pred_mask = np.zeros((H, W), dtype=bool)
    #         else:
    #             cells = np.asarray(get_cell_position_from_coords(
    #                 coords, self.env.ground_truth_info, check_negative=False
    #             ), dtype=int)
    #             valid = (cells[:, 0] >= 0) & (cells[:, 0] < W) & (cells[:, 1] >= 0) & (cells[:, 1] < H)
    #             cells = cells[valid]
    #             pred_mask = self._cells_to_mask(cells, H, W)

    #         # 1) 预测 vs 膨胀(GT)
    #         gt_dil = self._dilate_bool_mask(gt_free_mask, r)
    #         inter1 = np.logical_and(pred_mask, gt_dil).sum()
    #         union1 = np.logical_or(pred_mask, gt_dil).sum()
    #         iou_dil_gt = inter1 / union1 if union1 > 0 else np.nan
    #         iou_dil_gt_per_frame[s] = float(iou_dil_gt)

    #         # 2) 双向膨胀的 IoU
    #         pred_dil = self._dilate_bool_mask(pred_mask, r)
    #         inter2 = np.logical_and(pred_dil, gt_dil).sum()
    #         union2 = np.logical_or(pred_dil, gt_dil).sum()
    #         iou_sym = inter2 / union2 if union2 > 0 else np.nan
    #         iou_sym_per_frame[s] = float(iou_sym)

    #         # 3) 质心距离
    #         pcx, pcy = self._centroid_from_mask(pred_mask)
    #         gcx, gcy = self._centroid_from_mask(gt_free_mask)
    #         if np.isnan(pcx) or np.isnan(pcy) or np.isnan(gcx) or np.isnan(gcy):
    #             cen_dist = np.nan
    #         else:
    #             cen_dist = float(np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2))
    #         cen_dist_per_frame[s] = cen_dist

    #         # 4) 预测→GT 的平均最近邻距离
    #         pred_cells = np.argwhere(pred_mask)[:, [1, 0]]  # (x,y)
    #         gt_cells = np.argwhere(gt_free_mask)[:, [1, 0]]
    #         nn_mean = self._mean_nn_distance(pred_cells, gt_cells, sample_k=2000)
    #         nn_mean_per_frame[s] = float(nn_mean)

    #     # 写入 perf_metrics（分帧）
    #     self.perf_metrics['tolerant'] = {
    #         'iou_dil_gt': {str(k): {'v': v} for k, v in iou_dil_gt_per_frame.items()},
    #         'iou_sym':    {str(k): {'v': v} for k, v in iou_sym_per_frame.items()},
    #         'centroid_dist': {str(k): {'v': v} for k, v in cen_dist_per_frame.items()},
    #         'nn_mean': {str(k): {'v': v} for k, v in nn_mean_per_frame.items()},
    #         'r': int(r),
    #     }

    #     # 地图均值（去 NaN）
    #     def _nanmean(d):
    #         vals = [vv['v'] for vv in d.values() if vv['v'] == vv['v']]
    #         return float(np.mean(vals)) if len(vals) > 0 else float('nan')

    #     self.perf_metrics['tolerant_mean'] = {
    #         'iou_dil_gt': _nanmean(self.perf_metrics['tolerant']['iou_dil_gt']),
    #         'iou_sym':    _nanmean(self.perf_metrics['tolerant']['iou_sym']),
    #         'centroid_dist': _nanmean(self.perf_metrics['tolerant']['centroid_dist']),
    #         'nn_mean': _nanmean(self.perf_metrics['tolerant']['nn_mean']),
    #     }

    # ====================== 主流程 ======================
    def run_episode(self):
        unique_seed = int(time.time())
        set_random_seed(unique_seed)
        self.lama_rgb_image = None
        self.lama_binary_image = None
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))
        for robot in self.robot_list:
            robot.update_planning_state(self.env.robot_locations)

        # Get the first observation
        if DATA_TYPE == 'node':
            self.env.ground_truth_planner = GroundTruthPlanner(self.env.ground_truth_info, robot.node_manager)
            paths = self.env.get_ground_truth_paths()
            for path in paths:
                self.expert_path_x.append([p[0] for p in path])
                self.expert_path_y.append([p[1] for p in path])
            
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
            image = image.astype(np.float32) / 255
            image = np.expand_dims(image, axis=0)
            obs = {'image': image, 'agent_pos': state}
        else:
            raise ValueError('Invalid data type, check test_parameter.py')

        obs_deque = collections.deque([obs] * self.obs_horizon, maxlen=self.obs_horizon)

        step = 0
        for step in range(MAX_EPISODE_STEP):
            # self._maybe_take_snapshot(step)

            if DATA_TYPE == 'node':
                node_inputs = torch.stack([x['node_inputs'] for x in obs_deque])
                node_padding_mask = torch.stack([x['node_padding_mask'] for x in obs_deque])
                edge_mask = torch.stack([x['edge_mask'] for x in obs_deque])
                current_index = torch.stack([x['current_index'] for x in obs_deque])
                current_edge = torch.stack([x['current_edge'] for x in obs_deque])
                edge_padding_mask = torch.stack([x['edge_padding_mask'] for x in obs_deque])

                node_inputs = node_inputs.to(self.device, dtype=torch.float32)
                node_padding_mask = node_padding_mask.to(self.device, dtype=torch.int16)
                edge_mask = edge_mask.to(self.device, dtype=torch.int64)
                current_index = current_index.to(self.device, dtype=torch.int64)
                current_edge = current_edge.to(self.device, dtype=torch.int64)
                edge_padding_mask = edge_padding_mask.to(self.device, dtype=torch.int16)

                obs_dict = {'node_inputs': node_inputs.unsqueeze(0),
                            'node_padding_mask': node_padding_mask.unsqueeze(0),
                            'edge_mask': edge_mask.unsqueeze(0),
                            'current_index': current_index.unsqueeze(0),
                            'current_edge': current_edge.unsqueeze(0),
                            'edge_padding_mask': edge_padding_mask.unsqueeze(0)}
            elif DATA_TYPE == 'map':
                image = torch.stack([torch.tensor(x['image']) for x in obs_deque])
                agent_pos = torch.stack([torch.tensor(x['agent_pos']) for x in obs_deque])

                image = image.to(self.device, dtype=torch.float32)
                agent_pos = agent_pos.to(self.device, dtype=torch.float32)

                obs_dict = {'image': image.unsqueeze(0),
                            'agent_pos': agent_pos.unsqueeze(0)}
            else:
                raise ValueError('Invalid data type, check test_parameter.py')

            with torch.no_grad():
                action_dict = self.policy.predict_action(obs_dict)

            action_pred = action_dict['action_pred'].squeeze(0).cpu().numpy()
            action_pred = np.round(action_pred / NODE_RESOLUTION) * NODE_RESOLUTION

            start = self.obs_horizon - 1
            end = start + self.action_horizon
            action = action_pred[start:end, :]

            for action_step in range(self.action_horizon):
                if action_step == 0:
                    planned_location = deepcopy(self.env.robot_locations[0])
                    self.planned_path_x.append([planned_location[0]])
                    self.planned_path_y.append([planned_location[1]])
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

                if USE_DELTA_POSITION:
                    selected_coord = self.env.robot_locations[0] + action[action_step]
                else:
                    selected_coord = action[action_step]

                current_node = self.robot_list[0].node_manager.nodes_dict.find(self.env.robot_locations[0].tolist()).data

                # Collision avoidance
                if not any(np.all(selected_coord == neighbor) for neighbor in current_node.neighbor_list):
                    direction_vectors = np.cumsum(action_pred[start: start + 3], axis=0)
                    best_neighbor = None
                    best_average_angle = float('inf')
                    for neighbor_coords in current_node.neighbor_list:
                        if np.all(neighbor_coords == self.env.robot_locations[0]):
                            continue
                        neighbor_direction = neighbor_coords - self.env.robot_locations[0]
                        angles = []
                        for direction_vector in direction_vectors:
                            direction_magnitude = np.linalg.norm(direction_vector)
                            neighbor_magnitude = np.linalg.norm(neighbor_direction)
                            if direction_magnitude == 0 or neighbor_magnitude == 0:
                                continue
                            angle = np.arctan2(np.linalg.det([direction_vector, neighbor_direction]),
                                               np.dot(direction_vector, neighbor_direction))
                            angles.append(angle)
                        weights = np.arange(len(angles), 0, -1)
                        weighted_average_angle = np.average(np.abs(angles), weights=weights)
                        if weighted_average_angle < best_average_angle:
                            best_average_angle = weighted_average_angle
                            best_neighbor = neighbor_coords
                    selected_coord = best_neighbor

                self.env.step(selected_coord, 0)
                
                self.robot_list[0].update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[0]))
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
                    image = image.astype(np.float32) / 255
                    if len(image.shape) == 2:
                        image = np.expand_dims(image, axis=0)
                    obs = {'image': image, 'agent_pos': state}
                else:
                    raise ValueError('Invalid data type, check test_parameter.py')

                obs_deque.append(obs)

                if USE_EXPLORATION_RATE_FOR_DONE:
                    self.env.check_done()
                    done = self.env.done
                else:
                    done = self.robot_list[0].utility.sum() == 0

                if self.save_image:
                    self.plot_thesis(step)

                
            if done: # exit episode loop if done
                break
                
        self.perf_metrics['travel_dist'] = self.robot_list[0].travel_dist
        self.perf_metrics['success_rate'] = done
        
        if self.save_image: # save gif
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate, delete_images=True,fps=4)

        # self.perf_metrics['travel_dist'] = self.robot_list[0].travel_dist
        # self.perf_metrics['success_rate'] = done

        # # === 仅在本地图结束时统一评估，并做“去 NaN 平均” ===
        # self._evaluate_unknown_snapshots_strict(tol_m=1)  # 0.4m 容差
        # self._evaluate_unknown_snapshots_tolerant(r=2)      # r 可调 2~4

        # # 地图级别打印一次
        # def _fmt(x): return f"{x:.4f}" if x == x else "NaN"
        # print(f"[Map {self.global_step}] "
        #       f"strict-relaxed(mean, tol={self.perf_metrics['strict_tol_m']:.1f}m)={_fmt(self.perf_metrics['strict_mean'])}; "
        #       f"tolerant(iou_dil)={_fmt(self.perf_metrics['tolerant_mean']['iou_dil_gt'])}, "
        #       f"tolerant(iou_sym)={_fmt(self.perf_metrics['tolerant_mean']['iou_sym'])}, "
        #       f"centroid_dist={_fmt(self.perf_metrics['tolerant_mean']['centroid_dist'])}, "
        #       f"nn_mean={_fmt(self.perf_metrics['tolerant_mean']['nn_mean'])}")

        # # 返回 perf_metrics，供外部汇总
        # return self.perf_metrics

    # ====================== 可视化（保持你的实现） ======================
    def plot_belief(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(6, 6))
        plt.imshow(self.env.robot_belief, cmap='gray', origin='lower')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('{}/{}_{}_samples.png'.format(observe_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(observe_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()
    def plot_thesis(self, step):
        plt.switch_backend('agg')
        fig, ax = plt.subplots(figsize=(6, 6))

        # 底图：保持与 env 第一张图一致
        robot0 = self.robot_list[0]
        map_info = robot0.map_info
        ax.imshow(map_info.map, cmap='gray', origin='lower')

        # --- 固定坐标范围，避免 GIF 抖动 ---
        h, w = map_info.map.shape
        PAD = getattr(self, "viz_pad_cells", 5)  # 外扩的 cell 数，可按需调整
        if not hasattr(self, "viz_xlim"):
            self.viz_xlim = (-0.5 - PAD, w - 0.5 + PAD)
            self.viz_ylim = (-0.5 - PAD, h - 0.5 + PAD)
        ax.set_xlim(*self.viz_xlim)
        ax.set_ylim(*self.viz_ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_autoscale_on(False)
        ax.margins(0)
        ax.axis('off')

        # 规划路径（红线），和 env 第一张图一致
        if step < len(self.planned_path_x) and step < len(self.planned_path_y):
            px = (np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size
            py = (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size
            ax.plot(px, py, 'r', linewidth=1.5, alpha=0.9, zorder=4, label="planned path")

        # 机器人位置
        color_list = ['r', 'b', 'g', 'y']
        for robot in self.robot_list:
            c = color_list[robot.id]
            rcell = get_cell_position_from_coords(robot.location, robot.map_info)
            ax.plot(rcell[0], rcell[1], c + 'o', markersize=10, zorder=5, label="robot" if robot.id == 0 else None)

        # known/unknown 节点（保留原有绘制逻辑）
        handles = {}
        if hasattr(robot0, 'node_coords') and robot0.node_coords is not None:
            node_cells = get_cell_position_from_coords(robot0.node_coords, map_info)
            if hasattr(robot0, 'is_unknown_node') and robot0.is_unknown_node is not None:
                is_unknown = robot0.is_unknown_node.flatten().astype(bool)
                known_nodes   = node_cells[~is_unknown]
                unknown_nodes = node_cells[ is_unknown]
            else:
                known_nodes   = node_cells
                unknown_nodes = np.empty((0, 2), dtype=int)

            if known_nodes.shape[0] > 0:
                hk = ax.scatter(known_nodes[:, 0], known_nodes[:, 1], c='k', s=10, zorder=3, label='known nodes')
                handles.setdefault('known nodes', hk)
            if unknown_nodes.shape[0] > 0:
                hu = ax.scatter(unknown_nodes[:, 0], unknown_nodes[:, 1], c='lime', s=50, marker='o', zorder=3, label='unknown nodes')
                handles.setdefault('unknown nodes', hu)

        # 角标（标题里显示指标；保持原来的字体大小）
        plt.title(
            f"Explored rate: {self.env.explored_rate:.4g} | "
            f"Travel dist: {max(r.travel_dist for r in self.robot_list):.4g}",
            fontsize=10
        )

        # 固定版式，避免 tight_layout 引起的微调抖动
        fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)

        # 保存帧
        out_path = f'{gifs_path}/{self.global_step}_{step}_samples.png'
        plt.savefig(out_path, dpi=150)
        self.env.frame_files.append(out_path)
        plt.close()


    # def plot_thesis(self, step):
    #     plt.switch_backend('agg')
    #     fig, ax = plt.subplots(figsize=(6, 6))

    #     gt_map = self.env.ground_truth_info.map.copy()
    #     belief_map = self.env.robot_belief.copy()

    #     h, w = gt_map.shape
    #     base_img = np.zeros((h, w, 3))

    #     color_obstacle = [0.4, 0.4, 0.4]
    #     color_known_free = [0.8, 0.8, 0.8]
    #     color_unknown = [0.9, 0.9, 0.9]
    #     color_belief_known = [1.0, 1.0, 1.0]

    #     for y in range(h):
    #         for x in range(w):
    #             if gt_map[y, x] == OCCUPIED:
    #                 base_img[y, x] = color_obstacle
    #             elif gt_map[y, x] == FREE:
    #                 base_img[y, x] = color_known_free
    #             else:
    #                 base_img[y, x] = color_unknown

    #     known_mask = (belief_map == FREE)
    #     base_img[known_mask] = color_belief_known

    #     ax.imshow(base_img, origin='lower')

    #     color_list = ['r', 'b', 'g', 'y']
    #     for robot in self.robot_list:
    #         c = color_list[robot.id]
    #         if step < len(self.planned_path_x) and step < len(self.planned_path_y):
    #             path_x = (np.array(self.planned_path_x[step]) - self.env.belief_info.map_origin_x) / self.env.cell_size
    #             path_y = (np.array(self.planned_path_y[step]) - self.env.belief_info.map_origin_y) / self.env.cell_size
    #             ax.plot(path_x, path_y, 'r', linewidth=1.5, zorder=2, alpha=0.9)

    #         if hasattr(robot, 'node_coords') and robot.node_coords is not None:
    #             node_cells = get_cell_position_from_coords(robot.node_coords, robot.map_info)
    #             ax.scatter(node_cells[:, 0], node_cells[:, 1], c='k', s=10, label=f"robot{robot.id} nodes")

    #             if hasattr(robot, 'is_unknown_node') and robot.is_unknown_node is not None:
    #                 is_unknown = robot.is_unknown_node.flatten()
    #                 unknown_nodes = node_cells[is_unknown.astype(bool)]
    #                 ax.scatter(unknown_nodes[:, 0], unknown_nodes[:, 1], c='lime', s=50, marker='o', label=f"robot{robot.id} unknown")

    #         robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
    #         ax.plot(robot_cell[0], robot_cell[1], c + 'o', markersize=10, zorder=5)

    #     plt.axis('off')
    #     plt.tight_layout()

    #     plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
    #     frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
    #     self.env.frame_files.append(frame)
    #     plt.close()


# =============== 多地图总体汇总（可选） ===============
def print_overall_summary(per_map_metrics_list):
    """
    per_map_metrics_list: 由多个 TestWorker.run_episode() 的返回 perf_metrics 组成的 list
    功能：对每张地图的“严格均值/宽松均值”做最终总体平均（去 NaN），只打印一次。
    """
    strict_means = []
    iou_dil_means = []
    iou_sym_means = []
    cen_means = []
    nn_means = []

    for pm in per_map_metrics_list:
        if 'strict_mean' in pm:
            strict_means.append(pm['strict_mean'])
        tm = pm.get('tolerant_mean', {})
        iou_dil_means.append(tm.get('iou_dil_gt', float('nan')))
        iou_sym_means.append(tm.get('iou_sym', float('nan')))
        cen_means.append(tm.get('centroid_dist', float('nan')))
        nn_means.append(tm.get('nn_mean', float('nan')))

    def _overall(xs):
        v = [x for x in xs if x == x]
        return float(np.mean(v)) if len(v) > 0 else float('nan')

    def _fmt(x): return f"{x:.4f}" if x == x else "NaN"

    print("\n[ALL MAPS] === Overall (NaN removed) ===")
    print(f"strict-relaxed(mean, tol=0.4m): {_fmt(_overall(strict_means))} | "
          f"IoU(gt-dil): {_fmt(_overall(iou_dil_means))} | "
          f"IoU(sym): {_fmt(_overall(iou_sym_means))} | "
          f"centroid_dist: {_fmt(_overall(cen_means))} | "
          f"nn_mean: {_fmt(_overall(nn_means))}")
