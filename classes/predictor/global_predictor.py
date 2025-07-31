import numpy as np
import torch
import torch.nn.functional as F
import zmq
import pickle
from parameter import *
from classes.utils import *

class GlobalPredictor:
    def __init__(self, map_info):
        self.map_info = map_info
        self.resolution = NODE_RESOLUTION
        self.height = int(np.ceil(map_info.map.shape[0] * map_info.cell_size / self.resolution))
        self.width = int(np.ceil(map_info.map.shape[1] * map_info.cell_size / self.resolution))

        # 初始化图像：128 = 未知，uint8
        self.image = np.full((self.height, self.width), 128, dtype=np.uint8)

        # 初始化 zmq 连接
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:5555")

        # 记录已知节点坐标集合
        self.free_nodes_set = set()
        self.obs_nodes_set = set()

    def coords_to_index(self, coords):
        """将世界坐标转换为图像索引"""
        cell_x = ((coords[:, 0] - self.map_info.map_origin_x) / self.resolution).astype(int)
        cell_y = ((coords[:, 1] - self.map_info.map_origin_y) / self.resolution).astype(int)
        return cell_y, cell_x

    def update_nodes(self, free_node_coords, obstacle_node_coords):
        """增量更新图像像素值（128=未知，255=自由，0=障碍）"""

        # 1. 移除不再是障碍的节点
        to_remove_obs = self.obs_nodes_set - set(map(tuple, obstacle_node_coords))
        if to_remove_obs:
            ry, rx = self.coords_to_index(np.array(list(to_remove_obs)))
            self.image[ry, rx] = 128
            self.obs_nodes_set -= set(map(tuple, to_remove_obs))

        # 2. 移除不再是自由的节点
        to_remove_free = self.free_nodes_set - set(map(tuple, free_node_coords))
        if to_remove_free:
            ry, rx = self.coords_to_index(np.array(list(to_remove_free)))
            self.image[ry, rx] = 128
            self.free_nodes_set -= set(map(tuple, to_remove_free))

        # 3. 添加新的障碍节点
        new_obs = set(map(tuple, obstacle_node_coords)) - self.obs_nodes_set
        if new_obs:
            ry, rx = self.coords_to_index(np.array(list(new_obs)))
            self.image[ry, rx] = 0
            self.obs_nodes_set |= set(map(tuple, new_obs))

        # 4. 添加新的自由节点
        new_free = set(map(tuple, free_node_coords)) - self.free_nodes_set
        if new_free:
            ry, rx = self.coords_to_index(np.array(list(new_free)))
            self.image[ry, rx] = 255
            self.free_nodes_set |= set(map(tuple, new_free))

    def get_lama_input(self):
        """构造 LaMa 所需输入格式，中心 padding 到 8 的倍数"""
        img = self.image.astype(np.float32) / 255.0
        mask = (self.image == 128).astype(np.uint8)

        image_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask_tensor_o = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        h, w = image_tensor.shape[-2:]
        new_h = (h + 7) // 8 * 8
        new_w = (w + 7) // 8 * 8
        pad_h = new_h - h
        pad_w = new_w - w

        # 中心 padding：left, right, top, bottom
        padding = [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
        image_tensor = F.pad(image_tensor, padding, mode='constant', value=0)
        mask_tensor = F.pad(mask_tensor_o, padding, mode='constant', value=1)

        return image_tensor, mask_tensor,mask_tensor_o,(h, w)

    def request_lama_prediction(self):
        """向 LaMa Server 发送请求并返回裁剪后的补全图像"""
        image_tensor, mask_tensor, mask_tensor_o,(orig_h, orig_w) = self.get_lama_input()

        # 构造 RGB 图像
        image_rgb = image_tensor.repeat(1, 3, 1, 1)
        if image_rgb.shape[-2:] != mask_tensor.shape[-2:]:
            print(f"[警告] image 和 mask 尺寸不一致，将调整：{image_rgb.shape} vs {mask_tensor.shape}")
            mask_tensor = F.interpolate(mask_tensor.float(), size=image_rgb.shape[-2:], mode='nearest')

        image_np = image_rgb.cpu().numpy()     # [1,3,H,W]
        mask_np = mask_tensor.cpu().numpy()    # [1,1,H,W]
        print(f"[send] image_np shape: {image_np.shape}, mask_np shape: {mask_np.shape}")

        send_data = pickle.dumps({'image': image_np, 'mask': mask_np})
        self.socket.send(send_data)
        result_bytes = self.socket.recv()

        inpainted_image = pickle.loads(result_bytes)  # [H_pad, W_pad, 3]
        inpainted_image = self.crop_to_original_size(inpainted_image, orig_h, orig_w)
        return inpainted_image, mask_tensor_o

    @staticmethod
    def crop_to_original_size(image: np.ndarray, orig_h: int, orig_w: int):
        """从中心裁剪图像到原始大小"""
        h, w = image.shape[:2]
        start_y = (h - orig_h) // 2
        start_x = (w - orig_w) // 2
        return image[start_y:start_y + orig_h, start_x:start_x + orig_w, :]