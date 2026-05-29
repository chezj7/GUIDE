#!/usr/bin/env python3

import os
import sys
import torch
import zmq
import yaml
import pickle
import logging
import traceback
import numpy as np
import hydra
from omegaconf import OmegaConf
import time
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.utils import register_debug_signal_handlers

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

LOGGER = logging.getLogger(__name__)

@hydra.main(config_path='../configs/prediction', config_name='default.yaml')
def main(predict_config: OmegaConf):
    try:
        if sys.platform != 'win32':
            register_debug_signal_handlers()

        # === 1. 设置设备 ===
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Using device: {device}")

        # === 2. 加载训练配置 ===
        train_config_path = os.path.join(predict_config.model.path, 'config.yaml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        # 开启推理模式 & 禁用可视化器
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        # === 3. 加载模型 checkpoint ===
        checkpoint_path = os.path.join(
            predict_config.model.path,
            'models',
            predict_config.model.checkpoint
        )
        model = load_checkpoint(
            train_config,
            checkpoint_path,
            strict=False,
            map_location=device
        )

        if model is None:
            raise RuntimeError(f" Failed to load model from: {checkpoint_path}")

        model.freeze()
        model.to(device)
        model.eval()
        print("✅ LaMa model loaded successfully")

        # === 4. 初始化 ZMQ socket ===
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")
        print(" LaMa Server running at tcp://*:5555 ...")

        # === 5. 推理主循环 ===
        while True:
             
             #  计时开始
            msg = socket.recv()
            data = pickle.loads(msg)
            start_time = time.time()
            # 图像和掩膜格式为 Numpy: image=[1,3,H,W], mask=[1,1,H,W]
            image = torch.from_numpy(data['image']).float().to(device)
            mask = torch.from_numpy(data['mask']).float().to(device)
            
            with torch.no_grad():
                batch = {
                    'image': image,
                    'mask': (mask > 0).float()
                }
                batch = move_to_device(batch, device)
                output = model(batch)
                result = output[predict_config.out_key][0]  # [3, H, W]

            result_np = result.clamp(0, 1).mul(255).byte().cpu().numpy()  # 转为 uint8
            result_rgb = np.transpose(result_np, (1, 2, 0))  # [H, W, 3]
            socket.send_pyobj(result_rgb)  # 使用 pickle 回传 numpy 图像
            end_time = time.time()  # 计时结束
            print(f"Returned image: shape={result_rgb.shape} | Time used: {(end_time - start_time) * 1000:.2f} ms")
    except KeyboardInterrupt:
        LOGGER.warning(' Server interrupted by keyboard')
    except Exception as ex:
        LOGGER.critical(f' Exception in server: {ex}\n{traceback.format_exc()}')
        sys.exit(1)


if __name__ == '__main__':
    main()
