import os
from PIL import Image
from tqdm import tqdm

# 原始数据集路径
input_root = "/home/chezj/lama/lama/my_dataset"
# 输出数据集路径
output_root = "/home/chezj/lama/lama/my_dataset_1"
# 目标尺寸
target_size = (32, 32)
# 支持的图像格式
image_exts = ('.png', '.jpg', '.jpeg', '.bmp')

def resize_preserve_binary_rgb(input_path, output_path):
    try:
        # 打开为灰度图（即使原图是RGB）
        img = Image.open(input_path).convert("L")
        # 使用最近邻插值缩放，不引入中间灰度
        img_resized = img.resize(target_size, Image.NEAREST)

        # 转回 RGB，便于保存和兼容性
        img_rgb = img_resized.convert("RGB")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img_rgb.save(output_path)
    except Exception as e:
        print(f"[ERROR] Failed to process {input_path}: {e}")

def process_subfolder(subfolder):
    input_dir = os.path.join(input_root, subfolder)
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc=f"Processing {root}"):
            if file.lower().endswith(image_exts):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_root)
                output_path = os.path.join(output_root, relative_path)
                resize_preserve_binary_rgb(input_path, output_path)

# 批量处理
for subfolder in ['train', 'val', 'visual_test']:
    process_subfolder(subfolder)

print("✅ All RGB binary images resized to 32x32 and saved in my_dataset_1.")
