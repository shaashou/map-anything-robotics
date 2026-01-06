import os
import subprocess
import logging
import shutil

def colmap_bin2txt(input_path, output_path, output_type="TXT"):
    os.makedirs(output_path, exist_ok=True)
    convert_cmd = [
        "colmap", "model_converter",
        "--input_path", input_path,
        "--output_path", output_path,
        "--output_type", output_type
    ]
    try:
        subprocess.run(convert_cmd, check=True)
        print("Conversion successful!")
    except Exception as e:
        logging.error(f"Conversion failed: {e}")

def prepare_colmap_folder(base_path):
    # 创建 sparse/0 文件夹
    input_dir = os.path.join(base_path, "sparse/0")
    os.makedirs(input_dir, exist_ok=True)
    # 要移动的文件列表
    files_to_move = ["cameras.bin", "images.bin", "points3D.bin"]
    for fname in files_to_move:
        src = os.path.join(base_path, "sparse", fname)
        dst = os.path.join(input_dir, fname)
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found.")
    return input_dir

if __name__ == "__main__":
    # 修改为你的根目录路径
    base_path = "/home/zyk/projection/Dr-Splat/apple_ygr"
    input_dir = prepare_colmap_folder(base_path)
    output_dir = os.path.join(base_path, "sparse")
    colmap_bin2txt(input_dir, output_dir, "TXT")
