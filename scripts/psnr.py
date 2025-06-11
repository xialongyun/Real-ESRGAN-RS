import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def safe_ssim(img1, img2, max_window=7):
    """自动调整窗口大小的SSIM计算"""
    # 获取最小边尺寸
    min_side = min(img1.shape[:2])

    # 自动确定窗口大小（必须为奇数）
    win_size = min(max_window, min_side - 1 if min_side % 2 == 0 else min_side)
    win_size = max(3, win_size)  # 最小窗口为3x3

    # 处理多通道图像
    channel_axis = -1 if img1.ndim == 3 else None

    return ssim(img1, img2,
                win_size=win_size,
                channel_axis=channel_axis,
                data_range=img1.max() - img1.min())


def calculate_psnr_ssim(image1_path, image2_path):
    """改进版的计算函数"""
    try:
        # 读取图像
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            raise ValueError("图像读取失败，请检查文件路径")

        # 尺寸验证
        if img1.shape != img2.shape:
            raise ValueError(f"尺寸不匹配: {img1.shape} vs {img2.shape}")

        # 转换颜色空间
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # 计算指标
        psnr_value = psnr(img1, img2)
        ssim_value = safe_ssim(img1, img2)

        return psnr_value, ssim_value

    except Exception as e:
        print(f"处理 {os.path.basename(image1_path)} 时遇到问题: {str(e)}")
        # 返回空值或记录错误
        return None, None


def batch_compare(folder1, folder2):
    """带错误恢复的批量处理"""
    files1 = sorted([f for f in os.listdir(folder1) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    files2 = sorted([f for f in os.listdir(folder2) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # 文件验证
    if len(files1) != len(files2):
        print(f"警告: 文件数量不一致 ({len(files1)} vs {len(files2)})")

    results = []
    for f1, f2 in zip(files1, files2):
        path1 = os.path.join(folder1, f1)
        path2 = os.path.join(folder2, f2)

        psnr_val, ssim_val = calculate_psnr_ssim(path1, path2)

        if psnr_val is not None and ssim_val is not None:
            results.append((f1, psnr_val, ssim_val))
            print(f"处理成功: {f1}")
            print(f"  PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")
        else:
            print(f"处理失败: {f1} vs {f2}")

    return results



if __name__ == "__main__":
    # 输入文件夹路径
    original_folder = r"F:\xxx\dataset\sr\0509\high-original"
    processed_folder = r"F:\xxx\dataset\sr\0509\high-out"

    # 运行批量比较
    results = batch_compare(original_folder, processed_folder)

    # 可选：保存结果到CSV
    import csv_cal

    with open("image_quality.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "PSNR", "SSIM"])
        writer.writerows(results)