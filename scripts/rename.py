import os
import sys
from datetime import datetime


def rename_files(folder_path, prefix='file'):
    # 获取文件夹中的所有文件（排除子目录）
    try:
        entries = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return
    except PermissionError:
        print(f"错误：没有权限访问文件夹 '{folder_path}'。")
        return

    files = [entry for entry in entries if os.path.isfile(os.path.join(folder_path, entry))]

    if not files:
        print("文件夹中没有可重命名的文件。")
        return

    # 按修改时间排序文件
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)))

    # 计算序号位数（例如：100个文件需要3位）
    num_digits = len(str(len(files)))  # 根据文件数量自动调整位数

    # 遍历并重命名
    for idx, filename in enumerate(files, 1):
        # 分割文件名和扩展名
        name_part, ext = os.path.splitext(filename)

        # 生成新文件名（例如：file_001.jpg）
        new_name = f"{prefix}_{str(idx).zfill(num_digits)}{ext}"
        new_path = os.path.join(folder_path, new_name)

        # 原始文件完整路径
        old_path = os.path.join(folder_path, filename)

        # 避免覆盖现有文件
        if os.path.exists(new_path):
            print(f"警告：'{new_name}' 已存在，跳过重命名 '{filename}'")
            continue

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            print(f"重命名成功：'{filename}' -> '{new_name}'")
        except Exception as e:
            print(f"错误：无法重命名 '{filename}': {str(e)}")


if __name__ == "__main__":
    rename_files(r'F:\xxx\code\sr\Real-ESRGAN\Real-ESRGAN-master\datasets\GF04\original\16001024_200', '16001024')