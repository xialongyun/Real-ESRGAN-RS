import os


def batch_rename_files(dir_path):
    """递归重命名目录及其子目录中的所有文件（移除'_out'）"""
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            old_path = os.path.join(root, filename)

            if "_out" in filename:
                # 修复replace语法错误
                new_filename = filename.replace("_out", "")
                new_path = os.path.join(root, new_filename)

                try:
                    os.rename(old_path, new_path)
                    print(f"✅ {old_path} → {new_path}")
                except Exception as e:
                    print(f"❌ 重命名失败 {old_path}: {str(e)}")


# 使用示例（处理多级目录）
target_dir = r"C:\hainantile\18tiaose"
batch_rename_files(target_dir)
