import os
import cv2

def resize_images(input_dir, output_dir, target_size=(640, 480)):
    """
    遍历目录并调整所有图片尺寸
    :param input_dir: 输入目录路径
    :param output_dir: 输出目录路径
    :param target_size: 目标尺寸 (宽, 高)
    """
    # 支持的图片格式（不区分大小写）
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 遍历目录树
    for root, dirs, files in os.walk(input_dir):
        # 创建对应的输出目录
        rel_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)

        for file in files:
            # 检查文件扩展名
            ext = os.path.splitext(file)[1].lower()
            if ext not in valid_exts:
                continue

            # 处理文件路径
            input_path = os.path.join(root, file)
            output_path = os.path.join(current_output_dir, file)

            try:
                # 读取图片
                img = cv2.imread(input_path)
                if img is None:
                    print(f"警告：无法读取图片 {input_path}")
                    continue

                # 调整尺寸
                resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

                # 保存图片（保留原质量参数，适用于JPEG）
                cv2.imwrite(output_path, resized)
                print(f"已处理: {input_path} => {output_path}")

            except Exception as e:
                print(f"处理 {input_path} 时出错: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_folder = r"F:\xxx\code\sr\Real-ESRGAN\Real-ESRGAN-master\data\images-MLS-18-2024-2-out\18"
    output_folder = r"F:\xxx\code\sr\Real-ESRGAN\Real-ESRGAN-master\data\images-MLS-18-2024-2-out\18-resize"
    resize_images(input_folder, output_folder, target_size=(256, 256))