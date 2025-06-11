import argparse
import os
from functools import partial
from multiprocessing import freeze_support
from concurrent.futures import ProcessPoolExecutor

from inference_realesrgan_batch import process_batch

def split_into_batches(file_list, batch_step):
    if batch_step <= 0:
        raise ValueError("batch_step必须大于0")
    total = len(file_list)
    if total == 0:
        return []
    base_size, remainder = divmod(total, batch_step)
    batches = []
    start = 0
    for i in range(batch_step):
        end = start + base_size + (1 if i < remainder else 0)
        batches.append(file_list[start:end])
        start = end
    return batches

def main():
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='./data/18/', help='输入目录')
    parser.add_argument('-o', '--output', type=str, default='./data/18-batch-x2-true/', help='输出目录')
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus',
                      choices=['RealESRGAN_x4plus', 'RealESRNet_x4plus', 'RealESRGAN_x4plus_anime_6B',
                               'RealESRGAN_x2plus', 'realesr-animevideov3', 'realesr-general-x4v3'],
                      help='使用的模型')
    parser.add_argument('--outscale', type=int, default=2, help='超分辨率缩放比例')
    parser.add_argument('--batch_size', type=int, default=4, help='并行处理的批次数量')
    args = parser.parse_args()

    # 收集所有文件
    file_list = []
    for root, _, files in os.walk(args.input):
        for file in files:
            abs_path = os.path.join(root, file)
            file_list.append(abs_path)
    file_list.sort()

    # file_list_final = []
    # input_file_list = []
    # output_file_list = []
    # for root, _, files in os.walk(args.input):
    #     for file in files:
    #         abs_path = os.path.join(root, file)
    #         rel_path = os.path.relpath(abs_path, args.input)
    #         input_file_list.append((abs_path, rel_path))
    # input_file_list.sort()
    #
    # for root, _, files in os.walk(args.output):
    #     for file in files:
    #         abs_path = os.path.join(root, file)
    #         rel_path = os.path.relpath(abs_path, args.output)
    #         output_file_list.append(rel_path)
    # output_file_list.sort()
    #
    # for item in input_file_list:
    #     if item[1] not in output_file_list:
    #         file_list.append(item[0])
    # file_list.sort()

    # print(len(input_file_list), len(output_file_list), len(file_list))

    batches = split_into_batches(file_list, args.batch_size)

    # 准备worker函数的partial
    worker = partial(
        process_batch,
        model_name=args.model_name,
        input_dir=args.input,
        output_dir=args.output,
        outscale=args.outscale
    )

    # 启动多进程处理
    with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
        executor.map(worker, batches)

if __name__ == '__main__':
    main()