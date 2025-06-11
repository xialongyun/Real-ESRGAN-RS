import csv
import numpy as np


def analyze_csv(csv_path):
    """CSV文件统计分析函数"""
    psnr_values = []
    ssim_values = []

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # 读取标题行

        for row in reader:
            # 跳过空行和统计行
            if len(row) < 3 or row[0].lower() in ['统计指标', '平均值', '有效样本数']:
                continue

            try:
                psnr = float(row[1])
                ssim = float(row[2])
                psnr_values.append(psnr)
                ssim_values.append(ssim)
            except (ValueError, IndexError):
                print(f"跳过无效行: {row}")
                continue

    if not psnr_values:
        print("错误：没有找到有效数据")
        return None

    # 计算统计量
    stats = {
        "PSNR": {
            "max": np.max(psnr_values),
            "min": np.min(psnr_values),
            "mean": np.mean(psnr_values),
            "median": np.median(psnr_values),
            "std": np.std(psnr_values)
        },
        "SSIM": {
            "max": np.max(ssim_values),
            "min": np.min(ssim_values),
            "mean": np.mean(ssim_values),
            "median": np.median(ssim_values),
            "std": np.std(ssim_values)
        },
        "count": len(psnr_values)
    }

    # 打印报告
    print(f"\n分析报告：{csv_path}")
    print(f"有效样本数: {stats['count']}")
    print("\nPSNR统计:")
    print(f"  最大值: {stats['PSNR']['max']:.2f} dB")
    print(f"  最小值: {stats['PSNR']['min']:.2f} dB")
    print(f"  平均值: {stats['PSNR']['mean']:.2f} dB")
    print(f"  中位数: {stats['PSNR']['median']:.2f} dB")
    print(f"  标准差: {stats['PSNR']['std']:.2f} dB")

    print("\nSSIM统计:")
    print(f"  最大值: {stats['SSIM']['max']:.4f}")
    print(f"  最小值: {stats['SSIM']['min']:.4f}")
    print(f"  平均值: {stats['SSIM']['mean']:.4f}")
    print(f"  中位数: {stats['SSIM']['median']:.4f}")
    print(f"  标准差: {stats['SSIM']['std']:.4f}")

    return stats


if __name__ == "__main__":
    # 使用示例
    csv_file = "image_quality.csv"
    results = analyze_csv(csv_file)

    # 可选：保存统计报告到新文件
    if results:
        report_file = csv_file.replace(".csv", "_report.txt")
        with open(report_file, 'w') as f:
            f.write(f"Image Quality Analysis Report\n")
            f.write(f"Processed File: {csv_file}\n")
            f.write(f"Valid Samples: {results['count']}\n\n")

            f.write("PSNR Statistics (dB):\n")
            f.write(f"Max: {results['PSNR']['max']:.2f}\n")
            f.write(f"Min: {results['PSNR']['min']:.2f}\n")
            f.write(f"Mean: {results['PSNR']['mean']:.2f}\n")
            f.write(f"Median: {results['PSNR']['median']:.2f}\n")
            f.write(f"Std Dev: {results['PSNR']['std']:.2f}\n\n")

            f.write("SSIM Statistics:\n")
            f.write(f"Max: {results['SSIM']['max']:.4f}\n")
            f.write(f"Min: {results['SSIM']['min']:.4f}\n")
            f.write(f"Mean: {results['SSIM']['mean']:.4f}\n")
            f.write(f"Median: {results['SSIM']['median']:.4f}\n")
            f.write(f"Std Dev: {results['SSIM']['std']:.4f}\n")

        print(f"\n报告已保存至: {report_file}")