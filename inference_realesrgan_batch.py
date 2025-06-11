import os
import cv2
from tqdm import tqdm
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

def process_batch(batch_files, model_name, input_dir, output_dir, outscale):
    # 初始化模型配置
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRNet_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-animevideov3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = [
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
            'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
        ]
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 下载模型
    model_path = os.path.join('weights', f'{model_name}.pth')
    if not os.path.exists(model_path):
        for url in file_url:
            model_path = load_file_from_url(
                url=url,
                model_dir=os.path.join(os.path.dirname(__file__), 'weights'),
                progress=True,
                file_name=None
            )

    # 初始化超分辨率器
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None
    )

    # 处理批次中的每个文件
    for path in tqdm(batch_files, desc=f'Processing {model_name}'):
        if os.path.isdir(path):
            continue

        # 准备保存路径
        rel_path = os.path.relpath(path, input_dir)
        save_dir = os.path.join(output_dir, os.path.dirname(rel_path))
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.splitext(os.path.basename(path))[0]

        # 确定文件扩展名
        ext = os.path.splitext(path)[1]
        save_path = os.path.join(save_dir, f"{filename}{ext}")

        if os.path.exists(save_path):
            continue

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"无法读取图像: {path}")
            continue

        # 处理图像
        try:
            output_img, _ = upsampler.enhance(img, outscale=outscale)
        except Exception as e:
            print(f"处理失败: {path} - {str(e)}")
            continue

        cv2.imwrite(save_path, output_img)