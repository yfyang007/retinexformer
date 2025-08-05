# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer


#### usage ####
# python3 Enhancement/generate_dataset.py --input_dir /data1/yyf/data4lowlight/cvpr/mini_val/gt --result_dir /data1/yyf/result/test --opt /data1/yyf/Retinexformer/Options/Re_reverse.yml --weights /data1/yyf/Retinexformer/experiments/Re_reverse/best_psnr_29.65_8000.pth --dataset lol1 --self_ensemble
###############



from ast import arg
import numpy as np
import os
import argparse
from tqdm import tqdm
import cv2

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils

from natsort import natsorted
from glob import glob
from skimage import img_as_ubyte
from pdb import set_trace as stx
from skimage import metrics

from basicsr.models import create_model
from basicsr.utils.options import dict2str, parse

# class RetinexInferenceDataset(Dataset):
#     def __init__(self, input_dir, extensions=['*.png', '*.jpg', '*.webp', '*.JPEG']):
#         self.paths = []
#         for ext in extensions:
#             self.paths.extend(glob(os.path.join(input_dir, ext)))
#         self.paths = natsorted(self.paths)
    
#     def __len__(self):
#         return len(self.paths)
    
#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         # 加载图像（假设返回 uint8 图像）
#         img = utils.load_img(path)
#         # 转换为 float32 并归一化到 [0,1]
#         img = np.float32(img) / 255.
#         # 转换为 torch Tensor，形状 (C, H, W)
#         img = torch.from_numpy(img).permute(2, 0, 1)
#         # 返回字典，包含图像和对应路径
#         return {"lq": img, "lq_path": path}

        
def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

parser = argparse.ArgumentParser(
    description='Image Enhancement using Retinexformer')

parser.add_argument('--input_dir', default='./Enhancement/Datasets',
                    type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/',
                    type=str, help='Directory for results')
parser.add_argument(
    '--opt', type=str, default='Options/RetinexFormer_SDSD_indoor.yml', help='Path to option YAML file.')
parser.add_argument('--weights', default='pretrained_weights/SDSD_indoor.pth',
                    type=str, help='Path to weights')
parser.add_argument('--dataset', default='lolv1', type=str,
                    help='Test Dataset') 
parser.add_argument('--gpus', type=str, default="0", help='GPU devices.')
parser.add_argument('--GT_mean', action='store_true', help='Use the mean of GT to rectify the output of the model')
parser.add_argument('--self_ensemble', action='store_true', help='Use self-ensemble to obtain better results')

args = parser.parse_args()

# 指定 gpu
gpu_list = ','.join(str(x) for x in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

####### Load yaml #######
yaml_file = args.opt
weights = args.weights
print(f"dataset {args.dataset}")

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

opt = parse(args.opt, is_train=False)
opt['dist'] = False


x = yaml.load(open(args.opt, mode='r'), Loader=Loader)
s = x['network_g'].pop('type')
##########################


model_restoration = create_model(opt).net_g

# 加载模型
checkpoint = torch.load(weights)

# stx()

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']  # 适用于常见的 checkpoint 格式
else:
    state_dict = checkpoint['params']  # 适用于某些自定义 checkpoint

# 处理键名不匹配的问题
try:
    model_restoration.load_state_dict(state_dict)
except RuntimeError:  # 若键名不匹配，尝试添加 'module.'
    new_checkpoint = {'module.' + k: v for k, v in state_dict.items()}
    model_restoration.load_state_dict(new_checkpoint)


# try:
#     model_restoration.load_state_dict(checkpoint['params'])
# except:
#     new_checkpoint = {}
#     for k in checkpoint['params']:
#         new_checkpoint['module.' + k] = checkpoint['params'][k]
#     model_restoration.load_state_dict(new_checkpoint)

print("===>Testing using weights: ", weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# stx()

# 生成输出结果的文件
factor = 4
dataset = args.dataset
config = os.path.basename(args.opt).split('.')[0]
checkpoint_name = os.path.basename(args.weights).split('.')[0]
result_dir = args.result_dir
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
result_dir_input = os.path.join(args.result_dir, dataset, 'input')
result_dir_gt = os.path.join(args.result_dir, dataset, 'gt')

psnr = []
ssim = []
if dataset in ['SID', 'SMID', 'SDSD_indoor', 'SDSD_outdoor']:
    os.makedirs(result_dir_input, exist_ok=True)
    os.makedirs(result_dir_gt, exist_ok=True)
    if dataset == 'SID':
        from basicsr.data.SID_image_dataset import Dataset_SIDImage as Dataset
    elif dataset == 'SMID':
        from basicsr.data.SMID_image_dataset import Dataset_SMIDImage as Dataset
    else:
        from basicsr.data.SDSD_image_dataset import Dataset_SDSDImage as Dataset
    opt = opt['datasets']['val']
    opt['phase'] = 'test'
    if opt.get('scale') is None:
        opt['scale'] = 1
    # if '~' in opt['dataroot_gt']:
    #     opt['dataroot_gt'] = os.path.expanduser('~') + opt['dataroot_gt'][1:]
    if '~' in opt['dataroot_lq']:
        opt['dataroot_lq'] = os.path.expanduser('~') + opt['dataroot_lq'][1:]
    dataset = Dataset(opt)
    print(f'test dataset length: {len(dataset)}')
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    with torch.inference_mode():
        for data_batch in tqdm(dataloader):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_batch['lq']
            input_save = data_batch['lq'].cpu().permute(
                0, 2, 3, 1).squeeze(0).numpy()
            # target = data_batch['gt'].cpu().permute(
            #     0, 2, 3, 1).squeeze(0).numpy()
            inp_path = data_batch['lq_path'][0]

            # Padding in case images are not multiples of 4
            h, w = input_.shape[2], input_.shape[3]
            H, W = ((h + factor) // factor) * \
                factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

            if args.self_ensemble:
                restored = self_ensemble(input_, model_restoration)
            else:
                restored = model_restoration(input_)

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            restored = torch.clamp(restored, 0, 1).cpu(
            ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

            # if args.GT_mean:
            #     # This test setting is the same as KinD, LLFlow, and recent diffusion models
            #     # Please refer to Line 73 (https://github.com/zhangyhuaee/KinD/blob/master/evaluate_LOLdataset.py)
            #     mean_restored = cv2.cvtColor(restored.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            #     mean_target = cv2.cvtColor(target.astype(np.float32), cv2.COLOR_BGR2GRAY).mean()
            #     restored = np.clip(restored * (mean_target / mean_restored), 0, 1)

            # psnr.append(utils.PSNR(target, restored))
            # ssim.append(utils.calculate_ssim(
            #     img_as_ubyte(target), img_as_ubyte(restored)))
            type_id = os.path.dirname(inp_path).split('/')[-1]
            os.makedirs(os.path.join(result_dir, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_input, type_id), exist_ok=True)
            os.makedirs(os.path.join(result_dir_gt, type_id), exist_ok=True)
            utils.save_img((os.path.join(result_dir, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(restored))
            utils.save_img((os.path.join(result_dir_input, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(input_save))
            utils.save_img((os.path.join(result_dir_gt, type_id, os.path.splitext(
                os.path.split(inp_path)[-1])[0] + '.png')), img_as_ubyte(target))
else:

    input_dir = args.input_dir
    
    # target_dir = opt['datasets']['val']['dataroot_gt']
    # print(input_dir)
    # print(target_dir)

    input_paths = natsorted(
        glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')) + glob(os.path.join(input_dir , '*.webp')) + glob(os.path.join(input_dir, '*.JPEG')))
    print(len(input_paths))

    # target_paths = natsorted(glob(os.path.join(
    #     target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))
    with torch.inference_mode():
        for inp_path in tqdm(input_paths):
            # try:
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                img = np.float32(utils.load_img(inp_path)) / 255.
                # target = np.float32(utils.load_img(tar_path)) / 255.

                img = torch.from_numpy(img).permute(2, 0, 1)
                input_ = img.unsqueeze(0).cuda()

                # Padding in case images are not multiples of 4
                b, c, h, w = input_.shape
                H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
                padh, padw = H - h, W - w
                input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

                # split and test 
                print('use slice input ##############')
                def get_patch_coords(input_size, patch_size, stride):
                    """生成补丁起始坐标列表，自动处理边缘情况"""
                    coords = list(range(0, input_size - patch_size + 1, stride))
                    
                    # 添加最后一个补丁（如果需要）
                    if not coords or (coords[-1] + patch_size < input_size):
                        final_pos = max(0, input_size - patch_size)
                        if final_pos not in coords:
                            coords.append(final_pos)
                    return coords

                
                                
                def split_image(input_tensor,patch_size = 512 ,stride = 256):
                    H, W = input_tensor.shape[2:]

                    print(f'H is {H} and W is {W}')

                    # 生成坐标列表
                    x_coords = get_patch_coords(W, patch_size, stride)
                    y_coords = get_patch_coords(H, patch_size, stride)

                    print(f'x_coords is {x_coords} and y_coords is {y_coords}')
                    
                    # 执行分块
                    patches = []
                    for y in y_coords:
                        for x in x_coords:
                            patch = input_tensor[:,:,y:y+patch_size, x:x+patch_size]
                            patches.append(patch)
                    
                    return patches, (x_coords, y_coords)

                input_patches,coord_info = split_image(input_)
                print(f'len of input_patches is {len(input_patches)} ###############')
                result_patches = []
                for patch in input_patches:
                    if args.self_ensemble:
                        result_patches.append(self_ensemble(patch, model_restoration))
                    else:
                        result_patches.append(model_restoration(patch).cuda())

                def merge_patches(patches, original_shape, coord_info, patch_size=512):
                    """合并分块图像，使用加权平均处理重叠区域"""
                    H, W = original_shape[2:]
                    C = patches[0].shape[1] if len(original_shape) > 2 else 1
                    
                    # 初始化合并矩阵和计数器
                    merged_image = torch.zeros((1,C,H,W),dtype=torch.float32).cuda()
                    count_matrix = torch.zeros_like(merged_image).cuda()
                    
                    x_coords, y_coords = coord_info
                    patch_idx = 0
                    
                    # 逐块累加
                    for y in y_coords:
                        for x in x_coords:
                            # 计算实际覆盖区域
                            actual_y_end = min(y + patch_size, H)
                            actual_x_end = min(x + patch_size, W)
                            
                            # 获取当前补丁的视图
                            current_patch = patches[patch_idx]
                            
                            # 计算有效区域
                            ph = actual_y_end - y
                            pw = actual_x_end - x
                            valid_patch = current_patch[:ph, :pw]
                            
                            # 累加到合并图像
                            merged_image[:,:,y:actual_y_end, x:actual_x_end] += valid_patch
                            count_matrix[:,:,y:actual_y_end, x:actual_x_end] += 1
                            patch_idx += 1
                    
                    # 处理未覆盖区域（理论上不应该存在）
                    count_matrix = torch.maximum(count_matrix, torch.tensor(1, dtype=count_matrix.dtype, device=count_matrix.device))

                    
                    # 执行平均计算
                    merged_image = (merged_image / count_matrix)
                    print(merged_image.shape,'###############################',type(merged_image))
                    return merged_image

            # Unpad images to original dimensions
                restored = merge_patches(result_patches,input_.shape,coord_info)
                print(f'restoredshape is {restored.shape} ##################')

                restored = torch.clamp(restored, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                # Save results
                
                save_path = os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0] + '.png')
                utils.save_img(save_path, img_as_ubyte(restored))
                del restored
                torch.cuda.empty_cache()

            # except Exception as e:
            #     print(f"Error processing {inp_path}: {e}")
            #     continue  # 继续下一个图

# psnr = np.mean(np.array(psnr))
# ssim = np.mean(np.array(ssim))
# print("PSNR: %f " % (psnr))
# print("SSIM: %f " % (ssim))
print('degration done')
