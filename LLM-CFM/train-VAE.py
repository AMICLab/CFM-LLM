import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader.PETdataset import PETDataset
from models.ldm.models.autoencoder import AutoencoderKL
from models.ldm.modules.losses import *
import numpy as np
import csv
from pytorch_msssim import ssim
from pydicom import dcmread
from torch.optim.lr_scheduler import StepLR
from configs.config_PET import C
from tqdm import tqdm
from models.diffusion import *
import time
import random
# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.version.cuda)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(0))
# 设置随机种子
seed = 1777777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# # 定义保存案例图像的函数
def save_case_images(val_data_loader, model, save_dir, epoch):
    # 创建保存npy文件的根目录
    npy_save_dir = os.path.join(save_dir, 'npy_output', f'val_epoch_{epoch}')
    os.makedirs(npy_save_dir, exist_ok=True)

    model.eval()
    with torch.no_grad():
        # 初始化一个字典来存储每个病例的文件编号
        case_file_indices = {}

        for x_UNCORRECTED, NORMAL, UNCORRECTED_path in val_data_loader:
            # print(f"val nac SUV max: {torch.max(x_UNCORRECTED)}, min: {torch.min(x_UNCORRECTED)}")
            # print(f"val NORMAL SUV max: {torch.max(NORMAL)}, min: {torch.min(NORMAL)}")
            # 直接迭代val_data_loader
            # 从UNCORRECTED_path的路径中提取病例名称
            case_name = os.path.basename(os.path.dirname(os.path.dirname(UNCORRECTED_path[0])))
            case_npy_save_dir = os.path.join(npy_save_dir, case_name)
            os.makedirs(case_npy_save_dir, exist_ok=True)

            # 获取当前病例的文件编号，如果没有则初始化为1
            if case_name not in case_file_indices:
                case_file_indices[case_name] = 1
            case_file_index = case_file_indices[case_name]

            x_UNCORRECTED, NORMAL = x_UNCORRECTED.to(device), NORMAL.to(device)
            reconstructions, posterior = model(NORMAL)


            # 确保输出和目标的形状一致
            if reconstructions.shape != NORMAL.shape:
                print(f"Warning: Output shape {reconstructions.shape} does not match NORMAL shape {NORMAL.shape}")
                continue

            x_UNCORRECTED = x_UNCORRECTED.squeeze().cpu().detach().numpy()
            reconstructions = reconstructions.squeeze().cpu().detach().numpy()
            NORMAL = NORMAL.squeeze().cpu().detach().numpy()
            # print(f"after nac SUV max: {np.max(x_UNCORRECTED)}, min: {np.min(x_UNCORRECTED)}")
            # print(f"after NORMAL SUV max: {np.max(NORMAL)}, min: {np.min(NORMAL)}")
            # print(f"outputs SUV max: {np.max(outputs)}, min: {np.min(outputs)}")

            # 保存npy文件到对应的病例文件夹
            npy_x_UNCORRECTED_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_input.npy')
            np.save(npy_x_UNCORRECTED_save_path, x_UNCORRECTED)
            npy_reconstructions_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_output.npy')
            np.save(npy_reconstructions_save_path, reconstructions)
            npy_NORMAL_save_path = os.path.join(case_npy_save_dir, f'{case_file_index:08}_NORMAL.npy')
            np.save(npy_NORMAL_save_path, NORMAL)

            # 递增当前病例的文件编号
            case_file_indices[case_name] += 1

def calculate_psnr(mse):
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1 / torch.sqrt(mse))

def calculate_ssim(NORMAL, reconstructions):
    return ssim(NORMAL, reconstructions, data_range=1, size_average=True)

def calculate_mae(NORMAL, reconstructions):
    return torch.mean(torch.abs(NORMAL - reconstructions)).item()

# def get_last_layer(self):
#     return self.model.decoder.conv_out.weight

def get_last_layer(model):
    return model.decoder.conv_out.weight

def main():
    # 确保保存目录存在
    save_dir = '/home/siat/ycy/Diffusion/Ruijin_saved_models/vae'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    train_csv_file_path = os.path.join(save_dir, 'training.csv')
    val_csv_file_path = os.path.join(save_dir, 'validation.csv')

    with open(train_csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'MSE_loss', 'SSIM', 'MAE', 'PSNR'])

    with open(val_csv_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Epoch', 'MSE_loss', 'SSIM', 'MAE', 'PSNR'])

    train_dataset = PETDataset(
        data_list_path=C.train_source,
        base_path=C.base_path,
        transform=transforms.ToTensor()
    )
    train_data_loader = DataLoader(train_dataset, batch_size=C.batch_size, shuffle=True, num_workers=C.num_workers)

    val_dataset = PETDataset(
        data_list_path=C.eval_source,
        base_path=C.base_path,
        transform=transforms.ToTensor()
    )
    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=C.num_workers)

    ddconfig = {
        'double_z': True,
        'z_channels': 3,
        'resolution': 192,
        'in_channels': 1,
        'out_ch': 1,
        'ch': 128,
        'ch_mult': [1, 2, 4],  # num_down = len(ch_mult)-1
        'num_res_blocks': 2,
        'attn_resolutions': [],
        'dropout': 0.0
    }

    lossconfig = {
        "target": "lib.medzoo.ldm.modules.losses.LPIPSWithDiscriminator",
        "params": {
            "disc_start": 50001,
            "kl_weight": 0.000001,
            "disc_weight": 0.5
        }
    }
    model = AutoencoderKL(ddconfig,lossconfig,embed_dim=3,ckpt_path=None,ignore_keys=[],image_key="image",colorize_nlabels=None,monitor="val/rec_loss")
    criterion = LPIPSWithDiscriminator(disc_start=50001, kl_weight=0.000001, disc_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)

    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

        model.load_state_dict(torch.load(C.ckpt_path))

    model.to(device)
    print('Model and data loaders are ready.')

    # 加载第200个epoch的模型权重
    model.load_state_dict(
        torch.load(os.path.join(save_dir, 'model_epoch_25.pth'), map_location=device, weights_only=True))

    start_epoch = 26  # 从第201个epoch开始训练

    # 初始化global_step

    for epoch in range(start_epoch, C.nepochs + 1):
        model.train()
        train_loss = 0.0
        train_ssim_total = 0.0
        train_mae = 0.0
        train_batch_num = 0
        global_step = 0

        for x_UNCORRECTED, NORMAL, UNCORRECTED_path in tqdm(train_data_loader, desc='Training', leave=True):
            x_UNCORRECTED, NORMAL = x_UNCORRECTED.to(device), NORMAL.to(device)
            optimizer.zero_grad()
            reconstructions, posterior = model(NORMAL)
            # 定义get_last_layer函数

            # 更新global_step
            global_step += 1
            # 使用global_step和get_last_layer
            loss, log_dict_ae = criterion(NORMAL, reconstructions, posterior, optimizer_idx=0,
                                            global_step=global_step,
                                            last_layer=get_last_layer(model))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            with torch.no_grad():
                ssim_trainue = calculate_ssim(NORMAL, reconstructions)
                train_ssim_total += ssim_trainue

                train_mae += calculate_mae(NORMAL, reconstructions)

                train_batch_num += 1

        scheduler.step()

        average_loss = train_loss / train_batch_num
        average_ssim = train_ssim_total / train_batch_num
        average_mae = train_mae / train_batch_num
        # average_psnr = calculate_psnr(torch.tensor(average_loss))

        print(
            f'train MSE_loss: {average_loss:.4f}, SSIM: {average_ssim:.4f}, MAE: {average_mae:.4f}')
        # print(
        #     f'train MSE_loss: {average_loss:.4f}')

        with open(train_csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([epoch, average_loss, average_ssim, average_mae])
            # csv_writer.writerow([epoch, average_loss])

        # 验证集指标计算
        model.eval()  # 将模型设置为评估模式
        val_loss = 0.0
        val_ssim_total = 0.0
        val_mae = 0.0
        val_batch_num = 0
        global_step = 0
        with torch.no_grad():
            for x_UNCORRECTED, NORMAL, UNCORRECTED_path in tqdm(val_data_loader, desc='valing', leave=True):
                x_UNCORRECTED, NORMAL = x_UNCORRECTED.to(device), NORMAL.to(device)
                reconstructions, posterior = model(NORMAL)
                # def get_last_layer(model):
                #     return model.decoder.conv_out.weight
                # 更新global_step
                global_step += 1
                # 使用global_step和get_last_layer
                loss, log_dict_ae = criterion(NORMAL, reconstructions, posterior, optimizer_idx=0,
                                              global_step=global_step,
                                              last_layer=get_last_layer(model))
                val_loss += loss.item()

                ssim_value = calculate_ssim(NORMAL, reconstructions)
                val_ssim_total += ssim_value
                val_mae += calculate_mae(NORMAL, reconstructions)

                val_batch_num += 1

        average_val_loss = val_loss / val_batch_num
        average_val_ssim = val_ssim_total / val_batch_num
        average_val_mae = val_mae / val_batch_num
        # average_val_psnr = calculate_psnr(torch.tensor(average_val_loss))

        # print(
        #     f'val MSE_loss: {average_val_loss:.4f}, SSIM: {average_val_ssim:.4f}, MAE: {average_val_mae:.4f}, PSNR: {average_val_psnr:.4f}')
        print(
            f'val MSE_loss: {average_val_loss:.4f}, SSIM: {average_val_ssim:.4f}, MAE: {average_val_mae:.4f}')

        with open(val_csv_file_path, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            # csv_writer.writerow([epoch, average_val_loss, average_val_ssim, average_val_mae, average_val_psnr])
            csv_writer.writerow([epoch, average_val_loss, average_val_ssim, average_val_mae])

        if epoch % 25 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch}.pth'))
            save_case_images(val_data_loader, model, save_dir, epoch)

if __name__ == '__main__':
    main()