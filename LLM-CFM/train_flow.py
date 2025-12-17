"""
普通 PyTorch 循环版 Flow Matching 训练脚本
保存逻辑：每 20 epoch 保存一次图像，按病例名建子目录，8 位序号命名
"""
import os
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --------------- 你自己的模块 ---------------
from flow_matching.path import CondOTProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from models.kernel_fusion import KernelFusion
from models.unet import UNetModel
from models.ldm.models.autoencoder import AutoencoderKL
from dataloader.flow_dataset import PETDataset
from configs.config_PET import C
from pytorch_msssim import ssim
from torchmetrics.functional import peak_signal_noise_ratio, mean_absolute_error
import clip
from torchdiffeq import odeint

# --------------- 基础配置 ---------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 1777777
torch.manual_seed(seed)
np.random.seed(seed)

save_dir = "/home/siat/ycy/Flow_matching/Ruijin_saved_models/flow_plain"
os.makedirs(save_dir, exist_ok=True)
csv_log = os.path.join(save_dir, 'train_log.csv')
with open(csv_log, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'train_loss'])
csv_log1 = os.path.join(save_dir, 'val_log.csv')
with open(csv_log1, 'w', newline='') as f:
    csv.writer(f).writerow(['epoch', 'val_mse', 'val_psnr', 'val_ssim', 'val_mae'])

# --------------- 数据 ---------------
train_ds = PETDataset(
    data_list_path=C.train_source,
    base_path=C.base_path,
    transform=transforms.ToTensor(),
    caption_jsonl="/home/siat/ycy/Flow_matching/datasets/Ruijin-data/llava_med_answers.jsonl"
)
val_ds = PETDataset(
    data_list_path=C.eval_source,
    base_path=C.base_path,
    transform=transforms.ToTensor(),
    caption_jsonl="/home/siat/ycy/Flow_matching/datasets/Ruijin-data/llava_med_answers.jsonl"
)

train_loader = DataLoader(train_ds, batch_size=C.batch_size,
                          shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=1,
                          shuffle=False, num_workers=0, pin_memory=True)

# --------------- 模型 ---------------
# 1. VAE（冻结）
vae = AutoencoderKL(
    ddconfig={'double_z': True, 'z_channels': 3, 'resolution': 128,
              'in_channels': 1, 'out_ch': 1, 'ch': 128,
              'ch_mult': [1, 2, 4], 'num_res_blocks': 2,
              'attn_resolutions': [], 'dropout': 0.0},
    lossconfig={"target": "models.modules.losses.LPIPSWithDiscriminator",
                "params": {"disc_start": 50001, "kl_weight": 1e-6, "disc_weight": 0.5}},
    embed_dim=3, image_key="image", monitor="val/rec_loss"
)
vae.load_state_dict(torch.load("/home/siat/ycy/Flow_matching/Ruijin_saved_models/vae/model_epoch_300.pth",
                               map_location="cpu"), strict=False)
vae.eval().requires_grad_(False)
vae.to(device)

# 2. 可训练子网络
fusion = KernelFusion(z_ch=3, text_dim=512).to(device)
unet   = UNetModel(in_channels=15, out_channels=3,
                   model_channels=64, num_res_blocks=4,
                   attention_resolutions=[16, 8]).to(device)

# 3. CLIP（冻结）
clip_model, _ = clip.load("ViT-B/32", device="cpu")
clip_model.eval().requires_grad_(False)

# 4. Flow Matching 路径
path = CondOTProbPath()

# --------------- 优化器 ---------------
optimizer = torch.optim.AdamW(list(fusion.parameters()) + list(unet.parameters()),
                              lr=C.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# --------------- 工具函数 ---------------
@torch.no_grad()
def img2z(img):
    z = vae.encode(img).sample().mul_(0.18215)
    return torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)

@torch.no_grad()
def z2img(z):
    z = torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)
    return vae.decode(z / 0.18215)

@torch.no_grad()
def text2vec(text):
    tokens = clip.tokenize(text, truncate=True)
    return clip_model.encode_text(tokens).float()

def solve_ode(z_nac, text):
    text_vec = text2vec(text).to(z_nac.device)
    z_fused  = fusion(z_nac, text_vec)
    extra    = {"concat_conditioning": torch.cat([z_nac, z_fused], dim=1)}

    def ode_func(t, z_4d):
        t_batch = torch.full((z_4d.size(0),), t, device=z_4d.device, dtype=z_4d.dtype)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            v = unet(z_4d, t_batch, extra=extra)
        return v

    t_seq = torch.tensor([0.0, 1.0], device=z_nac.device)
    z1 = odeint(ode_func, z_nac.float(), t_seq,
                method='dopri5', rtol=1e-4, atol=1e-6)[-1]
    return torch.nan_to_num(z1, nan=0.0, posinf=5.0, neginf=-5.0)

# --------------- 保存图像（每 20 epoch） ---------------
def save_case_images_flow(epoch):
    npy_dir = os.path.join(save_dir, 'npy_output', f'val_epoch_{epoch}')
    os.makedirs(npy_dir, exist_ok=True)
    case_counters = {}

    fusion.eval(); unet.eval()
    with torch.no_grad():
        for nac, gt, paths, text in tqdm(val_loader, desc=f'Save images epoch {epoch}'):
            case_name = os.path.basename(os.path.dirname(os.path.dirname(paths[0])))
            case_dir  = os.path.join(npy_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)

            case_counters.setdefault(case_name, 1)
            idx = case_counters[case_name]

            nac, gt = nac.to(device), gt.to(device)
            z_nac   = img2z(nac)
            z1      = solve_ode(z_nac, text)
            recon   = z2img(z1)

            for i in range(nac.size(0)):          # batch=1 只循环 1 次
                base = f"{idx + i:08d}"
                np.save(os.path.join(case_dir, f"{base}_input.npy"),
                        nac[i].squeeze().cpu().numpy())
                np.save(os.path.join(case_dir, f"{base}_gt.npy"),
                        gt[i].squeeze().cpu().numpy())
                np.save(os.path.join(case_dir, f"{base}_output.npy"),
                        recon[i].squeeze().cpu().numpy())
            case_counters[case_name] += nac.size(0)

# --------------- 训练一个 epoch ---------------
def train_one_epoch(epoch):
    fusion.train(); unet.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} train')
    for nac, gt, _, text in pbar:
        nac, gt = nac.to(device), gt.to(device)
        z_nac   = img2z(nac)
        z_clean = img2z(gt)

        t = torch.rand(z_nac.size(0), device=device)
        path_sample = path.sample(t=t, x_0=z_nac, x_1=z_clean)
        z_t, v_target = path_sample.x_t, path_sample.dx_t

        text_vec = text2vec(text).to(device)
        z_fused  = fusion(z_nac, text_vec)
        extra    = {"concat_conditioning": torch.cat([z_nac, z_fused], dim=1)}

        with torch.amp.autocast('cuda', enabled=False):
            z_t = z_t.float()
            v_pred = unet(z_t, t, extra=extra)
            loss = nn.functional.mse_loss(v_pred, v_target.float())

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(fusion.parameters()) + list(unet.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

# --------------- 验证 ---------------
def validate(epoch):
    fusion.eval(); unet.eval()
    metrics = {'mse': 0, 'psnr': 0, 'ssim': 0, 'mae': 0}
    pbar = tqdm(val_loader, desc='val')
    for nac, gt, _, text in pbar:
        nac, gt = nac.to(device), gt.to(device)
        z_nac   = img2z(nac)
        z1      = solve_ode(z_nac, text)
        recon   = z2img(z1)

        mse  = nn.functional.mse_loss(recon, gt)
        psnr = peak_signal_noise_ratio(recon, gt, data_range=1.0)
        ssim_ = ssim(recon, gt, data_range=1.0, size_average=True)
        mae  = mean_absolute_error(recon, gt)

        metrics['mse']  += mse.item()
        metrics['psnr'] += psnr.item()
        metrics['ssim'] += ssim_.item()
        metrics['mae']  += mae.item()

    return {k: v / len(val_loader) for k, v in metrics.items()}

# --------------- 主循环 ---------------
def main():
    start_epoch = 1
    for epoch in range(start_epoch, C.nepochs + 1):
        # ---------- 训练 ----------
        train_loss = train_one_epoch(epoch)
        scheduler.step()

        print(f'Epoch {epoch} | train {train_loss:.4f}')

        with open(csv_log, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss])

        # 只每 50 epoch 做一次验证 + 保存模型 + 保存图像
        if epoch % 5 == 0:

            ckpt = {'fusion': fusion.state_dict(),
                    'unet': unet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
            torch.save(ckpt, os.path.join(save_dir, f'fm_epoch_{epoch:03d}.pth'))

            print(f'----- Validating & Saving at epoch {epoch} -----')
            val_dict = validate(epoch)  # 计算指标
            save_case_images_flow(epoch)  # 保存图像
            # 保存模型

            # 写 CSV（只在 50 的倍数写一次）
            with open(csv_log1, 'a', newline='') as f:
                csv.writer(f).writerow([epoch,
                                        val_dict['mse'], val_dict['psnr'],
                                        val_dict['ssim'], val_dict['mae']])

if __name__ == '__main__':
    main()