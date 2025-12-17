"""
测试脚本：使用第50轮权重进行推理，并保存结果图像
保存逻辑：按病例名建子目录，8位序号命名，保存input、gt、output
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# --------------- 你自己的模块 ---------------
from models.kernel_fusion import KernelFusion
from models.unet import UNetModel
from models.ldm.models.autoencoder import AutoencoderKL
from dataloader.flow_dataset import PETDataset
from configs.config_PET import C
import clip
from torchdiffeq import odeint

# --------------- 基础配置 ---------------
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 模型和输出路径（根据实际路径修改）
save_dir = "/home/siat/ycy/Flow_matching/Ruijin_saved_models/flow_plain"
test_output_dir = os.path.join(save_dir, 'test_SYM')
os.makedirs(test_output_dir, exist_ok=True)

# --------------- 模型初始化 ---------------
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
vae.eval().requires_grad_(False).to(device)

# 2. 子网络
fusion = KernelFusion(z_ch=3, text_dim=512).to(device)
unet = UNetModel(in_channels=15, out_channels=3,
                 model_channels=64, num_res_blocks=4,
                 attention_resolutions=[16, 8]).to(device)

# 3. CLIP（冻结）
clip_model, _ = clip.load("ViT-B/32", device="cpu")
clip_model.eval().requires_grad_(False)

# --------------- 加载第50轮权重 ---------------
checkpoint_path = os.path.join(save_dir, 'fm_epoch_060.pth')
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"权重文件不存在: {checkpoint_path}")

print(f"正在加载权重: {checkpoint_path}")
ckpt = torch.load(checkpoint_path, map_location=device)
fusion.load_state_dict(ckpt['fusion'])
unet.load_state_dict(ckpt['unet'])
fusion.eval()
unet.eval()
print("权重加载完成！模型已设置为评估模式")


# --------------- 工具函数（与训练脚本保持一致） ---------------
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


@torch.no_grad()
def solve_ode(z_nac, text):
    text_vec = text2vec(text).to(z_nac.device)
    z_fused = fusion(z_nac, text_vec)
    extra = {"concat_conditioning": torch.cat([z_nac, z_fused], dim=1)}

    def ode_func(t, z_4d):
        t_batch = torch.full((z_4d.size(0),), t, device=z_4d.device, dtype=z_4d.dtype)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            v = unet(z_4d, t_batch, extra=extra)
        return v

    t_seq = torch.tensor([0.0, 1.0], device=z_nac.device)
    z1 = odeint(ode_func, z_nac.float(), t_seq,
                method='dopri5', rtol=1e-4, atol=1e-6)[-1]
    return torch.nan_to_num(z1, nan=0.0, posinf=5.0, neginf=-5.0)


# --------------- 测试数据加载（请根据实际情况修改路径） ---------------
# 如果测试集与验证集相同，直接使用 C.eval_source
# 如果不同，请指定测试集路径
TEST_DATA_LIST = "/home/siat/ycy/Flow_matching/datasets/Ruijin-data/SYM.txt"  # 修改为实际测试集列表路径
TEST_BASE_PATH = "/home/siat/ycy/RUIJIN-Data-test"  # 修改为实际测试数据根目录

# 如果测试集和验证集一致，可以取消下面这行的注释
# TEST_DATA_LIST, TEST_BASE_PATH = C.eval_source, C.base_path

test_ds = PETDataset(
    data_list_path=TEST_DATA_LIST,
    base_path=TEST_BASE_PATH,
    transform=transforms.ToTensor(),
    caption_jsonl="/home/siat/ycy/Flow_matching/datasets/Ruijin-data/test_llava_med_answers.jsonl"
)

test_loader = DataLoader(test_ds, batch_size=1,
                         shuffle=False, num_workers=0, pin_memory=True)


# --------------- 测试并保存图像 ---------------
def test_and_save():
    case_counters = {}

    print(f"开始测试，共 {len(test_loader)} 个样本...")
    with torch.no_grad():
        for nac, gt, paths, text in tqdm(test_loader, desc='Testing'):
            # 获取病例名（与验证代码保持相同逻辑）
            case_name = os.path.basename(os.path.dirname(os.path.dirname(paths[0])))
            case_dir = os.path.join(test_output_dir, case_name)
            os.makedirs(case_dir, exist_ok=True)

            # 获取当前病例的序号
            case_counters.setdefault(case_name, 1)
            idx = case_counters[case_name]

            # 推理
            nac, gt = nac.to(device), gt.to(device)
            z_nac = img2z(nac)
            z1 = solve_ode(z_nac, text)
            recon = z2img(z1)

            # 保存数据（batch=1，循环一次）
            for i in range(nac.size(0)):
                base = f"{idx + i:08d}"

                # 保存图像数据（与验证代码完全一致）
                np.save(os.path.join(case_dir, f"{base}_input.npy"),
                        nac[i].squeeze().cpu().numpy())
                np.save(os.path.join(case_dir, f"{base}_gt.npy"),
                        gt[i].squeeze().cpu().numpy())
                np.save(os.path.join(case_dir, f"{base}_output.npy"),
                        recon[i].squeeze().cpu().numpy())

                # 可选：保存对应的文本描述
                # text_str = text[i] if isinstance(text, (list, tuple)) else text
                # with open(os.path.join(case_dir, f"{base}_text.txt"), 'w', encoding='utf-8') as f:
                #     f.write(str(text_str))

            case_counters[case_name] += nac.size(0)

    print(f"\n测试完成！结果已保存到: {test_output_dir}")
    print(f"共处理 {sum(case_counters.values()) - len(case_counters)} 个样本")


# --------------- 执行测试 ---------------
if __name__ == '__main__':
    test_and_save()