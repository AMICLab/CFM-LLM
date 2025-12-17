# PETdataset.py
import os
import json
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
from pydicom import dcmread
from torch.nn import functional as F

# ---------- 1. 新增：一次性读取 jsonl ----------
def build_caption_dict(jsonl_path: str) -> dict:
    cap_dict = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 去掉开头的 ./  再拆分
            parts = rec["image"].lstrip("./").split("/")
            if len(parts) < 2:          # 防御
                continue
            key = f"{parts[-2]}/{os.path.splitext(parts[-1])[0]}"  # CBL_PET128962-FPSMA/00000001_nac
            cap_dict[key] = rec["text"].strip()
    return cap_dict

# ---------- 2. 原函数：DICOM 时间转换 ----------
pydicom.config.dimse_protocol = 'GDCM'

def dicom_hhmmss(t: object) -> object:    # dicom存时间格式:小时、分钟、秒(每个占两位),这里转化回秒
    t = str(t)
    if len(t) == 5:     # 有些提取时漏了个0，小时的位置只有一位，这里把零补上
        t = '0'+t
    h_t = float(t[0:2])
    m_t = float(t[2:4])
    s_t = float(t[4:6])
    return h_t*3600+m_t*60+s_t

def get_suv(path_img,path_GT):
    dcm=dcmread(path_GT)
    RadiopharmaceuticalInformationSequence = dcm.RadiopharmaceuticalInformationSequence[0]
    RadiopharmaceuticalStartTime = str(RadiopharmaceuticalInformationSequence['RadiopharmaceuticalStartTime'].value)
    RadionuclideTotalDose = str(RadiopharmaceuticalInformationSequence['RadionuclideTotalDose'].value)
    RadionuclideHalfLife = str(RadiopharmaceuticalInformationSequence['RadionuclideHalfLife'].value)
    ##放在一起
    dcm_tag = str(dcm.SeriesTime)+'\n'+str(dcm.AcquisitionTime)+'\n'+str(dcm.PatientWeight)+'\n'+RadiopharmaceuticalStartTime+'\n'
    dcm_tag = dcm_tag+RadionuclideTotalDose+'\n'+RadionuclideHalfLife+'\n'+str(dcm.RescaleSlope)+'\n'+str(dcm.RescaleIntercept)
    dcm_tag = dcm_tag.split('\n')

    norm = False
    [ST, AT, PW, RST, RTD, RHL, RS, RI] = dcm_tag  # AT基本等于ST,RI一般是0
    decay_time = dicom_hhmmss(ST) - dicom_hhmmss(RST)
    decay_dose = float(RTD) * pow(2, -float(decay_time) / float(RHL))
    SUVbwScaleFactor = (1000 * float(PW)) / decay_dose

    # if norm:
    # PET_SUV = (PET * np.array(RS).astype('float') + np.array(RI).astype('float')) * SUVbwScaleFactor  # 标准公式做法
    # else:
    GT_SUV = dcm.pixel_array * SUVbwScaleFactor  # 非标准做法，但和软件得到的SUV较为一致
    dcm1 = dcmread(path_img)
    img_SUV = dcm1.pixel_array * SUVbwScaleFactor

    return img_SUV,GT_SUV


# ---------- 4. Dataset 类 ----------
class PETDataset(Dataset):
    def __init__(self, data_list_path: str, base_path: str,
                 caption_jsonl: str = None, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.data_list = self._read_data_list(data_list_path)

        self.cap_dict = {}
        if caption_jsonl and os.path.isfile(caption_jsonl):
            self.cap_dict = build_caption_dict(caption_jsonl)

    # 读取  nac_path, _, norm_path  三列
    def _read_data_list(self, path):
        data = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) == 3:
                    nac, _, norm = parts
                    data.append((os.path.join(self.base_path, nac.strip()),
                                 os.path.join(self.base_path, norm.strip())))
                else:
                    print(f"[Warn] skip line: {line}")
        return data

    def __getitem__(self, idx):
        nac_path, norm_path = self.data_list[idx]

        img_suv, gt_suv = get_suv(nac_path, norm_path)
        img_nd = np.array(img_suv, dtype=np.float32)[None, ...]
        gt_nd = np.array(gt_suv, dtype=np.float32)[None, ...]

        # 统一 192×192
        nac = F.interpolate(torch.from_numpy(img_nd).unsqueeze(0),
                                   size=(192, 192), mode='area').squeeze(0)
        gt = F.interpolate(torch.from_numpy(gt_nd).unsqueeze(0),
                                  size=(192, 192), mode='area').squeeze(0)

        # ------- 生成与 jsonl 完全一致的 key -------
        rel_path = os.path.splitext(os.path.relpath(nac_path, self.base_path))[0]  # CBL_PET128962-FPSMA/00000001_nac
        text = self.cap_dict.get(rel_path, "")

        return nac, gt, nac_path, text

    def __len__(self):
        return len(self.data_list)

