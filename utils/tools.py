import random
import os
import torch
import numpy as np
import cv2


# Cores originais UAVid (RGB)
Building   = np.array([128,  0,   0  ])  # label 0 — Building
Road       = np.array([128,  64,  128])  # label 2 — Road
Tree       = np.array([  0, 128,   0  ]) # label 1 — Tree
LowVeg     = np.array([128, 128,   0  ]) # label 4 — Clutter (vegetação baixa)
Moving_Car = np.array([ 64,   0, 128])   # label 3 — Dynamic
Static_Car = np.array([192,   0, 192])   # label 4 — Clutter (carro parado)
Human      = np.array([ 64,  64,   0  ]) # label 4 — Clutter (humano)
Clutter    = np.array([  0,   0,   0  ]) # label 4 — Clutter
Boundary   = np.array([255, 255, 255])   # label 255 — Boundary (ignorado)

# =============================================================
# 5 Classes:
#   0 = Building  → edifícios (separa das árvores)
#   1 = Tree      → árvores (separador natural entre edifícios)
#   2 = Road      → estrada
#   3 = Dynamic   → carros em movimento
#   4 = Clutter   → tudo o resto (LowVeg, Static_Car, Human, Clutter)
# =============================================================
num_classes = 5


def seed(seed_num):
    random.seed(seed_num)
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def rgb2label(label):
    """Converte máscara RGB original UAVid para labels 0-4."""
    label_seg = np.zeros(label.shape[:2], dtype=np.uint8)
    label_seg[np.all(label == Building,   axis=-1)] = 0  # Building
    label_seg[np.all(label == Tree,       axis=-1)] = 1  # Tree
    label_seg[np.all(label == Road,       axis=-1)] = 2  # Road
    label_seg[np.all(label == LowVeg,     axis=-1)] = 4  # Clutter
    label_seg[np.all(label == Moving_Car, axis=-1)] = 3  # Dynamic
    label_seg[np.all(label == Static_Car, axis=-1)] = 4  # Clutter
    label_seg[np.all(label == Human,      axis=-1)] = 4  # Clutter
    label_seg[np.all(label == Clutter,    axis=-1)] = 4  # Clutter
    label_seg[np.all(label == Boundary,   axis=-1)] = 255  # Boundary
    return label_seg


def uavid2rgb(mask):
    """Converte máscara de labels 0-4 para imagem RGB para visualização."""
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128,   0,   0]  # Building — vermelho
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [  0, 128,   0]  # Tree     — verde
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [128,  64, 128]  # Road     — roxo
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [  0,   0, 128]  # Dynamic  — azul
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [  0,   0,   0]  # Clutter  — preto
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb