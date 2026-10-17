"""
Script de diagnóstico — verifica qualidade dos TFRecords e bboxes
Corre antes de treinar para confirmar que os dados estão corretos.
"""
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image

# =============================================================
# 1. Verificar TFRecords
# =============================================================
def check_tfrecords(tfrecord_dir='./tfrecords_train'):
    print('\n' + '='*60)
    print('1. VERIFICAR TFRECORDS')
    print('='*60)

    files = glob.glob(os.path.join(tfrecord_dir, '*.tfrecord'))
    if not files:
        print(f'[ERRO] Nenhum TFRecord encontrado em: {tfrecord_dir}')
        return

    print(f'TFRecords encontrados: {len(files)}')
    for f in sorted(files):
        size = os.path.getsize(f) / 1024
        print(f'  {os.path.basename(f)}  —  {size:.1f} KB')

    # Contar total de exemplos
    total = 0
    label_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    bbox_sizes = []
    n_bboxes_per_image = []

    for f in files:
        for record in tf.python_io.tf_record_iterator(f):
            example = tf.train.Example()
            example.ParseFromString(record)
            feat = example.features.feature

            labels = list(feat['image/object/bbox/label'].int64_list.value)
            xmin   = list(feat['image/object/bbox/xmin'].float_list.value)
            xmax   = list(feat['image/object/bbox/xmax'].float_list.value)
            ymin   = list(feat['image/object/bbox/ymin'].float_list.value)
            ymax   = list(feat['image/object/bbox/ymax'].float_list.value)

            n_bboxes_per_image.append(len(labels))
            for l in labels:
                if l in label_counts:
                    label_counts[l] += 1

            for i in range(len(xmin)):
                w = xmax[i] - xmin[i]
                h = ymax[i] - ymin[i]
                bbox_sizes.append((w, h))

            total += 1

    print(f'\nTotal de imagens nos TFRecords: {total}')
    print(f'\nDistribuição de labels:')
    names = {1: 'Static', 2: 'Road', 3: 'Dynamic', 4: 'Clutter'}
    for lid, count in label_counts.items():
        print(f'  label {lid} ({names[lid]}): {count} bboxes')

    if n_bboxes_per_image:
        print(f'\nBboxes por imagem:')
        print(f'  min:    {min(n_bboxes_per_image)}')
        print(f'  max:    {max(n_bboxes_per_image)}')
        print(f'  média:  {np.mean(n_bboxes_per_image):.1f}')
        print(f'  mediana:{np.median(n_bboxes_per_image):.1f}')

    if bbox_sizes:
        ws = [b[0] for b in bbox_sizes]
        hs = [b[1] for b in bbox_sizes]
        print(f'\nTamanho das bboxes (normalizado 0-1):')
        print(f'  largura  — min:{min(ws):.3f}  max:{max(ws):.3f}  média:{np.mean(ws):.3f}')
        print(f'  altura   — min:{min(hs):.3f}  max:{max(hs):.3f}  média:{np.mean(hs):.3f}')

        # Bboxes muito grandes (> 80% da imagem) são problemáticas para SSD
        big = sum(1 for w, h in bbox_sizes if w > 0.8 or h > 0.8)
        print(f'\n  Bboxes muito grandes (>80% imagem): {big} ({big/len(bbox_sizes)*100:.1f}%)')
        if big / len(bbox_sizes) > 0.3:
            print('  [AVISO] Muitas bboxes grandes — o SSD tem dificuldade com objetos que ocupam')
            print('          grande parte da imagem. Considera aumentar MIN_AREA_PX para filtrar.')

        # Bboxes muito pequenas (< 2% da imagem)
        tiny = sum(1 for w, h in bbox_sizes if w < 0.02 and h < 0.02)
        print(f'  Bboxes muito pequenas (<2% imagem):  {tiny} ({tiny/len(bbox_sizes)*100:.1f}%)')
        if tiny / len(bbox_sizes) > 0.3:
            print('  [AVISO] Muitas bboxes pequenas — aumenta MIN_AREA_PX no uavid_to_tf.py')


# =============================================================
# 2. Verificar máscaras visualmente (amostra)
# =============================================================
def check_masks(mask_dir, n_samples=5):
    print('\n' + '='*60)
    print('2. VERIFICAR MÁSCARAS')
    print('='*60)

    # Procurar máscaras
    masks = []
    for root, dirs, files in os.walk(mask_dir):
        if 'Labels' in root and 'overlay' not in root.lower():
            for f in files:
                if f.endswith('.png'):
                    masks.append(os.path.join(root, f))

    if not masks:
        print(f'[AVISO] Nenhuma máscara encontrada em: {mask_dir}')
        return

    print(f'Máscaras encontradas: {len(masks)}')

    COLORS = {
        'Static':  np.array([128, 0,   0  ]),
        'Road':    np.array([0,   128, 0  ]),
        'Dynamic': np.array([0,   0,   128]),
        'Clutter': np.array([0,   0,   0  ]),
    }

    for mask_path in masks[:n_samples]:
        mask = np.array(Image.open(mask_path).convert('RGB'))
        h, w = mask.shape[:2]
        total_px = h * w
        print(f'\n  {os.path.basename(mask_path)} ({w}x{h}):')
        for name, color in COLORS.items():
            px = np.all(mask == color, axis=-1).sum()
            pct = px / total_px * 100
            print(f'    {name}: {px:8d} px ({pct:.1f}%)')


# =============================================================
# 3. Sugestões de correção
# =============================================================
def print_suggestions():
    print('\n' + '='*60)
    print('3. SUGESTÕES')
    print('='*60)
    print("""
Se a loss está muito alta (>50 consistentemente):

A) Bboxes muito grandes (Road/Static ocupam >80% imagem):
   → No uavid_to_tf.py, aumenta MIN_AREA_PX para Road e Static
     e adiciona um MAX_AREA_PX para ignorar regiões gigantes:
     
     MAX_AREA_PX = {
         'Static':  0.5,   # máx 50% da imagem
         'Road':    0.4,   # máx 40% da imagem
         'Dynamic': 0.3,
         'Clutter': 0.4,
     }

B) Demasiadas bboxes por imagem (>20):
   → Aumenta MIN_AREA_PX para reduzir o número de regiões pequenas

C) Learning rate ainda alta:
   → Muda para 0.00005 no config_train.py

D) Dataset muito pequeno para SSD convergir:
   → Considera data augmentation (já está no preprocessing do SSD)
   → Ou reduz o modelo para ssd_mobilenet_v1 com menos layers
""")


from PIL import Image
import numpy as np

img = np.array(Image.open('./datasets/uavid/uavid_train/seq1/Labels/000000.png').convert('RGB'))
print('Shape:', img.shape)

# Verificar cada cor exata
print('Static  [128,0,0]:', np.all(img == [128,0,0], axis=-1).sum())
print('Road    [0,128,0]:', np.all(img == [0,128,0], axis=-1).sum())
print('Dynamic [0,0,128]:', np.all(img == [0,0,128], axis=-1).sum())
print('Clutter [0,0,0]:  ', np.all(img == [0,0,0],   axis=-1).sum())

# Ver as cores exatas dos pixeis azuis
blue_mask = (img[:,:,2] > 100) & (img[:,:,0] < 50) & (img[:,:,1] < 50)
if blue_mask.any():
    blue_px = img[blue_mask]
    u, c = np.unique(blue_px, axis=0, return_counts=True)
    print('Cores azuis encontradas:')
    for i in np.argsort(-c)[:5]:
        print(f'  RGB{list(u[i])}: {c[i]} px')


# =============================================================
# Main
# =============================================================
if __name__ == '__main__':
    # Muda estes caminhos para os teus
    TFRECORD_DIR = './tfrecords_train'
    MASK_DIR     = './datasets/uavid'

    check_tfrecords(TFRECORD_DIR)
    check_masks(MASK_DIR)
    print_suggestions()
