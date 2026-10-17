import os
import sys
import random
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage

# =============================================================================
# Classes do segmentador UAVid — 4 classes agrupadas
#
# Cores RGB lidas com PIL.Image:
#   Static  → RGB[128, 0,   0  ]  → vermelho escuro
#   Road    → RGB[0,   128, 0  ]  → verde
#   Dynamic → RGB[0,   0,   128]  → azul
#   Clutter → RGB[0,   0,   0  ]  → preto
# =============================================================================

SEGMENTOR_CLASSES = {
    'Static':  ([128, 0,   0  ], 1),
    'Road':    ([0,   128, 0  ], 2),
    'Dynamic': ([0,   0,   128], 3),
    'Clutter': ([0,   0,   0  ], 4),
}

# Área mínima em píxeis²
MIN_AREA_PX = {
    'Static':  3000,   # menor que antes para apanhar mais regiões
    'Road':    8000,
    'Dynamic': 800,
    'Clutter': 5000,
}

# Área máxima como fração da imagem
# Static aumentado para 0.70 — deixa regiões grandes passar
MAX_AREA_FRACTION = {
    'Static':  0.70,   # era 0.40 — Static pode ocupar muito espaço
    'Road':    0.35,
    'Dynamic': 0.20,
    'Clutter': 0.20,
}

# Máximo de bboxes por classe por imagem
# Static aumentado para 8 — mais exemplos de Static no treino
MAX_BBOXES_PER_CLASS = {
    'Static':  8,    # era 4
    'Road':    2,
    'Dynamic': 4,    # era 6 — reduzido para equilibrar com Static
    'Clutter': 2,    # era 3
}

# Total máximo de bboxes por imagem
MAX_BBOXES_TOTAL = 15

# Conectividade para connected components
CONNECTIVITY_8 = {'Static', 'Road', 'Clutter'}

SAMPLES_PER_FILES = 50


# =============================================================================
# Extração de bounding boxes por connected components
# =============================================================================

def mask_to_bboxes(mask_path):
    """
    Lê uma máscara do segmentador UAVid com PIL (RGB) e devolve
    bboxes individuais por região conectada de cada classe.

    Static tem prioridade — as suas bboxes são adicionadas primeiro
    antes de aplicar o limite total MAX_BBOXES_TOTAL.
    """
    mask = np.array(Image.open(mask_path).convert('RGB'))
    h, w = mask.shape[:2]
    total_px = h * w

    struct_4 = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    struct_8 = np.ones((3, 3), dtype=bool)

    # Píxeis anotados = Static | Road | Dynamic
    annotated = (
        np.all(mask == [128, 0,   0  ], axis=-1) |
        np.all(mask == [0,   128, 0  ], axis=-1) |
        np.all(mask == [0,   0,   128], axis=-1)
    )

    # Recolher bboxes por classe — Static primeiro para ter prioridade
    all_bboxes = []
    all_labels = []

    # Ordem de prioridade: Static primeiro, depois Dynamic, Road, Clutter
    priority_order = ['Static', 'Dynamic', 'Road', 'Clutter']

    for class_name in priority_order:
        color, label_id = SEGMENTOR_CLASSES[class_name]

        if class_name == 'Clutter':
            binary = ~annotated
            binary[:5, :]  = False
            binary[-5:, :] = False
            binary[:, :5]  = False
            binary[:, -5:] = False
        else:
            binary = np.all(mask == np.array(color), axis=-1)

        if not binary.any():
            continue

        structure = struct_8 if class_name in CONNECTIVITY_8 else struct_4
        labeled, num_regions = ndimage.label(binary, structure=structure)

        min_area   = MIN_AREA_PX[class_name]
        max_area   = MAX_AREA_FRACTION[class_name] * total_px
        max_bboxes = MAX_BBOXES_PER_CLASS[class_name]

        class_bboxes = []
        for region_id in range(1, num_regions + 1):
            region = labeled == region_id
            area   = int(region.sum())

            if area < min_area or area > max_area:
                continue

            rows = np.where(region.any(axis=1))[0]
            cols = np.where(region.any(axis=0))[0]
            ymin_r, ymax_r = int(rows[0]),  int(rows[-1])
            xmin_r, xmax_r = int(cols[0]),  int(cols[-1])

            class_bboxes.append((area, (
                float(ymin_r) / h,
                float(xmin_r) / w,
                float(ymax_r) / h,
                float(xmax_r) / w,
            ), label_id))

        # Ordenar por área descendente — maiores primeiro
        class_bboxes.sort(key=lambda x: x[0], reverse=True)
        for _, bbox, lid in class_bboxes[:max_bboxes]:
            all_bboxes.append(bbox)
            all_labels.append(lid)

    # Limitar total — Static já está no início então é preservado
    if len(all_bboxes) > MAX_BBOXES_TOTAL:
        all_bboxes = all_bboxes[:MAX_BBOXES_TOTAL]
        all_labels = all_labels[:MAX_BBOXES_TOTAL]

    return all_bboxes, all_labels


# =============================================================================
# Serialização para TFRecord
# =============================================================================

def convert_to_example(image_data, shape, bboxes, labels):
    xmin = [b[1] for b in bboxes]
    ymin = [b[0] for b in bboxes]
    xmax = [b[3] for b in bboxes]
    ymax = [b[2] for b in bboxes]

    id_to_name = {v[1]: k.encode() for k, v in SEGMENTOR_CLASSES.items()}
    label_texts = [id_to_name[l] for l in labels]

    def int64_feat(v):
        if not isinstance(v, list): v = [v]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=v))
    def float_feat(v):
        if not isinstance(v, list): v = [v]
        return tf.train.Feature(float_list=tf.train.FloatList(value=v))
    def bytes_feat(v):
        if not isinstance(v, list): v = [v]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=v))

    return tf.train.Example(features=tf.train.Features(feature={
        'image/height':                  int64_feat(shape[0]),
        'image/width':                   int64_feat(shape[1]),
        'image/channels':                int64_feat(shape[2]),
        'image/shape':                   int64_feat(shape),
        'image/object/bbox/xmin':        float_feat(xmin),
        'image/object/bbox/xmax':        float_feat(xmax),
        'image/object/bbox/ymin':        float_feat(ymin),
        'image/object/bbox/ymax':        float_feat(ymax),
        'image/object/bbox/label':       int64_feat(labels),
        'image/object/bbox/label_text':  bytes_feat(label_texts),
        'image/object/bbox/difficult':   int64_feat([0] * len(labels)),
        'image/object/bbox/truncated':   int64_feat([0] * len(labels)),
        'image/format':                  bytes_feat(b'JPEG'),
        'image/encoded':                 bytes_feat(image_data),
    }))


# =============================================================================
# Pipeline principal
# =============================================================================

def run(dataset_dir, output_dir, split='train', name='uavid'):
    os.makedirs(output_dir, exist_ok=True)
    split_dir = os.path.join(dataset_dir, f'uavid_{split}')

    all_pairs = []
    for seq in sorted(os.listdir(split_dir)):
        img_dir = os.path.join(split_dir, seq, 'Images')
        lbl_dir = os.path.join(split_dir, seq, 'Labels')
        if not os.path.isdir(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir)):
            img_path  = os.path.join(img_dir, fname)
            mask_path = os.path.join(lbl_dir, fname)
            if os.path.exists(mask_path):
                all_pairs.append((img_path, mask_path))

    total = len(all_pairs)
    print(f'[UAVid→TF] Split: {split} | Imagens: {total}')
    print(f'[UAVid→TF] Classes (RGB via PIL):')
    for class_name, (color, lid) in SEGMENTOR_CLASSES.items():
        print(f'  label {lid} — {class_name}: RGB{color}')
    print(f'[UAVid→TF] Filtros:')
    for class_name in SEGMENTOR_CLASSES:
        print(f'  {class_name}: MIN={MIN_AREA_PX[class_name]}px  '
              f'MAX={MAX_AREA_FRACTION[class_name]*100:.0f}%  '
              f'MAX_BBOXES={MAX_BBOXES_PER_CLASS[class_name]}')
    print(f'  Total máx bboxes/imagem: {MAX_BBOXES_TOTAL}')
    print(f'  Prioridade: Static > Dynamic > Road > Clutter')

    i = 0
    fidx = 0
    skipped = 0

    while i < total:
        tf_filename = os.path.join(output_dir, f'{name}_{split}_{fidx:03d}.tfrecord')
        with tf.python_io.TFRecordWriter(tf_filename) as writer:
            j = 0
            while i < total and j < SAMPLES_PER_FILES:
                sys.stdout.write(f'\r>> Convertendo {i+1}/{total} (ignoradas: {skipped})')
                sys.stdout.flush()

                img_path, mask_path = all_pairs[i]
                bboxes, labels = mask_to_bboxes(mask_path)

                if len(bboxes) == 0:
                    skipped += 1
                    i += 1
                    continue

                image_data = tf.gfile.FastGFile(img_path, 'rb').read()
                img_pil    = Image.open(img_path)
                shape      = [img_pil.height, img_pil.width, 3]

                example = convert_to_example(image_data, shape, bboxes, labels)
                writer.write(example.SerializeToString())
                i += 1
                j += 1
        fidx += 1

    print(f'\n[UAVid→TF] Concluído! TFRecords: {fidx} | Ignoradas: {skipped}')


if __name__ == '__main__':
    run(
        dataset_dir='./datasets/uavid',
        output_dir='./tfrecords_train',
        split='train',
        name='uavid'
    )
