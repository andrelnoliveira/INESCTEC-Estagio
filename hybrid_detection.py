# ============================================================
# Deteção Híbrida UAVid — 5 Classes
#
# Building → contornos da máscara (separados das árvores)
# Tree     → contornos da máscara (separador natural entre edifícios)
# Dynamic  → SSD (carros em movimento)
# Road e Clutter → ignorados
# ============================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf
import tf_slim as slim
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

from preprocessing.preprocessing import *
from networks import network_factory
from postprocessing.demo_postprocessing import *
from configs.config_demo import *


# =============================================================
# CONFIGURAÇÃO
# =============================================================

MASK_PATH       = './000000.png'
CHECKPOINT_PATH = './tmp/tfmodel_finetune/model.ckpt-20000'
SSD_SCORE_THRESHOLD = 0.15
SAVE_OUTPUT     = './resultado_hibrido.png'
INPUT_FOLDER    = './Labels'
OUTPUT_FOLDER   = './outputs_hibrido'

# --- Parâmetros Building ---
BUILDING_MIN_AREA        = 5000
BUILDING_MAX_DIRECT_FRAC = 0.06
BUILDING_KMEANS_TARGET   = 0.05
BUILDING_MAX_BBOXES      = 15

# --- Parâmetros Tree ---
TREE_MIN_AREA        = 2000
TREE_MAX_DIRECT_FRAC = 0.05
TREE_KMEANS_TARGET   = 0.03
TREE_MAX_BBOXES      = 10

KMEANS_MAX_PTS = 4000


# =============================================================
# Cores das 5 classes (RGB via PIL)
#
# Segmentador novo (5 classes):
#   Building → RGB[128,  0,   0]  vermelho
#   Tree     → RGB[  0, 128,  0]  verde
#   Road     → RGB[128,  64, 128] roxo
#   Dynamic  → RGB[  0,   0, 128] azul
#   Clutter  → RGB[  0,   0,   0] preto
# =============================================================
COLORS = {
    'Building': np.array([128,  0,   0]),
    'Tree':     np.array([  0, 128,  0]),
    'Road':     np.array([128,  64, 128]),
    'Dynamic':  np.array([  0,   0, 128]),
    'Clutter':  np.array([  0,   0,   0]),
}

VIZ_MPL = {
    'Building': (0.9, 0.1, 0.1),  # vermelho
    'Tree':     (0.1, 0.7, 0.1),  # verde
    'Dynamic':  (0.1, 0.4, 0.9),  # azul
}


# =============================================================
# 1. K-means para separar regiões grandes
# =============================================================

def kmeans_split(region_mask, img_h, img_w, target_frac):
    ys, xs = np.where(region_mask)
    n_pts   = len(ys)
    region_frac = n_pts / (img_h * img_w)

    k = max(2, round(region_frac / target_frac))
    k = min(k, 20)

    step = max(1, n_pts // KMEANS_MAX_PTS)
    ys_s = ys[::step]
    xs_s = xs[::step]
    pts  = np.column_stack([xs_s / img_w, ys_s / img_h])

    km = MiniBatchKMeans(n_clusters=k, random_state=42,
                         n_init=3, max_iter=100)
    labels = km.fit_predict(pts)

    bboxes = []
    for c in range(k):
        cluster = pts[labels == c]
        if len(cluster) < 10:
            continue
        xmin = float(cluster[:, 0].min())
        ymin = float(cluster[:, 1].min())
        xmax = float(cluster[:, 0].max())
        ymax = float(cluster[:, 1].max())
        area_frac = (xmax - xmin) * (ymax - ymin)
        if area_frac < 0.001:
            continue
        bboxes.append({'ymin': ymin, 'xmin': xmin,
                       'ymax': ymax, 'xmax': xmax,
                       'area': area_frac})
    return bboxes


# =============================================================
# 2. Extração de bboxes por classe
# =============================================================

def extract_class_bboxes(mask_rgb, class_name,
                          min_area, max_direct_frac,
                          kmeans_target, max_bboxes):
    """Extrai bboxes de uma classe usando connected components + K-means."""
    h, w = mask_rgb.shape[:2]
    total_px = h * w

    binary = np.all(mask_rgb == COLORS[class_name], axis=-1)
    if not binary.any():
        return []

    struct_8 = np.ones((3, 3), dtype=bool)
    labeled, num_regions = ndimage.label(binary, structure=struct_8)

    all_bboxes = []
    max_direct  = max_direct_frac * total_px

    for region_id in range(1, num_regions + 1):
        region = labeled == region_id
        area   = int(region.sum())
        if area < min_area:
            continue

        if area <= max_direct:
            rows = np.where(region.any(axis=1))[0]
            cols = np.where(region.any(axis=0))[0]
            all_bboxes.append({
                'ymin': float(rows[0])  / h,
                'xmin': float(cols[0])  / w,
                'ymax': float(rows[-1]) / h,
                'xmax': float(cols[-1]) / w,
                'area': area / total_px,
            })
        else:
            sub = kmeans_split(region, h, w, kmeans_target)
            all_bboxes.extend(sub)

    all_bboxes.sort(key=lambda x: x['area'], reverse=True)
    result = [{k: v for k, v in b.items() if k != 'area'}
              for b in all_bboxes[:max_bboxes]]

    return result


def extract_static_bboxes(mask_rgb):
    """Extrai Building e Tree separadamente."""
    building = extract_class_bboxes(
        mask_rgb, 'Building',
        BUILDING_MIN_AREA, BUILDING_MAX_DIRECT_FRAC,
        BUILDING_KMEANS_TARGET, BUILDING_MAX_BBOXES
    )
    tree = extract_class_bboxes(
        mask_rgb, 'Tree',
        TREE_MIN_AREA, TREE_MAX_DIRECT_FRAC,
        TREE_KMEANS_TARGET, TREE_MAX_BBOXES
    )
    print(f'[Building] {len(building)} bboxes')
    print(f'[Tree]     {len(tree)} bboxes')
    return building, tree


# =============================================================
# 3. SSD para Dynamic
# =============================================================

def detect_dynamic_ssd(mask_rgb, checkpoint_path):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False,
                            gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        input_ph = tf.placeholder(tf.uint8, shape=(None, None, 3))
        oPreprocess = Preprocessing()
        image, _, _, bbox_img = oPreprocess.preprocess(
            input_ph, None, None, 'test')
        image = tf.expand_dims(image, 0)

        network = network_factory.load_network(FLAGS.model_name)
        with slim.arg_scope(network.arg_scope(
                data_format=FLAGS.demo_data_format)):
            predictions, localisations, _, _ = network.net(image)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            sess.run(tf.global_variables_initializer())

        saver.restore(sess, checkpoint_path)

        _, rp, rl, _ = sess.run(
            [image, predictions, localisations, bbox_img],
            feed_dict={input_ph: mask_rgb.astype(np.uint8)}
        )

        anchors = network.generate_anchors()
        oPost = Demo_Postprocessing()
        rc, rs, rb = oPost.select_top_bboxes(rp, rl, anchors)
        rc, rs, rb = oPost.sort_bboxes(rc, rs, rb)
        rc, rs, rb = oPost.non_maximum_supression_bboxes(rc, rs, rb)

        # SSD foi treinado com Dynamic = label 3 — mantém == 3
        dyn_b = [tuple(rb[i]) for i in range(len(rc))
                 if int(rc[i]) == 3 and float(rs[i]) >= SSD_SCORE_THRESHOLD]
        dyn_s = [float(rs[i]) for i in range(len(rc))
                 if int(rc[i]) == 3 and float(rs[i]) >= SSD_SCORE_THRESHOLD]

    print(f'[Dynamic]  {len(dyn_b)} deteções SSD')
    return dyn_b, dyn_s


# =============================================================
# 4. Visualização
# =============================================================

def draw_results(mask_rgb, building_bboxes, tree_bboxes,
                 dynamic_bboxes, dynamic_scores, save_path=None):
    h, w = mask_rgb.shape[:2]
    fig, ax = plt.subplots(1, figsize=(18, 10))
    ax.imshow(mask_rgb)
    ax.axis('off')

    # Building — vermelho tracejado
    for bbox in building_bboxes:
        rect = patches.Rectangle(
            (bbox['xmin'] * w, bbox['ymin'] * h),
            (bbox['xmax'] - bbox['xmin']) * w,
            (bbox['ymax'] - bbox['ymin']) * h,
            linewidth=2, edgecolor=VIZ_MPL['Building'],
            facecolor='none', alpha=0.9, linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(bbox['xmin'] * w, bbox['ymin'] * h - 4, 'Building',
                color='white', fontsize=6, fontweight='bold',
                bbox=dict(facecolor=VIZ_MPL['Building'], alpha=0.85,
                          pad=1, edgecolor='none'))

    # Tree — verde tracejado
    for bbox in tree_bboxes:
        rect = patches.Rectangle(
            (bbox['xmin'] * w, bbox['ymin'] * h),
            (bbox['xmax'] - bbox['xmin']) * w,
            (bbox['ymax'] - bbox['ymin']) * h,
            linewidth=2, edgecolor=VIZ_MPL['Tree'],
            facecolor='none', alpha=0.9, linestyle=':'
        )
        ax.add_patch(rect)
        ax.text(bbox['xmin'] * w, bbox['ymin'] * h - 4, 'Tree',
                color='white', fontsize=6, fontweight='bold',
                bbox=dict(facecolor=VIZ_MPL['Tree'], alpha=0.85,
                          pad=1, edgecolor='none'))

    # Dynamic — azul sólido
    for i, bbox in enumerate(dynamic_bboxes):
        ymin, xmin, ymax, xmax = bbox
        rect = patches.Rectangle(
            (xmin * w, ymin * h),
            (xmax - xmin) * w, (ymax - ymin) * h,
            linewidth=2, edgecolor=VIZ_MPL['Dynamic'],
            facecolor='none', alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(xmin * w, ymin * h - 4,
                f'Dynamic {dynamic_scores[i]:.2f}',
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=VIZ_MPL['Dynamic'], alpha=0.85,
                          pad=1, edgecolor='none'))

    total = len(building_bboxes) + len(tree_bboxes) + len(dynamic_bboxes)
    ax.set_title(
        f'Deteção Híbrida UAVid (5 classes) — '
        f'{len(building_bboxes)} Building + '
        f'{len(tree_bboxes)} Tree + '
        f'{len(dynamic_bboxes)} Dynamic = {total} total',
        fontsize=11, pad=10
    )

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color=VIZ_MPL['Building'], lw=2,
               linestyle='--', label='Building (máscara)'),
        Line2D([0], [0], color=VIZ_MPL['Tree'],     lw=2,
               linestyle=':',  label='Tree (máscara)'),
        Line2D([0], [0], color=VIZ_MPL['Dynamic'],  lw=2,
               label='Dynamic (SSD)'),
    ], loc='upper right', fontsize=9,
       facecolor='black', labelcolor='white')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[OK] Resultado: {save_path}')
    else:
        plt.show()
    plt.close()


def print_results(building_bboxes, tree_bboxes,
                  dynamic_bboxes, dynamic_scores):
    print('\n' + '='*65)
    print(f'{"#":<4} {"Classe":<20} {"Score":>6}  '
          f'{"BBox [ymin,xmin,ymax,xmax]"}')
    print('='*65)
    for i, b in enumerate(building_bboxes):
        print(f'{i+1:<4} {"Building":<20} {"---":>6}  '
              f'[{b["ymin"]:.3f}, {b["xmin"]:.3f}, '
              f'{b["ymax"]:.3f}, {b["xmax"]:.3f}]')
    for i, b in enumerate(tree_bboxes):
        j = len(building_bboxes) + i + 1
        print(f'{j:<4} {"Tree":<20} {"---":>6}  '
              f'[{b["ymin"]:.3f}, {b["xmin"]:.3f}, '
              f'{b["ymax"]:.3f}, {b["xmax"]:.3f}]')
    for i, b in enumerate(dynamic_bboxes):
        j = len(building_bboxes) + len(tree_bboxes) + i + 1
        print(f'{j:<4} {"Dynamic (SSD)":<20} '
              f'{dynamic_scores[i]:>6.3f}  '
              f'[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]')
    print('='*65)
    total = len(building_bboxes) + len(tree_bboxes) + len(dynamic_bboxes)
    print(f'Total: {len(building_bboxes)} Building + '
          f'{len(tree_bboxes)} Tree + '
          f'{len(dynamic_bboxes)} Dynamic = {total}\n')


# =============================================================
# Pipeline principal
# =============================================================

def run_hybrid(mask_path, checkpoint_path, save_output=None):
    print(f'\n[Deteção Híbrida UAVid — 5 Classes]')
    print(f'  Máscara    : {mask_path}')
    print(f'  Checkpoint : {checkpoint_path}')

    mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
    print(f'  Shape      : {mask_rgb.shape}')

    print('\n[INFO] Extraindo Building e Tree...')
    building_bboxes, tree_bboxes = extract_static_bboxes(mask_rgb)

    print('\n[INFO] Detetando Dynamic com SSD...')
    dynamic_bboxes, dynamic_scores = detect_dynamic_ssd(
        mask_rgb, checkpoint_path)

    print_results(building_bboxes, tree_bboxes,
                  dynamic_bboxes, dynamic_scores)
    draw_results(mask_rgb, building_bboxes, tree_bboxes,
                 dynamic_bboxes, dynamic_scores, save_path=save_output)

    return building_bboxes, tree_bboxes, dynamic_bboxes, dynamic_scores


# =============================================================
# Modo batch
# =============================================================

def run_batch(input_folder, output_folder, checkpoint_path):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in sorted(os.listdir(input_folder))
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print(f'[AVISO] Nenhuma imagem em: {input_folder}')
        return

    print(f'\n[Batch] {len(images)} imagens.')
    gpu_options = tf.GPUOptions(allow_growth=True)
    cfg = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

    with tf.Session(config=cfg) as sess:
        input_ph = tf.placeholder(tf.uint8, shape=(None, None, 3))
        oPreprocess = Preprocessing()
        image, _, _, bbox_img = oPreprocess.preprocess(
            input_ph, None, None, 'test')
        image = tf.expand_dims(image, 0)

        network = network_factory.load_network(FLAGS.model_name)
        with slim.arg_scope(network.arg_scope(
                data_format=FLAGS.demo_data_format)):
            predictions, localisations, _, _ = network.net(image)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            sess.run(tf.global_variables_initializer())

        saver.restore(sess, checkpoint_path)
        anchors = network.generate_anchors()
        oPost = Demo_Postprocessing()

        for idx, fname in enumerate(images):
            mask_path = os.path.join(input_folder, fname)
            out_path  = os.path.join(output_folder,
                         fname.replace('.png', '_hibrido.png'))

            mask_rgb = np.array(Image.open(mask_path).convert('RGB'))
            building_bboxes, tree_bboxes = extract_static_bboxes(mask_rgb)

            _, rp, rl, _ = sess.run(
                [image, predictions, localisations, bbox_img],
                feed_dict={input_ph: mask_rgb.astype(np.uint8)}
            )
            rc, rs, rb = oPost.select_top_bboxes(rp, rl, anchors)
            rc, rs, rb = oPost.sort_bboxes(rc, rs, rb)
            rc, rs, rb = oPost.non_maximum_supression_bboxes(rc, rs, rb)

            dyn_b = [tuple(rb[i]) for i in range(len(rc))
                     if int(rc[i]) == 3 and
                     float(rs[i]) >= SSD_SCORE_THRESHOLD]
            dyn_s = [float(rs[i]) for i in range(len(rc))
                     if int(rc[i]) == 3 and
                     float(rs[i]) >= SSD_SCORE_THRESHOLD]

            total = len(building_bboxes) + len(tree_bboxes) + len(dyn_b)
            print(f'[{idx+1}/{len(images)}] {fname} → '
                  f'{len(building_bboxes)} Building + '
                  f'{len(tree_bboxes)} Tree + '
                  f'{len(dyn_b)} Dynamic = {total}')
            draw_results(mask_rgb, building_bboxes, tree_bboxes,
                         dyn_b, dyn_s, save_path=out_path)

    print(f'\n[OK] Resultados em: {output_folder}')


# =============================================================
# Entry point
# =============================================================

if __name__ == '__main__':
    if INPUT_FOLDER is not None:
        run_batch(INPUT_FOLDER, OUTPUT_FOLDER, CHECKPOINT_PATH)
    else:
        run_hybrid(MASK_PATH, CHECKPOINT_PATH, SAVE_OUTPUT)
