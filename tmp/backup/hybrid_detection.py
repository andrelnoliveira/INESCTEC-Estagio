# ============================================================
# Deteção Híbrida UAVid
#
# Abordagem:
#   Static  → extração direta de contornos da máscara (sem SSD)
#   Dynamic → SSD deteta objetos pequenos (carros, pessoas)
#   Road    → ignorado (não é obstáculo relevante)
#   Clutter → ignorado
#
# Input: máscara pura do segmentador (RGB via PIL)
# Output: imagem com bounding boxes das duas fontes
# ============================================================

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import tensorflow as tf
import tf_slim as slim
from PIL import Image
from scipy import ndimage

from preprocessing.preprocessing import *
from networks import network_factory
from postprocessing.demo_postprocessing import *
from configs.config_demo import *


# =============================================================
# CONFIGURAÇÃO — edita aqui antes de correr
# =============================================================

# Caminho para a máscara pura do segmentador
MASK_PATH = './000000.png'

# Checkpoint SSD treinado (para Dynamic)
CHECKPOINT_PATH = './tmp/tfmodel_finetune/model.ckpt-20000'

# Score mínimo SSD para Dynamic
SSD_SCORE_THRESHOLD = 0.15

# Guardar resultado (None = mostrar janela)
SAVE_OUTPUT = './resultado_hibrido.png'

# Modo batch — pasta com máscaras (None = usar MASK_PATH)
INPUT_FOLDER  = None
OUTPUT_FOLDER = './outputs_hibrido'


# =============================================================
# Cores das classes (RGB via PIL)
# =============================================================
COLORS = {
    'Static':  np.array([128, 0,   0  ]),
    'Road':    np.array([0,   128, 0  ]),
    'Dynamic': np.array([0,   0,   128]),
    'Clutter': np.array([0,   0,   0  ]),
}

# Cores para visualização (BGR para OpenCV)
VIZ_BGR = {
    'Static':  (0,   0,   220),   # vermelho
    'Dynamic': (220, 100, 0  ),   # azul
}

# Cores para matplotlib (RGB normalizado)
VIZ_MPL = {
    'Static':  (0.9, 0.1, 0.1),
    'Dynamic': (0.1, 0.4, 0.9),
}

# Filtros para Static (extração por contornos)
STATIC_MIN_AREA_PX       = 5000    # área mínima em píxeis²
STATIC_MAX_AREA_FRACTION = 0.65    # máx fração da imagem
STATIC_MAX_BBOXES        = 10      # máx bboxes de Static por imagem


# =============================================================
# 1. Extração de bboxes de Static direto da máscara
# =============================================================

def extract_static_bboxes(mask_rgb):
    """
    Extrai bounding boxes de Static diretamente da máscara RGB.
    Usa connected components para separar regiões individuais.

    Args:
        mask_rgb: numpy array HxWx3 em RGB (lido com PIL)

    Returns:
        bboxes: list de (ymin, xmin, ymax, xmax) normalizados [0,1]
    """
    h, w = mask_rgb.shape[:2]
    total_px = h * w

    # Isolar píxeis de Static
    binary = np.all(mask_rgb == COLORS['Static'], axis=-1)
    if not binary.any():
        return []

    # Connected components com conectividade 8
    struct_8 = np.ones((3, 3), dtype=bool)
    labeled, num_regions = ndimage.label(binary, structure=struct_8)

    bboxes = []
    for region_id in range(1, num_regions + 1):
        region = labeled == region_id
        area   = int(region.sum())

        if area < STATIC_MIN_AREA_PX:
            continue
        if area > STATIC_MAX_AREA_FRACTION * total_px:
            continue

        rows = np.where(region.any(axis=1))[0]
        cols = np.where(region.any(axis=0))[0]
        bboxes.append((
            float(rows[0])  / h,
            float(cols[0])  / w,
            float(rows[-1]) / h,
            float(cols[-1]) / w,
            area
        ))

    # Ordenar por área e manter os maiores
    bboxes.sort(key=lambda x: x[4], reverse=True)
    bboxes = [(b[0], b[1], b[2], b[3]) for b in bboxes[:STATIC_MAX_BBOXES]]

    print(f'[Static] {len(bboxes)} regiões encontradas (contornos da máscara)')
    return bboxes


# =============================================================
# 2. SSD para Dynamic
# =============================================================

def run_ssd_inference(sess, image_tensor, predictions, localisations,
                      bbox_img, network, mask_img):
    """Corre o SSD e devolve bboxes de Dynamic."""

    anchors = network.generate_anchors()

    rimg, rpredictions, rlocalisations, rbbox_img = sess.run(
        [image_tensor, predictions, localisations, bbox_img],
        feed_dict={image_tensor.op.inputs[0]: mask_img}
    )

    oPostprocess = Demo_Postprocessing()
    rclasses, rscores, rbboxes = oPostprocess.select_top_bboxes(
        rpredictions, rlocalisations, anchors)
    rclasses, rscores, rbboxes = oPostprocess.sort_bboxes(rclasses, rscores, rbboxes)
    rclasses, rscores, rbboxes = oPostprocess.non_maximum_supression_bboxes(
        rclasses, rscores, rbboxes)

    # Filtrar só Dynamic (label 3) com score suficiente
    dynamic_bboxes = []
    dynamic_scores = []
    for i in range(len(rclasses)):
        if int(rclasses[i]) == 3 and float(rscores[i]) >= SSD_SCORE_THRESHOLD:
            dynamic_bboxes.append(tuple(rbboxes[i]))
            dynamic_scores.append(float(rscores[i]))

    print(f'[Dynamic] {len(dynamic_bboxes)} deteções SSD (score >= {SSD_SCORE_THRESHOLD})')
    return dynamic_bboxes, dynamic_scores


# =============================================================
# 3. Visualização
# =============================================================

def draw_hybrid_results(mask_img, static_bboxes, dynamic_bboxes,
                         dynamic_scores, save_path=None):
    """Desenha bboxes híbridas na imagem."""
    h, w = mask_img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(18, 10))
    ax.imshow(mask_img)
    ax.axis('off')

    total = 0

    # Static — contornos da máscara
    for bbox in static_bboxes:
        ymin, xmin, ymax, xmax = bbox
        bw = (xmax - xmin) * w
        bh = (ymax - ymin) * h
        rect = patches.Rectangle(
            (xmin * w, ymin * h), bw, bh,
            linewidth=2, edgecolor=VIZ_MPL['Static'],
            facecolor='none', alpha=0.9, linestyle='--'
        )
        ax.add_patch(rect)
        ax.text(xmin * w, ymin * h - 4, 'Static (máscara)',
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=VIZ_MPL['Static'], alpha=0.85,
                          pad=1, edgecolor='none'))
        total += 1

    # Dynamic — SSD
    for i, bbox in enumerate(dynamic_bboxes):
        ymin, xmin, ymax, xmax = bbox
        bw = (xmax - xmin) * w
        bh = (ymax - ymin) * h
        rect = patches.Rectangle(
            (xmin * w, ymin * h), bw, bh,
            linewidth=2, edgecolor=VIZ_MPL['Dynamic'],
            facecolor='none', alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(xmin * w, ymin * h - 4,
                f'Dynamic {dynamic_scores[i]:.2f}',
                color='white', fontsize=7, fontweight='bold',
                bbox=dict(facecolor=VIZ_MPL['Dynamic'], alpha=0.85,
                          pad=1, edgecolor='none'))
        total += 1

    ax.set_title(
        f'Deteção Híbrida UAVid — {len(static_bboxes)} Static (máscara) + '
        f'{len(dynamic_bboxes)} Dynamic (SSD) = {total} total',
        fontsize=12, pad=10
    )

    # Legenda
    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color=VIZ_MPL['Static'],  lw=2,
               linestyle='--', label='Static (contornos)'),
        Line2D([0], [0], color=VIZ_MPL['Dynamic'], lw=2,
               label='Dynamic (SSD)'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=9,
              facecolor='black', labelcolor='white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[OK] Resultado guardado: {save_path}')
    else:
        plt.show()
    plt.close()

    return total


def print_results(static_bboxes, dynamic_bboxes, dynamic_scores):
    print('\n' + '='*65)
    print(f'{"#":<4} {"Fonte":<20} {"Score":>6}  {"BBox [ymin,xmin,ymax,xmax]"}')
    print('='*65)
    for i, bbox in enumerate(static_bboxes):
        print(f'{i+1:<4} {"Static (máscara)":<20} {"---":>6}  '
              f'[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]')
    for i, bbox in enumerate(dynamic_bboxes):
        j = len(static_bboxes) + i + 1
        print(f'{j:<4} {"Dynamic (SSD)":<20} {dynamic_scores[i]:>6.3f}  '
              f'[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]')
    print('='*65)
    print(f'Total: {len(static_bboxes)} Static + {len(dynamic_bboxes)} Dynamic\n')


# =============================================================
# Pipeline principal
# =============================================================

def run_hybrid(mask_path, checkpoint_path, save_output=None):
    print(f'\n[Deteção Híbrida UAVid]')
    print(f'  Máscara    : {mask_path}')
    print(f'  Checkpoint : {checkpoint_path}')

    if not os.path.exists(mask_path):
        raise FileNotFoundError(f'Máscara não encontrada: {mask_path}')

    # Carregar máscara
    mask_pil = Image.open(mask_path).convert('RGB')
    mask_rgb = np.array(mask_pil)
    print(f'  Shape      : {mask_rgb.shape}')

    # --- 1. Static via contornos ---
    print('\n[INFO] Extraindo Static via contornos da máscara...')
    static_bboxes = extract_static_bboxes(mask_rgb)

    # --- 2. Dynamic via SSD ---
    print('\n[INFO] Carregando SSD para Dynamic...')
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False,
                            gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        input_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
        oPreprocess = Preprocessing()
        image, _, _, bbox_img = oPreprocess.preprocess(
            input_image, None, None, 'test')
        image = tf.expand_dims(image, 0)

        network = network_factory.load_network(FLAGS.model_name)
        with slim.arg_scope(network.arg_scope(
                data_format=FLAGS.demo_data_format)):
            predictions, localisations, _, _ = network.net(image)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            sess.run(tf.global_variables_initializer())

        try:
            saver.restore(sess, checkpoint_path)
            print('[OK] Pesos SSD carregados.')
        except Exception as e:
            print(f'[ERRO] Falha ao carregar checkpoint: {e}')
            raise

        # Preparar imagem para SSD
        img = mask_rgb.copy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        anchors = network.generate_anchors()
        _, rpredictions, rlocalisations, _ = sess.run(
            [image, predictions, localisations, bbox_img],
            feed_dict={input_image: img}
        )

        oPostprocess = Demo_Postprocessing()
        rclasses, rscores, rbboxes = oPostprocess.select_top_bboxes(
            rpredictions, rlocalisations, anchors)
        rclasses, rscores, rbboxes = oPostprocess.sort_bboxes(
            rclasses, rscores, rbboxes)
        rclasses, rscores, rbboxes = oPostprocess.non_maximum_supression_bboxes(
            rclasses, rscores, rbboxes)

        # Filtrar só Dynamic (label 3)
        dynamic_bboxes = []
        dynamic_scores = []
        for i in range(len(rclasses)):
            if (int(rclasses[i]) == 3 and
                    float(rscores[i]) >= SSD_SCORE_THRESHOLD):
                dynamic_bboxes.append(tuple(rbboxes[i]))
                dynamic_scores.append(float(rscores[i]))

        print(f'[Dynamic] {len(dynamic_bboxes)} deteções '
              f'(score >= {SSD_SCORE_THRESHOLD})')

    # --- 3. Resultados ---
    print_results(static_bboxes, dynamic_bboxes, dynamic_scores)

    # --- 4. Visualização ---
    draw_hybrid_results(mask_rgb, static_bboxes, dynamic_bboxes,
                        dynamic_scores, save_path=save_output)

    return static_bboxes, dynamic_bboxes, dynamic_scores


# =============================================================
# Modo batch
# =============================================================

def run_batch_hybrid(input_folder, output_folder, checkpoint_path):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in sorted(os.listdir(input_folder))
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f'[AVISO] Nenhuma imagem em: {input_folder}')
        return

    print(f'\n[Batch Híbrido] {len(images)} imagens.')

    gpu_options = tf.GPUOptions(allow_growth=True)
    cfg = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

    with tf.Session(config=cfg) as sess:
        input_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
        oPreprocess = Preprocessing()
        image, _, _, bbox_img = oPreprocess.preprocess(
            input_image, None, None, 'test')
        image = tf.expand_dims(image, 0)

        network = network_factory.load_network(FLAGS.model_name)
        with slim.arg_scope(network.arg_scope(
                data_format=FLAGS.demo_data_format)):
            predictions, localisations, _, _ = network.net(image)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            sess.run(tf.global_variables_initializer())

        saver.restore(sess, checkpoint_path)
        print('[OK] Pesos carregados.\n')

        anchors = network.generate_anchors()
        oPostprocess = Demo_Postprocessing()

        for idx, fname in enumerate(images):
            mask_path = os.path.join(input_folder, fname)
            out_path  = os.path.join(output_folder,
                                      fname.replace('.png', '_hibrido.png'))

            mask_rgb = np.array(
                Image.open(mask_path).convert('RGB'))

            # Static
            static_bboxes = extract_static_bboxes(mask_rgb)

            # Dynamic via SSD
            img = mask_rgb.copy()
            _, rpredictions, rlocalisations, _ = sess.run(
                [image, predictions, localisations, bbox_img],
                feed_dict={input_image: img}
            )
            rclasses, rscores, rbboxes = oPostprocess.select_top_bboxes(
                rpredictions, rlocalisations, anchors)
            rclasses, rscores, rbboxes = oPostprocess.sort_bboxes(
                rclasses, rscores, rbboxes)
            rclasses, rscores, rbboxes = \
                oPostprocess.non_maximum_supression_bboxes(
                    rclasses, rscores, rbboxes)

            dynamic_bboxes = []
            dynamic_scores = []
            for i in range(len(rclasses)):
                if (int(rclasses[i]) == 3 and
                        float(rscores[i]) >= SSD_SCORE_THRESHOLD):
                    dynamic_bboxes.append(tuple(rbboxes[i]))
                    dynamic_scores.append(float(rscores[i]))

            print(f'[{idx+1}/{len(images)}] {fname} → '
                  f'{len(static_bboxes)} Static + '
                  f'{len(dynamic_bboxes)} Dynamic')

            draw_hybrid_results(mask_rgb, static_bboxes,
                                dynamic_bboxes, dynamic_scores,
                                save_path=out_path)

    print(f'\n[OK] Batch concluído! Resultados em: {output_folder}')


# =============================================================
# Entry point
# =============================================================

if __name__ == '__main__':

    if INPUT_FOLDER is not None:
        run_batch_hybrid(
            input_folder=INPUT_FOLDER,
            output_folder=OUTPUT_FOLDER,
            checkpoint_path=CHECKPOINT_PATH,
        )
    else:
        run_hybrid(
            mask_path=MASK_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            save_output=SAVE_OUTPUT,
        )
