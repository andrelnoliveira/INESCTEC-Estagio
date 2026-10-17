# ============================================================
# Avaliação Completa — Segmentador + SSD/Hybrid
# UAVid 5 Classes: Building, Tree, Road, Dynamic, Clutter
#
# Métricas Segmentador (pixel-level):
#   mIoU, F1, Precision, Recall, OA por classe
#
# Métricas SSD/Hybrid (object-level):
#   mAP@0.5, IoU, Precision, Recall, F1 por classe
#   + dimensões das bboxes em píxeis
# ============================================================

import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scipy import ndimage


# =============================================================
# CONFIGURAÇÃO
# =============================================================

# Pasta com as máscaras do segmentador (predições)
PRED_MASK_DIR = './Labels'

# Pasta com as máscaras ground truth originais UAVid
# None se não tiveres GT
GT_MASK_DIR = './datasets/uavid/original'

# Pasta de saída
OUTPUT_DIR = './evaluation_results'

# =============================================================
# Filtros — devem ser iguais ao hybrid_detection.py
# =============================================================
BUILDING_MIN_AREA        = 5000
BUILDING_MAX_DIRECT_FRAC = 0.05   # mesmo que hybrid_detection.py
BUILDING_KMEANS_TARGET   = 0.04   # mesmo que hybrid_detection.py
BUILDING_MAX_BBOXES      = 15

TREE_MIN_AREA        = 2000
TREE_MAX_DIRECT_FRAC = 0.05
TREE_KMEANS_TARGET   = 0.03
TREE_MAX_BBOXES      = 10

DYNAMIC_MIN_AREA = 800
DYNAMIC_MAX_FRAC = 0.30

KMEANS_MAX_PTS = 4000

# =============================================================
# Cores RGB (PIL) — 5 classes segmentador novo
# =============================================================
PRED_COLORS = {
    'Building': np.array([128,  0,   0]),
    'Tree':     np.array([  0, 128,  0]),
    'Road':     np.array([128,  64, 128]),
    'Dynamic':  np.array([  0,   0, 128]),
    'Clutter':  np.array([  0,   0,   0]),
}

# Cores GT originais UAVid (8 classes → mapeadas para 5)
GT_COLORS_TO_CLASS = {
    (128,  0,   0): 'Building',   # Building
    (  0, 128,   0): 'Tree',      # Tree
    (128, 128,   0): 'Clutter',   # LowVeg → Clutter
    ( 64,  0,  128): 'Dynamic',   # Moving_Car → Dynamic
    (192,  0,  192): 'Clutter',   # Static_Car → Clutter
    ( 64, 64,    0): 'Clutter',   # Human → Clutter
    (128,  64,  128): 'Road',     # Road
    (  0,  0,    0): 'Clutter',   # Clutter
}

CLASSES = ['Building', 'Tree', 'Road', 'Dynamic', 'Clutter']


# =============================================================
# 1. Métricas Segmentador (pixel-level)
# =============================================================

def compute_segmentation_metrics(pred_mask_rgb, gt_mask_rgb):
    """
    Calcula métricas pixel-level entre predição e ground truth.
    Retorna dict com IoU, Precision, Recall, F1 por classe + OA.
    """
    h, w = pred_mask_rgb.shape[:2]
    total_px = h * w

    # Converter GT para labels 0-4
    gt_label = np.full((h, w), 4, dtype=np.uint8)  # default = Clutter
    for color_tuple, class_name in GT_COLORS_TO_CLASS.items():
        color = np.array(color_tuple)
        mask  = np.all(gt_mask_rgb == color, axis=-1)
        idx   = CLASSES.index(class_name)
        gt_label[mask] = idx

    # Converter pred para labels 0-4
    pred_label = np.full((h, w), 4, dtype=np.uint8)
    for i, class_name in enumerate(CLASSES):
        mask = np.all(pred_mask_rgb == PRED_COLORS[class_name], axis=-1)
        pred_label[mask] = i

    results = {}
    n_correct = 0

    for i, class_name in enumerate(CLASSES):
        tp = int(((pred_label == i) & (gt_label == i)).sum())
        fp = int(((pred_label == i) & (gt_label != i)).sum())
        fn = int(((pred_label != i) & (gt_label == i)).sum())
        tn = int(((pred_label != i) & (gt_label != i)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) \
                    if (precision + recall) > 0 else 0.0
        iou       = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        results[class_name] = {
            'TP': tp, 'FP': fp, 'FN': fn,
            'Precision': round(precision, 4),
            'Recall':    round(recall,    4),
            'F1':        round(f1,        4),
            'IoU':       round(iou,       4),
        }
        n_correct += tp

    results['OA'] = round(n_correct / total_px, 4)
    results['mIoU'] = round(
        np.mean([results[c]['IoU'] for c in CLASSES]), 4)
    results['mF1']  = round(
        np.mean([results[c]['F1']  for c in CLASSES]), 4)

    return results


# =============================================================
# 2. Extração de bboxes — igual ao hybrid_detection.py
# =============================================================

from sklearn.cluster import MiniBatchKMeans

def kmeans_split(region_mask, img_h, img_w, target_frac):
    ys, xs = np.where(region_mask)
    n_pts   = len(ys)
    region_frac = n_pts / (img_h * img_w)
    k = max(2, round(region_frac / target_frac))
    k = min(k, 20)
    step = max(1, n_pts // KMEANS_MAX_PTS)
    pts  = np.column_stack([xs[::step] / img_w, ys[::step] / img_h])
    km   = MiniBatchKMeans(n_clusters=k, random_state=42,
                           n_init=3, max_iter=100)
    labels = km.fit_predict(pts)
    bboxes = []
    for c in range(k):
        cl = pts[labels == c]
        if len(cl) < 10:
            continue
        xmin, ymin = float(cl[:,0].min()), float(cl[:,1].min())
        xmax, ymax = float(cl[:,0].max()), float(cl[:,1].max())
        if (xmax-xmin)*(ymax-ymin) < 0.001:
            continue
        bboxes.append({'ymin': ymin, 'xmin': xmin,
                       'ymax': ymax, 'xmax': xmax,
                       'width_px':  int((xmax-xmin)*img_w),
                       'height_px': int((ymax-ymin)*img_h),
                       'area_px':   int((xmax-xmin)*img_w*(ymax-ymin)*img_h)})
    return bboxes


def extract_class_bboxes(mask_rgb, class_name,
                          min_area, max_direct_frac,
                          kmeans_target, max_bboxes):
    h, w = mask_rgb.shape[:2]
    total_px = h * w
    binary = np.all(mask_rgb == PRED_COLORS[class_name], axis=-1)
    if not binary.any():
        return []
    struct_8 = np.ones((3,3), dtype=bool)
    labeled, n = ndimage.label(binary, structure=struct_8)
    all_bboxes = []
    max_direct = max_direct_frac * total_px

    for rid in range(1, n+1):
        region = labeled == rid
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
                'width_px':  int(cols[-1] - cols[0]),
                'height_px': int(rows[-1] - rows[0]),
                'area_px':   area,
            })
        else:
            sub = kmeans_split(region, h, w, kmeans_target)
            all_bboxes.extend(sub)

    all_bboxes.sort(
        key=lambda x: x['width_px']*x['height_px'], reverse=True)
    return all_bboxes[:max_bboxes]


def extract_bboxes_from_mask(mask_rgb, class_name,
                              min_area=None, max_frac=None):
    """Interface unificada para extrair bboxes por classe."""
    if class_name == 'Building':
        return extract_class_bboxes(
            mask_rgb, 'Building',
            BUILDING_MIN_AREA, BUILDING_MAX_DIRECT_FRAC,
            BUILDING_KMEANS_TARGET, BUILDING_MAX_BBOXES)
    elif class_name == 'Tree':
        return extract_class_bboxes(
            mask_rgb, 'Tree',
            TREE_MIN_AREA, TREE_MAX_DIRECT_FRAC,
            TREE_KMEANS_TARGET, TREE_MAX_BBOXES)
    elif class_name == 'Dynamic':
        return extract_class_bboxes(
            mask_rgb, 'Dynamic',
            DYNAMIC_MIN_AREA, DYNAMIC_MAX_FRAC,
            0.05, 20)
    return []


def compute_iou_bbox(a, b):
    """IoU entre dois dicts com ymin,xmin,ymax,xmax."""
    iy1 = max(a['ymin'], b['ymin'])
    ix1 = max(a['xmin'], b['xmin'])
    iy2 = min(a['ymax'], b['ymax'])
    ix2 = min(a['xmax'], b['xmax'])
    if iy2 <= iy1 or ix2 <= ix1:
        return 0.0
    inter = (iy2-iy1) * (ix2-ix1)
    area_a = (a['ymax']-a['ymin']) * (a['xmax']-a['xmin'])
    area_b = (b['ymax']-b['ymin']) * (b['xmax']-b['xmin'])
    return inter / (area_a + area_b - inter)


def compute_detection_metrics(pred_bboxes, gt_bboxes, iou_thresh=0.5):
    """
    Calcula Precision, Recall, F1, mAP para uma classe.
    pred_bboxes: lista de dicts com ymin,xmin,ymax,xmax,(score opcional)
    gt_bboxes:   lista de dicts com ymin,xmin,ymax,xmax
    """
    if not gt_bboxes:
        return {'Precision': 0, 'Recall': 0, 'F1': 0,
                'AP': 0, 'TP': 0, 'FP': len(pred_bboxes), 'FN': 0}
    if not pred_bboxes:
        return {'Precision': 0, 'Recall': 0, 'F1': 0,
                'AP': 0, 'TP': 0, 'FP': 0, 'FN': len(gt_bboxes)}

    matched_gt = set()
    tp = 0
    fp = 0

    for pb in pred_bboxes:
        best_iou = 0
        best_gt  = -1
        for j, gb in enumerate(gt_bboxes):
            if j in matched_gt:
                continue
            iou = compute_iou_bbox(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_gt  = j
        if best_iou >= iou_thresh and best_gt >= 0:
            tp += 1
            matched_gt.add(best_gt)
        else:
            fp += 1

    fn = len(gt_bboxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    return {
        'Precision': round(precision, 4),
        'Recall':    round(recall,    4),
        'F1':        round(f1,        4),
        'AP':        round(precision * recall, 4),
        'TP': tp, 'FP': fp, 'FN': fn,
    }


# =============================================================
# 3. Dimensões das bboxes em píxeis
# =============================================================

def bbox_dimensions(bboxes, img_h, img_w):
    """Calcula estatísticas de dimensão das bboxes em píxeis."""
    if not bboxes:
        return {}

    widths  = []
    heights = []
    areas   = []

    for b in bboxes:
        if 'width_px' in b:
            widths.append(b['width_px'])
            heights.append(b['height_px'])
            areas.append(b['area_px'])
        else:
            w_px = (b['xmax'] - b['xmin']) * img_w
            h_px = (b['ymax'] - b['ymin']) * img_h
            widths.append(w_px)
            heights.append(h_px)
            areas.append(w_px * h_px)

    def s(v):
        arr = np.array(v)
        return {'min': round(float(arr.min()), 1),
                'max': round(float(arr.max()), 1),
                'mean': round(float(arr.mean()), 1),
                'std':  round(float(arr.std()),  1),
                'median': round(float(np.median(arr)), 1)}

    return {
        'width_px':  s(widths),
        'height_px': s(heights),
        'area_px':   s(areas),
        'n_bboxes':  len(bboxes),
    }


# =============================================================
# 4. Relatório completo
# =============================================================

def print_segmentation_report(seg_results):
    sep = '='*70
    print(f'\n{sep}')
    print('  MÉTRICAS SEGMENTADOR (pixel-level)')
    print(sep)
    print(f'  OA (Overall Accuracy): {seg_results["OA"]*100:.2f}%')
    print(f'  mIoU:                  {seg_results["mIoU"]*100:.2f}%')
    print(f'  mF1:                   {seg_results["mF1"]*100:.2f}%')
    print(f'\n  {"Classe":<12} {"IoU":>7} {"Precision":>10} '
          f'{"Recall":>8} {"F1":>7}')
    print(f'  {"-"*50}')
    for c in CLASSES:
        r = seg_results[c]
        print(f'  {c:<12} {r["IoU"]*100:>6.2f}% '
              f'{r["Precision"]*100:>9.2f}% '
              f'{r["Recall"]*100:>7.2f}% '
              f'{r["F1"]*100:>6.2f}%')
    print(sep)


def print_detection_report(det_results, dim_results):
    sep = '='*70
    print(f'\n{sep}')
    print('  MÉTRICAS DETEÇÃO (object-level) — mAP@0.5')
    print(sep)

    aps = []
    for class_name in ['Building', 'Tree', 'Dynamic']:
        if class_name not in det_results:
            continue
        r = det_results[class_name]
        aps.append(r['AP'])
        print(f'\n  ── {class_name} ──')
        print(f'    Precision : {r["Precision"]*100:.2f}%')
        print(f'    Recall    : {r["Recall"]*100:.2f}%')
        print(f'    F1        : {r["F1"]*100:.2f}%')
        print(f'    AP@0.5    : {r["AP"]*100:.2f}%')
        print(f'    TP={r["TP"]}  FP={r["FP"]}  FN={r["FN"]}')

        if class_name in dim_results and dim_results[class_name]:
            d = dim_results[class_name]
            print(f'\n    Dimensões bboxes ({d["n_bboxes"]} bboxes):')
            print(f'    Largura  (px): min={d["width_px"]["min"]:.0f}  '
                  f'max={d["width_px"]["max"]:.0f}  '
                  f'média={d["width_px"]["mean"]:.0f}  '
                  f'std={d["width_px"]["std"]:.0f}')
            print(f'    Altura   (px): min={d["height_px"]["min"]:.0f}  '
                  f'max={d["height_px"]["max"]:.0f}  '
                  f'média={d["height_px"]["mean"]:.0f}  '
                  f'std={d["height_px"]["std"]:.0f}')
            print(f'    Área     (px): min={d["area_px"]["min"]:.0f}  '
                  f'max={d["area_px"]["max"]:.0f}  '
                  f'média={d["area_px"]["mean"]:.0f}')

    if aps:
        print(f'\n  mAP@0.5 (Building+Tree+Dynamic): '
              f'{np.mean(aps)*100:.2f}%')
    print(sep)


# =============================================================
# 5. Gráficos
# =============================================================

def plot_metrics(seg_agg, det_agg, dim_agg, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Avaliação Completa — UAVid 5 Classes',
                 fontsize=13, fontweight='bold')

    colors = ['#cc2222', '#22aa22', '#8855aa', '#2244cc', '#888888']

    # 1. IoU por classe (segmentador)
    ax = axes[0, 0]
    ious = [seg_agg.get(c, {}).get('mIoU', 0) * 100 for c in CLASSES]
    bars = ax.bar(CLASSES, ious, color=colors, alpha=0.85)
    ax.set_title('IoU por classe — Segmentador')
    ax.set_ylabel('IoU (%)')
    ax.set_ylim(0, 100)
    for bar, v in zip(bars, ious):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{v:.1f}%', ha='center', fontsize=8)

    # 2. Precision/Recall/F1 segmentador
    ax = axes[0, 1]
    x = np.arange(len(CLASSES))
    w = 0.25
    prec = [seg_agg.get(c, {}).get('mPrec', 0)*100 for c in CLASSES]
    rec  = [seg_agg.get(c, {}).get('mRec',  0)*100 for c in CLASSES]
    f1   = [seg_agg.get(c, {}).get('mF1',   0)*100 for c in CLASSES]
    ax.bar(x - w, prec, w, label='Precision', alpha=0.8, color='#4488cc')
    ax.bar(x,     rec,  w, label='Recall',    alpha=0.8, color='#44cc88')
    ax.bar(x + w, f1,   w, label='F1',        alpha=0.8, color='#cc8844')
    ax.set_title('Precision / Recall / F1 — Segmentador')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

    # 3. AP por classe (deteção)
    ax = axes[0, 2]
    det_classes = ['Building', 'Tree', 'Dynamic']
    aps = [det_agg.get(c, {}).get('AP', 0)*100 for c in det_classes]
    ax.bar(det_classes, aps,
           color=['#cc2222', '#22aa22', '#2244cc'], alpha=0.85)
    ax.set_title('AP@0.5 por classe — Deteção Híbrida')
    ax.set_ylabel('AP (%)')
    ax.set_ylim(0, 100)
    for i, v in enumerate(aps):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=9)

    # 4. Dimensões Building
    ax = axes[1, 0]
    if 'Building' in dim_agg and dim_agg['Building']:
        d = dim_agg['Building']
        labels_d = ['Largura', 'Altura']
        means = [d['width_px']['mean'], d['height_px']['mean']]
        stds  = [d['width_px']['std'],  d['height_px']['std']]
        ax.bar(labels_d, means, yerr=stds, color='#cc2222',
               alpha=0.8, capsize=5)
        ax.set_title(f'Dimensões Building (px)\n'
                     f'n={d["n_bboxes"]} bboxes')
        ax.set_ylabel('Píxeis')

    # 5. Dimensões Tree
    ax = axes[1, 1]
    if 'Tree' in dim_agg and dim_agg['Tree']:
        d = dim_agg['Tree']
        labels_d = ['Largura', 'Altura']
        means = [d['width_px']['mean'], d['height_px']['mean']]
        stds  = [d['width_px']['std'],  d['height_px']['std']]
        ax.bar(labels_d, means, yerr=stds, color='#22aa22',
               alpha=0.8, capsize=5)
        ax.set_title(f'Dimensões Tree (px)\n'
                     f'n={d["n_bboxes"]} bboxes')
        ax.set_ylabel('Píxeis')

    # 6. Dimensões Dynamic
    ax = axes[1, 2]
    if 'Dynamic' in dim_agg and dim_agg['Dynamic']:
        d = dim_agg['Dynamic']
        labels_d = ['Largura', 'Altura']
        means = [d['width_px']['mean'], d['height_px']['mean']]
        stds  = [d['width_px']['std'],  d['height_px']['std']]
        ax.bar(labels_d, means, yerr=stds, color='#2244cc',
               alpha=0.8, capsize=5)
        ax.set_title(f'Dimensões Dynamic (px)\n'
                     f'n={d["n_bboxes"]} bboxes')
        ax.set_ylabel('Píxeis')

    plt.tight_layout()
    path = os.path.join(output_dir, 'evaluation_complete.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] Gráficos: {path}')


# =============================================================
# 6. Main
# =============================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Encontrar máscaras preditas
    pred_paths = []
    for root, dirs, files in os.walk(PRED_MASK_DIR):
        if 'Labels' in root and 'overlay' not in root.lower():
            for f in sorted(files):
                if f.endswith('.png'):
                    pred_paths.append(os.path.join(root, f))
    if not pred_paths:
        pred_paths = sorted(glob.glob(
            os.path.join(PRED_MASK_DIR, '*.png')))

    print(f'[INFO] {len(pred_paths)} máscaras encontradas')

    # Acumuladores
    seg_accum = {c: {'IoU': [], 'Precision': [], 'Recall': [], 'F1': []}
                 for c in CLASSES}
    seg_accum['OA'] = []

    det_accum = {c: {'TP': 0, 'FP': 0, 'FN': 0}
                 for c in ['Building', 'Tree', 'Dynamic']}

    all_building_bboxes = []
    all_tree_bboxes     = []
    all_dynamic_bboxes  = []

    per_image_results = []

    for i, pred_path in enumerate(pred_paths):
        sys.stdout.write(f'\r  {i+1}/{len(pred_paths)}: '
                         f'{os.path.basename(pred_path)}   ')
        sys.stdout.flush()

        pred_rgb = np.array(Image.open(pred_path).convert('RGB'))
        h, w = pred_rgb.shape[:2]

        # Segmentador — só se tiver GT
        seg_result = {}
        if GT_MASK_DIR:
            fname    = os.path.basename(pred_path)
            abs_path = os.path.abspath(pred_path)
            seq      = abs_path.split(os.sep)[-3]
            gt_path  = os.path.join(GT_MASK_DIR, seq, 'Labels', fname)
            if os.path.exists(gt_path):
                gt_rgb = np.array(Image.open(gt_path).convert('RGB'))
                seg_result = compute_segmentation_metrics(pred_rgb, gt_rgb)
                seg_accum['OA'].append(seg_result['OA'])
                for c in CLASSES:
                    for m in ['IoU', 'Precision', 'Recall', 'F1']:
                        seg_accum[c][m].append(seg_result[c][m])

        # Bboxes preditas — usando mesmo método do hybrid_detection.py
        building_b = extract_bboxes_from_mask(pred_rgb, 'Building')
        tree_b     = extract_bboxes_from_mask(pred_rgb, 'Tree')
        dynamic_b  = extract_bboxes_from_mask(pred_rgb, 'Dynamic')

        all_building_bboxes.extend(building_b)
        all_tree_bboxes.extend(tree_b)
        all_dynamic_bboxes.extend(dynamic_b)

        # GT bboxes (se disponível)
        if GT_MASK_DIR and os.path.exists(gt_path):
            gt_building = extract_bboxes_from_mask(gt_rgb, 'Building')
            gt_tree     = extract_bboxes_from_mask(gt_rgb, 'Tree')
            gt_dynamic  = extract_bboxes_from_mask(gt_rgb, 'Dynamic')

            for class_name, pred_b, gt_b in [
                ('Building', building_b, gt_building),
                ('Tree',     tree_b,     gt_tree),
                ('Dynamic',  dynamic_b,  gt_dynamic),
            ]:
                dm = compute_detection_metrics(pred_b, gt_b)
                det_accum[class_name]['TP'] += dm['TP']
                det_accum[class_name]['FP'] += dm['FP']
                det_accum[class_name]['FN'] += dm['FN']

        per_image_results.append({
            'image':      os.path.basename(pred_path),
            'n_building': len(building_b),
            'n_tree':     len(tree_b),
            'n_dynamic':  len(dynamic_b),
            'seg':        seg_result,
        })

    print()

    # Agregar segmentador
    seg_agg = {}
    if seg_accum['OA']:
        seg_agg['OA']   = round(np.mean(seg_accum['OA']), 4)
        seg_agg['mIoU'] = round(
            np.mean([np.mean(seg_accum[c]['IoU']) for c in CLASSES]), 4)
        seg_agg['mF1']  = round(
            np.mean([np.mean(seg_accum[c]['F1'])  for c in CLASSES]), 4)
        for c in CLASSES:
            seg_agg[c] = {
                'mIoU':  round(np.mean(seg_accum[c]['IoU']),       4),
                'mPrec': round(np.mean(seg_accum[c]['Precision']), 4),
                'mRec':  round(np.mean(seg_accum[c]['Recall']),    4),
                'mF1':   round(np.mean(seg_accum[c]['F1']),        4),
            }
        print_segmentation_report(seg_agg)
    else:
        print('[INFO] GT não disponível — métricas segmentador ignoradas')
        print('[INFO] Para activar: define GT_MASK_DIR no topo do script')

    # Agregar deteção
    det_agg = {}
    for c in ['Building', 'Tree', 'Dynamic']:
        tp = det_accum[c]['TP']
        fp = det_accum[c]['FP']
        fn = det_accum[c]['FN']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else 0.0
        det_agg[c] = {
            'Precision': round(prec, 4),
            'Recall':    round(rec,  4),
            'F1':        round(f1,   4),
            'AP':        round(prec * rec, 4),
            'TP': tp, 'FP': fp, 'FN': fn,
        }

    # Dimensões
    dim_agg = {
        'Building': bbox_dimensions(all_building_bboxes, h, w),
        'Tree':     bbox_dimensions(all_tree_bboxes,     h, w),
        'Dynamic':  bbox_dimensions(all_dynamic_bboxes,  h, w),
    }

    print_detection_report(det_agg, dim_agg)

    # JSON
    json_path = os.path.join(OUTPUT_DIR, 'evaluation_complete.json')
    with open(json_path, 'w') as f:
        json.dump({
            'segmentation': seg_agg,
            'detection':    det_agg,
            'dimensions':   dim_agg,
            'per_image':    per_image_results,
        }, f, indent=2)
    print(f'[OK] JSON: {json_path}')

    # Gráficos
    plot_metrics(seg_agg, det_agg, dim_agg, OUTPUT_DIR)
    print(f'\n[OK] Tudo em: {OUTPUT_DIR}')
