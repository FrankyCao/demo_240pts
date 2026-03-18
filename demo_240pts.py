#!/usr/bin/env python3
"""
240点人脸关键点 Demo
逆向分析来源: demo/ios/FACameraDemo/FAGL/FAKGDetect.mm

V014D1A07 模型实际拥有 4 个输出张量:
  "landmark" : [1, 212, 1, 1]  → 106 基础点 × (x,y)
  "out134"   : [1, 268, 1, 1]  → 134 扩展点 × (x,y)
  "out40"    : [1,  80, 1, 1]  →  40 虹膜点 × (x,y)
  "score"    : [1,   1]        → 置信度

240点 = 106基础点 + 134扩展点
  extend_point[0..21]   = 左眼 22pts
  extend_point[22..43]  = 右眼 22pts
  extend_point[44..56]  = 左眉毛 13pts
  extend_point[57..69]  = 右眉毛 13pts
  extend_point[70..133] = 嘴唇 64pts (上唇/下唇/嘴角)

FAKGDetect.mm 中的重排序映射已在 build_240_points() 中完整复现。

运行:
  source venv_mnn/bin/activate
  python3 demo_240pts.py [--image 1.jpg]
"""

import argparse, sys, os
import numpy as np
import cv2
import MNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL    = os.path.join(BASE_DIR, "mnn_converted/V014D1A07.mnn")
DEFAULT_IMAGE = os.path.join(BASE_DIR, "1.jpg")


# ─────────────────────────────────────────────────────
# 颜色映射 (240点分区)
# ─────────────────────────────────────────────────────
REGION_COLORS = {
    "base_contour":    (200, 200, 200),  # 脸轮廓      pt[0-32]
    "base_brow_l":     (  0, 220,  80),  # 左眉        pt[33-41]
    "base_brow_r":     (  0, 160, 255),  # 右眉        pt[42-50]
    "base_nose":       (220,   0, 220),  # 鼻          pt[51-64]
    "base_eye_l":      (255, 200,   0),  # 左眼        pt[65-74]
    "base_eye_r":      (  0, 200, 255),  # 右眼        pt[75-84]
    "base_mouth":      (  0, 255,   0),  # 嘴          pt[85-103]
    "base_pupil":      (255, 255, 255),  # 瞳孔        pt[104-105]
    "ext_eye_l":       (255, 160,   0),  # 扩展左眼    ext[0-21]
    "ext_eye_r":       (  0, 160, 255),  # 扩展右眼    ext[22-43]
    "ext_brow_l":      (  0, 255, 100),  # 扩展左眉    ext[44-56]
    "ext_brow_r":      (  0, 100, 255),  # 扩展右眉    ext[57-69]
    "ext_lips":        (255,  80,  80),  # 嘴唇        ext[70-133]
    "iris":            (255, 255,   0),  # 虹膜        iris[0-39]
}


def base_color(idx):
    if idx <= 32:  return REGION_COLORS["base_contour"]
    if idx <= 41:  return REGION_COLORS["base_brow_l"]
    if idx <= 50:  return REGION_COLORS["base_brow_r"]
    if idx <= 64:  return REGION_COLORS["base_nose"]
    if idx <= 74:  return REGION_COLORS["base_eye_l"]
    if idx <= 84:  return REGION_COLORS["base_eye_r"]
    if idx <= 103: return REGION_COLORS["base_mouth"]
    return REGION_COLORS["base_pupil"]


def ext_color(idx):
    if idx <= 21:  return REGION_COLORS["ext_eye_l"]
    if idx <= 43:  return REGION_COLORS["ext_eye_r"]
    if idx <= 56:  return REGION_COLORS["ext_brow_l"]
    if idx <= 69:  return REGION_COLORS["ext_brow_r"]
    return REGION_COLORS["ext_lips"]


# ─────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────

def detect_face(img_bgr):
    """OpenCV Haar 人脸检测, 返回最大人脸 (x,y,w,h) 或 None"""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
    if len(faces) == 0:
        return None
    return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]


def crop_face(img_bgr, face_roi, pad_ratio=0.15):
    """从图像中裁剪人脸区域(含 padding), 返回 (face_crop, x1, y1, x2, y2)"""
    H, W = img_bgr.shape[:2]
    fx, fy, fw, fh = face_roi
    pad = int(max(fw, fh) * pad_ratio)
    x1 = max(0, fx - pad);  y1 = max(0, fy - pad)
    x2 = min(W, fx + fw + pad); y2 = min(H, fy + fh + pad)
    return img_bgr[y1:y2, x1:x2], x1, y1, x2, y2


def preprocess(face_bgr, size=112):
    """BGR → RGB, resize, [-1,1] 归一化 → NCHW float32"""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    rs  = cv2.resize(rgb, (size, size))
    arr = (rs.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
    return arr


def run_landmark_model(inp_np):
    """
    运行 V014D1A07 并返回全部4个输出的 numpy 数组:
    landmark[106,2], out134[134,2], out40[40,2], score
    """
    net  = MNN.Interpreter(MODEL)
    sess = net.createSession({})

    in_t = net.getSessionInput(sess, "input")
    host_in = MNN.Tensor([1, 3, 112, 112], MNN.Halide_Type_Float,
                         inp_np, MNN.Tensor_DimensionType_Caffe)
    in_t.copyFromHostTensor(host_in)
    net.runSession(sess)

    result = {}
    for name in ("landmark", "out134", "out40", "score"):
        o = net.getSessionOutput(sess, name)
        host_o = MNN.Tensor(o, MNN.Tensor_DimensionType_Caffe)
        o.copyToHostTensor(host_o)
        result[name] = np.array(host_o.getData())

    lm    = result["landmark"].reshape(106, 2)
    ext   = result["out134"].reshape(134, 2)
    iris  = result["out40"].reshape(40, 2)
    score = float(result["score"][0])
    return lm, ext, iris, score


# ─────────────────────────────────────────────────────
# FAKGDetect.mm 中的 extend_point 重排序逻辑
# ─────────────────────────────────────────────────────

def build_240_points(lm, ext):
    """
    严格复现 FAKGDetect.mm (line 77-185) 的索引映射,
    返回 pts_240: shape [240, 2], 坐标与 lm/ext 相同尺度(归一化[0,1])

    索引分配:
        0-105:   106 基础点 (landmark)
        106-127: 扩展左眼  22pts  (output  0-21)
        128-149: 扩展右眼  22pts  (output 22-43)
        150-162: 扩展左眉  13pts  (output 44-56)
        163-175: 扩展右眉  13pts  (output 57-69)
        176-239: 扩展嘴唇  64pts  (output 70-133)
    """
    # 按 FAKGDetect.mm 的循环精确复现 ★
    ordered_ext = []

    offset = 0
    # 左眼下半部分: 通用0-10 ← extend_point[21-k] (k=0..10)
    for k in range(0, 11):
        ordered_ext.append(ext[21 - k + offset])
    # 左眼上半部分: 通用11-21 ← extend_point[k-11] (k=11..21)
    for k in range(11, 22):
        ordered_ext.append(ext[k - 11 + offset])
    offset += 22   # offset=22

    # 右眼下半部分: 通用22-32 ← extend_point[43-k+offset] (k=22..32)
    for k in range(22, 33):
        ordered_ext.append(ext[43 - k + offset])
    # 右眼上半部分: 通用33-43 ← extend_point[k-33+offset] (k=33..43)
    for k in range(33, 44):
        ordered_ext.append(ext[k - 33 + offset])
    offset += 22   # offset=44

    # 左眉毛: 通用44-56 ← extend_point[k-44+offset] (k=44..56)
    for k in range(44, 57):
        ordered_ext.append(ext[k - 44 + offset])
    offset += 13   # offset=57

    # 右眉毛: 通用57-69 ← extend_point[k-57+offset] (k=57..69)
    for k in range(57, 70):
        ordered_ext.append(ext[k - 57 + offset])
    offset += 13   # offset=70

    # 嘴唇嘴角左: 通用70 ← extend_point[60+offset]
    ordered_ext.append(ext[60 + offset])
    # 嘴唇上: 通用71-85 ← extend_point[k-71+offset] (k=71..85)
    for k in range(71, 86):
        ordered_ext.append(ext[k - 71 + offset])
    # 嘴唇嘴角右: 通用86 ← extend_point[61+offset]
    ordered_ext.append(ext[61 + offset])
    # 嘴唇嘴角左2: 通用87 ← extend_point[62+offset]
    ordered_ext.append(ext[62 + offset])
    # 嘴唇上下: 通用88-102 ← extend_point[k-73+offset] (k=88..102)
    for k in range(88, 103):
        ordered_ext.append(ext[k - 73 + offset])
    # 嘴唇嘴角右2: 通用103 ← extend_point[63+offset]
    ordered_ext.append(ext[63 + offset])
    # 嘴唇下: 通用104-133 ← extend_point[k-74+offset] (k=104..133)
    for k in range(104, 134):
        ordered_ext.append(ext[k - 74 + offset])

    ordered_ext = np.array(ordered_ext)   # [134, 2]
    pts_240 = np.concatenate([lm, ordered_ext], axis=0)   # [240, 2]
    return pts_240


def map_to_image(pts_norm, x1, y1, x2, y2):
    """将归一化坐标 [0,1] 映射回原图像素坐标"""
    cw, ch = x2 - x1, y2 - y1
    px = pts_norm[:, 0] * cw + x1
    py = pts_norm[:, 1] * ch + y1
    return np.stack([px, py], axis=1)


# ─────────────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────────────

def draw_points(img, pts, colors_fn, radius=2, label_step=0):
    """在 img 上绘制 pts, colors_fn(idx)→BGR color"""
    vis = img.copy()
    for i, (px, py) in enumerate(pts):
        col = colors_fn(i)
        cv2.circle(vis, (int(px), int(py)), radius, col, -1)
        if label_step > 0 and i % label_step == 0:
            cv2.putText(vis, str(i), (int(px)+2, int(py)-2),
                        cv2.FONT_HERSHEY_PLAIN, 0.55, col, 1)
    return vis


def draw_legend(vis, region_map, start_y=15):
    y = start_y
    for name, col in region_map.items():
        cv2.rectangle(vis, (8, y), (20, y + 11), col, -1)
        cv2.putText(vis, name, (24, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, col, 1)
        y += 14
    return vis


def draw_contour(img, pts, indices, color, closed=True, thickness=1):
    """用折线/闭合曲线连接一组点索引"""
    pts_sel = np.array([[int(pts[i, 0]), int(pts[i, 1])] for i in indices])
    cv2.polylines(img, [pts_sel], isClosed=closed, color=color, thickness=thickness)


# ─────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="240点人脸关键点 Demo")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"❌ 无法读取图片: {args.image}"); sys.exit(1)
    H, W = img.shape[:2]
    print(f"输入图片: {args.image}  ({W}×{H})")

    # 1. 人脸检测
    face_roi = detect_face(img)
    if face_roi is None:
        print("❌ 未检测到人脸"); sys.exit(1)

    face_crop, cx1, cy1, cx2, cy2 = crop_face(img, face_roi)
    cw, ch = cx2 - cx1, cy2 - cy1
    print(f"人脸 ROI: ({cx1},{cy1})-({cx2},{cy2}), crop={cw}×{ch}")

    # 2. 运行模型
    inp = preprocess(face_crop, 112)
    lm, ext, iris, score = run_landmark_model(inp)
    print(f"模型推理 score={score:.4f}")
    print(f"  landmark [106,2] range x=[{lm[:,0].min():.3f},{lm[:,0].max():.3f}]")
    print(f"  out134   [134,2] range x=[{ext[:,0].min():.3f},{ext[:,0].max():.3f}]")
    print(f"  out40    [ 40,2] range x=[{iris[:,0].min():.3f},{iris[:,0].max():.3f}]")

    # 3. 构建 240 点 (严格按 FAKGDetect.mm 映射)
    pts_240_norm = build_240_points(lm, ext)     # [240, 2] 归一化
    pts_iris_norm = iris                          # [ 40, 2] 归一化
    pts_280_norm = np.concatenate([pts_240_norm, pts_iris_norm], axis=0)

    # 4. 映射到原图坐标
    pts_240 = map_to_image(pts_240_norm, cx1, cy1, cx2, cy2)  # [240, 2]
    pts_280 = map_to_image(pts_280_norm, cx1, cy1, cx2, cy2)  # [280, 2]

    # ── 图1: 106 基础点 vs 240 密集点对比 ──────────────────
    vis_base = img.copy()
    # 绘制人脸框
    cv2.rectangle(vis_base, (cx1, cy1), (cx2, cy2), (0, 200, 255), 1)
    for i in range(106):
        col = base_color(i)
        cv2.circle(vis_base, (int(pts_240[i, 0]), int(pts_240[i, 1])), 2, col, -1)
    draw_legend(vis_base, {k: v for k, v in REGION_COLORS.items() if k.startswith("base")})
    cv2.putText(vis_base, f"106 base pts  score={score:.3f}",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # ── 图2: 240 密集点 (含扩展点, 按区域着色) ─────────────
    vis_240 = img.copy()
    cv2.rectangle(vis_240, (cx1, cy1), (cx2, cy2), (0, 200, 255), 1)
    for i in range(240):
        if i < 106:
            col = base_color(i)
            r = 2
        else:
            col = ext_color(i - 106)
            r = 2
        cv2.circle(vis_240, (int(pts_240[i, 0]), int(pts_240[i, 1])), r, col, -1)
    draw_legend(vis_240, REGION_COLORS)
    cv2.putText(vis_240, f"240 pts (106 base + 134 ext)  score={score:.3f}",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # ── 图3: 280 点 (含虹膜) + 轮廓线连接 ──────────────────
    vis_280 = img.copy()
    cv2.rectangle(vis_280, (cx1, cy1), (cx2, cy2), (0, 200, 255), 1)

    # 连接轮廓线
    def safe_poly(vis, idxs, col, closed=True):
        sel = [(int(pts_280[i, 0]), int(pts_280[i, 1])) for i in idxs]
        cv2.polylines(vis, [np.array(sel)], isClosed=closed, color=col, thickness=1)

    # 脸轮廓 pt[0..32]
    safe_poly(vis_280, list(range(0, 33)), REGION_COLORS["base_contour"], closed=False)
    # 左眉 pt[33..41] + ext[44..56] (mapped to 240 idx 150..162)
    safe_poly(vis_280, list(range(33, 42)), REGION_COLORS["base_brow_l"], closed=False)
    safe_poly(vis_280, list(range(150, 163)), REGION_COLORS["ext_brow_l"], closed=False)
    # 右眉
    safe_poly(vis_280, list(range(42, 51)), REGION_COLORS["base_brow_r"], closed=False)
    safe_poly(vis_280, list(range(163, 176)), REGION_COLORS["ext_brow_r"], closed=False)
    # 鼻
    safe_poly(vis_280, list(range(51, 65)), REGION_COLORS["base_nose"], closed=False)
    # 左眼 base + ext
    safe_poly(vis_280, list(range(65, 75)), REGION_COLORS["base_eye_l"], closed=True)
    safe_poly(vis_280, list(range(106, 128)), REGION_COLORS["ext_eye_l"], closed=True)
    # 右眼
    safe_poly(vis_280, list(range(75, 85)), REGION_COLORS["base_eye_r"], closed=True)
    safe_poly(vis_280, list(range(128, 150)), REGION_COLORS["ext_eye_r"], closed=True)
    # 嘴 base
    safe_poly(vis_280, list(range(85, 104)), REGION_COLORS["base_mouth"], closed=True)
    # 扩展嘴唇
    safe_poly(vis_280, list(range(176, 240)), REGION_COLORS["ext_lips"], closed=True)
    # 虹膜 (20pts per eye)
    safe_poly(vis_280, list(range(240, 260)), REGION_COLORS["iris"], closed=True)
    safe_poly(vis_280, list(range(260, 280)), REGION_COLORS["iris"], closed=True)

    # 再绘制所有点
    for i in range(280):
        if i < 106:    col = base_color(i);           r = 2
        elif i < 240:  col = ext_color(i - 106);      r = 2
        else:          col = REGION_COLORS["iris"];    r = 3
        cv2.circle(vis_280, (int(pts_280[i, 0]), int(pts_280[i, 1])), r, col, -1)

    draw_legend(vis_280, REGION_COLORS)
    cv2.putText(vis_280, f"280 pts (240 + 40 iris)  score={score:.3f}",
                (8, H - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    # ── 图4: 带标号的扩展点 (方便验证映射正确性) ────────────
    vis_label = img.copy()
    cv2.rectangle(vis_label, (cx1, cy1), (cx2, cy2), (0, 200, 255), 1)
    for i in range(240):
        px, py = int(pts_240[i, 0]), int(pts_240[i, 1])
        if i < 106:
            col = base_color(i); r = 2
        else:
            col = ext_color(i - 106); r = 2
        cv2.circle(vis_label, (px, py), r, col, -1)
        # 只标注扩展点的索引 (可读性)
        if i >= 106 and (i - 106) % 5 == 0:
            cv2.putText(vis_label, str(i - 106), (px + 2, py - 2),
                        cv2.FONT_HERSHEY_PLAIN, 0.5, col, 1)

    # ── 保存 ──────────────────────────────────────────────
    out = BASE_DIR
    pairs = [
        (vis_base,  "result_4a_106pts.jpg"),
        (vis_240,   "result_4b_240pts.jpg"),
        (vis_280,   "result_4c_280pts_contour.jpg"),
        (vis_label, "result_4d_240pts_labeled.jpg"),
    ]
    for img_out, fname in pairs:
        path = os.path.join(out, fname)
        cv2.imwrite(path, img_out)
        print(f"  ✓ {path}")

    print(f"\n 240点统计:")
    print(f"  基础点[0-105]: 106pts")
    print(f"  扩展左眼[106-127]: 22pts")
    print(f"  扩展右眼[128-149]: 22pts")
    print(f"  扩展左眉[150-162]: 13pts")
    print(f"  扩展右眉[163-175]: 13pts")
    print(f"  扩展嘴唇[176-239]: 64pts")
    print(f"  虹膜[240-279]: 40pts (280点模式)")
    print(f"\n 完成!")


if __name__ == "__main__":
    main()
