import warnings
import os
import cv2
from glob import glob
from tqdm import tqdm

warnings.filterwarnings('ignore')
from ultralytics import YOLOMM

# ==========================
# 固定颜色（BGR）——六类版本
# ==========================
# 0: Car         → 紫色
# 1: Truck       → 粉色
# 2: People      → 橙色
# 3: Motorcycle  → 黄色
# 4: Lamp        → 青色
# 5: Bus         → 绿色
CLASS_COLORS = {
    0: (255, 0, 255),     # Car - 紫色
    1: (203, 192, 255),   # Truck - 粉色
    2: (0, 165, 255),     # People - 橙色
    3: (0, 255, 255),     # Motorcycle - 黄色
    4: (255, 255, 0),     # Lamp - 青色
    5: (0, 255, 0),       # Bus - 绿色
}

def get_color(cls_id: int):
    """根据类别 ID 返回预定义颜色，未定义类别返回白色。"""
    return CLASS_COLORS.get(cls_id, (255, 255, 255))


def draw_boxes_on_image(img, results):
    """在图像上绘制检测框和标签（保持你原本的简单效果）"""
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())

            label = f"{names.get(cls, str(cls))} {conf:.2f}"
            color = get_color(cls)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img


def save_txt_results(results, txt_path):
    """保存 YOLO txt (xc, yc, w, h, conf)"""
    if len(results) == 0:
        return

    h, w = results[0].orig_img.shape[:2]
    lines = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            lines.append(
                f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.4f}"
            )

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == '__main__':
    # ========== 加载模型 ==========
    model = YOLOMM('weights/m3fd_best.pt')

    # ========== 输入目录 ==========
    rgb_dir = 'dataset/M3FD/images/test'
    ir_dir = 'dataset/M3FD/images_ir/test'

    # ========== 输出目录 ==========
    output_rgb_dir = 'runs/M3FD/detect/MSDA-YOLO/rgb'
    output_ir_dir = 'runs/M3FD/detect/MSDA-YOLO/ir'
    output_txt_dir = 'runs/M3FD/detect/MSDA-YOLO/txt'

    os.makedirs(output_rgb_dir, exist_ok=True)
    os.makedirs(output_ir_dir, exist_ok=True)
    os.makedirs(output_txt_dir, exist_ok=True)

    # ========== 搜索测试集图像 ==========
    rgb_images = sorted(
        glob(os.path.join(rgb_dir, '*.jpg')) +
        glob(os.path.join(rgb_dir, '*.png'))
    )

    print(f"共找到 {len(rgb_images)} 张图片待处理\n")

    # ========== 推理循环 ==========
    for rgb_path in tqdm(rgb_images, desc="推理中"):
        filename = os.path.basename(rgb_path)
        ir_path = os.path.join(ir_dir, filename)

        if not os.path.exists(ir_path):
            print(f"跳过: 缺少 IR 图像 {filename}")
            continue

        txt_name = filename.rsplit('.', 1)[0] + '.txt'

        results = model.predict(
            source=[rgb_path, ir_path],
            imgsz=640,
            verbose=False,
            save=False,
        )

        rgb_img = cv2.imread(rgb_path)
        ir_img = cv2.imread(ir_path)

        rgb_result = draw_boxes_on_image(rgb_img.copy(), results)
        ir_result = draw_boxes_on_image(ir_img.copy(), results)

        cv2.imwrite(os.path.join(output_rgb_dir, filename), rgb_result)
        cv2.imwrite(os.path.join(output_ir_dir, filename), ir_result)

        save_txt_results(results, os.path.join(output_txt_dir, txt_name))

    print("\n推理完成！")
    print(f"RGB 结果保存在: {output_rgb_dir}")
    print(f"IR  结果保存在:  {output_ir_dir}")
    print(f"TXT 结果保存在:   {output_txt_dir}")
