import os
import cv2

def visualize_yolo_annotations(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    class_colors: dict = None
):
    """
    Dibuja las anotaciones YOLO sobre las imágenes.

    Args:
        images_dir: Carpeta con las imágenes (.jpg, .png).
        labels_dir: Carpeta con los archivos .txt de YOLO (mismo nombre base que la imagen).
        output_dir: Carpeta donde se guardarán las imágenes anotadas.
        class_colors: Opcional dict {class_id: (B, G, R)} para colorear cada clase.
                      Si es None, todas las cajas serán rojas.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Color por defecto si no se especifica
    default_color = (0, 0, 255)

    for img_name in sorted(os.listdir(images_dir)):
        if not img_name.lower().endswith(('.jpg','jpeg','png')):
            continue

        base, _ = os.path.splitext(img_name)
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, base + '.txt')

        # Cargar imagen
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ No se pudo leer la imagen: {img_path}")
            continue
        h, w = img.shape[:2]

        # Si existe un .txt, leerlo y dibujar cajas
        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, x_c, y_c, w_n, h_n = parts
                    class_id = int(class_id)
                    x_c, y_c, w_n, h_n = map(float, (x_c, y_c, w_n, h_n))

                    # Convertir YOLO norm → píxeles
                    box_w = w_n * w
                    box_h = h_n * h
                    x1 = int((x_c * w) - box_w / 2)
                    y1 = int((y_c * h) - box_h / 2)
                    x2 = int(x1 + box_w)
                    y2 = int(y1 + box_h)

                    color = default_color
                    if class_colors and class_id in class_colors:
                        color = class_colors[class_id]

                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        img,
                        str(class_id),
                        (x1, max(10, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1
                    )
        else:
            print(f"⚠️  Etiqueta faltante para {img_name}")

        # Guardar imagen anotada
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)
        print(f"✅ Anotada guardada en {out_path}")

# Ejemplo de uso:
if __name__ == "__main__":
    visualize_yolo_annotations(
        images_dir="./images",
        labels_dir="./labels",
        output_dir="./annotated",
        class_colors={0: (255,0,0), 1: (0,255,0), 2: (0,0,255), 3: (255,255,0)}
    )
