from ultralytics import YOLO
from PIL import ImageDraw, ImageFont, Image

def model_yolo(image):

    # for image in images:
    try:
        image = Image.open(image)

        draw = ImageDraw.Draw(image)

        default_model = YOLO("yolo12x.pt")
        horse_model = YOLO("my_model/horse.pt")

        first_analise = default_model(image)

        horse_detect = any(first_analise[0].names[int(box.cls[0])] == 'horse' for box in first_analise[0].boxes)

        if horse_detect:
            horses = horse_model(image)

            for box in horses[0].boxes:
                conf = float(box.conf[0])

                if conf < 0.5:
                    continue

                # Extrai info da detecção válida
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label: str = horses[0].names[cls_id].replace("_", " ").title()

                # Desenha apenas as boxes válidas
                font = ImageFont.truetype(r"font/roboto.ttf", size=30)
                # draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
                # draw.text((x1 + 5, y1 + 20), f"{label} {conf:.2f}", fill=(0, 255, 255), font=font)

            return {
                "imagem": image,
                "label": label,
                "conf": conf
            }

        else:
            return "🚫 Nenhum cavalo detectado pelo modelo pré-treinado."
        
    except Exception as e:
        return f"🚫 Erro ao processar a imagem: {e}"
