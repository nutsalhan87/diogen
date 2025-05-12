import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

from .types import DiogenAnswer, PlateReadSuccess

def draw_boxes_on_image(image_tensor: torch.Tensor, diogen_answer: DiogenAnswer) -> None:
    # Преобразуем тензор в numpy array и меняем оси для matplotlib (C, H, W) -> (H, W, C)
    image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Создаем фигуру и оси
    _fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Цвета для bounding boxes
    truck_color = 'red'
    plate_color = 'green'
    text_color = 'white'
    
    for truck in diogen_answer:
        # Рисуем bounding box для грузовика
        (x1, y1), (x2, y2) = truck.xyxy
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=truck_color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Рисуем bounding boxes для номеров
        for plate in truck.plates:
            (px1, py1), (px2, py2) = plate.xyxy
            p_width = px2 - px1
            p_height = py2 - py1
            plate_rect = patches.Rectangle(
                (px1, py1), p_width, p_height,
                linewidth=1, edgecolor=plate_color, facecolor='none'
            )
            ax.add_patch(plate_rect)
            
            # Добавляем текст с номером, если он есть
            if isinstance(plate.read_attempt, PlateReadSuccess):
                plate_text = f"{plate.read_attempt.number} ({plate.read_attempt.confidence:.2f})"
                ax.text(
                    px1, py1 - 5, plate_text,
                    color=text_color, fontsize=8,
                    bbox=dict(facecolor=plate_color, alpha=0.7, edgecolor='none')
                )
    
    # Сохраняем изображение
    filename = f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()