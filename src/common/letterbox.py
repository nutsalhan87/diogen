import torch

class LetterboxTransform:
    def __init__(self, imgs: torch.Tensor, target_size: int = 640):
        """
        Args:
            imgs: Tensor shape (B, 3, H, W) - батч с изображениями
            target_size: Размер, к которому приводим изображение (квадрат)
        """
        self.original_shapes = [img.shape[1:] for img in imgs]  # List[(H, W)]
        self.target_size = target_size
        self.scales = []
        self.paddings = []
        self.resized = self._process_images(imgs)
    
    def _process_images(self, imgs: torch.Tensor) -> torch.Tensor:
        """Преобразует изображения к target_size с letterbox"""
        B, C, H, W = imgs.shape
        processed_imgs = torch.zeros((B, C, self.target_size, self.target_size), 
                                  dtype=imgs.dtype, device=imgs.device)
        
        for i in range(B):
            # Вычисляем масштаб
            scale = min(self.target_size / H, self.target_size / W)
            h_new = int(H * scale)
            w_new = int(W * scale)
            
            # Вычисляем padding
            pad_h = (self.target_size - h_new) / 2
            pad_w = (self.target_size - w_new) / 2
            
            self.scales.append((scale, scale))
            self.paddings.append((pad_w, pad_h))
            
            # Масштабируем и вставляем по центру
            img_resized = torch.nn.functional.interpolate(
                imgs[i:i+1],
                size=(h_new, w_new),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            
            processed_imgs[i, :, int(pad_h):int(pad_h)+h_new, int(pad_w):int(pad_w)+w_new] = img_resized
        
        return processed_imgs
    
    def reverse_boxes(self, batch_idx: int, boxes: torch.Tensor) -> torch.Tensor:
        """
        Преобразует координаты boxes для одного изображения обратно к оригинальному размеру
        
        Args:
            batch_idx: индекс изображения в батче
            boxes: Tensor shape (N, 4) - координаты в формате x1,y1,x2,y2
            
        Returns:
            Tensor shape (N, 4) - координаты в оригинальном размере
        """
        scale_w, scale_h = self.scales[batch_idx]
        pad_w, pad_h = self.paddings[batch_idx]
        H, W = self.original_shapes[batch_idx]
        
        # Преобразуем координаты
        x1 = (boxes[:, 0] - pad_w) / scale_w
        y1 = (boxes[:, 1] - pad_h) / scale_h
        x2 = (boxes[:, 2] - pad_w) / scale_w
        y2 = (boxes[:, 3] - pad_h) / scale_h
        
        # Обрезаем координаты и возвращаем
        return torch.stack([
            torch.clamp(x1, 0, W),
            torch.clamp(y1, 0, H),
            torch.clamp(x2, 0, W),
            torch.clamp(y2, 0, H)
        ], dim=1)