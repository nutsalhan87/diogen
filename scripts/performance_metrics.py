import os
import torch
import torchvision
import time
import psutil
from typing import Dict
from diogen.model.pipeline import Pipeline


def load_images_from_directory(directory: str) -> list[torch.Tensor]:
    """Загружает все изображения из директории и возвращает список тензоров"""
    image_tensors = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(directory, filename)
            try:
                img = torchvision.io.decode_image(
                    torchvision.io.read_file(filepath),
                    torchvision.io.ImageReadMode.RGB
                ) / 255.0
                image_tensors.append(img)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return image_tensors

def get_cpu_memory_usage() -> int:
    """Возвращает используемую память процесса в МБ"""
    process = psutil.Process()
    return process.memory_info().rss // (1024 ** 2)

def test_cpu_performance(test_images: list[torch.Tensor], n_repeats: int = 10) -> Dict:
    """Тестирование производительности на CPU"""
    base_cpu_mem = get_cpu_memory_usage()

    model = Pipeline()
    model.to('cpu')

    start_time = time.perf_counter_ns()
    for _ in range(n_repeats):
        for test_image in test_images:
            img = test_image.to('cpu')
            model.predict(img)
            
    elapsed_time_ms = (time.perf_counter_ns() - start_time) / 1e6 / n_repeats / len(test_images)
    
    cpu_mem = get_cpu_memory_usage() - base_cpu_mem
    
    return {
        'device': 'cpu',
        'inference_time_ms': elapsed_time_ms,
        'memory_usage_mb': cpu_mem
    }

def test_gpu_performance(test_images: list[torch.Tensor], n_repeats: int = 10) -> Dict:
    """Тестирование производительности на GPU"""
    model = Pipeline()
    model.to('cuda')
    
    start_time = time.perf_counter_ns()
    for _ in range(n_repeats):
        for test_image in test_images:
            img = test_image.to('cuda')
            model.predict(img)
    torch.cuda.synchronize()
    elapsed_time_ms = (time.perf_counter_ns() - start_time) / 1e6 / n_repeats / len(test_images)
    
    torch.cuda.empty_cache()
    gpu_mem = torch.cuda.memory_allocated() // (1024 ** 2)
    
    return {
        'device': 'cuda',
        'inference_time_ms': elapsed_time_ms,
        'memory_usage_mb': gpu_mem
    }

def main():
    test_images_dir = "data/perf_test_images"
    test_images = load_images_from_directory(test_images_dir)
    if not test_images:
        raise ValueError(f"No images found in {test_images_dir} directory")
    
    cpu_results = test_cpu_performance(test_images, 50)
    print("CPU Results:", cpu_results)
    
    gpu_results = test_gpu_performance(test_images, 50)
    print("GPU Results:", gpu_results)


if __name__ == '__main__':
    main()