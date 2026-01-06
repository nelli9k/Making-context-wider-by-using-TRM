import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import psutil
import os

# Перевірка пристрою
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

# Параметри
d_model = 256  # Розмірність ембедінгів для toy-TRM
num_layers = 2  # Кількість шарів
vocab_size = 5  # Спрощений словник для maze: 0=порожньо, 1=стіна, 2=старт, 3=кінець, 4=шлях
batch_size = 4  # Батч
maze_size = 30  # Розмір лабіринту 30x30
seq_len = maze_size * maze_size  # Токенізований як плоска послідовність ~900
max_iterations = 20  # Макс ітерацій для TRM
runs = 5  # Кількість запусків для усереднення

# Генерація випадкового лабіринту (toy-версія, без реального генератора)
def generate_maze(batch_size, size=30):
    maze = torch.randint(0, vocab_size, (batch_size, size * size), device=device)
    # Встановлюємо старт і кінець
    maze[:, 0] = 2
    maze[:, -1] = 3
    return maze

# Базова MHA для порівняння
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq, _ = x.shape
        q = self.proj_q(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        v = self.proj_v(x).view(batch, seq, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = nn.functional.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch, seq, -1)
        return self.proj_o(out)

class BaselineTransformer(nn.Module):
    def __init__(self, d_model, num_heads, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = MultiHeadAttention(d_model, num_heads=4)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        attn_out = self.attn(emb)
        logits = self.head(attn_out)
        return logits

# Toy-TRM модель
class ToyTRM(nn.Module):
    def __init__(self, d_model, num_layers, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, nhead=4) for _ in range(num_layers)])
        self.head_y = nn.Linear(d_model, vocab_size)
        self.head_z = nn.Linear(d_model, d_model)

    def forward_single(self, x, y_prev, z_prev):
        batch, seq = x.shape
        x_emb = self.embed(x.long())
        y_emb = self.embed(y_prev.long())
        z_emb = z_prev.unsqueeze(1).repeat(1, seq, 1)
        combined = torch.cat([x_emb, y_emb, z_emb], dim=2)

        combined = nn.Linear(3 * d_model, d_model, device=combined.device)(combined)
        for layer in self.layers:
            combined = layer(combined)
        z_new = self.head_z(combined.mean(dim=1))
        y_new_logits = self.head_y(combined)
        return y_new_logits, z_new

    def recursive_refine(self, x, max_steps=20, halt_thres=0.1):
        batch, seq = x.shape
        y = torch.zeros(batch, seq, dtype=torch.long, device=device)
        z = torch.zeros(batch, d_model, device=device)
        for step in range(max_steps):
            y_logits, z_new = self.forward_single(x, y, z)
            y_new = y_logits.argmax(-1)
            change_ratio = (y_new != y).float().mean()
            if change_ratio < halt_thres:
                break
            y = y_new
            z = z_new
        return y

# Функція для вимірювання
def measure_model(model, name, is_trm=False):
    model = model.to(device)
    model.eval()

    x = generate_maze(batch_size, maze_size)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    mem_before = 0.0
    if device.type == 'cuda':
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

    total_time = 0.0
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            if is_trm:
                _ = model.recursive_refine(x)
            else:
                _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time += time.perf_counter() - start

    avg_time = total_time / runs

    peak_mem_mb = 0.0
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        process = psutil.Process(os.getpid())
        peak_mem_mb = process.memory_info().rss / (1024 ** 2)

    print(f"\n{name}:")
    print(f"  Середній час: {avg_time:.4f} с")
    if device.type == 'cuda':
        print(f"  Пікова пам'ять (CUDA): {peak_mem_mb:.1f} MB")
    else:
        print(f"  Приблизна пам'ять (RSS): {peak_mem_mb:.1f} MB")

    return avg_time, peak_mem_mb

# Простий тест точності (імітація успіху шляху: % клітинок з 'шляхом' без 'стіни')
def simple_maze_accuracy(output, target=None):
    path_ratio = (output == 4).float().mean() * 100
    wall_collision = (output == 1).float().mean() * 100
    return path_ratio - wall_collision

# Запуск експериментів
print("\nЗапуск порівняння для Maze-Hard (maze_size = {}, batch = {})".format(maze_size, batch_size))

baseline = BaselineTransformer(d_model, num_heads=4, vocab_size=vocab_size)
trm = ToyTRM(d_model, num_layers, vocab_size)

# Вимірювання
measure_model(baseline, "Базовий MHA (повний контекст)", is_trm=False)
measure_model(trm, "Toy-TRM (рекурсивний)", is_trm=True)

# Точність (dummy)
x_test = generate_maze(1, maze_size)
baseline_out = baseline(x_test).argmax(-1)
trm_out = trm.recursive_refine(x_test)

print("\nПриблизна точність на dummy Maze-Hard (чим вище — тим краще):")
print("Базовий MHA:", simple_maze_accuracy(baseline_out))
print("Toy-TRM:", simple_maze_accuracy(trm_out))