# Паралельні обчислення на відеокарті за допомогою `wgpu` та Rust

Репозиторій містить вихідний код трьох самостійних Rust-проєктів, підготовлених у межах самостійної роботи №2 з дисципліни «Паралельне програмування» (ХНУРЕ, кафедра ПІ).

## Структура репозиторію

| Проєкт | Каталог | Опис |
| :--- | :--- | :--- |
| **Проект 1** | [`projects/01_gpu_check`](projects/01_gpu_check) | Перевірка підтримки GPGPU через `wgpu`, виведення характеристик відеокарти та базовий обчислювальний шейдер WGSL (піднесення масиву до квадрата). |
| **Проект 2** | [`projects/02_map_benchmark`](projects/02_map_benchmark) | Бенчмарк нелінійної обробки 1D-сигналу за шаблоном **Map** (`CPU 1T` vs `CPU std::thread::scope` vs `wgpu` з апаратними `TIMESTAMP_QUERY`). |
| **Проект 3** | [`projects/03_matmul_tiled`](projects/03_matmul_tiled) | Бенчмарк множення квадратних матриць $N \times N$ (`CPU 1T i-k-j` vs `CPU std::thread::scope` vs `GPU Naive` vs `GPU Tiled 32x32` з регістровим блокуванням `WPT = 4`, `stride = 33` та branchless `select()`). |

## Системні вимоги

- **Rust 1.85+** (`edition = "2024"`), встановлений через [`rustup`](https://rustup.rs/).
- **Windows 10/11**: Visual Studio Build Tools (*Desktop development with C++*) та драйвер DirectX 12 або Vulkan (за відсутності фізичного GPU працює вбудований програмний драйвер Microsoft WARP).
- **Linux**: `build-essential` та драйвер Vulkan (`libvulkan1`).
- **macOS / iOS**: нативний бекенд Apple Metal.

## Швидкий запуск

Клонуйте репозиторій:

```bash
git clone https://github.com/megaredb/tiny-rust-gpu-guide.git
cd tiny-rust-gpu-guide
```

Запустіть потрібний проєкт у режимі оптимізації `--release`:

```bash
# Проект 1: перевірка GPU та базовий WGSL-шейдер
cargo run --release --manifest-path projects/01_gpu_check/Cargo.toml

# Проект 2: бенчмарк Map-обробки 1D-сигналу
cargo run --release --manifest-path projects/02_map_benchmark/Cargo.toml

# Проект 3: бенчмарк множення матриць (Naive vs Tiled)
cargo run --release --manifest-path projects/03_matmul_tiled/Cargo.toml
```

Або перейдіть безпосередньо в каталог конкретного проєкту (наприклад, `cd projects/01_gpu_check`) і виконайте `cargo run --release`.
