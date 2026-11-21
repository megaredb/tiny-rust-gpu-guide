use std::borrow::Cow;
use std::sync::mpsc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 256;

/// Функція обробки одного відліку сигналу на CPU (ідентична до алгоритму у shader.wgsl).
#[inline(always)]
fn process_sample(mut v: f32) -> f32 {
    for k in 1u32..=32u32 {
        let fk = k as f32;
        let s = (v * 1.15 + fk * 0.13).sin();
        let c = (v * 0.85 - fk * 0.07).cos();
        v = s * 0.6 + c * 0.4;
    }
    v
}

/// Однопотокова реалізація на CPU.
fn cpu_single_thread(input: &[f32]) -> Vec<f32> {
    input.iter().map(|&x| process_sample(x)).collect()
}

/// Багатопотокова реалізація на CPU за допомогою std::thread::scope.
fn cpu_multi_thread(input: &[f32], num_threads: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];
    let chunk_size = input.len().div_ceil(num_threads).max(1);

    std::thread::scope(|scope| {
        for (in_chunk, out_chunk) in input.chunks(chunk_size).zip(output.chunks_mut(chunk_size)) {
            scope.spawn(move || {
                for (src, dst) in in_chunk.iter().zip(out_chunk.iter_mut()) {
                    *dst = process_sample(*src);
                }
            });
        }
    });

    output
}

struct TimestampResources {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    period_ns: f32,
}

struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    timestamps: Option<TimestampResources>,
}

impl GpuContext {
    async fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .expect("Не знайдено сумісного GPU-адаптера");

        let info = adapter.get_info();
        let supports_timestamps = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let required_features = if supports_timestamps {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("GPU для Map-бенчмарку"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Не вдалося створити логічний пристрій wgpu");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Шейдер обробки 1D-сигналу"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Обчислювальний конвеєр Map"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

        // Ресурси апаратного таймера створюються один раз під час ініціалізації
        let timestamps = if supports_timestamps {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Набір міток часу GPU"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Буфер розділення міток часу"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging-буфер міток часу"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            Some(TimestampResources {
                query_set,
                resolve_buf,
                read_buf,
                period_ns: queue.get_timestamp_period(),
            })
        } else {
            None
        };

        println!(
            "GPU: {} ({:?}) | Апаратні Timestamp Queries: {}",
            info.name, info.backend, supports_timestamps
        );

        Self {
            device,
            queue,
            pipeline,
            bind_group_layout,
            timestamps,
        }
    }

    /// Виконує обчислення на GPU за одне подання черги (single submit) та повертає:
    /// (результат, чистий час шейдера в мс, повний час з копіюванням RAM <-> VRAM у мс).
    fn run_benchmark(&self, input: &[f32]) -> (Vec<f32>, f64, f64) {
        let byte_size = std::mem::size_of_val(input) as wgpu::BufferAddress;
        let num_groups = (input.len() as u32).div_ceil(WORKGROUP_SIZE);

        let total_start = Instant::now();

        // 1. Створення буферів та передача вхідного масиву з RAM у VRAM
        let input_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Вхідний storage-буфер"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Вихідний storage-буфер"),
            size: byte_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging-буфер для читання на CPU"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Група прив'язки Map"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // 2. Єдиний CommandEncoder для обчислень, фіксації міток часу та копіювання результату
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Кодувальник обчислень і копіювання"),
            });

        {
            let timestamp_writes =
                self.timestamps
                    .as_ref()
                    .map(|ts| wgpu::ComputePassTimestampWrites {
                        query_set: &ts.query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Прохід обробки сигналу"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(num_groups, 1, 1);
        }

        if let Some(ts) = &self.timestamps {
            encoder.resolve_query_set(&ts.query_set, 0..2, &ts.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&ts.resolve_buf, 0, &ts.read_buf, 0, 16);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, byte_size);

        let submit_start = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        // Запитуємо асинхронне відображення обох буферів до єдиного виклику device.poll
        let out_slice = staging_buffer.slice(..);
        let (out_tx, out_rx) = mpsc::channel();
        out_slice.map_async(wgpu::MapMode::Read, move |res| {
            out_tx.send(res).unwrap();
        });

        let ts_rx = self.timestamps.as_ref().map(|ts| {
            let slice = ts.read_buf.slice(..);
            let (tx, rx) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |res| {
                tx.send(res).unwrap();
            });
            rx
        });

        // Єдине очікування завершення всіх команд на GPU
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("Помилка очікування GPU");
        let fallback_ms = submit_start.elapsed().as_secs_f64() * 1000.0;

        out_rx
            .recv()
            .unwrap()
            .expect("Не вдалося відобразити вихідний буфер");

        let view = out_slice.get_mapped_range().unwrap();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging_buffer.unmap();

        let gpu_total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let gpu_kernel_ms = if let (Some(ts), Some(rx)) = (&self.timestamps, ts_rx) {
            rx.recv()
                .unwrap()
                .expect("Не вдалося відобразити буфер міток часу");
            let ts_view = ts.read_buf.slice(..).get_mapped_range().unwrap();
            let ticks: &[u64] = bytemuck::cast_slice(&ts_view);
            let elapsed_ticks = ticks[1].wrapping_sub(ticks[0]);
            let ms = (elapsed_ticks as f64 * ts.period_ns as f64) / 1_000_000.0;
            drop(ts_view);
            ts.read_buf.unmap();
            ms
        } else {
            fallback_ms
        };

        (result, gpu_kernel_ms, gpu_total_ms)
    }
}

fn main() {
    println!("=== Проект 2: Map-обробка 1D-сигналу (CPU vs std::thread vs wgpu) ===");

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("Доступно логічних потоків CPU: {num_threads}");

    let gpu = pollster::block_on(GpuContext::new());

    // Прогрів (warmup) на 500 000 елементів для виведення GPU на робочу частоту
    let warmup_data = vec![0.5f32; 500_000];
    let _ = gpu.run_benchmark(&warmup_data);
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    let sizes = [10_000usize, 100_000, 1_000_000, 10_000_000];

    println!();
    println!(
        "{:>10} | {:>11} | {:>11} | {:>12} | {:>12} | {:>9} | {:>9} | {:>10}",
        "N",
        "CPU 1T (мс)",
        "CPU MT (мс)",
        "GPU ядро",
        "GPU+PCIe",
        "S_ядро",
        "S_повне",
        "Макс. пох."
    );
    println!("{}", "-".repeat(102));

    for &n in &sizes {
        let input: Vec<f32> = (0..n).map(|i| ((i % 1000) as f32) * 0.005).collect();

        // Виконуємо GPU-замір до тривалих CPU-тестів, щоб мобільна відеокарта
        // не переходила в енергоощадний стан (P8) під час очікування CPU
        let (res_gpu, t_gpu_kernel, t_gpu_total) = gpu.run_benchmark(&input);
        // Гарантуємо звільнення тимчасових буферів VRAM перед наступним заміром
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

        let t1_start = Instant::now();
        let res_cpu_1t = cpu_single_thread(&input);
        let t_cpu_1t = t1_start.elapsed().as_secs_f64() * 1000.0;

        let mt_start = Instant::now();
        let res_cpu_mt = cpu_multi_thread(&input, num_threads);
        let t_cpu_mt = mt_start.elapsed().as_secs_f64() * 1000.0;

        let max_err = res_cpu_1t
            .iter()
            .zip(res_gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(res_cpu_1t, res_cpu_mt, "Розбіжність між CPU 1T та CPU MT!");
        assert!(
            max_err < 1e-4,
            "Занадто велика похибка GPU ({max_err}) для N={n}"
        );

        let speedup_kernel = t_cpu_1t / t_gpu_kernel;
        let speedup_total = t_cpu_1t / t_gpu_total;

        println!(
            "{:>10} | {:>11.3} | {:>11.3} | {:>9.3} мс | {:>9.3} мс | {:>8.2}x | {:>8.2}x | {:>10.2e}",
            n,
            t_cpu_1t,
            t_cpu_mt,
            t_gpu_kernel,
            t_gpu_total,
            speedup_kernel,
            speedup_total,
            max_err
        );
    }
}
