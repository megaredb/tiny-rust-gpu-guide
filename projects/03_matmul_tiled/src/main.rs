use std::borrow::Cow;
use std::sync::mpsc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const NAIVE_WG_SIZE: u32 = 16;
const TILED_TILE_SIZE: u32 = 32;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Dimensions {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Однопотокове множення матриць на CPU з кеш-дружнім порядком циклів (i-k-j).
fn cpu_matmul_single(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        let a_row = &a[i * n..(i + 1) * n];
        let c_row = &mut c[i * n..(i + 1) * n];
        for k in 0..n {
            let a_ik = a_row[k];
            let b_row = &b[k * n..(k + 1) * n];
            for j in 0..n {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
    c
}

/// Багатопотокове множення матриць на CPU за допомогою std::thread::scope.
fn cpu_matmul_multi(a: &[f32], b: &[f32], n: usize, num_threads: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    let rows_per_thread = n.div_ceil(num_threads).max(1);

    std::thread::scope(|scope| {
        for (chunk_idx, c_chunk) in c.chunks_mut(rows_per_thread * n).enumerate() {
            let start_row = chunk_idx * rows_per_thread;
            let chunk_rows = c_chunk.len() / n;
            let a_chunk = &a[start_row * n..(start_row + chunk_rows) * n];

            scope.spawn(move || {
                for local_i in 0..chunk_rows {
                    let a_row = &a_chunk[local_i * n..(local_i + 1) * n];
                    let c_row = &mut c_chunk[local_i * n..(local_i + 1) * n];
                    for k in 0..n {
                        let a_ik = a_row[k];
                        let b_row = &b[k * n..(k + 1) * n];
                        for j in 0..n {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            });
        }
    });

    c
}

#[derive(Copy, Clone)]
enum KernelKind {
    Naive,
    Tiled,
}

struct TimestampResources {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    read_buf: wgpu::Buffer,
    period_ns: f32,
}

struct GpuMatmulContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    naive_pipeline: wgpu::ComputePipeline,
    tiled_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    timestamps: Option<TimestampResources>,
}

impl GpuMatmulContext {
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
                label: Some("GPU для множення матриць"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("Не вдалося створити логічний пристрій wgpu");

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Шейдер множення матриць"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Схема прив'язки матриць"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Макет конвеєра матриць"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let naive_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Конвеєр matmul_naive"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("matmul_naive"),
            compilation_options: Default::default(),
            cache: None,
        });

        let tiled_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Конвеєр matmul_tiled"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("matmul_tiled"),
            compilation_options: Default::default(),
            cache: None,
        });

        let timestamps = if supports_timestamps {
            let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Таймер GPU"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            });
            let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Буфер розділення міток"),
                size: 16,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging міток"),
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
            naive_pipeline,
            tiled_pipeline,
            bind_group_layout,
            timestamps,
        }
    }

    /// Запускає обраний шейдер множення матриць за один submit і повертає:
    /// (матриця C, чистий час шейдера в мс, повний час із передачею RAM <-> VRAM у мс).
    fn run_matmul(&self, a: &[f32], b: &[f32], n: usize, kind: KernelKind) -> (Vec<f32>, f64, f64) {
        let matrix_bytes = (n * n * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let groups_xy = match kind {
            KernelKind::Naive => (n as u32).div_ceil(NAIVE_WG_SIZE),
            KernelKind::Tiled => (n as u32).div_ceil(TILED_TILE_SIZE),
        };

        let dims = Dimensions {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let total_start = Instant::now();

        let uniform_buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Uniform розміру матриці"),
                contents: bytemuck::bytes_of(&dims),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let buf_a = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Буфер матриці A"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_b = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Буфер матриці B"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let buf_c = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Буфер матриці C"),
            size: matrix_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging-буфер матриці C"),
            size: matrix_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Група прив'язки матриць"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_c.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Кодувальник множення матриць"),
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
                label: Some("Прохід множення матриць"),
                timestamp_writes,
            });
            let pipeline = match kind {
                KernelKind::Naive => &self.naive_pipeline,
                KernelKind::Tiled => &self.tiled_pipeline,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(groups_xy, groups_xy, 1);
        }

        if let Some(ts) = &self.timestamps {
            encoder.resolve_query_set(&ts.query_set, 0..2, &ts.resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&ts.resolve_buf, 0, &ts.read_buf, 0, 16);
        }

        encoder.copy_buffer_to_buffer(&buf_c, 0, &staging_buf, 0, matrix_bytes);

        let submit_start = Instant::now();
        self.queue.submit(Some(encoder.finish()));

        let out_slice = staging_buf.slice(..);
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

        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("Помилка очікування GPU");
        let fallback_ms = submit_start.elapsed().as_secs_f64() * 1000.0;

        out_rx.recv().unwrap().unwrap();
        let view = out_slice.get_mapped_range().unwrap();
        let result: Vec<f32> = bytemuck::cast_slice(&view).to_vec();
        drop(view);
        staging_buf.unmap();

        let gpu_total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        let gpu_kernel_ms = if let (Some(ts), Some(rx)) = (&self.timestamps, ts_rx) {
            rx.recv().unwrap().unwrap();
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    println!("=== Проект 3: Множення матриць N x N (CPU vs Naive GPU vs Tiled GPU) ===");

    let num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!("Доступно логічних потоків CPU: {num_threads}");

    let gpu = pollster::block_on(GpuMatmulContext::new());

    // Прогрів обох конвеєрів на матриці 512x512 для виведення GPU на робочу частоту
    let warm = vec![0.5f32; 512 * 512];
    let _ = gpu.run_matmul(&warm, &warm, 512, KernelKind::Naive);
    let _ = gpu.run_matmul(&warm, &warm, 512, KernelKind::Tiled);
    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

    let sizes = [256usize, 512, 1024, 2048];

    println!();
    println!(
        "{:>6} | {:>10} | {:>10} | {:>11} | {:>11} | {:>11} | {:>11} | {:>9} | {:>11}",
        "N",
        "CPU 1T",
        "CPU MT",
        "Naive ядро",
        "Naive+PCIe",
        "Tiled ядро",
        "Tiled+PCIe",
        "Tiled/Naiv",
        "S_повне(T)"
    );
    println!("{}", "-".repeat(114));

    for &n in &sizes {
        let a: Vec<f32> = (0..n * n)
            .map(|idx| ((idx * 17 + 3) % 100) as f32 * 0.005)
            .collect();
        let b: Vec<f32> = (0..n * n)
            .map(|idx| ((idx * 31 + 7) % 100) as f32 * 0.005)
            .collect();

        let t1 = Instant::now();
        let c_cpu_1t = cpu_matmul_single(&a, &b, n);
        let t_cpu_1t = t1.elapsed().as_secs_f64() * 1000.0;

        let t2 = Instant::now();
        let c_cpu_mt = cpu_matmul_multi(&a, &b, n, num_threads);
        let t_cpu_mt = t2.elapsed().as_secs_f64() * 1000.0;

        let (c_naive, t_naive_kernel, t_naive_total) = gpu.run_matmul(&a, &b, n, KernelKind::Naive);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

        let (c_tiled, t_tiled_kernel, t_tiled_total) = gpu.run_matmul(&a, &b, n, KernelKind::Tiled);
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());

        let err_mt = max_abs_diff(&c_cpu_1t, &c_cpu_mt);
        let err_naive = max_abs_diff(&c_cpu_1t, &c_naive);
        let err_tiled = max_abs_diff(&c_cpu_1t, &c_tiled);

        assert!(err_mt < 1e-3, "Похибка CPU MT ({err_mt}) для N={n}");
        assert!(
            err_naive < 1e-2,
            "Похибка Naive GPU ({err_naive}) для N={n}"
        );
        assert!(
            err_tiled < 1e-2,
            "Похибка Tiled GPU ({err_tiled}) для N={n}"
        );

        let tiled_vs_naive = t_naive_kernel / t_tiled_kernel;
        let speedup_total = t_cpu_1t / t_tiled_total;

        println!(
            "{:>6} | {:>7.2} мс | {:>7.2} мс | {:>8.3} мс | {:>8.3} мс | {:>8.3} мс | {:>8.3} мс | {:>8.2}x | {:>10.2}x",
            n,
            t_cpu_1t,
            t_cpu_mt,
            t_naive_kernel,
            t_naive_total,
            t_tiled_kernel,
            t_tiled_total,
            tiled_vs_naive,
            speedup_total
        );
    }
}
