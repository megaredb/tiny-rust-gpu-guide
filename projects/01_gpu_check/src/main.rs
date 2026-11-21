use std::borrow::Cow;
use std::sync::mpsc;
use wgpu::util::DeviceExt;

fn main() {
    // pollster дозволяє виконати асинхронні виклики wgpu синхронно у функції main
    pollster::block_on(run());
}

async fn run() {
    println!("=== Проект 1: Перевірка підтримки GPGPU через wgpu ===");

    // 1. Ініціалізуємо екземпляр wgpu з бекендами за замовчуванням (DX12/Vulkan на Windows, Vulkan на Linux, Metal на macOS)
    let instance = wgpu::Instance::default();

    // 2. Обираємо найпродуктивніший графічний адаптер (дискретну або інтегровану відеокарту)
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await
        .expect("Не знайдено сумісного GPU-адаптера (перевірте драйвери Vulkan або DirectX 12)");

    // Виводимо технічні характеристики знайденої відеокарти
    let info = adapter.get_info();
    let limits = adapter.limits();
    let features = adapter.features();

    println!("Знайдено GPU: {}", info.name);
    println!("Графічний бекенд: {:?}", info.backend);
    println!("Тип пристрою: {:?}", info.device_type);
    println!("Драйвер: {} ({})", info.driver, info.driver_info);
    println!(
        "Підтримка апаратних Timestamp Queries: {}",
        features.contains(wgpu::Features::TIMESTAMP_QUERY)
    );
    println!(
        "Макс. розмір робочої групи (X, Y, Z): ({}, {}, {})",
        limits.max_compute_workgroup_size_x,
        limits.max_compute_workgroup_size_y,
        limits.max_compute_workgroup_size_z
    );
    println!(
        "Макс. потоків у робочій групі: {}",
        limits.max_compute_invocations_per_workgroup
    );
    println!(
        "Макс. розмір storage-буфера: {} МБ",
        limits.max_storage_buffer_binding_size / (1024 * 1024)
    );

    // 3. Запитуємо логічний пристрій (Device) і чергу команд (Queue)
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("Обчислювальний пристрій"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Не вдалося створити логічний пристрій wgpu");

    // 4. Компілюємо WGSL-шейдер (вбудований у бінарник через include_str!)
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Тестовий обчислювальний шейдер"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    // 5. Готуємо тестові дані в оперативній пам'яті (RAM)
    let input_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let buffer_size = std::mem::size_of_val(input_data.as_slice()) as wgpu::BufferAddress;

    // Створюємо буфер у відеопам'яті (VRAM) і завантажуємо туди початкові числа
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Робочий storage-буфер"),
        contents: bytemuck::cast_slice(&input_data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });

    // Створюємо проміжний (staging) буфер, доступний для читання з CPU
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Буфер зворотного читання (staging)"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 6. Створюємо обчислювальний конвеєр (Compute Pipeline)
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Тестовий обчислювальний конвеєр"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Прив'язуємо буфер до @group(0) @binding(0) у шейдері
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Група прив'язки даних"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    // 7. Записуємо команди для GPU у кодувальник (CommandEncoder)
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Кодувальник обчислень"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Тестовий обчислювальний прохід"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        // Запускаємо 1 робочу групу (64 потоки покривають усі 8 елементів)
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    // Копіюємо результат із швидкої VRAM у staging-буфер для передачі на CPU
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, buffer_size);

    // Надсилаємо сформований пакет команд у чергу відеокарти
    queue.submit(Some(encoder.finish()));

    // 8. Відображаємо (map) staging-буфер у адресний простір CPU та чекаємо завершення
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    // Блокуємо потік, доки GPU не виконає команди і не поверне дані
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("Помилка очікування завершення роботи GPU");
    receiver
        .recv()
        .unwrap()
        .expect("Не вдалося відобразити буфер для читання");

    let mapped_data = buffer_slice
        .get_mapped_range()
        .expect("Не вдалося отримати доступ до відображеної пам'яті буфера");
    let result: Vec<u32> = bytemuck::cast_slice(&mapped_data).to_vec();

    // Звільняємо відображення перед подальшим використанням чи видаленням буфера
    drop(mapped_data);
    staging_buffer.unmap();

    println!("Вхідні дані:  {:?}", input_data);
    println!("Результат GPU: {:?}", result);

    let expected: Vec<u32> = input_data.iter().map(|x| x * x).collect();
    assert_eq!(
        result, expected,
        "Результат GPU не збігається з очікуваним!"
    );
    println!("Успіх! Відеокарта коректно виконала обчислювальний шейдер WGSL.");
}
