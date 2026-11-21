// Обчислювальні шейдери множення квадратних матриць C = A * B (розміром N x N).
// Містить дві точки входу:
// 1. matmul_naive — наївна реалізація з прямим читанням із глобальної storage-пам'яті.
// 2. matmul_tiled — оптимізована блочна реалізація (Tiling 32x32 + регістрове блокування WPT=4
//    + захист від банк-конфліктів stride=33 + безрозгалужувальне завантаження select()).

const TILE_SIZE: u32 = 32u;
const WPT: u32 = 4u; // Кількість елементів результату на один потік (Work Per Thread)

struct Dimensions {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<uniform> dims: Dimensions;

@group(0) @binding(1)
var<storage, read> matrix_a: array<f32>;

@group(0) @binding(2)
var<storage, read> matrix_b: array<f32>;

@group(0) @binding(3)
var<storage, read_write> matrix_c: array<f32>;

// Розділювана пам'ять робочої групи (on-chip SRAM) для плиток 32x32.
// Додатковий 33-й стовпець (padding) усуває банк-конфлікти 32 банків розділюваної пам'яті.
var<workgroup> tile_a: array<array<f32, 33>, 32>;
var<workgroup> tile_b: array<array<f32, 33>, 32>;

@compute @workgroup_size(16, 16, 1)
fn matmul_naive(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let n = dims.n;

    if (row >= n || col >= n) {
        return;
    }

    var acc: f32 = 0.0;
    for (var k: u32 = 0u; k < n; k = k + 1u) {
        acc = fma(matrix_a[row * n + k], matrix_b[k * n + col], acc);
    }

    matrix_c[row * n + col] = acc;
}

// Робоча група 32x8 (256 потоків) обчислює плитку матриці C розміром 32x32:
// кожен потік обчислює WPT = 4 сусідні елементи по вертикалі у регістрах,
// а 32 потоки вздовж осі X читають рядок матриці однією коалесцентною транзакцією з VRAM.
@compute @workgroup_size(32, 8, 1)
fn matmul_tiled(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>,
) {
    let local_col = local_id.x; // 0..31 (горизонтальна координата потоку в робочій групі)
    let local_row_base = local_id.y * WPT; // 0, 4, 8, ..., 28
    let global_col = group_id.x * TILE_SIZE + local_col;
    let global_row_base = group_id.y * TILE_SIZE + local_row_base;
    let n = dims.n;
    let max_idx = n * n - 1u;

    let num_tiles = (n + TILE_SIZE - 1u) / TILE_SIZE;

    // Чотири регістрові акумулятори для WPT = 4 рядків одного стовпця
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let tiled_col = t * TILE_SIZE + local_col;

        // Кооперативне безрозгалужувальне (branchless) завантаження 4 рядків плитки A та B
        for (var w: u32 = 0u; w < WPT; w = w + 1u) {
            let l_row = local_row_base + w;
            let g_row = global_row_base + w;
            let tiled_row = t * TILE_SIZE + l_row;

            let a_valid = g_row < n && tiled_col < n;
            let a_addr = min(g_row * n + tiled_col, max_idx);
            tile_a[l_row][local_col] = select(0.0, matrix_a[a_addr], a_valid);

            let b_valid = tiled_row < n && global_col < n;
            let b_addr = min(tiled_row * n + global_col, max_idx);
            tile_b[l_row][local_col] = select(0.0, matrix_b[b_addr], b_valid);
        }

        // Бар'єр очікування запису плитки всіма 256 потоками групи
        workgroupBarrier();

        // Обчислення у швидкій on-chip SRAM із повторним використанням b_val у регістрі
        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            let b_val = tile_b[k][local_col];
            acc0 = fma(tile_a[local_row_base + 0u][k], b_val, acc0);
            acc1 = fma(tile_a[local_row_base + 1u][k], b_val, acc1);
            acc2 = fma(tile_a[local_row_base + 2u][k], b_val, acc2);
            acc3 = fma(tile_a[local_row_base + 3u][k], b_val, acc3);
        }

        // Бар'єр перед завантаженням наступної плитки
        workgroupBarrier();
    }

    if (global_col < n) {
        if (global_row_base + 0u < n) {
            matrix_c[(global_row_base + 0u) * n + global_col] = acc0;
        }
        if (global_row_base + 1u < n) {
            matrix_c[(global_row_base + 1u) * n + global_col] = acc1;
        }
        if (global_row_base + 2u < n) {
            matrix_c[(global_row_base + 2u) * n + global_col] = acc2;
        }
        if (global_row_base + 3u < n) {
            matrix_c[(global_row_base + 3u) * n + global_col] = acc3;
        }
    }
}
