// Мінімальний обчислювальний шейдер для перевірки працездатності GPU.
// Читає масив беззнакових 32-бітних чисел і підносить кожен елемент до квадрата.

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    // Захист від виходу за межі масиву, якщо розмір не кратний 64
    if (idx < arrayLength(&data)) {
        let val = data[idx];
        data[idx] = val * val;
    }
}

