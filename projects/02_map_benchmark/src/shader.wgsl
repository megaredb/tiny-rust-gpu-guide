// Обчислювальний шейдер для задачі Map: нелінійна багатогармонійна обробка 1D-сигналу.
// Кожен потік незалежно обробляє один відлік сигналу без умовних розгалужень у циклі,
// використовуючи апаратну інструкцію злитого множення-додавання fma(a, b, c).

@group(0) @binding(0)
var<storage, read> input_signal: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_signal: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&input_signal)) {
        return;
    }

    var v = input_signal[idx];
    for (var k: u32 = 1u; k <= 32u; k = k + 1u) {
        let fk = f32(k);
        let s = sin(fma(v, 1.15, fk * 0.13));
        let c = cos(fma(v, 0.85, -fk * 0.07));
        v = fma(s, 0.6, c * 0.4);
    }

    output_signal[idx] = v;
}
