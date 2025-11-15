const std = @import("std");
const zm = @import("../root.zig");
const testing = @import("../testing.zig");
const core = zm.core;

const expect = testing.expect;
const expectVecEqual = testing.expectVecEqual;

const F32x4 = core.F32x4;
const F32x8 = core.F32x8;
const Mat = core.Mat;

const f32x4 = core.f32x4;
const f32x8s = core.f32x8s;
const load = core.load;
const store = core.store;
const arrNPtr = core.arrNPtr;
const loadArr3 = core.loadArr3;
const loadArr3w = core.loadArr3w;

test "zmath.load" {
    const a = [7]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    var ptr = &a;
    var i: u32 = 0;
    const v0 = load(a[i..], F32x4, 2);
    try expectVecEqual(v0, F32x4{ 1.0, 2.0, 0.0, 0.0 });
    i += 2;
    const v1 = load(a[i .. i + 2], F32x4, 2);
    try expectVecEqual(v1, F32x4{ 3.0, 4.0, 0.0, 0.0 });
    const v2 = load(a[5..7], F32x4, 2);
    try expectVecEqual(v2, F32x4{ 6.0, 7.0, 0.0, 0.0 });
    const v3 = load(ptr[1..], F32x4, 2);
    try expectVecEqual(v3, F32x4{ 2.0, 3.0, 0.0, 0.0 });
    i += 1;
    const v4 = load(ptr[i .. i + 2], F32x4, 2);
    try expectVecEqual(v4, F32x4{ 4.0, 5.0, 0.0, 0.0 });
}

test "zmath.store" {
    var a = [7]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
    const v = load(a[1..], F32x4, 3);
    store(a[2..], v, 4);
    try expect(a[0] == 1.0);
    try expect(a[1] == 2.0);
    try expect(a[2] == 2.0);
    try expect(a[3] == 3.0);
    try expect(a[4] == 4.0);
    try expect(a[5] == 0.0);
}

test "zmath.arrNPtr" {
    {
        const mat = Mat{
            f32x4(1.0, 0.0, 0.0, 0.0),
            f32x4(0.0, 1.0, 0.0, 0.0),
            f32x4(0.0, 0.0, 1.0, 0.0),
            f32x4(0.0, 0.0, 0.0, 1.0),
        };
        const f32ptr = arrNPtr(&mat);
        try expect(f32ptr[0] == 1.0);
        try expect(f32ptr[5] == 1.0);
        try expect(f32ptr[10] == 1.0);
        try expect(f32ptr[15] == 1.0);
    }
    {
        const v8 = f32x8s(1.0);
        const f32ptr = arrNPtr(&v8);
        try expect(f32ptr[1] == 1.0);
        try expect(f32ptr[7] == 1.0);
    }
}

test "zmath.loadArr" {
    {
        const camera_position = [3]f32{ 1.0, 2.0, 3.0 };
        const simd_reg = loadArr3(camera_position);
        try expectVecEqual(simd_reg, f32x4(1.0, 2.0, 3.0, 0.0));
    }
    {
        const camera_position = [3]f32{ 1.0, 2.0, 3.0 };
        const simd_reg = loadArr3w(camera_position, 1.0);
        try expectVecEqual(simd_reg, f32x4(1.0, 2.0, 3.0, 1.0));
    }
}
