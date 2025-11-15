const std = @import("std");
const builtin = @import("builtin");
const zm = @import("../root.zig");
const core = zm.core;
const vector = zm.vector;
const vector_internal = @import("../vector.zig");
const testing = @import("../testing.zig");

const math = std.math;
inline fn expect(actual: anytype) !void {
    try testing.expect(actual);
}
const expectVecEqual = testing.expectVecEqual;
const expectVecApproxEqAbs = testing.expectVecApproxEqAbs;
const approxEqAbs = testing.approxEqAbs;

const F32x4 = core.F32x4;
const F32x8 = core.F32x8;
const F32x16 = core.F32x16;
const Boolx4 = core.Boolx4;
const Boolx8 = core.Boolx8;
const Boolx16 = core.Boolx16;
const Vec = core.Vec;
const Mat = core.Mat;
const Quat = core.Quat;

const f32x4 = core.f32x4;
const f32x8 = core.f32x8;
const f32x16 = core.f32x16;
const f32x4s = core.f32x4s;
const f32x8s = core.f32x8s;
const f32x16s = core.f32x16s;
const boolx4 = core.boolx4;
const boolx8 = core.boolx8;
const boolx16 = core.boolx16;
const splat = core.splat;
const splatInt = core.splatInt;
const veclen = core.veclen;
const vecToArr2 = core.vecToArr2;
const vecToArr3 = core.vecToArr3;
const vecToArr4 = core.vecToArr4;

const all = vector.all;
const any = vector.any;
const isNearEqual = vector.isNearEqual;
const isNan = vector.isNan;
const isInf = vector.isInf;
const isInBounds = vector.isInBounds;
const andInt = vector.andInt;
const andNotInt = vector.andNotInt;
const orInt = vector.orInt;
const norInt = vector.norInt;
const xorInt = vector.xorInt;
const minFast = vector.minFast;
const maxFast = vector.maxFast;
const min = vector.min;
const max = vector.max;
const round = vector.round;
const floor = vector.floor;
const trunc = vector.trunc;
const ceil = vector.ceil;
const clamp = vector.clamp;
const clampFast = vector.clampFast;
const saturate = vector.saturate;
const saturateFast = vector.saturateFast;
const lerp = vector.lerp;
const lerpV = vector.lerpV;
const lerpInverse = vector.lerpInverse;
const lerpInverseV = vector.lerpInverseV;
const mapLinear = vector.mapLinear;
const mapLinearV = vector.mapLinearV;
const lerpOverTime = vector.lerpOverTime;
const lerpVOverTime = vector.lerpVOverTime;
const sqrt = vector.sqrt;
const abs = vector.abs;
const mod = vector.mod;
const modAngle = vector.modAngle;
const mulAdd = vector.mulAdd;
const select = vector.select;
const sin = vector.sin;
const cos = vector.cos;
const sincos = vector.sincos;
const asin = vector.asin;
const acos = vector.acos;
const atan = vector.atan;
const atan2 = vector.atan2;
const cmulSoa = vector.cmulSoa;
const linePointDistance = vector_internal.linePointDistance;
const sin32 = vector_internal.sin32;
const cos32 = vector_internal.cos32;
const sincos32 = vector_internal.sincos32;
const asin32 = vector_internal.asin32;
const acos32 = vector_internal.acos32;
const swizzle = vector.swizzle;
const dot2 = vector.dot2;
const dot3 = vector.dot3;
const dot4 = vector.dot4;
const cross3 = vector.cross3;
const lengthSq2 = vector.lengthSq2;
const lengthSq3 = vector.lengthSq3;
const lengthSq4 = vector.lengthSq4;
const length2 = vector.length2;
const length3 = vector.length3;
const length4 = vector.length4;
const normalize2 = vector.normalize2;
const normalize3 = vector.normalize3;
const normalize4 = vector.normalize4;
const mul = vector.mul;
const floatToIntAndBack = vector_internal.floatToIntAndBack;

test "zmath.all" {
    try expect(all(boolx8(true, true, true, true, true, false, true, false), 5) == true);
    try expect(all(boolx8(true, true, true, true, true, false, true, false), 6) == false);
    try expect(all(boolx8(true, true, true, true, false, false, false, false), 4) == true);
    try expect(all(boolx4(true, true, true, false), 3) == true);
    try expect(all(boolx4(true, true, true, false), 1) == true);
    try expect(all(boolx4(true, false, false, false), 1) == true);
    try expect(all(boolx4(false, true, false, false), 1) == false);
    try expect(all(boolx8(true, true, true, true, true, false, true, false), 0) == false);
    try expect(all(boolx4(false, true, false, false), 0) == false);
    try expect(all(boolx4(true, true, true, true), 0) == true);
}

test "zmath.any" {
    try expect(any(boolx8(true, true, true, true, true, false, true, false), 0) == true);
    try expect(any(boolx8(false, false, false, true, true, false, true, false), 3) == false);
    try expect(any(boolx8(false, false, false, false, false, true, false, false), 4) == false);
}

test "zmath.isNearEqual" {
    {
        const v0 = f32x4(1.0, 2.0, -3.0, 4.001);
        const v1 = f32x4(1.0, 2.1, 3.0, 4.0);
        const b = isNearEqual(v0, v1, splat(F32x4, 0.01));
        try expect(@reduce(.And, b == boolx4(true, false, false, true)));
    }
    {
        const v0 = f32x8(1.0, 2.0, -3.0, 4.001, 1.001, 2.3, -0.0, 0.0);
        const v1 = f32x8(1.0, 2.1, 3.0, 4.0, -1.001, 2.1, 0.0, 0.0);
        const b = isNearEqual(v0, v1, splat(F32x8, 0.01));
        try expect(@reduce(.And, b == boolx8(true, false, false, true, false, false, true, true)));
    }
    try expect(all(isNearEqual(
        splat(F32x4, math.inf(f32)),
        splat(F32x4, math.inf(f32)),
        splat(F32x4, 0.0001),
    ), 0) == false);
    try expect(all(isNearEqual(
        splat(F32x4, -math.inf(f32)),
        splat(F32x4, math.inf(f32)),
        splat(F32x4, 0.0001),
    ), 0) == false);
    try expect(all(isNearEqual(
        splat(F32x4, -math.inf(f32)),
        splat(F32x4, -math.inf(f32)),
        splat(F32x4, 0.0001),
    ), 0) == false);
    try expect(all(isNearEqual(
        splat(F32x4, -math.nan(f32)),
        splat(F32x4, math.inf(f32)),
        splat(F32x4, 0.0001),
    ), 0) == false);
}

test "zmath.isNan" {
    {
        const v0 = f32x4(math.inf(f32), math.nan(f32), math.nan(f32), 7.0);
        const b = isNan(v0);
        try expect(@reduce(.And, b == boolx4(false, true, true, false)));
    }
    {
        const v0 = f32x8(0, math.nan(f32), 0, 0, math.inf(f32), math.nan(f32), math.snan(f32), 7.0);
        const b = isNan(v0);
        try expect(@reduce(.And, b == boolx8(false, true, false, false, false, true, true, false)));
    }
}

test "zmath.isInf" {
    {
        const v0 = f32x4(math.inf(f32), math.nan(f32), math.snan(f32), 7.0);
        const b = isInf(v0);
        try expect(@reduce(.And, b == boolx4(true, false, false, false)));
    }
    {
        const v0 = f32x8(0, math.inf(f32), 0, 0, math.inf(f32), math.nan(f32), math.snan(f32), 7.0);
        const b = isInf(v0);
        try expect(@reduce(.And, b == boolx8(false, true, false, false, true, false, false, false)));
    }
}

test "zmath.isInBounds" {
    {
        const v0 = f32x4(0.5, -2.0, -1.0, 1.9);
        const v1 = f32x4(-1.6, -2.001, -1.0, 1.9);
        const bounds = f32x4(1.0, 2.0, 1.0, 2.0);
        const b0 = isInBounds(v0, bounds);
        const b1 = isInBounds(v1, bounds);
        try expect(@reduce(.And, b0 == boolx4(true, true, true, true)));
        try expect(@reduce(.And, b1 == boolx4(false, false, true, true)));
    }
    {
        const v0 = f32x8(2.0, 1.0, 2.0, 1.0, 0.5, -2.0, -1.0, 1.9);
        const bounds = f32x8(1.0, 1.0, 1.0, math.inf(f32), 1.0, math.nan(f32), 1.0, 2.0);
        const b0 = isInBounds(v0, bounds);
        try expect(@reduce(.And, b0 == boolx8(false, true, false, true, true, false, true, true)));
    }
}

test "zmath.andInt" {
    {
        const v0 = f32x4(0, @as(f32, @bitCast(~@as(u32, 0))), 0, @as(f32, @bitCast(~@as(u32, 0))));
        const v1 = f32x4(1.0, 2.0, 3.0, math.inf(f32));
        const v = andInt(v0, v1);
        try expect(v[3] == math.inf(f32));
        try expectVecEqual(v, f32x4(0.0, 2.0, 0.0, math.inf(f32)));
    }
    {
        const v0 = f32x8(0, 0, 0, 0, 0, @as(f32, @bitCast(~@as(u32, 0))), 0, @as(f32, @bitCast(~@as(u32, 0))));
        const v1 = f32x8(0, 0, 0, 0, 1.0, 2.0, 3.0, math.inf(f32));
        const v = andInt(v0, v1);
        try expect(v[7] == math.inf(f32));
        try expectVecEqual(v, f32x8(0, 0, 0, 0, 0.0, 2.0, 0.0, math.inf(f32)));
    }
}

test "zmath.andNotInt" {
    {
        const v0 = f32x4(1.0, 2.0, 3.0, 4.0);
        const v1 = f32x4(0, @as(f32, @bitCast(~@as(u32, 0))), 0, @as(f32, @bitCast(~@as(u32, 0))));
        const v = andNotInt(v1, v0);
        try expectVecEqual(v, f32x4(1.0, 0.0, 3.0, 0.0));
    }
    {
        const v0 = f32x8(0, 0, 0, 0, 1.0, 2.0, 3.0, 4.0);
        const v1 = f32x8(0, 0, 0, 0, 0, @as(f32, @bitCast(~@as(u32, 0))), 0, @as(f32, @bitCast(~@as(u32, 0))));
        const v = andNotInt(v1, v0);
        try expectVecEqual(v, f32x8(0, 0, 0, 0, 1.0, 0.0, 3.0, 0.0));
    }
}

test "zmath.orInt" {
    {
        const v0 = f32x4(0, @as(f32, @bitCast(~@as(u32, 0))), 0, 0);
        const v1 = f32x4(1.0, 2.0, 3.0, 4.0);
        const v = orInt(v0, v1);
        try expect(v[0] == 1.0);
        try expect(@as(u32, @bitCast(v[1])) == ~@as(u32, 0));
        try expect(v[2] == 3.0);
        try expect(v[3] == 4.0);
    }
    {
        const v0 = f32x8(0, 0, 0, 0, 0, @as(f32, @bitCast(~@as(u32, 0))), 0, 0);
        const v1 = f32x8(0, 0, 0, 0, 1.0, 2.0, 3.0, 4.0);
        const v = orInt(v0, v1);
        try expect(v[4] == 1.0);
        try expect(@as(u32, @bitCast(v[5])) == ~@as(u32, 0));
        try expect(v[6] == 3.0);
        try expect(v[7] == 4.0);
    }
}

test "zmath.xorInt" {
    {
        const v0 = f32x4(1.0, @as(f32, @bitCast(~@as(u32, 0))), 0, 0);
        const v1 = f32x4(1.0, 0, 0, 0);
        const v = xorInt(v0, v1);
        try expect(v[0] == 0.0);
        try expect(@as(u32, @bitCast(v[1])) == ~@as(u32, 0));
        try expect(v[2] == 0.0);
        try expect(v[3] == 0.0);
    }
    {
        const v0 = f32x8(0, 0, 0, 0, 1.0, @as(f32, @bitCast(~@as(u32, 0))), 0, 0);
        const v1 = f32x8(0, 0, 0, 0, 1.0, 0, 0, 0);
        const v = xorInt(v0, v1);
        try expect(v[4] == 0.0);
        try expect(@as(u32, @bitCast(v[5])) == ~@as(u32, 0));
        try expect(v[6] == 0.0);
        try expect(v[7] == 0.0);
    }
}

test "zmath.minFast" {
    {
        const v0 = f32x4(1.0, 3.0, 2.0, 7.0);
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = minFast(v0, v1);
        try expectVecEqual(v, f32x4(1.0, 1.0, 2.0, 7.0));
    }
    {
        const v0 = f32x4(1.0, math.nan(f32), 5.0, math.snan(f32));
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = minFast(v0, v1);
        try expect(v[0] == 1.0);
        try expect(v[1] == 1.0);
        try expect(!math.isNan(v[1]));
        try expect(v[2] == 4.0);
        try expect(v[3] == math.inf(f32));
        try expect(!math.isNan(v[3]));
    }
}

test "zmath.maxFast" {
    {
        const v0 = f32x4(1.0, 3.0, 2.0, 7.0);
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = maxFast(v0, v1);
        try expectVecEqual(v, f32x4(2.0, 3.0, 4.0, math.inf(f32)));
    }
    {
        const v0 = f32x4(1.0, math.nan(f32), 5.0, math.snan(f32));
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = maxFast(v0, v1);
        try expect(v[0] == 2.0);
        try expect(v[1] == 1.0);
        try expect(v[2] == 5.0);
        try expect(v[3] == math.inf(f32));
        try expect(!math.isNan(v[3]));
    }
}

test "zmath.min" {
    // Calling math.inf causes test to fail!
    if (builtin.target.os.tag == .macos and builtin.target.cpu.arch == .aarch64) return error.SkipZigTest;
    {
        const v0 = f32x4(1.0, 3.0, 2.0, 7.0);
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = min(v0, v1);
        try expectVecEqual(v, f32x4(1.0, 1.0, 2.0, 7.0));
    }
    {
        const v0 = f32x8(0, 0, -2.0, 0, 1.0, 3.0, 2.0, 7.0);
        const v1 = f32x8(0, 1.0, 0, 0, 2.0, 1.0, 4.0, math.inf(f32));
        const v = min(v0, v1);
        try expectVecEqual(v, f32x8(0.0, 0.0, -2.0, 0.0, 1.0, 1.0, 2.0, 7.0));
    }
    {
        const v0 = f32x4(1.0, math.nan(f32), 5.0, math.snan(f32));
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = min(v0, v1);
        try expect(v[0] == 1.0);
        try expect(v[1] == 1.0);
        try expect(!math.isNan(v[1]));
        try expect(v[2] == 4.0);
        try expect(v[3] == math.inf(f32));
        try expect(!math.isNan(v[3]));
    }

    {
        const v0 = f32x4(-math.inf(f32), math.inf(f32), math.inf(f32), math.snan(f32));
        const v1 = f32x4(math.snan(f32), -math.inf(f32), math.snan(f32), math.nan(f32));
        const v = min(v0, v1);
        try expect(v[0] == -math.inf(f32));
        try expect(v[1] == -math.inf(f32));
        try expect(v[2] == math.inf(f32));
        try expect(!math.isNan(v[2]));
        try expect(math.isNan(v[3]));
        try expect(!math.isInf(v[3]));
    }
}

test "zmath.max" {
    // Calling math.inf causes test to fail!
    if (builtin.target.os.tag == .macos and builtin.target.cpu.arch == .aarch64) return error.SkipZigTest;
    {
        const v0 = f32x4(1.0, 3.0, 2.0, 7.0);
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = max(v0, v1);
        try expectVecEqual(v, f32x4(2.0, 3.0, 4.0, math.inf(f32)));
    }
    {
        const v0 = f32x8(0, 0, -2.0, 0, 1.0, 3.0, 2.0, 7.0);
        const v1 = f32x8(0, 1.0, 0, 0, 2.0, 1.0, 4.0, math.inf(f32));
        const v = max(v0, v1);
        try expectVecEqual(v, f32x8(0.0, 1.0, 0.0, 0.0, 2.0, 3.0, 4.0, math.inf(f32)));
    }
    {
        const v0 = f32x4(1.0, math.nan(f32), 5.0, math.snan(f32));
        const v1 = f32x4(2.0, 1.0, 4.0, math.inf(f32));
        const v = max(v0, v1);
        try expect(v[0] == 2.0);
        try expect(v[1] == 1.0);
        try expect(v[2] == 5.0);
        try expect(v[3] == math.inf(f32));
        try expect(!math.isNan(v[3]));
    }
    {
        const v0 = f32x4(-math.inf(f32), math.inf(f32), math.inf(f32), math.snan(f32));
        const v1 = f32x4(math.snan(f32), -math.inf(f32), math.snan(f32), math.nan(f32));
        const v = max(v0, v1);
        try expect(v[0] == -math.inf(f32));
        try expect(v[1] == math.inf(f32));
        try expect(v[2] == math.inf(f32));
        try expect(!math.isNan(v[2]));
        try expect(math.isNan(v[3]));
        try expect(!math.isInf(v[3]));
    }
}

test "zmath.round" {
    {
        try expect(all(round(splat(F32x4, math.inf(f32))) == splat(F32x4, math.inf(f32)), 0));
        try expect(all(round(splat(F32x4, -math.inf(f32))) == splat(F32x4, -math.inf(f32)), 0));
        try expect(all(isNan(round(splat(F32x4, math.nan(f32)))), 0));
        try expect(all(isNan(round(splat(F32x4, -math.nan(f32)))), 0));
        try expect(all(isNan(round(splat(F32x4, math.snan(f32)))), 0));
        try expect(all(isNan(round(splat(F32x4, -math.snan(f32)))), 0));
    }
    {
        const v = round(f32x16(1.1, -1.1, -1.5, 1.5, 2.1, 2.8, 2.9, 4.1, 5.8, 6.1, 7.9, 8.9, 10.1, 11.2, 12.7, 13.1));
        try expectVecApproxEqAbs(
            v,
            f32x16(1.0, -1.0, -2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 6.0, 6.0, 8.0, 9.0, 10.0, 11.0, 13.0, 13.0),
            0.0,
        );
    }
    var v = round(f32x4(1.1, -1.1, -1.5, 1.5));
    try expectVecEqual(v, f32x4(1.0, -1.0, -2.0, 2.0));

    const v1 = f32x4(-10_000_000.1, -math.inf(f32), 10_000_001.5, math.inf(f32));
    v = round(v1);
    try expect(v[3] == math.inf(f32));
    try expectVecEqual(v, f32x4(-10_000_000.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));

    const v2 = f32x4(-math.snan(f32), math.snan(f32), math.nan(f32), -math.inf(f32));
    v = round(v2);
    try expect(math.isNan(v2[0]));
    try expect(math.isNan(v2[1]));
    try expect(math.isNan(v2[2]));
    try expect(v2[3] == -math.inf(f32));

    const v3 = f32x4(1001.5, -201.499, -10000.99, -101.5);
    v = round(v3);
    try expectVecEqual(v, f32x4(1002.0, -201.0, -10001.0, -102.0));

    const v4 = f32x4(-1_388_609.9, 1_388_609.5, 1_388_109.01, 2_388_609.5);
    v = round(v4);
    try expectVecEqual(v, f32x4(-1_388_610.0, 1_388_610.0, 1_388_109.0, 2_388_610.0));

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const vr = round(splat(F32x4, f));
        const fr = @round(splat(F32x4, f));
        const vr8 = round(splat(F32x8, f));
        const fr8 = @round(splat(F32x8, f));
        const vr16 = round(splat(F32x16, f));
        const fr16 = @round(splat(F32x16, f));
        try expectVecEqual(vr, fr);
        try expectVecEqual(vr8, fr8);
        try expectVecEqual(vr16, fr16);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.trunc" {
    {
        try expect(all(trunc(splat(F32x4, math.inf(f32))) == splat(F32x4, math.inf(f32)), 0));
        try expect(all(trunc(splat(F32x4, -math.inf(f32))) == splat(F32x4, -math.inf(f32)), 0));
        try expect(all(isNan(trunc(splat(F32x4, math.nan(f32)))), 0));
        try expect(all(isNan(trunc(splat(F32x4, -math.nan(f32)))), 0));
        try expect(all(isNan(trunc(splat(F32x4, math.snan(f32)))), 0));
        try expect(all(isNan(trunc(splat(F32x4, -math.snan(f32)))), 0));
    }
    {
        const v = trunc(f32x16(1.1, -1.1, -1.5, 1.5, 2.1, 2.8, 2.9, 4.1, 5.8, 6.1, 7.9, 8.9, 10.1, 11.2, 12.7, 13.1));
        try expectVecApproxEqAbs(
            v,
            f32x16(1.0, -1.0, -1.0, 1.0, 2.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0),
            0.0,
        );
    }
    var v = trunc(f32x4(1.1, -1.1, -1.5, 1.5));
    try expectVecEqual(v, f32x4(1.0, -1.0, -1.0, 1.0));

    v = trunc(f32x4(-10_000_002.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));
    try expectVecEqual(v, f32x4(-10_000_002.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));

    v = trunc(f32x4(-math.snan(f32), math.snan(f32), math.nan(f32), -math.inf(f32)));
    try expect(math.isNan(v[0]));
    try expect(math.isNan(v[1]));
    try expect(math.isNan(v[2]));
    try expect(v[3] == -math.inf(f32));

    v = trunc(f32x4(1000.5001, -201.499, -10000.99, 100.750001));
    try expectVecEqual(v, f32x4(1000.0, -201.0, -10000.0, 100.0));

    v = trunc(f32x4(-7_388_609.5, 7_388_609.1, 8_388_109.5, -8_388_509.5));
    try expectVecEqual(v, f32x4(-7_388_609.0, 7_388_609.0, 8_388_109.0, -8_388_509.0));

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const vr = trunc(splat(F32x4, f));
        const fr = @trunc(splat(F32x4, f));
        const vr8 = trunc(splat(F32x8, f));
        const fr8 = @trunc(splat(F32x8, f));
        const vr16 = trunc(splat(F32x16, f));
        const fr16 = @trunc(splat(F32x16, f));
        try expectVecEqual(vr, fr);
        try expectVecEqual(vr8, fr8);
        try expectVecEqual(vr16, fr16);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.floor" {
    {
        try expect(all(floor(splat(F32x4, math.inf(f32))) == splat(F32x4, math.inf(f32)), 0));
        try expect(all(floor(splat(F32x4, -math.inf(f32))) == splat(F32x4, -math.inf(f32)), 0));
        try expect(all(isNan(floor(splat(F32x4, math.nan(f32)))), 0));
        try expect(all(isNan(floor(splat(F32x4, -math.nan(f32)))), 0));
        try expect(all(isNan(floor(splat(F32x4, math.snan(f32)))), 0));
        try expect(all(isNan(floor(splat(F32x4, -math.snan(f32)))), 0));
    }
    {
        const v = floor(f32x16(1.1, -1.1, -1.5, 1.5, 2.1, 2.8, 2.9, 4.1, 5.8, 6.1, 7.9, 8.9, 10.1, 11.2, 12.7, 13.1));
        try expectVecApproxEqAbs(
            v,
            f32x16(1.0, -2.0, -2.0, 1.0, 2.0, 2.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 11.0, 12.0, 13.0),
            0.0,
        );
    }
    var v = floor(f32x4(1.5, -1.5, -1.7, -2.1));
    try expectVecEqual(v, f32x4(1.0, -2.0, -2.0, -3.0));

    v = floor(f32x4(-10_000_002.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));
    try expectVecEqual(v, f32x4(-10_000_002.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));

    v = floor(f32x4(-math.snan(f32), math.snan(f32), math.nan(f32), -math.inf(f32)));
    try expect(math.isNan(v[0]));
    try expect(math.isNan(v[1]));
    try expect(math.isNan(v[2]));
    try expect(v[3] == -math.inf(f32));

    v = floor(f32x4(1000.5001, -201.499, -10000.99, 100.75001));
    try expectVecEqual(v, f32x4(1000.0, -202.0, -10001.0, 100.0));

    v = floor(f32x4(-7_388_609.5, 7_388_609.1, 8_388_109.5, -8_388_509.5));
    try expectVecEqual(v, f32x4(-7_388_610.0, 7_388_609.0, 8_388_109.0, -8_388_510.0));

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const vr = floor(splat(F32x4, f));
        const fr = @floor(splat(F32x4, f));
        const vr8 = floor(splat(F32x8, f));
        const fr8 = @floor(splat(F32x8, f));
        const vr16 = floor(splat(F32x16, f));
        const fr16 = @floor(splat(F32x16, f));
        try expectVecEqual(vr, fr);
        try expectVecEqual(vr8, fr8);
        try expectVecEqual(vr16, fr16);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.ceil" {
    {
        try expect(all(ceil(splat(F32x4, math.inf(f32))) == splat(F32x4, math.inf(f32)), 0));
        try expect(all(ceil(splat(F32x4, -math.inf(f32))) == splat(F32x4, -math.inf(f32)), 0));
        try expect(all(isNan(ceil(splat(F32x4, math.nan(f32)))), 0));
        try expect(all(isNan(ceil(splat(F32x4, -math.nan(f32)))), 0));
        try expect(all(isNan(ceil(splat(F32x4, math.snan(f32)))), 0));
        try expect(all(isNan(ceil(splat(F32x4, -math.snan(f32)))), 0));
    }
    {
        const v = ceil(f32x16(1.1, -1.1, -1.5, 1.5, 2.1, 2.8, 2.9, 4.1, 5.8, 6.1, 7.9, 8.9, 10.1, 11.2, 12.7, 13.1));
        try expectVecApproxEqAbs(
            v,
            f32x16(2.0, -1.0, -1.0, 2.0, 3.0, 3.0, 3.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0, 12.0, 13.0, 14.0),
            0.0,
        );
    }
    var v = ceil(f32x4(1.5, -1.5, -1.7, -2.1));
    try expectVecEqual(v, f32x4(2.0, -1.0, -1.0, -2.0));

    v = ceil(f32x4(-10_000_002.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));
    try expectVecEqual(v, f32x4(-10_000_002.1, -math.inf(f32), 10_000_001.5, math.inf(f32)));

    v = ceil(f32x4(-math.snan(f32), math.snan(f32), math.nan(f32), -math.inf(f32)));
    try expect(math.isNan(v[0]));
    try expect(math.isNan(v[1]));
    try expect(math.isNan(v[2]));
    try expect(v[3] == -math.inf(f32));

    v = ceil(f32x4(1000.5001, -201.499, -10000.99, 100.75001));
    try expectVecEqual(v, f32x4(1001.0, -201.0, -10000.0, 101.0));

    v = ceil(f32x4(-1_388_609.5, 1_388_609.1, 1_388_109.9, -1_388_509.9));
    try expectVecEqual(v, f32x4(-1_388_609.0, 1_388_610.0, 1_388_110.0, -1_388_509.0));

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const vr = ceil(splat(F32x4, f));
        const fr = @ceil(splat(F32x4, f));
        const vr8 = ceil(splat(F32x8, f));
        const fr8 = @ceil(splat(F32x8, f));
        const vr16 = ceil(splat(F32x16, f));
        const fr16 = @ceil(splat(F32x16, f));
        try expectVecEqual(vr, fr);
        try expectVecEqual(vr8, fr8);
        try expectVecEqual(vr16, fr16);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.clamp" {
    // Calling math.inf causes test to fail!
    if (builtin.target.os.tag == .macos and builtin.target.cpu.arch == .aarch64) return error.SkipZigTest;
    {
        const v0 = f32x4(-1.0, 0.2, 1.1, -0.3);
        const v = clamp(v0, splat(F32x4, -0.5), splat(F32x4, 0.5));
        try expectVecApproxEqAbs(v, f32x4(-0.5, 0.2, 0.5, -0.3), 0.0001);
    }
    {
        const v0 = f32x8(-2.0, 0.25, -0.25, 100.0, -1.0, 0.2, 1.1, -0.3);
        const v = clamp(v0, splat(F32x8, -0.5), splat(F32x8, 0.5));
        try expectVecApproxEqAbs(v, f32x8(-0.5, 0.25, -0.25, 0.5, -0.5, 0.2, 0.5, -0.3), 0.0001);
    }
    {
        const v0 = f32x4(-math.inf(f32), math.inf(f32), math.nan(f32), math.snan(f32));
        const v = clamp(v0, f32x4(-100.0, 0.0, -100.0, 0.0), f32x4(0.0, 100.0, 0.0, 100.0));
        try expectVecApproxEqAbs(v, f32x4(-100.0, 100.0, -100.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(math.inf(f32), math.inf(f32), -math.nan(f32), -math.snan(f32));
        const v = clamp(v0, splat(F32x4, -1.0), splat(F32x4, 1.0));
        try expectVecApproxEqAbs(v, f32x4(1.0, 1.0, -1.0, -1.0), 0.0001);
    }
}

test "zmath.clampFast" {
    {
        const v0 = f32x4(-1.0, 0.2, 1.1, -0.3);
        const v = clampFast(v0, splat(F32x4, -0.5), splat(F32x4, 0.5));
        try expectVecApproxEqAbs(v, f32x4(-0.5, 0.2, 0.5, -0.3), 0.0001);
    }
}

test "zmath.saturate" {
    // Calling math.inf causes test to fail!
    if (builtin.target.os.tag == .macos and builtin.target.cpu.arch == .aarch64) return error.SkipZigTest;
    {
        const v0 = f32x4(-1.0, 0.2, 1.1, -0.3);
        const v = saturate(v0);
        try expectVecApproxEqAbs(v, f32x4(0.0, 0.2, 1.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x8(0.0, 0.0, 2.0, -2.0, -1.0, 0.2, 1.1, -0.3);
        const v = saturate(v0);
        try expectVecApproxEqAbs(v, f32x8(0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 1.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(-math.inf(f32), math.inf(f32), math.nan(f32), math.snan(f32));
        const v = saturate(v0);
        try expectVecApproxEqAbs(v, f32x4(0.0, 1.0, 0.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(math.inf(f32), math.inf(f32), -math.nan(f32), -math.snan(f32));
        const v = saturate(v0);
        try expectVecApproxEqAbs(v, f32x4(1.0, 1.0, 0.0, 0.0), 0.0001);
    }
}

test "zmath.saturateFast" {
    {
        const v0 = f32x4(-1.0, 0.2, 1.1, -0.3);
        const v = saturateFast(v0);
        try expectVecApproxEqAbs(v, f32x4(0.0, 0.2, 1.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x8(0.0, 0.0, 2.0, -2.0, -1.0, 0.2, 1.1, -0.3);
        const v = saturateFast(v0);
        try expectVecApproxEqAbs(v, f32x8(0.0, 0.0, 1.0, 0.0, 0.0, 0.2, 1.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(-math.inf(f32), math.inf(f32), math.nan(f32), math.snan(f32));
        const v = saturateFast(v0);
        try expectVecApproxEqAbs(v, f32x4(0.0, 1.0, 0.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(math.inf(f32), math.inf(f32), -math.nan(f32), -math.snan(f32));
        const v = saturateFast(v0);
        try expectVecApproxEqAbs(v, f32x4(1.0, 1.0, 0.0, 0.0), 0.0001);
    }
}

test "zmath.lerpInverse" {
    try expect(approxEqAbs(lerpInverseV(10.0, 100.0, 10.0), 0, 0.0005));
    try expect(approxEqAbs(lerpInverseV(10.0, 100.0, 100.0), 1, 0.0005));
    try expect(approxEqAbs(lerpInverseV(10.0, 100.0, 55.0), 0.5, 0.05));
    try expectVecApproxEqAbs(lerpInverse(f32x4(0, 0, 10, 10), f32x4(100, 200, 100, 100), 10.0), f32x4(0.1, 0.05, 0, 0), 0.0005);
}

test "zmath.lerpOverTime" {
    try expect(approxEqAbs(lerpVOverTime(0.0, 1.0, 1.0, 1.0), 0.5, 0.0005));
    try expect(approxEqAbs(lerpVOverTime(0.5, 1.0, 1.0, 1.0), 0.75, 0.0005));
    try expect(approxEqAbs(lerpVOverTime(0.0, 1.0, 1.0, 0.0), 0.0, 0.0005));
    try expect(approxEqAbs(lerpVOverTime(0.0, 1.0, 1.0, std.math.inf(f32)), 1.0, 0.0005));
    try expectVecApproxEqAbs(lerpOverTime(f32x4(0, 0, 10, 10), f32x4(100, 200, 100, 100), 1.0, 1.0), f32x4(50, 100, 55, 55), 0.0005);
}

test "zmath.mapLinear" {
    try expect(approxEqAbs(mapLinearV(0, 0, 1.2, 10, 100), 10, 0.0005));
    try expect(approxEqAbs(mapLinearV(1.2, 0, 1.2, 10, 100), 100, 0.0005));
    try expect(approxEqAbs(mapLinearV(0.6, 0, 1.2, 10, 100), 55, 0.0005));
    try expectVecApproxEqAbs(mapLinearV(splat(F32x4, 0), splat(F32x4, 0), splat(F32x4, 1.2), splat(F32x4, 10), splat(F32x4, 100)), splat(F32x4, 10), 0.0005);
    try expectVecApproxEqAbs(mapLinear(f32x4(0, 0, 0.6, 1.2), 0, 1.2, 10, 100), f32x4(10, 10, 55, 100), 0.0005);
}

test "zmath.mod" {
    try expectVecApproxEqAbs(mod(splat(F32x4, 3.1), splat(F32x4, 1.7)), splat(F32x4, 1.4), 0.0005);
    try expectVecApproxEqAbs(mod(splat(F32x4, -3.0), splat(F32x4, 2.0)), splat(F32x4, -1.0), 0.0005);
    try expectVecApproxEqAbs(mod(splat(F32x4, -3.0), splat(F32x4, -2.0)), splat(F32x4, -1.0), 0.0005);
    try expectVecApproxEqAbs(mod(splat(F32x4, 3.0), splat(F32x4, -2.0)), splat(F32x4, 1.0), 0.0005);
    try expect(all(isNan(mod(splat(F32x4, math.inf(f32)), splat(F32x4, 1.0))), 0));
    try expect(all(isNan(mod(splat(F32x4, -math.inf(f32)), splat(F32x4, 123.456))), 0));
    try expect(all(isNan(mod(splat(F32x4, math.nan(f32)), splat(F32x4, 123.456))), 0));
    try expect(all(isNan(mod(splat(F32x4, math.snan(f32)), splat(F32x4, 123.456))), 0));
    try expect(all(isNan(mod(splat(F32x4, -math.snan(f32)), splat(F32x4, 123.456))), 0));
    try expect(all(isNan(mod(splat(F32x4, 123.456), splat(F32x4, math.inf(f32)))), 0));
    try expect(all(isNan(mod(splat(F32x4, 123.456), splat(F32x4, -math.inf(f32)))), 0));
    try expect(all(isNan(mod(splat(F32x4, math.inf(f32)), splat(F32x4, math.inf(f32)))), 0));
    try expect(all(isNan(mod(splat(F32x4, 123.456), splat(F32x4, math.nan(f32)))), 0));
    try expect(all(isNan(mod(splat(F32x4, math.inf(f32)), splat(F32x4, math.nan(f32)))), 0));
}

test "zmath.modAngle" {
    try expectVecApproxEqAbs(modAngle(splat(F32x4, math.tau)), splat(F32x4, 0.0), 0.0005);
    try expectVecApproxEqAbs(modAngle(splat(F32x4, 0.0)), splat(F32x4, 0.0), 0.0005);
    try expectVecApproxEqAbs(modAngle(splat(F32x4, math.pi)), splat(F32x4, math.pi), 0.0005);
    try expectVecApproxEqAbs(modAngle(splat(F32x4, 11 * math.pi)), splat(F32x4, math.pi), 0.0005);
    try expectVecApproxEqAbs(modAngle(splat(F32x4, 3.5 * math.pi)), splat(F32x4, -0.5 * math.pi), 0.0005);
    try expectVecApproxEqAbs(modAngle(splat(F32x4, 2.5 * math.pi)), splat(F32x4, 0.5 * math.pi), 0.0005);
}

test "zmath.sin" {
    const epsilon = 0.0001;

    try expectVecApproxEqAbs(sin(splat(F32x4, 0.5 * math.pi)), splat(F32x4, 1.0), epsilon);
    try expectVecApproxEqAbs(sin(splat(F32x4, 0.0)), splat(F32x4, 0.0), epsilon);
    try expectVecApproxEqAbs(sin(splat(F32x4, -0.0)), splat(F32x4, -0.0), epsilon);
    try expectVecApproxEqAbs(sin(splat(F32x4, 89.123)), splat(F32x4, 0.916166), epsilon);
    try expectVecApproxEqAbs(sin(splat(F32x8, 89.123)), splat(F32x8, 0.916166), epsilon);
    try expectVecApproxEqAbs(sin(splat(F32x16, 89.123)), splat(F32x16, 0.916166), epsilon);
    try expect(all(isNan(sin(splat(F32x4, math.inf(f32)))), 0) == true);
    try expect(all(isNan(sin(splat(F32x4, -math.inf(f32)))), 0) == true);
    try expect(all(isNan(sin(splat(F32x4, math.nan(f32)))), 0) == true);
    try expect(all(isNan(sin(splat(F32x4, math.snan(f32)))), 0) == true);

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const vr = sin(splat(F32x4, f));
        const fr = @sin(splat(F32x4, f));
        const vr8 = sin(splat(F32x8, f));
        const fr8 = @sin(splat(F32x8, f));
        const vr16 = sin(splat(F32x16, f));
        const fr16 = @sin(splat(F32x16, f));
        try expectVecApproxEqAbs(vr, fr, epsilon);
        try expectVecApproxEqAbs(vr8, fr8, epsilon);
        try expectVecApproxEqAbs(vr16, fr16, epsilon);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.cos" {
    const epsilon = 0.0001;

    try expectVecApproxEqAbs(cos(splat(F32x4, 0.5 * math.pi)), splat(F32x4, 0.0), epsilon);
    try expectVecApproxEqAbs(cos(splat(F32x4, 0.0)), splat(F32x4, 1.0), epsilon);
    try expectVecApproxEqAbs(cos(splat(F32x4, -0.0)), splat(F32x4, 1.0), epsilon);
    try expect(all(isNan(cos(splat(F32x4, math.inf(f32)))), 0) == true);
    try expect(all(isNan(cos(splat(F32x4, -math.inf(f32)))), 0) == true);
    try expect(all(isNan(cos(splat(F32x4, math.nan(f32)))), 0) == true);
    try expect(all(isNan(cos(splat(F32x4, math.snan(f32)))), 0) == true);

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const vr = cos(splat(F32x4, f));
        const fr = @cos(splat(F32x4, f));
        const vr8 = cos(splat(F32x8, f));
        const fr8 = @cos(splat(F32x8, f));
        const vr16 = cos(splat(F32x16, f));
        const fr16 = @cos(splat(F32x16, f));
        try expectVecApproxEqAbs(vr, fr, epsilon);
        try expectVecApproxEqAbs(vr8, fr8, epsilon);
        try expectVecApproxEqAbs(vr16, fr16, epsilon);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.sincos32xN" {
    const epsilon = 0.0001;

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const sc = sincos(splat(F32x4, f));
        const sc8 = sincos(splat(F32x8, f));
        const sc16 = sincos(splat(F32x16, f));
        const s4 = @sin(splat(F32x4, f));
        const s8 = @sin(splat(F32x8, f));
        const s16 = @sin(splat(F32x16, f));
        const c4 = @cos(splat(F32x4, f));
        const c8 = @cos(splat(F32x8, f));
        const c16 = @cos(splat(F32x16, f));
        try expectVecApproxEqAbs(sc[0], s4, epsilon);
        try expectVecApproxEqAbs(sc8[0], s8, epsilon);
        try expectVecApproxEqAbs(sc16[0], s16, epsilon);
        try expectVecApproxEqAbs(sc[1], c4, epsilon);
        try expectVecApproxEqAbs(sc8[1], c8, epsilon);
        try expectVecApproxEqAbs(sc16[1], c16, epsilon);
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.atan" {
    const epsilon = 0.0001;
    {
        const v = f32x4(0.25, 0.5, 1.0, 1.25);
        const e = f32x4(math.atan(v[0]), math.atan(v[1]), math.atan(v[2]), math.atan(v[3]));
        try expectVecApproxEqAbs(e, atan(v), epsilon);
    }
    {
        const v = f32x8(-0.25, 0.5, -1.0, 1.25, 100.0, -200.0, 300.0, 400.0);
        // zig fmt: off
        const e = f32x8(
            math.atan(v[0]), math.atan(v[1]), math.atan(v[2]), math.atan(v[3]),
            math.atan(v[4]), math.atan(v[5]), math.atan(v[6]), math.atan(v[7]),
        );
        // zig fmt: on
        try expectVecApproxEqAbs(e, atan(v), epsilon);
    }
    {
        // zig fmt: off
        const v = f32x16(
            -0.25, 0.5, -1.0, 0.0, 0.1, -0.2, 30.0, 400.0,
            -0.25, 0.5, -1.0, -0.0, -0.05, -0.125, 0.0625, 4000.0
        );
        const e = f32x16(
            math.atan(v[0]), math.atan(v[1]), math.atan(v[2]), math.atan(v[3]),
            math.atan(v[4]), math.atan(v[5]), math.atan(v[6]), math.atan(v[7]),
            math.atan(v[8]), math.atan(v[9]), math.atan(v[10]), math.atan(v[11]),
            math.atan(v[12]), math.atan(v[13]), math.atan(v[14]), math.atan(v[15]),
        );
        // zig fmt: on
        try expectVecApproxEqAbs(e, atan(v), epsilon);
    }
    {
        try expectVecApproxEqAbs(atan(splat(F32x4, math.inf(f32))), splat(F32x4, 0.5 * math.pi), epsilon);
        try expectVecApproxEqAbs(atan(splat(F32x4, -math.inf(f32))), splat(F32x4, -0.5 * math.pi), epsilon);
        try expect(all(isNan(atan(splat(F32x4, math.nan(f32)))), 0) == true);
        try expect(all(isNan(atan(splat(F32x4, -math.nan(f32)))), 0) == true);
    }
}

test "zmath.atan2" {
    // From DirectXMath XMVectorATan2():
    //
    // Return the inverse tangent of Y / X in the range of -Pi to Pi with the following exceptions:

    //     Y == 0 and X is Negative         -> Pi with the sign of Y
    //     y == 0 and x is positive         -> 0 with the sign of y
    //     Y != 0 and X == 0                -> Pi / 2 with the sign of Y
    //     Y != 0 and X is Negative         -> atan(y/x) + (PI with the sign of Y)
    //     X == -Infinity and Finite Y      -> Pi with the sign of Y
    //     X == +Infinity and Finite Y      -> 0 with the sign of Y
    //     Y == Infinity and X is Finite    -> Pi / 2 with the sign of Y
    //     Y == Infinity and X == -Infinity -> 3Pi / 4 with the sign of Y
    //     Y == Infinity and X == +Infinity -> Pi / 4 with the sign of Y

    const epsilon = 0.0001;
    try expectVecApproxEqAbs(atan2(splat(F32x4, 0.0), splat(F32x4, -1.0)), splat(F32x4, math.pi), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, -0.0), splat(F32x4, -1.0)), splat(F32x4, -math.pi), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, 1.0), splat(F32x4, 0.0)), splat(F32x4, 0.5 * math.pi), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, -1.0), splat(F32x4, 0.0)), splat(F32x4, -0.5 * math.pi), epsilon);
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, 1.0), splat(F32x4, -1.0)),
        splat(F32x4, math.atan(@as(f32, -1.0)) + math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, -10.0), splat(F32x4, -2.0)),
        splat(F32x4, math.atan(@as(f32, 5.0)) - math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(atan2(splat(F32x4, 1.0), splat(F32x4, -math.inf(f32))), splat(F32x4, math.pi), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, -1.0), splat(F32x4, -math.inf(f32))), splat(F32x4, -math.pi), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, 1.0), splat(F32x4, math.inf(f32))), splat(F32x4, 0.0), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, -1.0), splat(F32x4, math.inf(f32))), splat(F32x4, -0.0), epsilon);
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, math.inf(f32)), splat(F32x4, 2.0)),
        splat(F32x4, 0.5 * math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, -math.inf(f32)), splat(F32x4, 2.0)),
        splat(F32x4, -0.5 * math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, math.inf(f32)), splat(F32x4, -math.inf(f32))),
        splat(F32x4, 0.75 * math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, -math.inf(f32)), splat(F32x4, -math.inf(f32))),
        splat(F32x4, -0.75 * math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, math.inf(f32)), splat(F32x4, math.inf(f32))),
        splat(F32x4, 0.25 * math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(splat(F32x4, -math.inf(f32)), splat(F32x4, math.inf(f32))),
        splat(F32x4, -0.25 * math.pi),
        epsilon,
    );
    try expectVecApproxEqAbs(
        atan2(
            f32x8(0.0, -math.inf(f32), -0.0, 2.0, math.inf(f32), math.inf(f32), 1.0, -math.inf(f32)),
            f32x8(-2.0, math.inf(f32), 1.0, 0.0, 10.0, -math.inf(f32), 1.0, -math.inf(f32)),
        ),
        f32x8(
            math.pi,
            -0.25 * math.pi,
            -0.0,
            0.5 * math.pi,
            0.5 * math.pi,
            0.75 * math.pi,
            math.atan(@as(f32, 1.0)),
            -0.75 * math.pi,
        ),
        epsilon,
    );
    try expectVecApproxEqAbs(atan2(splat(F32x4, 0.0), splat(F32x4, 0.0)), splat(F32x4, 0.0), epsilon);
    try expectVecApproxEqAbs(atan2(splat(F32x4, -0.0), splat(F32x4, 0.0)), splat(F32x4, 0.0), epsilon);
    try expect(all(isNan(atan2(splat(F32x4, 1.0), splat(F32x4, math.nan(f32)))), 0) == true);
    try expect(all(isNan(atan2(splat(F32x4, -1.0), splat(F32x4, math.nan(f32)))), 0) == true);
    try expect(all(isNan(atan2(splat(F32x4, math.nan(f32)), splat(F32x4, -1.0))), 0) == true);
    try expect(all(isNan(atan2(splat(F32x4, -math.nan(f32)), splat(F32x4, 1.0))), 0) == true);
}

test "zmath.dot2" {
    const v0 = f32x4(-1.0, 2.0, 300.0, -2.0);
    const v1 = f32x4(4.0, 5.0, 600.0, 2.0);
    const v = dot2(v0, v1);
    try expectVecApproxEqAbs(v, splat(F32x4, 6.0), 0.0001);
}

test "zmath.dot3" {
    const v0 = f32x4(-1.0, 2.0, 3.0, 1.0);
    const v1 = f32x4(4.0, 5.0, 6.0, 1.0);
    const v = dot3(v0, v1);
    try expectVecApproxEqAbs(v, splat(F32x4, 24.0), 0.0001);
}

test "zmath.dot4" {
    const v0 = f32x4(-1.0, 2.0, 3.0, -2.0);
    const v1 = f32x4(4.0, 5.0, 6.0, 2.0);
    const v = dot4(v0, v1);
    try expectVecApproxEqAbs(v, splat(F32x4, 20.0), 0.0001);
}

test "zmath.cross3" {
    {
        const v0 = f32x4(1.0, 0.0, 0.0, 1.0);
        const v1 = f32x4(0.0, 1.0, 0.0, 1.0);
        const v = cross3(v0, v1);
        try expectVecApproxEqAbs(v, f32x4(0.0, 0.0, 1.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(1.0, 0.0, 0.0, 1.0);
        const v1 = f32x4(0.0, -1.0, 0.0, 1.0);
        const v = cross3(v0, v1);
        try expectVecApproxEqAbs(v, f32x4(0.0, 0.0, -1.0, 0.0), 0.0001);
    }
    {
        const v0 = f32x4(-3.0, 0, -2.0, 1.0);
        const v1 = f32x4(5.0, -1.0, 2.0, 1.0);
        const v = cross3(v0, v1);
        try expectVecApproxEqAbs(v, f32x4(-2.0, -4.0, 3.0, 0.0), 0.0001);
    }
}

test "zmath.length3" {
    {
        const v = length3(f32x4(1.0, -2.0, 3.0, 1000.0));
        try expectVecApproxEqAbs(v, splat(F32x4, math.sqrt(14.0)), 0.001);
    }
    {
        const v = length3(f32x4(1.0, math.nan(f32), math.nan(f32), 1000.0));
        try expect(all(isNan(v), 0));
    }
    {
        const v = length3(f32x4(1.0, math.inf(f32), 3.0, 1000.0));
        try expect(all(isInf(v), 0));
    }
    {
        const v = length3(f32x4(3.0, 2.0, 1.0, math.nan(f32)));
        try expectVecApproxEqAbs(v, splat(F32x4, math.sqrt(14.0)), 0.001);
    }
}

test "zmath.normalize3" {
    {
        const v0 = f32x4(1.0, -2.0, 3.0, 1000.0);
        const v = normalize3(v0);
        try expectVecApproxEqAbs(v, v0 * splat(F32x4, 1.0 / math.sqrt(14.0)), 0.0005);
    }
    {
        try expect(any(isNan(normalize3(f32x4(1.0, math.inf(f32), 1.0, 1.0))), 0));
        try expect(any(isNan(normalize3(f32x4(-math.inf(f32), math.inf(f32), 0.0, 0.0))), 0));
        try expect(any(isNan(normalize3(f32x4(-math.nan(f32), math.snan(f32), 0.0, 0.0))), 0));
        try expect(any(isNan(normalize3(f32x4(0, 0, 0, 0))), 0));
    }
}

test "zmath.normalize4" {
    {
        const v0 = f32x4(1.0, -2.0, 3.0, 10.0);
        const v = normalize4(v0);
        try expectVecApproxEqAbs(v, v0 * splat(F32x4, 1.0 / math.sqrt(114.0)), 0.0005);
    }
    {
        try expect(any(isNan(normalize4(f32x4(1.0, math.inf(f32), 1.0, 1.0))), 0));
        try expect(any(isNan(normalize4(f32x4(-math.inf(f32), math.inf(f32), 0.0, 0.0))), 0));
        try expect(any(isNan(normalize4(f32x4(-math.nan(f32), math.snan(f32), 0.0, 0.0))), 0));
        try expect(any(isNan(normalize4(f32x4(0, 0, 0, 0))), 0));
    }
}

test "zmath.linePointDistance" {
    {
        const linept0 = f32x4(-1.0, -2.0, -3.0, 1.0);
        const linept1 = f32x4(1.0, 2.0, 3.0, 1.0);
        const pt = f32x4(1.0, 1.0, 1.0, 1.0);
        const v = linePointDistance(linept0, linept1, pt);
        try expectVecApproxEqAbs(v, splat(F32x4, 0.654), 0.001);
    }
}

test "zmath.sincos32" {
    const epsilon = 0.0001;

    try expect(math.isNan(sincos32(math.inf(f32))[0]));
    try expect(math.isNan(sincos32(math.inf(f32))[1]));
    try expect(math.isNan(sincos32(-math.inf(f32))[0]));
    try expect(math.isNan(sincos32(-math.inf(f32))[1]));
    try expect(math.isNan(sincos32(math.nan(f32))[0]));
    try expect(math.isNan(sincos32(-math.nan(f32))[1]));

    try expect(math.isNan(sin32(math.inf(f32))));
    try expect(math.isNan(cos32(math.inf(f32))));
    try expect(math.isNan(sin32(-math.inf(f32))));
    try expect(math.isNan(cos32(-math.inf(f32))));
    try expect(math.isNan(sin32(math.nan(f32))));
    try expect(math.isNan(cos32(-math.nan(f32))));

    var f: f32 = -100.0;
    var i: u32 = 0;
    while (i < 100) : (i += 1) {
        const sc = sincos32(f);
        const s0 = sin32(f);
        const c0 = cos32(f);
        const s = @sin(f);
        const c = @cos(f);
        try expect(approxEqAbs(sc[0], s, epsilon));
        try expect(approxEqAbs(sc[1], c, epsilon));
        try expect(approxEqAbs(s0, s, epsilon));
        try expect(approxEqAbs(c0, c, epsilon));
        f += 0.12345 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.asin32" {
    const epsilon = 0.0001;

    try expect(approxEqAbs(asin(@as(f32, -1.1)), -0.5 * math.pi, epsilon));
    try expect(approxEqAbs(asin(@as(f32, 1.1)), 0.5 * math.pi, epsilon));
    try expect(approxEqAbs(asin(@as(f32, -1000.1)), -0.5 * math.pi, epsilon));
    try expect(approxEqAbs(asin(@as(f32, 100000.1)), 0.5 * math.pi, epsilon));
    try expect(math.isNan(asin(math.inf(f32))));
    try expect(math.isNan(asin(-math.inf(f32))));
    try expect(math.isNan(asin(math.nan(f32))));
    try expect(math.isNan(asin(-math.nan(f32))));

    try expectVecApproxEqAbs(asin(splat(F32x8, -100.0)), splat(F32x8, -0.5 * math.pi), epsilon);
    try expectVecApproxEqAbs(asin(splat(F32x16, 100.0)), splat(F32x16, 0.5 * math.pi), epsilon);
    try expect(all(isNan(asin(splat(F32x4, math.inf(f32)))), 0) == true);
    try expect(all(isNan(asin(splat(F32x4, -math.inf(f32)))), 0) == true);
    try expect(all(isNan(asin(splat(F32x4, math.nan(f32)))), 0) == true);
    try expect(all(isNan(asin(splat(F32x4, math.snan(f32)))), 0) == true);

    var f: f32 = -1.0;
    var i: u32 = 0;
    while (i < 8) : (i += 1) {
        const r0 = asin32(f);
        const r1 = math.asin(f);
        const r4 = asin(splat(F32x4, f));
        const r8 = asin(splat(F32x8, f));
        const r16 = asin(splat(F32x16, f));
        try expect(approxEqAbs(r0, r1, epsilon));
        try expectVecApproxEqAbs(r4, splat(F32x4, r1), epsilon);
        try expectVecApproxEqAbs(r8, splat(F32x8, r1), epsilon);
        try expectVecApproxEqAbs(r16, splat(F32x16, r1), epsilon);
        f += 0.09 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.acos32" {
    const epsilon = 0.1;

    try expect(approxEqAbs(acos(@as(f32, -1.1)), math.pi, epsilon));
    try expect(approxEqAbs(acos(@as(f32, -10000.1)), math.pi, epsilon));
    try expect(approxEqAbs(acos(@as(f32, 1.1)), 0.0, epsilon));
    try expect(approxEqAbs(acos(@as(f32, 1000.1)), 0.0, epsilon));
    try expect(math.isNan(acos(math.inf(f32))));
    try expect(math.isNan(acos(-math.inf(f32))));
    try expect(math.isNan(acos(math.nan(f32))));
    try expect(math.isNan(acos(-math.nan(f32))));

    try expectVecApproxEqAbs(acos(splat(F32x8, -100.0)), splat(F32x8, math.pi), epsilon);
    try expectVecApproxEqAbs(acos(splat(F32x16, 100.0)), splat(F32x16, 0.0), epsilon);
    try expect(all(isNan(acos(splat(F32x4, math.inf(f32)))), 0) == true);
    try expect(all(isNan(acos(splat(F32x4, -math.inf(f32)))), 0) == true);
    try expect(all(isNan(acos(splat(F32x4, math.nan(f32)))), 0) == true);
    try expect(all(isNan(acos(splat(F32x4, math.snan(f32)))), 0) == true);

    var f: f32 = -1.0;
    var i: u32 = 0;
    while (i < 8) : (i += 1) {
        const r0 = acos32(f);
        const r1 = math.acos(f);
        const r4 = acos(splat(F32x4, f));
        const r8 = acos(splat(F32x8, f));
        const r16 = acos(splat(F32x16, f));
        try expect(approxEqAbs(r0, r1, epsilon));
        try expectVecApproxEqAbs(r4, splat(F32x4, r1), epsilon);
        try expectVecApproxEqAbs(r8, splat(F32x8, r1), epsilon);
        try expectVecApproxEqAbs(r16, splat(F32x16, r1), epsilon);
        f += 0.09 * @as(f32, @floatFromInt(i));
    }
}

test "zmath.floatToIntAndBack" {
    {
        const v = floatToIntAndBack(f32x4(1.1, 2.9, 3.0, -4.5));
        try expectVecEqual(v, f32x4(1.0, 2.0, 3.0, -4.0));
    }
    {
        const v = floatToIntAndBack(f32x8(1.1, 2.9, 3.0, -4.5, 2.5, -2.5, 1.1, -100.2));
        try expectVecEqual(v, f32x8(1.0, 2.0, 3.0, -4.0, 2.0, -2.0, 1.0, -100.0));
    }
    {
        const v = floatToIntAndBack(f32x4(math.inf(f32), 2.9, math.nan(f32), math.snan(f32)));
        try expect(v[1] == 2.0);
    }
}
