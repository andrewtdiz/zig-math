const std = @import("std");
const builtin = @import("builtin");
const math = std.math;

const core = @import("core.zig");

const F32x4 = core.F32x4;
const F32x8 = core.F32x8;
const F32x16 = core.F32x16;
const Boolx4 = core.Boolx4;
const Boolx8 = core.Boolx8;
const Boolx16 = core.Boolx16;
const Vec = core.Vec;
const Mat = core.Mat;
const Quat = core.Quat;

const cpu_arch = core.cpu_arch;
const has_avx = core.has_avx;
const has_avx512f = core.has_avx512f;
const has_fma = core.has_fma;

const veclen = core.veclen;
const splat = core.splat;
const splatInt = core.splatInt;
const f32x4 = core.f32x4;
const f32x4s = core.f32x4s;
const f32x8 = core.f32x8;
const f32x8s = core.f32x8s;
const f32x16 = core.f32x16;
const f32x16s = core.f32x16s;
const boolx4 = core.boolx4;
const boolx8 = core.boolx8;
const boolx16 = core.boolx16;
const load = core.load;
const loadArr3 = core.loadArr3;
const store = core.store;
const approxEqAbs = core.approxEqAbs;
const expect = std.testing.expect;

// 2. Functions that work on all vector components (F32xN = F32x4 or F32x8 or F32x16)
//
// ------------------------------------------------------------------------------
pub fn all(vb: anytype, comptime len: u32) bool {
    const T = @TypeOf(vb);
    if (len > veclen(T)) {
        @compileError("zmath.all(): 'len' is greater than vector len of type " ++ @typeName(T));
    }
    const loop_len = if (len == 0) veclen(T) else len;
    const ab: [veclen(T)]bool = vb;
    comptime var i: u32 = 0;
    var result = true;
    inline while (i < loop_len) : (i += 1) {
        result = result and ab[i];
    }
    return result;
}

pub fn any(vb: anytype, comptime len: u32) bool {
    const T = @TypeOf(vb);
    if (len > veclen(T)) {
        @compileError("zmath.any(): 'len' is greater than vector len of type " ++ @typeName(T));
    }
    const loop_len = if (len == 0) veclen(T) else len;
    const ab: [veclen(T)]bool = vb;
    comptime var i: u32 = 0;
    var result = false;
    inline while (i < loop_len) : (i += 1) {
        result = result or ab[i];
    }
    return result;
}

pub inline fn isNearEqual(
    v0: anytype,
    v1: anytype,
    epsilon: anytype,
) @Vector(veclen(@TypeOf(v0)), bool) {
    const T = @TypeOf(v0, v1, epsilon);
    const delta = v0 - v1;
    const temp = maxFast(delta, splat(T, 0.0) - delta);
    return temp <= epsilon;
}

pub inline fn isNan(
    v: anytype,
) @Vector(veclen(@TypeOf(v)), bool) {
    return v != v;
}

pub inline fn isInf(
    v: anytype,
) @Vector(veclen(@TypeOf(v)), bool) {
    const T = @TypeOf(v);
    return abs(v) == splat(T, math.inf(f32));
}

pub inline fn isInBounds(
    v: anytype,
    bounds: anytype,
) @Vector(veclen(@TypeOf(v)), bool) {
    const T = @TypeOf(v, bounds);
    const Tu = @Vector(veclen(T), u1);
    const Tr = @Vector(veclen(T), bool);

    // 2 x cmpleps, xorps, load, andps
    const b0 = v <= bounds;
    const b1 = (bounds * splat(T, -1.0)) <= v;
    const b0u = @as(Tu, @bitCast(b0));
    const b1u = @as(Tu, @bitCast(b1));
    return @as(Tr, @bitCast(b0u & b1u));
}

pub inline fn andInt(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = @Vector(veclen(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(v0u & v1u)); // andps
}

pub inline fn andNotInt(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = @Vector(veclen(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(~v0u & v1u)); // andnps
}

pub inline fn orInt(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = @Vector(veclen(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(v0u | v1u)); // orps
}

pub inline fn norInt(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = @Vector(veclen(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(~(v0u | v1u))); // por, pcmpeqd, pxor
}

pub inline fn xorInt(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    const Tu = @Vector(veclen(T), u32);
    const v0u = @as(Tu, @bitCast(v0));
    const v1u = @as(Tu, @bitCast(v1));
    return @as(T, @bitCast(v0u ^ v1u)); // xorps
}

pub inline fn minFast(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    return select(v0 < v1, v0, v1); // minps
}

pub inline fn maxFast(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    return select(v0 > v1, v0, v1); // maxps
}

pub inline fn min(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    // This will handle inf & nan
    return @min(v0, v1); // minps, cmpunordps, andps, andnps, orps
}

pub inline fn max(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    // This will handle inf & nan
    return @max(v0, v1); // maxps, cmpunordps, andps, andnps, orps
}

pub fn round(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (cpu_arch == .x86_64 and has_avx) {
        if (T == F32x4) {
            return asm ("vroundps $0, %%xmm0, %%xmm0"
                : [ret] "={xmm0}" (-> T),
                : [v] "{xmm0}" (v),
            );
        } else if (T == F32x8) {
            return asm ("vroundps $0, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> T),
                : [v] "{ymm0}" (v),
            );
        } else if (T == F32x16 and has_avx512f) {
            return asm ("vrndscaleps $0, %%zmm0, %%zmm0"
                : [ret] "={zmm0}" (-> T),
                : [v] "{zmm0}" (v),
            );
        } else if (T == F32x16 and !has_avx512f) {
            const arr: [16]f32 = v;
            var ymm0 = @as(F32x8, arr[0..8].*);
            var ymm1 = @as(F32x8, arr[8..16].*);
            ymm0 = asm ("vroundps $0, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> F32x8),
                : [v] "{ymm0}" (ymm0),
            );
            ymm1 = asm ("vroundps $0, %%ymm1, %%ymm1"
                : [ret] "={ymm1}" (-> F32x8),
                : [v] "{ymm1}" (ymm1),
            );
            return @shuffle(f32, ymm0, ymm1, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
        }
    } else {
        const sign = andInt(v, splatNegativeZero(T));
        const magic = orInt(splatNoFraction(T), sign);
        var r1 = v + magic;
        r1 = r1 - magic;
        const r2 = abs(v);
        const mask = r2 <= splatNoFraction(T);
        return select(mask, r1, v);
    }
}

pub fn trunc(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (cpu_arch == .x86_64 and has_avx) {
        if (T == F32x4) {
            return asm ("vroundps $3, %%xmm0, %%xmm0"
                : [ret] "={xmm0}" (-> T),
                : [v] "{xmm0}" (v),
            );
        } else if (T == F32x8) {
            return asm ("vroundps $3, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> T),
                : [v] "{ymm0}" (v),
            );
        } else if (T == F32x16 and has_avx512f) {
            return asm ("vrndscaleps $3, %%zmm0, %%zmm0"
                : [ret] "={zmm0}" (-> T),
                : [v] "{zmm0}" (v),
            );
        } else if (T == F32x16 and !has_avx512f) {
            const arr: [16]f32 = v;
            var ymm0 = @as(F32x8, arr[0..8].*);
            var ymm1 = @as(F32x8, arr[8..16].*);
            ymm0 = asm ("vroundps $3, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> F32x8),
                : [v] "{ymm0}" (ymm0),
            );
            ymm1 = asm ("vroundps $3, %%ymm1, %%ymm1"
                : [ret] "={ymm1}" (-> F32x8),
                : [v] "{ymm1}" (ymm1),
            );
            return @shuffle(f32, ymm0, ymm1, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
        }
    } else {
        const mask = abs(v) < splatNoFraction(T);
        const result = floatToIntAndBack(v);
        return select(mask, result, v);
    }
}

pub fn floor(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (cpu_arch == .x86_64 and has_avx) {
        if (T == F32x4) {
            return asm ("vroundps $1, %%xmm0, %%xmm0"
                : [ret] "={xmm0}" (-> T),
                : [v] "{xmm0}" (v),
            );
        } else if (T == F32x8) {
            return asm ("vroundps $1, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> T),
                : [v] "{ymm0}" (v),
            );
        } else if (T == F32x16 and has_avx512f) {
            return asm ("vrndscaleps $1, %%zmm0, %%zmm0"
                : [ret] "={zmm0}" (-> T),
                : [v] "{zmm0}" (v),
            );
        } else if (T == F32x16 and !has_avx512f) {
            const arr: [16]f32 = v;
            var ymm0 = @as(F32x8, arr[0..8].*);
            var ymm1 = @as(F32x8, arr[8..16].*);
            ymm0 = asm ("vroundps $1, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> F32x8),
                : [v] "{ymm0}" (ymm0),
            );
            ymm1 = asm ("vroundps $1, %%ymm1, %%ymm1"
                : [ret] "={ymm1}" (-> F32x8),
                : [v] "{ymm1}" (ymm1),
            );
            return @shuffle(f32, ymm0, ymm1, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
        }
    } else {
        const mask = abs(v) < splatNoFraction(T);
        var result = floatToIntAndBack(v);
        const larger_mask = result > v;
        const larger = select(larger_mask, splat(T, -1.0), splat(T, 0.0));
        result = result + larger;
        return select(mask, result, v);
    }
}

pub fn ceil(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    if (cpu_arch == .x86_64 and has_avx) {
        if (T == F32x4) {
            return asm ("vroundps $2, %%xmm0, %%xmm0"
                : [ret] "={xmm0}" (-> T),
                : [v] "{xmm0}" (v),
            );
        } else if (T == F32x8) {
            return asm ("vroundps $2, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> T),
                : [v] "{ymm0}" (v),
            );
        } else if (T == F32x16 and has_avx512f) {
            return asm ("vrndscaleps $2, %%zmm0, %%zmm0"
                : [ret] "={zmm0}" (-> T),
                : [v] "{zmm0}" (v),
            );
        } else if (T == F32x16 and !has_avx512f) {
            const arr: [16]f32 = v;
            var ymm0 = @as(F32x8, arr[0..8].*);
            var ymm1 = @as(F32x8, arr[8..16].*);
            ymm0 = asm ("vroundps $2, %%ymm0, %%ymm0"
                : [ret] "={ymm0}" (-> F32x8),
                : [v] "{ymm0}" (ymm0),
            );
            ymm1 = asm ("vroundps $2, %%ymm1, %%ymm1"
                : [ret] "={ymm1}" (-> F32x8),
                : [v] "{ymm1}" (ymm1),
            );
            return @shuffle(f32, ymm0, ymm1, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 });
        }
    } else {
        const mask = abs(v) < splatNoFraction(T);
        var result = floatToIntAndBack(v);
        const smaller_mask = result < v;
        const smaller = select(smaller_mask, splat(T, -1.0), splat(T, 0.0));
        result = result - smaller;
        return select(mask, result, v);
    }
}

pub inline fn clamp(v: anytype, vmin: anytype, vmax: anytype) @TypeOf(v, vmin, vmax) {
    var result = max(vmin, v);
    result = min(vmax, result);
    return result;
}

pub inline fn clampFast(v: anytype, vmin: anytype, vmax: anytype) @TypeOf(v, vmin, vmax) {
    var result = maxFast(vmin, v);
    result = minFast(vmax, result);
    return result;
}

pub inline fn saturate(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    var result = max(v, splat(T, 0.0));
    result = min(result, splat(T, 1.0));
    return result;
}

pub inline fn saturateFast(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    var result = maxFast(v, splat(T, 0.0));
    result = minFast(result, splat(T, 1.0));
    return result;
}

pub inline fn sqrt(v: anytype) @TypeOf(v) {
    return @sqrt(v); // sqrtps
}

pub inline fn abs(v: anytype) @TypeOf(v) {
    return @abs(v); // load, andps
}

pub inline fn select(mask: anytype, v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    return @select(f32, mask, v0, v1);
}

pub inline fn lerp(v0: anytype, v1: anytype, t: f32) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    return v0 + (v1 - v0) * splat(T, t); // subps, shufps, addps, mulps
}

pub inline fn lerpV(v0: anytype, v1: anytype, t: anytype) @TypeOf(v0, v1, t) {
    return v0 + (v1 - v0) * t; // subps, addps, mulps
}

pub inline fn lerpInverse(v0: anytype, v1: anytype, t: anytype) @TypeOf(v0, v1) {
    const T = @TypeOf(v0, v1);
    return (splat(T, t) - v0) / (v1 - v0);
}

pub inline fn lerpInverseV(v0: anytype, v1: anytype, t: anytype) @TypeOf(v0, v1, t) {
    return (t - v0) / (v1 - v0);
}

// Frame rate independent lerp (or "damp"), for approaching things over time.
// Reference: https://www.gamedeveloper.com/programming/improved-lerp-smoothing-
pub inline fn lerpOverTime(v0: anytype, v1: anytype, rate: anytype, dt: anytype) @TypeOf(v0, v1) {
    const t = std.math.exp2(-rate * dt);
    return lerp(v1, v0, t);
}

pub inline fn lerpVOverTime(v0: anytype, v1: anytype, rate: anytype, dt: anytype) @TypeOf(v0, v1, rate, dt) {
    const t = std.math.exp2(-rate * dt);
    return lerpV(v1, v0, t);
}

/// To transform a vector of values from one range to another.
pub inline fn mapLinear(v: anytype, min1: anytype, max1: anytype, min2: anytype, max2: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    const min1V = splat(T, min1);
    const max1V = splat(T, max1);
    const min2V = splat(T, min2);
    const max2V = splat(T, max2);
    const dV = max1V - min1V;
    return min2V + (v - min1V) * (max2V - min2V) / dV;
}

pub inline fn mapLinearV(v: anytype, min1: anytype, max1: anytype, min2: anytype, max2: anytype) @TypeOf(v, min1, max1, min2, max2) {
    const d = max1 - min1;
    return min2 + (v - min1) * (max2 - min2) / d;
}

pub const F32x4Component = enum { x, y, z, w };

pub inline fn swizzle(
    v: F32x4,
    comptime x: F32x4Component,
    comptime y: F32x4Component,
    comptime z: F32x4Component,
    comptime w: F32x4Component,
) F32x4 {
    return @shuffle(f32, v, undefined, [4]i32{ @intFromEnum(x), @intFromEnum(y), @intFromEnum(z), @intFromEnum(w) });
}

pub inline fn mod(v0: anytype, v1: anytype) @TypeOf(v0, v1) {
    // vdivps, vroundps, vmulps, vsubps
    return v0 - v1 * trunc(v0 / v1);
}

pub fn modAngle(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return switch (T) {
        f32 => modAngle32(v),
        F32x4, F32x8, F32x16 => modAngle32xN(v),
        else => @compileError("zmath.modAngle() not implemented for " ++ @typeName(T)),
    };
}

pub inline fn modAngle32xN(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return v - splat(T, math.tau) * round(v * splat(T, 1.0 / math.tau)); // 2 x vmulps, 2 x load, vroundps, vaddps
}

pub inline fn mulAdd(v0: anytype, v1: anytype, v2: anytype) @TypeOf(v0, v1, v2) {
    const T = @TypeOf(v0, v1, v2);
    if (@import("zmath_options").enable_cross_platform_determinism) {
        return v0 * v1 + v2; // Compiler will generate mul, add sequence (no fma even if the target supports it).
    } else {
        if (cpu_arch == .x86_64 and has_avx and has_fma) {
            return @mulAdd(T, v0, v1, v2);
        } else {
            // NOTE(mziulek): On .x86_64 without HW fma instructions @mulAdd maps to really slow code!
            return v0 * v1 + v2;
        }
    }
}

fn sin32xN(v: anytype) @TypeOf(v) {
    // 11-degree minimax approximation
    const T = @TypeOf(v);

    var x = modAngle(v);
    const sign = andInt(x, splatNegativeZero(T));
    const c = orInt(sign, splat(T, math.pi));
    const absx = andNotInt(sign, x);
    const rflx = c - x;
    const comp = absx <= splat(T, 0.5 * math.pi);
    x = select(comp, x, rflx);
    const x2 = x * x;

    var result = mulAdd(splat(T, -2.3889859e-08), x2, splat(T, 2.7525562e-06));
    result = mulAdd(result, x2, splat(T, -0.00019840874));
    result = mulAdd(result, x2, splat(T, 0.0083333310));
    result = mulAdd(result, x2, splat(T, -0.16666667));
    result = mulAdd(result, x2, splat(T, 1.0));
    return x * result;
}

fn cos32xN(v: anytype) @TypeOf(v) {
    // 10-degree minimax approximation
    const T = @TypeOf(v);

    var x = modAngle(v);
    var sign = andInt(x, splatNegativeZero(T));
    const c = orInt(sign, splat(T, math.pi));
    const absx = andNotInt(sign, x);
    const rflx = c - x;
    const comp = absx <= splat(T, 0.5 * math.pi);
    x = select(comp, x, rflx);
    sign = select(comp, splat(T, 1.0), splat(T, -1.0));
    const x2 = x * x;

    var result = mulAdd(splat(T, -2.6051615e-07), x2, splat(T, 2.4760495e-05));
    result = mulAdd(result, x2, splat(T, -0.0013888378));
    result = mulAdd(result, x2, splat(T, 0.041666638));
    result = mulAdd(result, x2, splat(T, -0.5));
    result = mulAdd(result, x2, splat(T, 1.0));
    return sign * result;
}

pub fn sin(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return switch (T) {
        f32 => sin32(v),
        F32x4, F32x8, F32x16 => sin32xN(v),
        else => @compileError("zmath.sin() not implemented for " ++ @typeName(T)),
    };
}

pub fn cos(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return switch (T) {
        f32 => cos32(v),
        F32x4, F32x8, F32x16 => cos32xN(v),
        else => @compileError("zmath.cos() not implemented for " ++ @typeName(T)),
    };
}

pub fn sincos(v: anytype) [2]@TypeOf(v) {
    const T = @TypeOf(v);
    return switch (T) {
        f32 => sincos32(v),
        F32x4, F32x8, F32x16 => sincos32xN(v),
        else => @compileError("zmath.sincos() not implemented for " ++ @typeName(T)),
    };
}

pub fn asin(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return switch (T) {
        f32 => asin32(v),
        F32x4, F32x8, F32x16 => asin32xN(v),
        else => @compileError("zmath.asin() not implemented for " ++ @typeName(T)),
    };
}

pub fn acos(v: anytype) @TypeOf(v) {
    const T = @TypeOf(v);
    return switch (T) {
        f32 => acos32(v),
        F32x4, F32x8, F32x16 => acos32xN(v),
        else => @compileError("zmath.acos() not implemented for " ++ @typeName(T)),
    };
}

fn sincos32xN(v: anytype) [2]@TypeOf(v) {
    const T = @TypeOf(v);

    var x = modAngle(v);
    var sign = andInt(x, splatNegativeZero(T));
    const c = orInt(sign, splat(T, math.pi));
    const absx = andNotInt(sign, x);
    const rflx = c - x;
    const comp = absx <= splat(T, 0.5 * math.pi);
    x = select(comp, x, rflx);
    sign = select(comp, splat(T, 1.0), splat(T, -1.0));
    const x2 = x * x;

    var sresult = mulAdd(splat(T, -2.3889859e-08), x2, splat(T, 2.7525562e-06));
    sresult = mulAdd(sresult, x2, splat(T, -0.00019840874));
    sresult = mulAdd(sresult, x2, splat(T, 0.0083333310));
    sresult = mulAdd(sresult, x2, splat(T, -0.16666667));
    sresult = x * mulAdd(sresult, x2, splat(T, 1.0));

    var cresult = mulAdd(splat(T, -2.6051615e-07), x2, splat(T, 2.4760495e-05));
    cresult = mulAdd(cresult, x2, splat(T, -0.0013888378));
    cresult = mulAdd(cresult, x2, splat(T, 0.041666638));
    cresult = mulAdd(cresult, x2, splat(T, -0.5));
    cresult = sign * mulAdd(cresult, x2, splat(T, 1.0));

    return .{ sresult, cresult };
}

fn asin32xN(v: anytype) @TypeOf(v) {
    // 7-degree minimax approximation
    const T = @TypeOf(v);

    const x = abs(v);
    const root = sqrt(maxFast(splat(T, 0.0), splat(T, 1.0) - x));

    var t0 = mulAdd(splat(T, -0.0012624911), x, splat(T, 0.0066700901));
    t0 = mulAdd(t0, x, splat(T, -0.0170881256));
    t0 = mulAdd(t0, x, splat(T, 0.0308918810));
    t0 = mulAdd(t0, x, splat(T, -0.0501743046));
    t0 = mulAdd(t0, x, splat(T, 0.0889789874));
    t0 = mulAdd(t0, x, splat(T, -0.2145988016));
    t0 = root * mulAdd(t0, x, splat(T, 1.5707963050));

    const t1 = splat(T, math.pi) - t0;
    return splat(T, 0.5 * math.pi) - select(v >= splat(T, 0.0), t0, t1);
}

fn acos32xN(v: anytype) @TypeOf(v) {
    // 7-degree minimax approximation
    const T = @TypeOf(v);

    const x = abs(v);
    const root = sqrt(maxFast(splat(T, 0.0), splat(T, 1.0) - x));

    var t0 = mulAdd(splat(T, -0.0012624911), x, splat(T, 0.0066700901));
    t0 = mulAdd(t0, x, splat(T, -0.0170881256));
    t0 = mulAdd(t0, x, splat(T, 0.0308918810));
    t0 = mulAdd(t0, x, splat(T, -0.0501743046));
    t0 = mulAdd(t0, x, splat(T, 0.0889789874));
    t0 = mulAdd(t0, x, splat(T, -0.2145988016));
    t0 = root * mulAdd(t0, x, splat(T, 1.5707963050));

    const t1 = splat(T, math.pi) - t0;
    return select(v >= splat(T, 0.0), t0, t1);
}

pub fn atan(v: anytype) @TypeOf(v) {
    // 17-degree minimax approximation
    const T = @TypeOf(v);

    const vabs = abs(v);
    const vinv = splat(T, 1.0) / v;
    var sign = select(v > splat(T, 1.0), splat(T, 1.0), splat(T, -1.0));
    const comp = vabs <= splat(T, 1.0);
    sign = select(comp, splat(T, 0.0), sign);
    const x = select(comp, v, vinv);
    const x2 = x * x;

    var result = mulAdd(splat(T, 0.0028662257), x2, splat(T, -0.0161657367));
    result = mulAdd(result, x2, splat(T, 0.0429096138));
    result = mulAdd(result, x2, splat(T, -0.0752896400));
    result = mulAdd(result, x2, splat(T, 0.1065626393));
    result = mulAdd(result, x2, splat(T, -0.1420889944));
    result = mulAdd(result, x2, splat(T, 0.1999355085));
    result = mulAdd(result, x2, splat(T, -0.3333314528));
    result = x * mulAdd(result, x2, splat(T, 1.0));

    const result1 = sign * splat(T, 0.5 * math.pi) - result;
    return select(sign == splat(T, 0.0), result, result1);
}

pub fn atan2(vy: anytype, vx: anytype) @TypeOf(vx, vy) {
    const T = @TypeOf(vx, vy);
    const Tu = @Vector(veclen(T), u32);

    const vx_is_positive =
        (@as(Tu, @bitCast(vx)) & @as(Tu, @splat(0x8000_0000))) == @as(Tu, @splat(0));

    const vy_sign = andInt(vy, splatNegativeZero(T));
    const c0_25pi = orInt(vy_sign, @as(T, @splat(0.25 * math.pi)));
    const c0_50pi = orInt(vy_sign, @as(T, @splat(0.50 * math.pi)));
    const c0_75pi = orInt(vy_sign, @as(T, @splat(0.75 * math.pi)));
    const c1_00pi = orInt(vy_sign, @as(T, @splat(1.00 * math.pi)));

    var r1 = select(vx_is_positive, vy_sign, c1_00pi);
    var r2 = select(vx == splat(T, 0.0), c0_50pi, splatInt(T, 0xffff_ffff));
    const r3 = select(vy == splat(T, 0.0), r1, r2);
    const r4 = select(vx_is_positive, c0_25pi, c0_75pi);
    const r5 = select(isInf(vx), r4, c0_50pi);
    const result = select(isInf(vy), r5, r3);
    const result_valid = @as(Tu, @bitCast(result)) == @as(Tu, @splat(0xffff_ffff));

    const v = vy / vx;
    const r0 = atan(v);

    r1 = select(vx_is_positive, splatNegativeZero(T), c1_00pi);
    r2 = r0 + r1;

    return select(result_valid, r2, result);
}

// ------------------------------------------------------------------------------
//
// 3. 2D, 3D, 4D vector functions
//
// ------------------------------------------------------------------------------
pub inline fn dot2(v0: Vec, v1: Vec) F32x4 {
    var xmm0 = v0 * v1; // | x0*x1 | y0*y1 | -- | -- |
    const xmm1 = swizzle(xmm0, .y, .x, .x, .x); // | y0*y1 | -- | -- | -- |
    xmm0 = f32x4(xmm0[0] + xmm1[0], xmm0[1], xmm0[2], xmm0[3]); // | x0*x1 + y0*y1 | -- | -- | -- |
    return swizzle(xmm0, .x, .x, .x, .x);
}

pub inline fn dot3(v0: Vec, v1: Vec) F32x4 {
    const dot = v0 * v1;
    return f32x4s(dot[0] + dot[1] + dot[2]);
}

pub inline fn dot4(v0: Vec, v1: Vec) F32x4 {
    var xmm0 = v0 * v1; // | x0*x1 | y0*y1 | z0*z1 | w0*w1 |
    var xmm1 = swizzle(xmm0, .y, .x, .w, .x); // | y0*y1 | -- | w0*w1 | -- |
    xmm1 = xmm0 + xmm1; // | x0*x1 + y0*y1 | -- | z0*z1 + w0*w1 | -- |
    xmm0 = swizzle(xmm1, .z, .x, .x, .x); // | z0*z1 + w0*w1 | -- | -- | -- |
    xmm0 = f32x4(xmm0[0] + xmm1[0], xmm0[1], xmm0[2], xmm0[2]); // addss
    return swizzle(xmm0, .x, .x, .x, .x);
}

pub inline fn cross3(v0: Vec, v1: Vec) Vec {
    var xmm0 = swizzle(v0, .y, .z, .x, .w);
    var xmm1 = swizzle(v1, .z, .x, .y, .w);
    var result = xmm0 * xmm1;
    xmm0 = swizzle(xmm0, .y, .z, .x, .w);
    xmm1 = swizzle(xmm1, .z, .x, .y, .w);
    result = result - xmm0 * xmm1;
    return andInt(result, f32x4_mask3);
}

pub inline fn lengthSq2(v: Vec) F32x4 {
    return dot2(v, v);
}
pub inline fn lengthSq3(v: Vec) F32x4 {
    return dot3(v, v);
}
pub inline fn lengthSq4(v: Vec) F32x4 {
    return dot4(v, v);
}

pub inline fn length2(v: Vec) F32x4 {
    return sqrt(dot2(v, v));
}
pub inline fn length3(v: Vec) F32x4 {
    return sqrt(dot3(v, v));
}
pub inline fn length4(v: Vec) F32x4 {
    return sqrt(dot4(v, v));
}

pub inline fn normalize2(v: Vec) Vec {
    return v / sqrt(dot2(v, v));
}
pub inline fn normalize3(v: Vec) Vec {
    return v / sqrt(dot3(v, v));
}
pub inline fn normalize4(v: Vec) Vec {
    return v / sqrt(dot4(v, v));
}

pub fn linePointDistance(linept0: Vec, linept1: Vec, pt: Vec) F32x4 {
    const ptvec = pt - linept0;
    const linevec = linept1 - linept0;
    const scale = dot3(ptvec, linevec) / lengthSq3(linevec);
    return length3(ptvec - linevec * scale);
}

pub fn sin32(v: f32) f32 {
    var y = v - math.tau * @round(v * 1.0 / math.tau);

    if (y > 0.5 * math.pi) {
        y = math.pi - y;
    } else if (y < -math.pi * 0.5) {
        y = -math.pi - y;
    }
    const y2 = y * y;

    // 11-degree minimax approximation
    var sinv = mulAdd(@as(f32, -2.3889859e-08), y2, 2.7525562e-06);
    sinv = mulAdd(sinv, y2, -0.00019840874);
    sinv = mulAdd(sinv, y2, 0.0083333310);
    sinv = mulAdd(sinv, y2, -0.16666667);
    return y * mulAdd(sinv, y2, 1.0);
}
pub fn cos32(v: f32) f32 {
    var y = v - math.tau * @round(v * 1.0 / math.tau);

    const sign = blk: {
        if (y > 0.5 * math.pi) {
            y = math.pi - y;
            break :blk @as(f32, -1.0);
        } else if (y < -math.pi * 0.5) {
            y = -math.pi - y;
            break :blk @as(f32, -1.0);
        } else {
            break :blk @as(f32, 1.0);
        }
    };
    const y2 = y * y;

    // 10-degree minimax approximation
    var cosv = mulAdd(@as(f32, -2.6051615e-07), y2, 2.4760495e-05);
    cosv = mulAdd(cosv, y2, -0.0013888378);
    cosv = mulAdd(cosv, y2, 0.041666638);
    cosv = mulAdd(cosv, y2, -0.5);
    return sign * mulAdd(cosv, y2, 1.0);
}
pub fn sincos32(v: f32) [2]f32 {
    var y = v - math.tau * @round(v * 1.0 / math.tau);

    const sign = blk: {
        if (y > 0.5 * math.pi) {
            y = math.pi - y;
            break :blk @as(f32, -1.0);
        } else if (y < -math.pi * 0.5) {
            y = -math.pi - y;
            break :blk @as(f32, -1.0);
        } else {
            break :blk @as(f32, 1.0);
        }
    };
    const y2 = y * y;

    // 11-degree minimax approximation
    var sinv = mulAdd(@as(f32, -2.3889859e-08), y2, 2.7525562e-06);
    sinv = mulAdd(sinv, y2, -0.00019840874);
    sinv = mulAdd(sinv, y2, 0.0083333310);
    sinv = mulAdd(sinv, y2, -0.16666667);
    sinv = y * mulAdd(sinv, y2, 1.0);

    // 10-degree minimax approximation
    var cosv = mulAdd(@as(f32, -2.6051615e-07), y2, 2.4760495e-05);
    cosv = mulAdd(cosv, y2, -0.0013888378);
    cosv = mulAdd(cosv, y2, 0.041666638);
    cosv = mulAdd(cosv, y2, -0.5);
    cosv = sign * mulAdd(cosv, y2, 1.0);

    return .{ sinv, cosv };
}

pub fn asin32(v: f32) f32 {
    const x = @abs(v);
    var omx = 1.0 - x;
    if (omx < 0.0) {
        omx = 0.0;
    }
    const root = @sqrt(omx);

    // 7-degree minimax approximation
    var result = mulAdd(@as(f32, -0.0012624911), x, 0.0066700901);
    result = mulAdd(result, x, -0.0170881256);
    result = mulAdd(result, x, 0.0308918810);
    result = mulAdd(result, x, -0.0501743046);
    result = mulAdd(result, x, 0.0889789874);
    result = mulAdd(result, x, -0.2145988016);
    result = root * mulAdd(result, x, 1.5707963050);

    return if (v >= 0.0) 0.5 * math.pi - result else result - 0.5 * math.pi;
}

pub fn acos32(v: f32) f32 {
    const x = @abs(v);
    var omx = 1.0 - x;
    if (omx < 0.0) {
        omx = 0.0;
    }
    const root = @sqrt(omx);

    // 7-degree minimax approximation
    var result = mulAdd(@as(f32, -0.0012624911), x, 0.0066700901);
    result = mulAdd(result, x, -0.0170881256);
    result = mulAdd(result, x, 0.0308918810);
    result = mulAdd(result, x, -0.0501743046);
    result = mulAdd(result, x, 0.0889789874);
    result = mulAdd(result, x, -0.2145988016);
    result = root * mulAdd(result, x, 1.5707963050);

    return if (v >= 0.0) result else math.pi - result;
}

pub fn modAngle32(in_angle: f32) f32 {
    const angle = in_angle + math.pi;
    var temp: f32 = @abs(angle);
    temp = temp - (2.0 * math.pi * @as(f32, @floatFromInt(@as(i32, @intFromFloat(temp / math.pi)))));
    temp = temp - math.pi;
    if (angle < 0.0) {
        temp = -temp;
    }
    return temp;
}

pub fn cmulSoa(re0: anytype, im0: anytype, re1: anytype, im1: anytype) [2]@TypeOf(re0, im0, re1, im1) {
    const re0_re1 = re0 * re1;
    const re0_im1 = re0 * im1;
    return .{
        mulAdd(-im0, im1, re0_re1), // re
        mulAdd(re1, im0, re0_im1), // im
    };
}
// ------------------------------------------------------------------------------
//
// Private helpers shared across vector routines
//
// ------------------------------------------------------------------------------
pub const f32x4_sign_mask1: F32x4 = F32x4{ @as(f32, @bitCast(@as(u32, 0x8000_0000))), 0, 0, 0 };
pub const f32x4_mask2: F32x4 = F32x4{
    @as(f32, @bitCast(@as(u32, 0xffff_ffff))),
    @as(f32, @bitCast(@as(u32, 0xffff_ffff))),
    0,
    0,
};
pub const f32x4_mask3: F32x4 = F32x4{
    @as(f32, @bitCast(@as(u32, 0xffff_ffff))),
    @as(f32, @bitCast(@as(u32, 0xffff_ffff))),
    @as(f32, @bitCast(@as(u32, 0xffff_ffff))),
    0,
};

inline fn splatNegativeZero(comptime T: type) T {
    return @splat(@as(f32, @bitCast(@as(u32, 0x8000_0000))));
}
inline fn splatNoFraction(comptime T: type) T {
    return @splat(@as(f32, 8_388_608.0));
}
inline fn splatAbsMask(comptime T: type) T {
    return @splat(@as(f32, @bitCast(@as(u32, 0x7fff_ffff))));
}

pub fn floatToIntAndBack(v: anytype) @TypeOf(v) {
    // This routine won't handle nan, inf and numbers greater than 8_388_608.0 (will generate undefined values).
    @setRuntimeSafety(false);

    const T = @TypeOf(v);
    const len = veclen(T);

    var vi32: [len]i32 = undefined;
    comptime var i: u32 = 0;
    // vcvttps2dq
    inline while (i < len) : (i += 1) {
        vi32[i] = @as(i32, @intFromFloat(v[i]));
    }

    var vf32: [len]f32 = undefined;
    i = 0;
    // vcvtdq2ps
    inline while (i < len) : (i += 1) {
        vf32[i] = @as(f32, @floatFromInt(vi32[i]));
    }

    return vf32;
}
