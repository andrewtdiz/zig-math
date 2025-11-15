const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const assert = std.debug.assert;
const expect = std.testing.expect;

const core = @import("core.zig");
const vector = @import("vector.zig");

const F32x4 = core.F32x4;
const Boolx4 = core.Boolx4;
const boolx4 = core.boolx4;
const Vec = core.Vec;
const Mat = core.Mat;
const Quat = core.Quat;

const splat = core.splat;
const f32x4 = core.f32x4;
const f32x4s = core.f32x4s;
const f32x8 = core.f32x8;
const f32x16 = core.f32x16;
const load = core.load;
const store = core.store;
const loadArr3 = core.loadArr3;

const mulAdd = vector.mulAdd;
const dot2 = vector.dot2;
const dot3 = vector.dot3;
const dot4 = vector.dot4;
const length3 = vector.length3;
const lengthSq3 = vector.lengthSq3;
const lengthSq4 = vector.lengthSq4;
const length4 = vector.length4;
const normalize3 = vector.normalize3;
const normalize4 = vector.normalize4;
const cross3 = vector.cross3;
const sqrt = vector.sqrt;
const sin = vector.sin;
const cos = vector.cos;
const sincos = vector.sincos;
const asin = vector.asin;
const atan2 = vector.atan2;
const acos = vector.acos;
const saturate = vector.saturate;
const select = vector.select;
const swizzle = vector.swizzle;
const all = vector.all;
const isInf = vector.isInf;
const isNearEqual = vector.isNearEqual;
const andInt = vector.andInt;
const xorInt = vector.xorInt;
const min = vector.min;
const max = vector.max;
const floor = vector.floor;
const mod = vector.mod;
const f32x4_mask2 = vector.f32x4_mask2;
const f32x4_mask3 = vector.f32x4_mask3;
const f32x4_sign_mask1 = vector.f32x4_sign_mask1;

// 4. Matrix functions
//
// ------------------------------------------------------------------------------
pub fn identity() Mat {
    const static = struct {
        const identity = Mat{
            f32x4(1.0, 0.0, 0.0, 0.0),
            f32x4(0.0, 1.0, 0.0, 0.0),
            f32x4(0.0, 0.0, 1.0, 0.0),
            f32x4(0.0, 0.0, 0.0, 1.0),
        };
    };
    return static.identity;
}

pub fn matFromArr(arr: [16]f32) Mat {
    return Mat{
        f32x4(arr[0], arr[1], arr[2], arr[3]),
        f32x4(arr[4], arr[5], arr[6], arr[7]),
        f32x4(arr[8], arr[9], arr[10], arr[11]),
        f32x4(arr[12], arr[13], arr[14], arr[15]),
    };
}

fn mulRetType(comptime Ta: type, comptime Tb: type) type {
    if (Ta == Mat and Tb == Mat) {
        return Mat;
    } else if ((Ta == f32 and Tb == Mat) or (Ta == Mat and Tb == f32)) {
        return Mat;
    } else if ((Ta == Vec and Tb == Mat) or (Ta == Mat and Tb == Vec)) {
        return Vec;
    }
    @compileError("zmath.mul() not implemented for types: " ++ @typeName(Ta) ++ @typeName(Tb));
}

pub fn mul(a: anytype, b: anytype) mulRetType(@TypeOf(a), @TypeOf(b)) {
    const Ta = @TypeOf(a);
    const Tb = @TypeOf(b);
    if (Ta == Mat and Tb == Mat) {
        return mulMat(a, b);
    } else if (Ta == f32 and Tb == Mat) {
        const va = splat(F32x4, a);
        return Mat{ va * b[0], va * b[1], va * b[2], va * b[3] };
    } else if (Ta == Mat and Tb == f32) {
        const vb = splat(F32x4, b);
        return Mat{ a[0] * vb, a[1] * vb, a[2] * vb, a[3] * vb };
    } else if (Ta == Vec and Tb == Mat) {
        return vecMulMat(a, b);
    } else if (Ta == Mat and Tb == Vec) {
        return matMulVec(a, b);
    } else {
        @compileError("zmath.mul() not implemented for types: " ++ @typeName(Ta) ++ ", " ++ @typeName(Tb));
    }
}

fn vecMulMat(v: Vec, m: Mat) Vec {
    const vx = @shuffle(f32, v, undefined, [4]i32{ 0, 0, 0, 0 });
    const vy = @shuffle(f32, v, undefined, [4]i32{ 1, 1, 1, 1 });
    const vz = @shuffle(f32, v, undefined, [4]i32{ 2, 2, 2, 2 });
    const vw = @shuffle(f32, v, undefined, [4]i32{ 3, 3, 3, 3 });
    return vx * m[0] + vy * m[1] + vz * m[2] + vw * m[3];
}

fn matMulVec(m: Mat, v: Vec) Vec {
    return .{ dot4(m[0], v)[0], dot4(m[1], v)[0], dot4(m[2], v)[0], dot4(m[3], v)[0] };
}

fn mulMat(m0: Mat, m1: Mat) Mat {
    var result: Mat = undefined;
    comptime var row: u32 = 0;
    inline while (row < 4) : (row += 1) {
        const vx = swizzle(m0[row], .x, .x, .x, .x);
        const vy = swizzle(m0[row], .y, .y, .y, .y);
        const vz = swizzle(m0[row], .z, .z, .z, .z);
        const vw = swizzle(m0[row], .w, .w, .w, .w);
        result[row] = mulAdd(vx, m1[0], vz * m1[2]) + mulAdd(vy, m1[1], vw * m1[3]);
    }
    return result;
}

pub fn transpose(m: Mat) Mat {
    const temp1 = @shuffle(f32, m[0], m[1], [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
    const temp3 = @shuffle(f32, m[0], m[1], [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
    const temp2 = @shuffle(f32, m[2], m[3], [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 1) });
    const temp4 = @shuffle(f32, m[2], m[3], [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
    return .{
        @shuffle(f32, temp1, temp2, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) }),
        @shuffle(f32, temp1, temp2, [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) }),
        @shuffle(f32, temp3, temp4, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) }),
        @shuffle(f32, temp3, temp4, [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) }),
    };
}

pub fn rotationX(angle: f32) Mat {
    const sc = sincos(angle);
    return .{
        f32x4(1.0, 0.0, 0.0, 0.0),
        f32x4(0.0, sc[1], sc[0], 0.0),
        f32x4(0.0, -sc[0], sc[1], 0.0),
        f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

pub fn rotationY(angle: f32) Mat {
    const sc = sincos(angle);
    return .{
        f32x4(sc[1], 0.0, -sc[0], 0.0),
        f32x4(0.0, 1.0, 0.0, 0.0),
        f32x4(sc[0], 0.0, sc[1], 0.0),
        f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

pub fn rotationZ(angle: f32) Mat {
    const sc = sincos(angle);
    return .{
        f32x4(sc[1], sc[0], 0.0, 0.0),
        f32x4(-sc[0], sc[1], 0.0, 0.0),
        f32x4(0.0, 0.0, 1.0, 0.0),
        f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

pub fn translation(x: f32, y: f32, z: f32) Mat {
    return .{
        f32x4(1.0, 0.0, 0.0, 0.0),
        f32x4(0.0, 1.0, 0.0, 0.0),
        f32x4(0.0, 0.0, 1.0, 0.0),
        f32x4(x, y, z, 1.0),
    };
}
pub fn translationV(v: Vec) Mat {
    return translation(v[0], v[1], v[2]);
}

pub fn scaling(x: f32, y: f32, z: f32) Mat {
    return .{
        f32x4(x, 0.0, 0.0, 0.0),
        f32x4(0.0, y, 0.0, 0.0),
        f32x4(0.0, 0.0, z, 0.0),
        f32x4(0.0, 0.0, 0.0, 1.0),
    };
}
pub fn scalingV(v: Vec) Mat {
    return scaling(v[0], v[1], v[2]);
}

pub fn lookToLh(eyepos: Vec, eyedir: Vec, updir: Vec) Mat {
    const az = normalize3(eyedir);
    const ax = normalize3(cross3(updir, az));
    const ay = normalize3(cross3(az, ax));
    return .{
        f32x4(ax[0], ay[0], az[0], 0),
        f32x4(ax[1], ay[1], az[1], 0),
        f32x4(ax[2], ay[2], az[2], 0),
        f32x4(-dot3(ax, eyepos)[0], -dot3(ay, eyepos)[0], -dot3(az, eyepos)[0], 1.0),
    };
}
pub fn lookToRh(eyepos: Vec, eyedir: Vec, updir: Vec) Mat {
    return lookToLh(eyepos, -eyedir, updir);
}
pub fn lookAtLh(eyepos: Vec, focuspos: Vec, updir: Vec) Mat {
    return lookToLh(eyepos, focuspos - eyepos, updir);
}
pub fn lookAtRh(eyepos: Vec, focuspos: Vec, updir: Vec) Mat {
    return lookToLh(eyepos, eyepos - focuspos, updir);
}

pub fn perspectiveFovLh(fovy: f32, aspect: f32, near: f32, far: f32) Mat {
    const scfov = sincos(0.5 * fovy);

    assert(near > 0.0 and far > 0.0);
    assert(!math.approxEqAbs(f32, scfov[0], 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));
    assert(!math.approxEqAbs(f32, aspect, 0.0, 0.01));

    const h = scfov[1] / scfov[0];
    const w = h / aspect;
    const r = far / (far - near);
    return .{
        f32x4(w, 0.0, 0.0, 0.0),
        f32x4(0.0, h, 0.0, 0.0),
        f32x4(0.0, 0.0, r, 1.0),
        f32x4(0.0, 0.0, -r * near, 0.0),
    };
}
pub fn perspectiveFovRh(fovy: f32, aspect: f32, near: f32, far: f32) Mat {
    const scfov = sincos(0.5 * fovy);

    assert(near > 0.0 and far > 0.0);
    assert(!math.approxEqAbs(f32, scfov[0], 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));
    assert(!math.approxEqAbs(f32, aspect, 0.0, 0.01));

    const h = scfov[1] / scfov[0];
    const w = h / aspect;
    const r = far / (near - far);
    return .{
        f32x4(w, 0.0, 0.0, 0.0),
        f32x4(0.0, h, 0.0, 0.0),
        f32x4(0.0, 0.0, r, -1.0),
        f32x4(0.0, 0.0, r * near, 0.0),
    };
}

// Produces Z values in [-1.0, 1.0] range (OpenGL defaults)
pub fn perspectiveFovLhGl(fovy: f32, aspect: f32, near: f32, far: f32) Mat {
    const scfov = sincos(0.5 * fovy);

    assert(near > 0.0 and far > 0.0);
    assert(!math.approxEqAbs(f32, scfov[0], 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));
    assert(!math.approxEqAbs(f32, aspect, 0.0, 0.01));

    const h = scfov[1] / scfov[0];
    const w = h / aspect;
    const r = far - near;
    return .{
        f32x4(w, 0.0, 0.0, 0.0),
        f32x4(0.0, h, 0.0, 0.0),
        f32x4(0.0, 0.0, (near + far) / r, 1.0),
        f32x4(0.0, 0.0, 2.0 * near * far / -r, 0.0),
    };
}

// Produces Z values in [-1.0, 1.0] range (OpenGL defaults)
pub fn perspectiveFovRhGl(fovy: f32, aspect: f32, near: f32, far: f32) Mat {
    const scfov = sincos(0.5 * fovy);

    assert(near > 0.0 and far > 0.0);
    assert(!math.approxEqAbs(f32, scfov[0], 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));
    assert(!math.approxEqAbs(f32, aspect, 0.0, 0.01));

    const h = scfov[1] / scfov[0];
    const w = h / aspect;
    const r = near - far;
    return .{
        f32x4(w, 0.0, 0.0, 0.0),
        f32x4(0.0, h, 0.0, 0.0),
        f32x4(0.0, 0.0, (near + far) / r, -1.0),
        f32x4(0.0, 0.0, 2.0 * near * far / r, 0.0),
    };
}

pub fn orthographicLh(w: f32, h: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, w, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, h, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = 1 / (far - near);
    return .{
        f32x4(2 / w, 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / h, 0.0, 0.0),
        f32x4(0.0, 0.0, r, 0.0),
        f32x4(0.0, 0.0, -r * near, 1.0),
    };
}

pub fn orthographicRh(w: f32, h: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, w, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, h, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = 1 / (near - far);
    return .{
        f32x4(2 / w, 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / h, 0.0, 0.0),
        f32x4(0.0, 0.0, r, 0.0),
        f32x4(0.0, 0.0, r * near, 1.0),
    };
}

// Produces Z values in [-1.0, 1.0] range (OpenGL defaults)
pub fn orthographicLhGl(w: f32, h: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, w, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, h, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = far - near;
    return .{
        f32x4(2 / w, 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / h, 0.0, 0.0),
        f32x4(0.0, 0.0, 2 / r, 0.0),
        f32x4(0.0, 0.0, (near + far) / -r, 1.0),
    };
}

// Produces Z values in [-1.0, 1.0] range (OpenGL defaults)
pub fn orthographicRhGl(w: f32, h: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, w, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, h, 0.0, 0.001));
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = near - far;
    return .{
        f32x4(2 / w, 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / h, 0.0, 0.0),
        f32x4(0.0, 0.0, 2 / r, 0.0),
        f32x4(0.0, 0.0, (near + far) / r, 1.0),
    };
}

pub fn orthographicOffCenterLh(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = 1 / (far - near);
    return .{
        f32x4(2 / (right - left), 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / (top - bottom), 0.0, 0.0),
        f32x4(0.0, 0.0, r, 0.0),
        f32x4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), -r * near, 1.0),
    };
}

pub fn orthographicOffCenterRh(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = 1 / (near - far);
    return .{
        f32x4(2 / (right - left), 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / (top - bottom), 0.0, 0.0),
        f32x4(0.0, 0.0, r, 0.0),
        f32x4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), r * near, 1.0),
    };
}

// Produces Z values in [-1.0, 1.0] range (OpenGL defaults)
pub fn orthographicOffCenterLhGl(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = far - near;
    return .{
        f32x4(2 / (right - left), 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / (top - bottom), 0.0, 0.0),
        f32x4(0.0, 0.0, 2 / r, 0.0),
        f32x4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), (near + far) / -r, 1.0),
    };
}

// Produces Z values in [-1.0, 1.0] range (OpenGL defaults)
pub fn orthographicOffCenterRhGl(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat {
    assert(!math.approxEqAbs(f32, far, near, 0.001));

    const r = near - far;
    return .{
        f32x4(2 / (right - left), 0.0, 0.0, 0.0),
        f32x4(0.0, 2 / (top - bottom), 0.0, 0.0),
        f32x4(0.0, 0.0, 2 / r, 0.0),
        f32x4(-(right + left) / (right - left), -(top + bottom) / (top - bottom), (near + far) / r, 1.0),
    };
}

pub fn determinant(m: Mat) F32x4 {
    var v0 = swizzle(m[2], .y, .x, .x, .x);
    var v1 = swizzle(m[3], .z, .z, .y, .y);
    var v2 = swizzle(m[2], .y, .x, .x, .x);
    var v3 = swizzle(m[3], .w, .w, .w, .z);
    var v4 = swizzle(m[2], .z, .z, .y, .y);
    var v5 = swizzle(m[3], .w, .w, .w, .z);

    var p0 = v0 * v1;
    var p1 = v2 * v3;
    var p2 = v4 * v5;

    v0 = swizzle(m[2], .z, .z, .y, .y);
    v1 = swizzle(m[3], .y, .x, .x, .x);
    v2 = swizzle(m[2], .w, .w, .w, .z);
    v3 = swizzle(m[3], .y, .x, .x, .x);
    v4 = swizzle(m[2], .w, .w, .w, .z);
    v5 = swizzle(m[3], .z, .z, .y, .y);

    p0 = mulAdd(-v0, v1, p0);
    p1 = mulAdd(-v2, v3, p1);
    p2 = mulAdd(-v4, v5, p2);

    v0 = swizzle(m[1], .w, .w, .w, .z);
    v1 = swizzle(m[1], .z, .z, .y, .y);
    v2 = swizzle(m[1], .y, .x, .x, .x);

    const s = m[0] * f32x4(1.0, -1.0, 1.0, -1.0);
    var r = v0 * p0;
    r = mulAdd(-v1, p1, r);
    r = mulAdd(v2, p2, r);
    return dot4(s, r);
}

pub fn inverse(a: anytype) @TypeOf(a) {
    const T = @TypeOf(a);
    return switch (T) {
        Mat => inverseMat(a),
        Quat => inverseQuat(a),
        else => @compileError("zmath.inverse() not implemented for " ++ @typeName(T)),
    };
}

fn inverseMat(m: Mat) Mat {
    return inverseDet(m, null);
}

pub fn inverseDet(m: Mat, out_det: ?*F32x4) Mat {
    const mt = transpose(m);
    var v0: [4]F32x4 = undefined;
    var v1: [4]F32x4 = undefined;

    v0[0] = swizzle(mt[2], .x, .x, .y, .y);
    v1[0] = swizzle(mt[3], .z, .w, .z, .w);
    v0[1] = swizzle(mt[0], .x, .x, .y, .y);
    v1[1] = swizzle(mt[1], .z, .w, .z, .w);
    v0[2] = @shuffle(f32, mt[2], mt[0], [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });
    v1[2] = @shuffle(f32, mt[3], mt[1], [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });

    var d0 = v0[0] * v1[0];
    var d1 = v0[1] * v1[1];
    var d2 = v0[2] * v1[2];

    v0[0] = swizzle(mt[2], .z, .w, .z, .w);
    v1[0] = swizzle(mt[3], .x, .x, .y, .y);
    v0[1] = swizzle(mt[0], .z, .w, .z, .w);
    v1[1] = swizzle(mt[1], .x, .x, .y, .y);
    v0[2] = @shuffle(f32, mt[2], mt[0], [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });
    v1[2] = @shuffle(f32, mt[3], mt[1], [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });

    d0 = mulAdd(-v0[0], v1[0], d0);
    d1 = mulAdd(-v0[1], v1[1], d1);
    d2 = mulAdd(-v0[2], v1[2], d2);

    v0[0] = swizzle(mt[1], .y, .z, .x, .y);
    v1[0] = @shuffle(f32, d0, d2, [4]i32{ ~@as(i32, 1), 1, 3, 0 });
    v0[1] = swizzle(mt[0], .z, .x, .y, .x);
    v1[1] = @shuffle(f32, d0, d2, [4]i32{ 3, ~@as(i32, 1), 1, 2 });
    v0[2] = swizzle(mt[3], .y, .z, .x, .y);
    v1[2] = @shuffle(f32, d1, d2, [4]i32{ ~@as(i32, 3), 1, 3, 0 });
    v0[3] = swizzle(mt[2], .z, .x, .y, .x);
    v1[3] = @shuffle(f32, d1, d2, [4]i32{ 3, ~@as(i32, 3), 1, 2 });

    var c0 = v0[0] * v1[0];
    var c2 = v0[1] * v1[1];
    var c4 = v0[2] * v1[2];
    var c6 = v0[3] * v1[3];

    v0[0] = swizzle(mt[1], .z, .w, .y, .z);
    v1[0] = @shuffle(f32, d0, d2, [4]i32{ 3, 0, 1, ~@as(i32, 0) });
    v0[1] = swizzle(mt[0], .w, .z, .w, .y);
    v1[1] = @shuffle(f32, d0, d2, [4]i32{ 2, 1, ~@as(i32, 0), 0 });
    v0[2] = swizzle(mt[3], .z, .w, .y, .z);
    v1[2] = @shuffle(f32, d1, d2, [4]i32{ 3, 0, 1, ~@as(i32, 2) });
    v0[3] = swizzle(mt[2], .w, .z, .w, .y);
    v1[3] = @shuffle(f32, d1, d2, [4]i32{ 2, 1, ~@as(i32, 2), 0 });

    c0 = mulAdd(-v0[0], v1[0], c0);
    c2 = mulAdd(-v0[1], v1[1], c2);
    c4 = mulAdd(-v0[2], v1[2], c4);
    c6 = mulAdd(-v0[3], v1[3], c6);

    v0[0] = swizzle(mt[1], .w, .x, .w, .x);
    v1[0] = @shuffle(f32, d0, d2, [4]i32{ 2, ~@as(i32, 1), ~@as(i32, 0), 2 });
    v0[1] = swizzle(mt[0], .y, .w, .x, .z);
    v1[1] = @shuffle(f32, d0, d2, [4]i32{ ~@as(i32, 1), 0, 3, ~@as(i32, 0) });
    v0[2] = swizzle(mt[3], .w, .x, .w, .x);
    v1[2] = @shuffle(f32, d1, d2, [4]i32{ 2, ~@as(i32, 3), ~@as(i32, 2), 2 });
    v0[3] = swizzle(mt[2], .y, .w, .x, .z);
    v1[3] = @shuffle(f32, d1, d2, [4]i32{ ~@as(i32, 3), 0, 3, ~@as(i32, 2) });

    const c1 = mulAdd(-v0[0], v1[0], c0);
    const c3 = mulAdd(v0[1], v1[1], c2);
    const c5 = mulAdd(-v0[2], v1[2], c4);
    const c7 = mulAdd(v0[3], v1[3], c6);

    c0 = mulAdd(v0[0], v1[0], c0);
    c2 = mulAdd(-v0[1], v1[1], c2);
    c4 = mulAdd(v0[2], v1[2], c4);
    c6 = mulAdd(-v0[3], v1[3], c6);

    var mr = Mat{
        f32x4(c0[0], c1[1], c0[2], c1[3]),
        f32x4(c2[0], c3[1], c2[2], c3[3]),
        f32x4(c4[0], c5[1], c4[2], c5[3]),
        f32x4(c6[0], c7[1], c6[2], c7[3]),
    };

    const det = dot4(mr[0], mt[0]);
    if (out_det != null) {
        out_det.?.* = det;
    }

    if (math.approxEqAbs(f64, det[0], 0.0, math.floatEps(f64))) {
        return .{
            f32x4(0.0, 0.0, 0.0, 0.0),
            f32x4(0.0, 0.0, 0.0, 0.0),
            f32x4(0.0, 0.0, 0.0, 0.0),
            f32x4(0.0, 0.0, 0.0, 0.0),
        };
    }

    const scale = splat(F32x4, 1.0) / det;
    mr[0] *= scale;
    mr[1] *= scale;
    mr[2] *= scale;
    mr[3] *= scale;
    return mr;
}

pub fn matFromNormAxisAngle(axis: Vec, angle: f32) Mat {
    const sincos_angle = sincos(angle);

    const c2 = splat(F32x4, 1.0 - sincos_angle[1]);
    const c1 = splat(F32x4, sincos_angle[1]);
    const c0 = splat(F32x4, sincos_angle[0]);

    const n0 = swizzle(axis, .y, .z, .x, .w);
    const n1 = swizzle(axis, .z, .x, .y, .w);

    var v0 = c2 * n0 * n1;
    const r0 = c2 * axis * axis + c1;
    const r1 = c0 * axis + v0;
    var r2 = v0 - c0 * axis;

    v0 = andInt(r0, f32x4_mask3);

    var v1 = @shuffle(f32, r1, r2, [4]i32{ 0, 2, ~@as(i32, 1), ~@as(i32, 2) });
    v1 = swizzle(v1, .y, .z, .w, .x);

    var v2 = @shuffle(f32, r1, r2, [4]i32{ 1, 1, ~@as(i32, 0), ~@as(i32, 0) });
    v2 = swizzle(v2, .x, .z, .x, .z);

    r2 = @shuffle(f32, v0, v1, [4]i32{ 0, 3, ~@as(i32, 0), ~@as(i32, 1) });
    r2 = swizzle(r2, .x, .z, .w, .y);

    var m: Mat = undefined;
    m[0] = r2;

    r2 = @shuffle(f32, v0, v1, [4]i32{ 1, 3, ~@as(i32, 2), ~@as(i32, 3) });
    r2 = swizzle(r2, .z, .x, .w, .y);
    m[1] = r2;

    v2 = @shuffle(f32, v2, v0, [4]i32{ 0, 1, ~@as(i32, 2), ~@as(i32, 3) });
    m[2] = v2;
    m[3] = f32x4(0.0, 0.0, 0.0, 1.0);
    return m;
}
pub fn matFromAxisAngle(axis: Vec, angle: f32) Mat {
    assert(!all(axis == splat(F32x4, 0.0), 3));
    assert(!all(isInf(axis), 3));
    const normal = normalize3(axis);
    return matFromNormAxisAngle(normal, angle);
}

pub fn matFromQuat(quat: Quat) Mat {
    const q0 = quat + quat;
    var q1 = quat * q0;

    var v0 = swizzle(q1, .y, .x, .x, .w);
    v0 = andInt(v0, f32x4_mask3);

    var v1 = swizzle(q1, .z, .z, .y, .w);
    v1 = andInt(v1, f32x4_mask3);

    const r0 = (f32x4(1.0, 1.0, 1.0, 0.0) - v0) - v1;

    v0 = swizzle(quat, .x, .x, .y, .w);
    v1 = swizzle(q0, .z, .y, .z, .w);
    v0 = v0 * v1;

    v1 = swizzle(quat, .w, .w, .w, .w);
    const v2 = swizzle(q0, .y, .z, .x, .w);
    v1 = v1 * v2;

    const r1 = v0 + v1;
    const r2 = v0 - v1;

    v0 = @shuffle(f32, r1, r2, [4]i32{ 1, 2, ~@as(i32, 0), ~@as(i32, 1) });
    v0 = swizzle(v0, .x, .z, .w, .y);
    v1 = @shuffle(f32, r1, r2, [4]i32{ 0, 0, ~@as(i32, 2), ~@as(i32, 2) });
    v1 = swizzle(v1, .x, .z, .x, .z);

    q1 = @shuffle(f32, r0, v0, [4]i32{ 0, 3, ~@as(i32, 0), ~@as(i32, 1) });
    q1 = swizzle(q1, .x, .z, .w, .y);

    var m: Mat = undefined;
    m[0] = q1;

    q1 = @shuffle(f32, r0, v0, [4]i32{ 1, 3, ~@as(i32, 2), ~@as(i32, 3) });
    q1 = swizzle(q1, .z, .x, .w, .y);
    m[1] = q1;

    q1 = @shuffle(f32, v1, r0, [4]i32{ 0, 1, ~@as(i32, 2), ~@as(i32, 3) });
    m[2] = q1;
    m[3] = f32x4(0.0, 0.0, 0.0, 1.0);
    return m;
}

pub fn matFromRollPitchYaw(pitch: f32, yaw: f32, roll: f32) Mat {
    return matFromRollPitchYawV(f32x4(pitch, yaw, roll, 0.0));
}
pub fn matFromRollPitchYawV(angles: Vec) Mat {
    return matFromQuat(quatFromRollPitchYawV(angles));
}

pub fn matToQuat(m: Mat) Quat {
    return quatFromMat(m);
}

pub inline fn loadMat(mem: []const f32) Mat {
    return .{
        load(mem[0..4], F32x4, 0),
        load(mem[4..8], F32x4, 0),
        load(mem[8..12], F32x4, 0),
        load(mem[12..16], F32x4, 0),
    };
}

pub inline fn storeMat(mem: []f32, m: Mat) void {
    store(mem[0..4], m[0], 0);
    store(mem[4..8], m[1], 0);
    store(mem[8..12], m[2], 0);
    store(mem[12..16], m[3], 0);
}

pub inline fn loadMat43(mem: []const f32) Mat {
    return .{
        f32x4(mem[0], mem[1], mem[2], 0.0),
        f32x4(mem[3], mem[4], mem[5], 0.0),
        f32x4(mem[6], mem[7], mem[8], 0.0),
        f32x4(mem[9], mem[10], mem[11], 1.0),
    };
}

pub inline fn storeMat43(mem: []f32, m: Mat) void {
    store(mem[0..3], m[0], 3);
    store(mem[3..6], m[1], 3);
    store(mem[6..9], m[2], 3);
    store(mem[9..12], m[3], 3);
}

pub inline fn loadMat34(mem: []const f32) Mat {
    return .{
        load(mem[0..4], F32x4, 0),
        load(mem[4..8], F32x4, 0),
        load(mem[8..12], F32x4, 0),
        f32x4(0.0, 0.0, 0.0, 1.0),
    };
}

pub inline fn storeMat34(mem: []f32, m: Mat) void {
    store(mem[0..4], m[0], 0);
    store(mem[4..8], m[1], 0);
    store(mem[8..12], m[2], 0);
}

pub inline fn matToArr(m: Mat) [16]f32 {
    var array: [16]f32 = undefined;
    storeMat(array[0..], m);
    return array;
}

pub inline fn matToArr43(m: Mat) [12]f32 {
    var array: [12]f32 = undefined;
    storeMat43(array[0..], m);
    return array;
}

pub inline fn matToArr34(m: Mat) [12]f32 {
    var array: [12]f32 = undefined;
    storeMat34(array[0..], m);
    return array;
}
// ------------------------------------------------------------------------------
//
// 5. Quaternion functions
//
// ------------------------------------------------------------------------------
pub fn qmul(q0: Quat, q1: Quat) Quat {
    var result = swizzle(q1, .w, .w, .w, .w);
    var q1x = swizzle(q1, .x, .x, .x, .x);
    var q1y = swizzle(q1, .y, .y, .y, .y);
    var q1z = swizzle(q1, .z, .z, .z, .z);
    result = result * q0;
    var q0_shuf = swizzle(q0, .w, .z, .y, .x);
    q1x = q1x * q0_shuf;
    q0_shuf = swizzle(q0_shuf, .y, .x, .w, .z);
    result = mulAdd(q1x, f32x4(1.0, -1.0, 1.0, -1.0), result);
    q1y = q1y * q0_shuf;
    q0_shuf = swizzle(q0_shuf, .w, .z, .y, .x);
    q1y = q1y * f32x4(1.0, 1.0, -1.0, -1.0);
    q1z = q1z * q0_shuf;
    q1y = mulAdd(q1z, f32x4(-1.0, 1.0, 1.0, -1.0), q1y);
    return result + q1y;
}

pub fn quatToMat(quat: Quat) Mat {
    return matFromQuat(quat);
}

pub fn quatToAxisAngle(quat: Quat, axis: *Vec, angle: *f32) void {
    axis.* = quat;
    angle.* = 2.0 * acos(quat[3]);
}

pub fn quatFromMat(m: Mat) Quat {
    const r0 = m[0];
    const r1 = m[1];
    const r2 = m[2];
    const r00 = swizzle(r0, .x, .x, .x, .x);
    const r11 = swizzle(r1, .y, .y, .y, .y);
    const r22 = swizzle(r2, .z, .z, .z, .z);

    const x2gey2 = (r11 - r00) <= splat(F32x4, 0.0);
    const z2gew2 = (r11 + r00) <= splat(F32x4, 0.0);
    const x2py2gez2pw2 = r22 <= splat(F32x4, 0.0);

    var t0 = mulAdd(r00, f32x4(1.0, -1.0, -1.0, 1.0), splat(F32x4, 1.0));
    var t1 = r11 * f32x4(-1.0, 1.0, -1.0, 1.0);
    var t2 = mulAdd(r22, f32x4(-1.0, -1.0, 1.0, 1.0), t0);
    const x2y2z2w2 = t1 + t2;

    t0 = @shuffle(f32, r0, r1, [4]i32{ 1, 2, ~@as(i32, 2), ~@as(i32, 1) });
    t1 = @shuffle(f32, r1, r2, [4]i32{ 0, 0, ~@as(i32, 0), ~@as(i32, 1) });
    t1 = swizzle(t1, .x, .z, .w, .y);
    const xyxzyz = t0 + t1;

    t0 = @shuffle(f32, r2, r1, [4]i32{ 1, 0, ~@as(i32, 0), ~@as(i32, 0) });
    t1 = @shuffle(f32, r1, r0, [4]i32{ 2, 2, ~@as(i32, 2), ~@as(i32, 1) });
    t1 = swizzle(t1, .x, .z, .w, .y);
    const xwywzw = (t0 - t1) * f32x4(-1.0, 1.0, -1.0, 1.0);

    t0 = @shuffle(f32, x2y2z2w2, xyxzyz, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 0) });
    t1 = @shuffle(f32, x2y2z2w2, xwywzw, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 0) });
    t2 = @shuffle(f32, xyxzyz, xwywzw, [4]i32{ 1, 2, ~@as(i32, 0), ~@as(i32, 1) });

    const tensor0 = @shuffle(f32, t0, t2, [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });
    const tensor1 = @shuffle(f32, t0, t2, [4]i32{ 2, 1, ~@as(i32, 1), ~@as(i32, 3) });
    const tensor2 = @shuffle(f32, t2, t1, [4]i32{ 0, 1, ~@as(i32, 0), ~@as(i32, 2) });
    const tensor3 = @shuffle(f32, t2, t1, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 1) });

    t0 = select(x2gey2, tensor0, tensor1);
    t1 = select(z2gew2, tensor2, tensor3);
    t2 = select(x2py2gez2pw2, t0, t1);

    return t2 / length4(t2);
}

pub fn quatFromNormAxisAngle(axis: Vec, angle: f32) Quat {
    const n = f32x4(axis[0], axis[1], axis[2], 1.0);
    const sc = sincos(0.5 * angle);
    return n * f32x4(sc[0], sc[0], sc[0], sc[1]);
}
pub fn quatFromAxisAngle(axis: Vec, angle: f32) Quat {
    assert(!all(axis == splat(F32x4, 0.0), 3));
    assert(!all(isInf(axis), 3));
    const normal = normalize3(axis);
    return quatFromNormAxisAngle(normal, angle);
}

pub inline fn qidentity() Quat {
    return f32x4(@as(f32, 0.0), @as(f32, 0.0), @as(f32, 0.0), @as(f32, 1.0));
}

pub inline fn conjugate(quat: Quat) Quat {
    return quat * f32x4(-1.0, -1.0, -1.0, 1.0);
}

fn inverseQuat(quat: Quat) Quat {
    const l = lengthSq4(quat);
    const conj = conjugate(quat);
    return select(l <= splat(F32x4, math.floatEps(f32)), splat(F32x4, 0.0), conj / l);
}

// Algorithm from: https://github.com/g-truc/glm/blob/master/glm/detail/type_quat.inl
pub fn rotate(q: Quat, v: Vec) Vec {
    const w = splat(F32x4, q[3]);
    const axis = f32x4(q[0], q[1], q[2], 0.0);
    const uv = cross3(axis, v);
    return v + ((uv * w) + cross3(axis, uv)) * splat(F32x4, 2.0);
}

pub fn slerp(q0: Quat, q1: Quat, t: f32) Quat {
    return slerpV(q0, q1, splat(F32x4, t));
}
pub fn slerpV(q0: Quat, q1: Quat, t: F32x4) Quat {
    var cos_omega = dot4(q0, q1);
    const sign = select(cos_omega < splat(F32x4, 0.0), splat(F32x4, -1.0), splat(F32x4, 1.0));

    cos_omega = cos_omega * sign;
    const sin_omega = sqrt(splat(F32x4, 1.0) - cos_omega * cos_omega);

    const omega = atan2(sin_omega, cos_omega);

    var v01 = t;
    v01 = xorInt(andInt(v01, f32x4_mask2), f32x4_sign_mask1);
    v01 = f32x4(1.0, 0.0, 0.0, 0.0) + v01;

    var s0 = sin(v01 * omega) / sin_omega;
    s0 = select(cos_omega < splat(F32x4, 1.0 - 0.00001), s0, v01);

    const s1 = swizzle(s0, .y, .y, .y, .y);
    s0 = swizzle(s0, .x, .x, .x, .x);

    return q0 * s0 + sign * q1 * s1;
}

// Converts q back to euler angles, assuming a YXZ rotation order.
// See: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToEuler
pub fn quatToRollPitchYaw(q: Quat) [3]f32 {
    var angles: [3]f32 = undefined;

    const p = swizzle(q, .w, .y, .x, .z);
    const sign = -1.0;

    const singularity = p[0] * p[2] + sign * p[1] * p[3];
    if (singularity > 0.499) {
        angles[0] = math.pi * 0.5;
        angles[1] = 2.0 * math.atan2(p[1], p[0]);
        angles[2] = 0.0;
    } else if (singularity < -0.499) {
        angles[0] = -math.pi * 0.5;
        angles[1] = 2.0 * math.atan2(p[1], p[0]);
        angles[2] = 0.0;
    } else {
        const sq = p * p;
        const y = splat(F32x4, 2.0) * f32x4(p[0] * p[1] - sign * p[2] * p[3], p[0] * p[3] - sign * p[1] * p[2], 0.0, 0.0);
        const x = splat(F32x4, 1.0) - (splat(F32x4, 2.0) * f32x4(sq[1] + sq[2], sq[2] + sq[3], 0.0, 0.0));
        const res = atan2(y, x);
        angles[0] = math.asin(2.0 * singularity);
        angles[1] = res[0];
        angles[2] = res[1];
    }

    return angles;
}

pub fn quatFromRollPitchYaw(pitch: f32, yaw: f32, roll: f32) Quat {
    return quatFromRollPitchYawV(f32x4(pitch, yaw, roll, 0.0));
}
pub fn quatFromRollPitchYawV(angles: Vec) Quat { // | pitch | yaw | roll | 0 |
    const sc = sincos(splat(Vec, 0.5) * angles);
    const p0 = @shuffle(f32, sc[1], sc[0], [4]i32{ ~@as(i32, 0), 0, 0, 0 });
    const p1 = @shuffle(f32, sc[0], sc[1], [4]i32{ ~@as(i32, 0), 0, 0, 0 });
    const y0 = @shuffle(f32, sc[1], sc[0], [4]i32{ 1, ~@as(i32, 1), 1, 1 });
    const y1 = @shuffle(f32, sc[0], sc[1], [4]i32{ 1, ~@as(i32, 1), 1, 1 });
    const r0 = @shuffle(f32, sc[1], sc[0], [4]i32{ 2, 2, ~@as(i32, 2), 2 });
    const r1 = @shuffle(f32, sc[0], sc[1], [4]i32{ 2, 2, ~@as(i32, 2), 2 });
    const q1 = p1 * f32x4(1.0, -1.0, -1.0, 1.0) * y1;
    const q0 = p0 * y0 * r0;
    return mulAdd(q1, r1, q0);
}

// ------------------------------------------------------------------------------
//
// 6. Color functions
//
// ------------------------------------------------------------------------------
pub fn adjustSaturation(color: F32x4, saturation: f32) F32x4 {
    const luminance = dot3(f32x4(0.2125, 0.7154, 0.0721, 0.0), color);
    var result = mulAdd(color - luminance, f32x4s(saturation), luminance);
    result[3] = color[3];
    return result;
}

pub fn adjustContrast(color: F32x4, contrast: f32) F32x4 {
    var result = mulAdd(color - f32x4s(0.5), f32x4s(contrast), f32x4s(0.5));
    result[3] = color[3];
    return result;
}

pub fn rgbToHsl(rgb: F32x4) F32x4 {
    const r = swizzle(rgb, .x, .x, .x, .x);
    const g = swizzle(rgb, .y, .y, .y, .y);
    const b = swizzle(rgb, .z, .z, .z, .z);

    const minv = min(r, min(g, b));
    const maxv = max(r, max(g, b));

    const l = (minv + maxv) * f32x4s(0.5);
    const d = maxv - minv;
    const la = select(boolx4(true, true, true, false), l, rgb);

    if (all(d < f32x4s(math.floatEps(f32)), 3)) {
        return select(boolx4(true, true, false, false), f32x4s(0.0), la);
    } else {
        var s: F32x4 = undefined;
        var h: F32x4 = undefined;

        const d2 = minv + maxv;

        if (all(l > f32x4s(0.5), 3)) {
            s = d / (f32x4s(2.0) - d2);
        } else {
            s = d / d2;
        }

        if (all(r == maxv, 3)) {
            h = (g - b) / d;
        } else if (all(g == maxv, 3)) {
            h = f32x4s(2.0) + (b - r) / d;
        } else {
            h = f32x4s(4.0) + (r - g) / d;
        }

        h /= f32x4s(6.0);

        if (all(h < f32x4s(0.0), 3)) {
            h += f32x4s(1.0);
        }

        const lha = select(boolx4(true, true, false, false), h, la);
        return select(boolx4(true, false, true, true), lha, s);
    }
}

fn hueToClr(p: F32x4, q: F32x4, h: F32x4) F32x4 {
    var t = h;

    if (all(t < f32x4s(0.0), 3))
        t += f32x4s(1.0);

    if (all(t > f32x4s(1.0), 3))
        t -= f32x4s(1.0);

    if (all(t < f32x4s(1.0 / 6.0), 3))
        return mulAdd(q - p, f32x4s(6.0) * t, p);

    if (all(t < f32x4s(0.5), 3))
        return q;

    if (all(t < f32x4s(2.0 / 3.0), 3))
        return mulAdd(q - p, f32x4s(6.0) * (f32x4s(2.0 / 3.0) - t), p);

    return p;
}

pub fn hslToRgb(hsl: F32x4) F32x4 {
    const s = swizzle(hsl, .y, .y, .y, .y);
    const l = swizzle(hsl, .z, .z, .z, .z);

    if (all(isNearEqual(s, f32x4s(0.0), f32x4s(math.floatEps(f32))), 3)) {
        return select(boolx4(true, true, true, false), l, hsl);
    } else {
        const h = swizzle(hsl, .x, .x, .x, .x);
        var q: F32x4 = undefined;
        if (all(l < f32x4s(0.5), 3)) {
            q = l * (f32x4s(1.0) + s);
        } else {
            q = (l + s) - (l * s);
        }

        const p = f32x4s(2.0) * l - q;

        const r = hueToClr(p, q, h + f32x4s(1.0 / 3.0));
        const g = hueToClr(p, q, h);
        const b = hueToClr(p, q, h - f32x4s(1.0 / 3.0));

        const rg = select(boolx4(true, false, false, false), r, g);
        const ba = select(boolx4(true, true, true, false), b, hsl);
        return select(boolx4(true, true, false, false), rg, ba);
    }
}

pub fn rgbToHsv(rgb: F32x4) F32x4 {
    const r = swizzle(rgb, .x, .x, .x, .x);
    const g = swizzle(rgb, .y, .y, .y, .y);
    const b = swizzle(rgb, .z, .z, .z, .z);

    const minv = min(r, min(g, b));
    const v = max(r, max(g, b));
    const d = v - minv;
    const s = if (all(isNearEqual(v, f32x4s(0.0), f32x4s(math.floatEps(f32))), 3)) f32x4s(0.0) else d / v;

    if (all(d < f32x4s(math.floatEps(f32)), 3)) {
        const hv = select(boolx4(true, false, false, false), f32x4s(0.0), v);
        const hva = select(boolx4(true, true, true, false), hv, rgb);
        return select(boolx4(true, false, true, true), hva, s);
    } else {
        var h: F32x4 = undefined;
        if (all(r == v, 3)) {
            h = (g - b) / d;
            if (all(g < b, 3))
                h += f32x4s(6.0);
        } else if (all(g == v, 3)) {
            h = f32x4s(2.0) + (b - r) / d;
        } else {
            h = f32x4s(4.0) + (r - g) / d;
        }

        h /= f32x4s(6.0);
        const hv = select(boolx4(true, false, false, false), h, v);
        const hva = select(boolx4(true, true, true, false), hv, rgb);
        return select(boolx4(true, false, true, true), hva, s);
    }
}

pub fn hsvToRgb(hsv: F32x4) F32x4 {
    const h = swizzle(hsv, .x, .x, .x, .x);
    const s = swizzle(hsv, .y, .y, .y, .y);
    const v = swizzle(hsv, .z, .z, .z, .z);

    const h6 = h * f32x4s(6.0);
    const i = floor(h6);
    const f = h6 - i;

    const p = v * (f32x4s(1.0) - s);
    const q = v * (f32x4s(1.0) - f * s);
    const t = v * (f32x4s(1.0) - (f32x4s(1.0) - f) * s);

    const ii = @as(i32, @intFromFloat(mod(i, f32x4s(6.0))[0]));
    const rgb = switch (ii) {
        0 => blk: {
            const vt = select(boolx4(true, false, false, false), v, t);
            break :blk select(boolx4(true, true, false, false), vt, p);
        },
        1 => blk: {
            const qv = select(boolx4(true, false, false, false), q, v);
            break :blk select(boolx4(true, true, false, false), qv, p);
        },
        2 => blk: {
            const pv = select(boolx4(true, false, false, false), p, v);
            break :blk select(boolx4(true, true, false, false), pv, t);
        },
        3 => blk: {
            const pq = select(boolx4(true, false, false, false), p, q);
            break :blk select(boolx4(true, true, false, false), pq, v);
        },
        4 => blk: {
            const tp = select(boolx4(true, false, false, false), t, p);
            break :blk select(boolx4(true, true, false, false), tp, v);
        },
        5 => blk: {
            const vp = select(boolx4(true, false, false, false), v, p);
            break :blk select(boolx4(true, true, false, false), vp, q);
        },
        else => unreachable,
    };
    return select(boolx4(true, true, true, false), rgb, hsv);
}

pub fn rgbToSrgb(rgb: F32x4) F32x4 {
    const static = struct {
        const cutoff = f32x4(0.0031308, 0.0031308, 0.0031308, 1.0);
        const linear = f32x4(12.92, 12.92, 12.92, 1.0);
        const scale = f32x4(1.055, 1.055, 1.055, 1.0);
        const bias = f32x4(0.055, 0.055, 0.055, 1.0);
        const rgamma = 1.0 / 2.4;
    };
    var v = saturate(rgb);
    const v0 = v * static.linear;
    const v1 = static.scale * f32x4(
        math.pow(f32, v[0], static.rgamma),
        math.pow(f32, v[1], static.rgamma),
        math.pow(f32, v[2], static.rgamma),
        v[3],
    ) - static.bias;
    v = select(v < static.cutoff, v0, v1);
    return select(boolx4(true, true, true, false), v, rgb);
}

pub fn srgbToRgb(srgb: F32x4) F32x4 {
    const static = struct {
        const cutoff = f32x4(0.04045, 0.04045, 0.04045, 1.0);
        const rlinear = f32x4(1.0 / 12.92, 1.0 / 12.92, 1.0 / 12.92, 1.0);
        const scale = f32x4(1.0 / 1.055, 1.0 / 1.055, 1.0 / 1.055, 1.0);
        const bias = f32x4(0.055, 0.055, 0.055, 1.0);
        const gamma = 2.4;
    };
    var v = saturate(srgb);
    const v0 = v * static.rlinear;
    var v1 = static.scale * (v + static.bias);
    v1 = f32x4(
        math.pow(f32, v1[0], static.gamma),
        math.pow(f32, v1[1], static.gamma),
        math.pow(f32, v1[2], static.gamma),
        v1[3],
    );
    v = select(v > static.cutoff, v1, v0);
    return select(boolx4(true, true, true, false), v, srgb);
}

// ------------------------------------------------------------------------------
//
/// Collection of useful functions building on top of, and extending, core zmath.
/// https://github.com/michal-z/zig-gamedev/tree/main/libs/zmath
///
/// ------------------------------------------------------------------------------
/// 1. Matrix functions
/// ------------------------------------------------------------------------------
///
/// As an example, in a left handed Y-up system:
///   getAxisX is equivalent to the right vector
///   getAxisY is equivalent to the up vector
///   getAxisZ is equivalent to the forward vector
///
/// getTranslationVec(m: Mat) Vec
/// getAxisX(m: Mat) Vec
/// getAxisY(m: Mat) Vec
/// getAxisZ(m: Mat) Vec
///
/// ==============================================================================
pub const util = struct {
    pub fn getTranslationVec(m: Mat) Vec {
        var _translation = m[3];
        _translation[3] = 0;
        return _translation;
    }

    pub fn setTranslationVec(m: *Mat, _translation: Vec) void {
        const w = m[3][3];
        m[3] = _translation;
        m[3][3] = w;
    }

    pub fn getScaleVec(m: Mat) Vec {
        const scale_x = length3(f32x4(m[0][0], m[1][0], m[2][0], 0))[0];
        const scale_y = length3(f32x4(m[0][1], m[1][1], m[2][1], 0))[0];
        const scale_z = length3(f32x4(m[0][2], m[1][2], m[2][2], 0))[0];
        return f32x4(scale_x, scale_y, scale_z, 0);
    }

    pub fn getRotationQuat(_m: Mat) Quat {
        // Ortho normalize given matrix.
        const c1 = normalize3(f32x4(_m[0][0], _m[1][0], _m[2][0], 0));
        const c2 = normalize3(f32x4(_m[0][1], _m[1][1], _m[2][1], 0));
        const c3 = normalize3(f32x4(_m[0][2], _m[1][2], _m[2][2], 0));
        var m = _m;
        m[0][0] = c1[0];
        m[1][0] = c1[1];
        m[2][0] = c1[2];
        m[0][1] = c2[0];
        m[1][1] = c2[1];
        m[2][1] = c2[2];
        m[0][2] = c3[0];
        m[1][2] = c3[1];
        m[2][2] = c3[2];

        // Extract rotation
        return quatFromMat(m);
    }

    pub fn getAxisX(m: Mat) Vec {
        return normalize3(f32x4(m[0][0], m[0][1], m[0][2], 0.0));
    }

    pub fn getAxisY(m: Mat) Vec {
        return normalize3(f32x4(m[1][0], m[1][1], m[1][2], 0.0));
    }

    pub fn getAxisZ(m: Mat) Vec {
        return normalize3(f32x4(m[2][0], m[2][1], m[2][2], 0.0));
    }
}; // util

// ------------------------------------------------------------------------------
