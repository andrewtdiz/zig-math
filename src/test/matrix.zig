const std = @import("std");
const builtin = @import("builtin");
const zm = @import("../root.zig");
const core = zm.core;
const vector = zm.vector;
const matrix = zm.matrix;
const util = matrix.util;
const testing = @import("../testing.zig");

const math = std.math;
const expect = std.testing.expect;
const expectVecEqual = testing.expectVecEqual;
const expectVecApproxEqAbs = testing.expectVecApproxEqAbs;
const approxEqAbs = testing.approxEqAbs;

const F32x4 = core.F32x4;
const F32x8 = core.F32x8;
const F32x16 = core.F32x16;
const Vec = core.Vec;
const Mat = core.Mat;
const Quat = core.Quat;

const f32x4 = core.f32x4;
const f32x4s = core.f32x4s;
const boolx4 = core.boolx4;
const splat = core.splat;
const load = core.load;
const store = core.store;
const loadArr3 = core.loadArr3;
const loadArr4 = core.loadArr4;
const loadArr3w = core.loadArr3w;
const vecToArr2 = core.vecToArr2;
const vecToArr3 = core.vecToArr3;
const vecToArr4 = core.vecToArr4;
const loadMat = matrix.loadMat;

const all = vector.all;
const normalize3 = vector.normalize3;
const normalize4 = vector.normalize4;
const cross3 = vector.cross3;
const dot3 = vector.dot3;
const dot4 = vector.dot4;
const length3 = vector.length3;
const length4 = vector.length4;
const lengthSq3 = vector.lengthSq3;
const lengthSq4 = vector.lengthSq4;
const sin = vector.sin;
const cos = vector.cos;
const asin = vector.asin;
const atan2 = vector.atan2;
const sqrt = vector.sqrt;
const swizzle = vector.swizzle;
const select = vector.select;
const min = vector.min;
const max = vector.max;
const mulAdd = vector.mulAdd;
const saturate = vector.saturate;
const cmulSoa = vector.cmulSoa;
const lerp = vector.lerp;
const f32x4_mask2 = vector.f32x4_mask2;
const f32x4_mask3 = vector.f32x4_mask3;
const f32x4_sign_mask1 = vector.f32x4_sign_mask1;

const identity = matrix.identity;
const matFromArr = matrix.matFromArr;
const matFromAxisAngle = matrix.matFromAxisAngle;
const matFromNormAxisAngle = matrix.matFromNormAxisAngle;
const matFromQuat = matrix.matFromQuat;
const matToQuat = matrix.matToQuat;
const mul = matrix.mul;
const transpose = matrix.transpose;
const rotationX = matrix.rotationX;
const rotationY = matrix.rotationY;
const rotationZ = matrix.rotationZ;
const translation = matrix.translation;
const translationV = matrix.translationV;
const scaling = matrix.scaling;
const scalingV = matrix.scalingV;
const lookToLh = matrix.lookToLh;
const lookAtLh = matrix.lookAtLh;
const lookToRh = matrix.lookToRh;
const lookAtRh = matrix.lookAtRh;
const perspectiveFovLh = matrix.perspectiveFovLh;
const perspectiveFovRh = matrix.perspectiveFovRh;
const perspectiveFovLhGl = matrix.perspectiveFovLhGl;
const perspectiveFovRhGl = matrix.perspectiveFovRhGl;
const orthographicLh = matrix.orthographicLh;
const orthographicRh = matrix.orthographicRh;
const orthographicLhGl = matrix.orthographicLhGl;
const orthographicRhGl = matrix.orthographicRhGl;
const orthographicOffCenterLh = matrix.orthographicOffCenterLh;
const orthographicOffCenterRh = matrix.orthographicOffCenterRh;
const orthographicOffCenterLhGl = matrix.orthographicOffCenterLhGl;
const orthographicOffCenterRhGl = matrix.orthographicOffCenterRhGl;
const determinant = matrix.determinant;
const inverse = matrix.inverse;
const inverseDet = matrix.inverseDet;
const quatFromAxisAngle = matrix.quatFromAxisAngle;
const quatFromNormAxisAngle = matrix.quatFromNormAxisAngle;
const quatFromMat = matrix.quatFromMat;
const quatFromRollPitchYaw = matrix.quatFromRollPitchYaw;
const matFromRollPitchYaw = matrix.matFromRollPitchYaw;
const quatToRollPitchYaw = matrix.quatToRollPitchYaw;
const quatToMat = matrix.quatToMat;
const quatToAxisAngle = matrix.quatToAxisAngle;
const quatFromRollPitchYawV = matrix.quatFromRollPitchYawV;
const qmul = matrix.qmul;
const rotate = matrix.rotate;
const slerp = matrix.slerp;
const slerpV = matrix.slerpV;
const qidentity = matrix.qidentity;
const conjugate = matrix.conjugate;
const inverseQuat = matrix.inverseQuat;
const rgbToHsl = matrix.rgbToHsl;
const hslToRgb = matrix.hslToRgb;
const rgbToHsv = matrix.rgbToHsv;
const hsvToRgb = matrix.hsvToRgb;
const rgbToSrgb = matrix.rgbToSrgb;
const srgbToRgb = matrix.srgbToRgb;
const getAxisX = util.getAxisX;
const getAxisY = util.getAxisY;
const getAxisZ = util.getAxisZ;
const getTranslationVec = util.getTranslationVec;
const setTranslationVec = util.setTranslationVec;
const getScaleVec = util.getScaleVec;
const getRotationQuat = util.getRotationQuat;

test "zmath.mul" {
    {
        const m = Mat{
            f32x4(0.1, 0.2, 0.3, 0.4),
            f32x4(0.5, 0.6, 0.7, 0.8),
            f32x4(0.9, 1.0, 1.1, 1.2),
            f32x4(1.3, 1.4, 1.5, 1.6),
        };
        const ms = mul(@as(f32, 2.0), m);
        try expectVecApproxEqAbs(ms[0], f32x4(0.2, 0.4, 0.6, 0.8), 0.0001);
        try expectVecApproxEqAbs(ms[1], f32x4(1.0, 1.2, 1.4, 1.6), 0.0001);
        try expectVecApproxEqAbs(ms[2], f32x4(1.8, 2.0, 2.2, 2.4), 0.0001);
        try expectVecApproxEqAbs(ms[3], f32x4(2.6, 2.8, 3.0, 3.2), 0.0001);
    }
}

test "zmath.vecMulMat" {
    const m = Mat{
        f32x4(1.0, 0.0, 0.0, 0.0),
        f32x4(0.0, 1.0, 0.0, 0.0),
        f32x4(0.0, 0.0, 1.0, 0.0),
        f32x4(2.0, 3.0, 4.0, 1.0),
    };
    const vm = mul(f32x4(1.0, 2.0, 3.0, 1.0), m);
    const mv = mul(m, f32x4(1.0, 2.0, 3.0, 1.0));
    const v = mul(transpose(m), f32x4(1.0, 2.0, 3.0, 1.0));
    try expectVecApproxEqAbs(vm, f32x4(3.0, 5.0, 7.0, 1.0), 0.0001);
    try expectVecApproxEqAbs(mv, f32x4(1.0, 2.0, 3.0, 21.0), 0.0001);
    try expectVecApproxEqAbs(v, f32x4(3.0, 5.0, 7.0, 1.0), 0.0001);
}

test "zmath.matrix.mul" {
    const a = Mat{
        f32x4(0.1, 0.2, 0.3, 0.4),
        f32x4(0.5, 0.6, 0.7, 0.8),
        f32x4(0.9, 1.0, 1.1, 1.2),
        f32x4(1.3, 1.4, 1.5, 1.6),
    };
    const b = Mat{
        f32x4(1.7, 1.8, 1.9, 2.0),
        f32x4(2.1, 2.2, 2.3, 2.4),
        f32x4(2.5, 2.6, 2.7, 2.8),
        f32x4(2.9, 3.0, 3.1, 3.2),
    };
    const c = mul(a, b);
    try expectVecApproxEqAbs(c[0], f32x4(2.5, 2.6, 2.7, 2.8), 0.0001);
    try expectVecApproxEqAbs(c[1], f32x4(6.18, 6.44, 6.7, 6.96), 0.0001);
    try expectVecApproxEqAbs(c[2], f32x4(9.86, 10.28, 10.7, 11.12), 0.0001);
    try expectVecApproxEqAbs(c[3], f32x4(13.54, 14.12, 14.7, 15.28), 0.0001);
}

test "zmath.matrix.transpose" {
    const m = Mat{
        f32x4(1.0, 2.0, 3.0, 4.0),
        f32x4(5.0, 6.0, 7.0, 8.0),
        f32x4(9.0, 10.0, 11.0, 12.0),
        f32x4(13.0, 14.0, 15.0, 16.0),
    };
    const mt = transpose(m);
    try expectVecApproxEqAbs(mt[0], f32x4(1.0, 5.0, 9.0, 13.0), 0.0001);
    try expectVecApproxEqAbs(mt[1], f32x4(2.0, 6.0, 10.0, 14.0), 0.0001);
    try expectVecApproxEqAbs(mt[2], f32x4(3.0, 7.0, 11.0, 15.0), 0.0001);
    try expectVecApproxEqAbs(mt[3], f32x4(4.0, 8.0, 12.0, 16.0), 0.0001);
}

test "zmath.matrix.lookToLh" {
    const m = lookToLh(f32x4(0.0, 0.0, -3.0, 1.0), f32x4(0.0, 0.0, 1.0, 0.0), f32x4(0.0, 1.0, 0.0, 0.0));
    try expectVecApproxEqAbs(m[0], f32x4(1.0, 0.0, 0.0, 0.0), 0.001);
    try expectVecApproxEqAbs(m[1], f32x4(0.0, 1.0, 0.0, 0.0), 0.001);
    try expectVecApproxEqAbs(m[2], f32x4(0.0, 0.0, 1.0, 0.0), 0.001);
    try expectVecApproxEqAbs(m[3], f32x4(0.0, 0.0, 3.0, 1.0), 0.001);
}

test "zmath.matrix.determinant" {
    const m = Mat{
        f32x4(10.0, -9.0, -12.0, 1.0),
        f32x4(7.0, -12.0, 11.0, 1.0),
        f32x4(-10.0, 10.0, 3.0, 1.0),
        f32x4(1.0, 2.0, 3.0, 4.0),
    };
    try expectVecApproxEqAbs(determinant(m), splat(F32x4, 2939.0), 0.0001);
}

test "zmath.matrix.inverse" {
    const m = Mat{
        f32x4(10.0, -9.0, -12.0, 1.0),
        f32x4(7.0, -12.0, 11.0, 1.0),
        f32x4(-10.0, 10.0, 3.0, 1.0),
        f32x4(1.0, 2.0, 3.0, 4.0),
    };
    var det: F32x4 = undefined;
    const mi = inverseDet(m, &det);
    try expectVecApproxEqAbs(det, splat(F32x4, 2939.0), 0.0001);

    try expectVecApproxEqAbs(mi[0], f32x4(-0.170806, -0.13576, -0.349439, 0.164001), 0.0001);
    try expectVecApproxEqAbs(mi[1], f32x4(-0.163661, -0.14801, -0.253147, 0.141204), 0.0001);
    try expectVecApproxEqAbs(mi[2], f32x4(-0.0871045, 0.00646478, -0.0785982, 0.0398095), 0.0001);
    try expectVecApproxEqAbs(mi[3], f32x4(0.18986, 0.103096, 0.272882, 0.10854), 0.0001);
}

test "zmath.matrix.matFromAxisAngle" {
    {
        const m0 = matFromAxisAngle(f32x4(1.0, 0.0, 0.0, 0.0), math.pi * 0.25);
        const m1 = rotationX(math.pi * 0.25);
        try expectVecApproxEqAbs(m0[0], m1[0], 0.001);
        try expectVecApproxEqAbs(m0[1], m1[1], 0.001);
        try expectVecApproxEqAbs(m0[2], m1[2], 0.001);
        try expectVecApproxEqAbs(m0[3], m1[3], 0.001);
    }
    {
        const m0 = matFromAxisAngle(f32x4(0.0, 1.0, 0.0, 0.0), math.pi * 0.125);
        const m1 = rotationY(math.pi * 0.125);
        try expectVecApproxEqAbs(m0[0], m1[0], 0.001);
        try expectVecApproxEqAbs(m0[1], m1[1], 0.001);
        try expectVecApproxEqAbs(m0[2], m1[2], 0.001);
        try expectVecApproxEqAbs(m0[3], m1[3], 0.001);
    }
    {
        const m0 = matFromAxisAngle(f32x4(0.0, 0.0, 1.0, 0.0), math.pi * 0.333);
        const m1 = rotationZ(math.pi * 0.333);
        try expectVecApproxEqAbs(m0[0], m1[0], 0.001);
        try expectVecApproxEqAbs(m0[1], m1[1], 0.001);
        try expectVecApproxEqAbs(m0[2], m1[2], 0.001);
        try expectVecApproxEqAbs(m0[3], m1[3], 0.001);
    }
}

test "zmath.matrix.matFromQuat" {
    {
        const m = matFromQuat(f32x4(0.0, 0.0, 0.0, 1.0));
        try expectVecApproxEqAbs(m[0], f32x4(1.0, 0.0, 0.0, 0.0), 0.0001);
        try expectVecApproxEqAbs(m[1], f32x4(0.0, 1.0, 0.0, 0.0), 0.0001);
        try expectVecApproxEqAbs(m[2], f32x4(0.0, 0.0, 1.0, 0.0), 0.0001);
        try expectVecApproxEqAbs(m[3], f32x4(0.0, 0.0, 0.0, 1.0), 0.0001);
    }
}

test "zmath.loadMat" {
    const a = [18]f32{
        1.0,  2.0,  3.0,  4.0,
        5.0,  6.0,  7.0,  8.0,
        9.0,  10.0, 11.0, 12.0,
        13.0, 14.0, 15.0, 16.0,
        17.0, 18.0,
    };
    const m = loadMat(a[1..]);
    try expectVecEqual(m[0], f32x4(2.0, 3.0, 4.0, 5.0));
    try expectVecEqual(m[1], f32x4(6.0, 7.0, 8.0, 9.0));
    try expectVecEqual(m[2], f32x4(10.0, 11.0, 12.0, 13.0));
    try expectVecEqual(m[3], f32x4(14.0, 15.0, 16.0, 17.0));
}

test "zmath.quaternion.mul" {
    {
        const q0 = f32x4(2.0, 3.0, 4.0, 1.0);
        const q1 = f32x4(3.0, 2.0, 1.0, 4.0);
        try expectVecApproxEqAbs(qmul(q0, q1), f32x4(16.0, 4.0, 22.0, -12.0), 0.0001);
    }
}

test "zmath.quaternion.quatToAxisAngle" {
    {
        const q0 = quatFromNormAxisAngle(f32x4(1.0, 0.0, 0.0, 0.0), 0.25 * math.pi);
        var axis: Vec = f32x4(4.0, 3.0, 2.0, 1.0);
        var angle: f32 = 10.0;
        quatToAxisAngle(q0, &axis, &angle);
        try expect(math.approxEqAbs(f32, axis[0], @sin(@as(f32, 0.25) * math.pi * 0.5), 0.0001));
        try expect(axis[1] == 0.0);
        try expect(axis[2] == 0.0);
        try expect(math.approxEqAbs(f32, angle, 0.25 * math.pi, 0.0001));
    }
}

test "zmath.quatFromMat" {
    {
        const q0 = quatFromAxisAngle(f32x4(1.0, 0.0, 0.0, 0.0), 0.25 * math.pi);
        const q1 = quatFromMat(rotationX(0.25 * math.pi));
        try expectVecApproxEqAbs(q0, q1, 0.0001);
    }
    {
        const q0 = quatFromAxisAngle(f32x4(1.0, 2.0, 0.5, 0.0), 0.25 * math.pi);
        const q1 = quatFromMat(matFromAxisAngle(f32x4(1.0, 2.0, 0.5, 0.0), 0.25 * math.pi));
        try expectVecApproxEqAbs(q0, q1, 0.0001);
    }
    {
        const q0 = quatFromRollPitchYaw(0.1 * math.pi, -0.2 * math.pi, 0.3 * math.pi);
        const q1 = quatFromMat(matFromRollPitchYaw(0.1 * math.pi, -0.2 * math.pi, 0.3 * math.pi));
        try expectVecApproxEqAbs(q0, q1, 0.0001);
    }
}

test "zmath.quaternion.quatFromNormAxisAngle" {
    {
        const q0 = quatFromAxisAngle(f32x4(1.0, 0.0, 0.0, 0.0), 0.25 * math.pi);
        const q1 = quatFromAxisAngle(f32x4(0.0, 1.0, 0.0, 0.0), 0.125 * math.pi);
        const m0 = rotationX(0.25 * math.pi);
        const m1 = rotationY(0.125 * math.pi);
        const mr0 = quatToMat(qmul(q0, q1));
        const mr1 = mul(m0, m1);
        try expectVecApproxEqAbs(mr0[0], mr1[0], 0.0001);
        try expectVecApproxEqAbs(mr0[1], mr1[1], 0.0001);
        try expectVecApproxEqAbs(mr0[2], mr1[2], 0.0001);
        try expectVecApproxEqAbs(mr0[3], mr1[3], 0.0001);
    }
    {
        const m0 = quatToMat(quatFromAxisAngle(f32x4(1.0, 2.0, 0.5, 0.0), 0.25 * math.pi));
        const m1 = matFromAxisAngle(f32x4(1.0, 2.0, 0.5, 0.0), 0.25 * math.pi);
        try expectVecApproxEqAbs(m0[0], m1[0], 0.0001);
        try expectVecApproxEqAbs(m0[1], m1[1], 0.0001);
        try expectVecApproxEqAbs(m0[2], m1[2], 0.0001);
        try expectVecApproxEqAbs(m0[3], m1[3], 0.0001);
    }
}

test "zmath.quaternion.inverseQuat" {
    try expectVecApproxEqAbs(
        inverse(f32x4(2.0, 3.0, 4.0, 1.0)),
        f32x4(-1.0 / 15.0, -1.0 / 10.0, -2.0 / 15.0, 1.0 / 30.0),
        0.0001,
    );
    try expectVecApproxEqAbs(inverse(qidentity()), qidentity(), 0.0001);
}

test "zmath.quaternion.rotate" {
    const quat = quatFromRollPitchYaw(0.1 * math.pi, 0.2 * math.pi, 0.3 * math.pi);
    const mat = matFromQuat(quat);
    const forward = f32x4(0.0, 0.0, -1.0, 0.0);
    const up = f32x4(0.0, 1.0, 0.0, 0.0);
    const right = f32x4(1.0, 0.0, 0.0, 0.0);
    try expectVecApproxEqAbs(rotate(quat, forward), mul(forward, mat), 0.0001);
    try expectVecApproxEqAbs(rotate(quat, up), mul(up, mat), 0.0001);
    try expectVecApproxEqAbs(rotate(quat, right), mul(right, mat), 0.0001);
}

test "zmath.quaternion.slerp" {
    const from = f32x4(0.0, 0.0, 0.0, 1.0);
    const to = f32x4(0.5, 0.5, -0.5, 0.5);
    const result = slerp(from, to, 0.5);
    try expectVecApproxEqAbs(result, f32x4(0.28867513, 0.28867513, -0.28867513, 0.86602540), 0.0001);
}

test "zmath.quaternion.quatToRollPitchYaw" {
    {
        const expected = f32x4(0.1 * math.pi, 0.2 * math.pi, 0.3 * math.pi, 0.0);
        const quat = quatFromRollPitchYaw(expected[0], expected[1], expected[2]);
        const result = quatToRollPitchYaw(quat);
        try expectVecApproxEqAbs(loadArr3(result), expected, 0.0001);
    }

    {
        const expected = f32x4(0.3 * math.pi, 0.1 * math.pi, 0.2 * math.pi, 0.0);
        const quat = quatFromRollPitchYaw(expected[0], expected[1], expected[2]);
        const result = quatToRollPitchYaw(quat);
        try expectVecApproxEqAbs(loadArr3(result), expected, 0.0001);
    }

    // North pole singularity
    {
        const angle = f32x4(0.5 * math.pi, 0.2 * math.pi, 0.3 * math.pi, 0.0);
        const expected = f32x4(0.5 * math.pi, -0.1 * math.pi, 0.0, 0.0);
        const quat = quatFromRollPitchYaw(angle[0], angle[1], angle[2]);
        const result = quatToRollPitchYaw(quat);
        try expectVecApproxEqAbs(loadArr3(result), expected, 0.0001);
    }

    // South pole singularity
    {
        const angle = f32x4(-0.5 * math.pi, 0.2 * math.pi, 0.3 * math.pi, 0.0);
        const expected = f32x4(-0.5 * math.pi, 0.5 * math.pi, 0.0, 0.0);
        const quat = quatFromRollPitchYaw(angle[0], angle[1], angle[2]);
        const result = quatToRollPitchYaw(quat);
        try expectVecApproxEqAbs(loadArr3(result), expected, 0.0001);
    }
}

test "zmath.quaternion.quatFromRollPitchYawV" {
    {
        const m0 = quatToMat(quatFromRollPitchYawV(f32x4(0.25 * math.pi, 0.0, 0.0, 0.0)));
        const m1 = rotationX(0.25 * math.pi);
        try expectVecApproxEqAbs(m0[0], m1[0], 0.0001);
        try expectVecApproxEqAbs(m0[1], m1[1], 0.0001);
        try expectVecApproxEqAbs(m0[2], m1[2], 0.0001);
        try expectVecApproxEqAbs(m0[3], m1[3], 0.0001);
    }
    {
        const m0 = quatToMat(quatFromRollPitchYaw(0.1 * math.pi, 0.2 * math.pi, 0.3 * math.pi));
        const m1 = mul(
            rotationZ(0.3 * math.pi),
            mul(rotationX(0.1 * math.pi), rotationY(0.2 * math.pi)),
        );
        try expectVecApproxEqAbs(m0[0], m1[0], 0.0001);
        try expectVecApproxEqAbs(m0[1], m1[1], 0.0001);
        try expectVecApproxEqAbs(m0[2], m1[2], 0.0001);
        try expectVecApproxEqAbs(m0[3], m1[3], 0.0001);
    }
}

test "zmath.color.rgbToHsl" {
    try expectVecApproxEqAbs(rgbToHsl(f32x4(0.2, 0.4, 0.8, 1.0)), f32x4(0.6111, 0.6, 0.5, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsl(f32x4(1.0, 0.0, 0.0, 0.5)), f32x4(0.0, 1.0, 0.5, 0.5), 0.0001);
    try expectVecApproxEqAbs(rgbToHsl(f32x4(0.0, 1.0, 0.0, 0.25)), f32x4(0.3333, 1.0, 0.5, 0.25), 0.0001);
    try expectVecApproxEqAbs(rgbToHsl(f32x4(0.0, 0.0, 1.0, 1.0)), f32x4(0.6666, 1.0, 0.5, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsl(f32x4(0.0, 0.0, 0.0, 1.0)), f32x4(0.0, 0.0, 0.0, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsl(f32x4(1.0, 1.0, 1.0, 1.0)), f32x4(0.0, 0.0, 1.0, 1.0), 0.0001);
}

test "zmath.color.hslToRgb" {
    try expectVecApproxEqAbs(f32x4(0.2, 0.4, 0.8, 1.0), hslToRgb(f32x4(0.6111, 0.6, 0.5, 1.0)), 0.0001);
    try expectVecApproxEqAbs(f32x4(1.0, 0.0, 0.0, 0.5), hslToRgb(f32x4(0.0, 1.0, 0.5, 0.5)), 0.0001);
    try expectVecApproxEqAbs(f32x4(0.0, 1.0, 0.0, 0.25), hslToRgb(f32x4(0.3333, 1.0, 0.5, 0.25)), 0.0005);
    try expectVecApproxEqAbs(f32x4(0.0, 0.0, 1.0, 1.0), hslToRgb(f32x4(0.6666, 1.0, 0.5, 1.0)), 0.0005);
    try expectVecApproxEqAbs(f32x4(0.0, 0.0, 0.0, 1.0), hslToRgb(f32x4(0.0, 0.0, 0.0, 1.0)), 0.0001);
    try expectVecApproxEqAbs(f32x4(1.0, 1.0, 1.0, 1.0), hslToRgb(f32x4(0.0, 0.0, 1.0, 1.0)), 0.0001);
    try expectVecApproxEqAbs(hslToRgb(rgbToHsl(f32x4(1.0, 1.0, 1.0, 1.0))), f32x4(1.0, 1.0, 1.0, 1.0), 0.0005);
    try expectVecApproxEqAbs(
        hslToRgb(rgbToHsl(f32x4(0.82198, 0.1839, 0.632, 1.0))),
        f32x4(0.82198, 0.1839, 0.632, 1.0),
        0.0005,
    );
    try expectVecApproxEqAbs(
        rgbToHsl(hslToRgb(f32x4(0.82198, 0.1839, 0.632, 1.0))),
        f32x4(0.82198, 0.1839, 0.632, 1.0),
        0.0005,
    );
    try expectVecApproxEqAbs(
        rgbToHsl(hslToRgb(f32x4(0.1839, 0.82198, 0.632, 1.0))),
        f32x4(0.1839, 0.82198, 0.632, 1.0),
        0.0005,
    );
    try expectVecApproxEqAbs(
        hslToRgb(rgbToHsl(f32x4(0.1839, 0.632, 0.82198, 1.0))),
        f32x4(0.1839, 0.632, 0.82198, 1.0),
        0.0005,
    );
}

test "zmath.color.rgbToHsv" {
    try expectVecApproxEqAbs(rgbToHsv(f32x4(0.2, 0.4, 0.8, 1.0)), f32x4(0.6111, 0.75, 0.8, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(0.4, 0.2, 0.8, 1.0)), f32x4(0.7222, 0.75, 0.8, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(0.4, 0.8, 0.2, 1.0)), f32x4(0.2777, 0.75, 0.8, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(1.0, 0.0, 0.0, 0.5)), f32x4(0.0, 1.0, 1.0, 0.5), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(0.0, 1.0, 0.0, 0.25)), f32x4(0.3333, 1.0, 1.0, 0.25), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(0.0, 0.0, 1.0, 1.0)), f32x4(0.6666, 1.0, 1.0, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(0.0, 0.0, 0.0, 1.0)), f32x4(0.0, 0.0, 0.0, 1.0), 0.0001);
    try expectVecApproxEqAbs(rgbToHsv(f32x4(1.0, 1.0, 1.0, 1.0)), f32x4(0.0, 0.0, 1.0, 1.0), 0.0001);
}

test "zmath.color.hsvToRgb" {
    const epsilon = 0.0005;
    try expectVecApproxEqAbs(f32x4(0.2, 0.4, 0.8, 1.0), hsvToRgb(f32x4(0.6111, 0.75, 0.8, 1.0)), epsilon);
    try expectVecApproxEqAbs(f32x4(0.4, 0.2, 0.8, 1.0), hsvToRgb(f32x4(0.7222, 0.75, 0.8, 1.0)), epsilon);
    try expectVecApproxEqAbs(f32x4(0.4, 0.8, 0.2, 1.0), hsvToRgb(f32x4(0.2777, 0.75, 0.8, 1.0)), epsilon);
    try expectVecApproxEqAbs(f32x4(1.0, 0.0, 0.0, 0.5), hsvToRgb(f32x4(0.0, 1.0, 1.0, 0.5)), epsilon);
    try expectVecApproxEqAbs(f32x4(0.0, 1.0, 0.0, 0.25), hsvToRgb(f32x4(0.3333, 1.0, 1.0, 0.25)), epsilon);
    try expectVecApproxEqAbs(f32x4(0.0, 0.0, 1.0, 1.0), hsvToRgb(f32x4(0.6666, 1.0, 1.0, 1.0)), epsilon);
    try expectVecApproxEqAbs(f32x4(0.0, 0.0, 0.0, 1.0), hsvToRgb(f32x4(0.0, 0.0, 0.0, 1.0)), epsilon);
    try expectVecApproxEqAbs(f32x4(1.0, 1.0, 1.0, 1.0), hsvToRgb(f32x4(0.0, 0.0, 1.0, 1.0)), epsilon);
    try expectVecApproxEqAbs(
        hsvToRgb(rgbToHsv(f32x4(0.1839, 0.632, 0.82198, 1.0))),
        f32x4(0.1839, 0.632, 0.82198, 1.0),
        epsilon,
    );
    try expectVecApproxEqAbs(
        hsvToRgb(rgbToHsv(f32x4(0.82198, 0.1839, 0.632, 1.0))),
        f32x4(0.82198, 0.1839, 0.632, 1.0),
        epsilon,
    );
    try expectVecApproxEqAbs(
        rgbToHsv(hsvToRgb(f32x4(0.82198, 0.1839, 0.632, 1.0))),
        f32x4(0.82198, 0.1839, 0.632, 1.0),
        epsilon,
    );
    try expectVecApproxEqAbs(
        rgbToHsv(hsvToRgb(f32x4(0.1839, 0.82198, 0.632, 1.0))),
        f32x4(0.1839, 0.82198, 0.632, 1.0),
        epsilon,
    );
}

test "zmath.color.rgbToSrgb" {
    const epsilon = 0.001;
    try expectVecApproxEqAbs(rgbToSrgb(f32x4(0.2, 0.4, 0.8, 1.0)), f32x4(0.484, 0.665, 0.906, 1.0), epsilon);
}

test "zmath.color.srgbToRgb" {
    const epsilon = 0.0007;
    try expectVecApproxEqAbs(f32x4(0.2, 0.4, 0.8, 1.0), srgbToRgb(f32x4(0.484, 0.665, 0.906, 1.0)), epsilon);
    try expectVecApproxEqAbs(
        rgbToSrgb(srgbToRgb(f32x4(0.1839, 0.82198, 0.632, 1.0))),
        f32x4(0.1839, 0.82198, 0.632, 1.0),
        epsilon,
    );
}

test "zmath.util.mat.translation" {
    // zig fmt: off
        const mat_data = [18]f32{
            1.0,
            2.0, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 9.0,
            10.0,11.0, 12.0,13.0,
            14.0, 15.0, 16.0, 17.0,
            18.0,
        };
        // zig fmt: on
    const mat = loadMat(mat_data[1..]);
    try expectVecApproxEqAbs(getTranslationVec(mat), f32x4(14.0, 15.0, 16.0, 0.0), 0.0001);
}

test "zmath.util.mat.scale" {
    const mat = mul(scaling(3, 4, 5), translation(6, 7, 8));
    const scale = getScaleVec(mat);
    try expectVecApproxEqAbs(scale, f32x4(3.0, 4.0, 5.0, 0.0), 0.0001);
}

test "zmath.util.mat.rotation" {
    const rotate_origin = matFromRollPitchYaw(0.1, 1.2, 2.3);
    const mat = mul(mul(rotate_origin, scaling(3, 4, 5)), translation(6, 7, 8));
    const rotate_get = getRotationQuat(mat);
    const v0 = mul(f32x4s(1), rotate_origin);
    const v1 = mul(f32x4s(1), quatToMat(rotate_get));
    try expectVecApproxEqAbs(v0, v1, 0.0001);
}

test "zmath.util.mat.z_vec" {
    const degToRad = std.math.degreesToRadians;
    var mat_id = identity();
    var z_vec = getAxisZ(mat_id);
    try expectVecApproxEqAbs(z_vec, f32x4(0.0, 0.0, 1.0, 0), 0.0001);
    const rot_yaw = rotationY(degToRad(90));
    mat_id = mul(mat_id, rot_yaw);
    z_vec = getAxisZ(mat_id);
    try expectVecApproxEqAbs(z_vec, f32x4(1.0, 0.0, 0.0, 0), 0.0001);
}

test "zmath.util.mat.y_vec" {
    const degToRad = std.math.degreesToRadians;
    var mat_id = identity();
    var y_vec = getAxisY(mat_id);
    try expectVecApproxEqAbs(y_vec, f32x4(0.0, 1.0, 0.0, 0), 0.01);
    const rot_yaw = rotationY(degToRad(90));
    mat_id = mul(mat_id, rot_yaw);
    y_vec = getAxisY(mat_id);
    try expectVecApproxEqAbs(y_vec, f32x4(0.0, 1.0, 0.0, 0), 0.01);
    const rot_pitch = rotationX(degToRad(90));
    mat_id = mul(mat_id, rot_pitch);
    y_vec = getAxisY(mat_id);
    try expectVecApproxEqAbs(y_vec, f32x4(0.0, 0.0, 1.0, 0), 0.01);
}

test "zmath.util.mat.right" {
    const degToRad = std.math.degreesToRadians;
    var mat_id = identity();
    var right = getAxisX(mat_id);
    try expectVecApproxEqAbs(right, f32x4(1.0, 0.0, 0.0, 0), 0.01);
    const rot_yaw = rotationY(degToRad(90));
    mat_id = mul(mat_id, rot_yaw);
    right = getAxisX(mat_id);
    try expectVecApproxEqAbs(right, f32x4(0.0, 0.0, -1.0, 0), 0.01);
    const rot_pitch = rotationX(degToRad(90));
    mat_id = mul(mat_id, rot_pitch);
    right = getAxisX(mat_id);
    try expectVecApproxEqAbs(right, f32x4(0.0, 1.0, 0.0, 0), 0.01);
}
