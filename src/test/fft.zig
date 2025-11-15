const std = @import("std");
const zm = @import("../root.zig");
const core = zm.core;
const fft_api = zm.fft;
const fft_internal = @import("../fft.zig");
const testing = @import("../testing.zig");

const expectVecApproxEqAbs = testing.expectVecApproxEqAbs;
const expect = std.testing.expect;

const F32x4 = core.F32x4;
const f32x4 = core.f32x4;
const f32x4s = core.f32x4s;
const approxEqAbs = testing.approxEqAbs;

const fft4 = fft_internal.fft4;
const fft8 = fft_internal.fft8;
const fft16 = fft_internal.fft16;
const fftN = fft_internal.fftN;
const fftInitUnityTable = fft_api.fftInitUnityTable;
const fftUnswizzle = fft_internal.fftUnswizzle;
const fftButterflyDit4_1 = fft_internal.fftButterflyDit4_1;
const fftButterflyDit4_4 = fft_internal.fftButterflyDit4_4;
const fftRun = fft_api.fft;
const ifftRun = fft_api.ifft;

test "zmath.fft4" {
    const epsilon = 0.0001;
    var re = [_]F32x4{f32x4(1.0, 2.0, 3.0, 4.0)};
    var im = [_]F32x4{f32x4s(0.0)};
    fft4(re[0..], im[0..], 1);

    var re_uns: [1]F32x4 = undefined;
    var im_uns: [1]F32x4 = undefined;
    fftUnswizzle(re[0..], re_uns[0..]);
    fftUnswizzle(im[0..], im_uns[0..]);

    try expectVecApproxEqAbs(re_uns[0], f32x4(10.0, -2.0, -2.0, -2.0), epsilon);
    try expectVecApproxEqAbs(im_uns[0], f32x4(0.0, 2.0, 0.0, -2.0), epsilon);
}

test "zmath.fft8" {
    const epsilon = 0.0001;
    var re = [_]F32x4{ f32x4(1.0, 2.0, 3.0, 4.0), f32x4(5.0, 6.0, 7.0, 8.0) };
    var im = [_]F32x4{ f32x4s(0.0), f32x4s(0.0) };
    fft8(re[0..], im[0..], 1);

    var re_uns: [2]F32x4 = undefined;
    var im_uns: [2]F32x4 = undefined;
    fftUnswizzle(re[0..], re_uns[0..]);
    fftUnswizzle(im[0..], im_uns[0..]);

    try expectVecApproxEqAbs(re_uns[0], f32x4(36.0, -4.0, -4.0, -4.0), epsilon);
    try expectVecApproxEqAbs(re_uns[1], f32x4(-4.0, -4.0, -4.0, -4.0), epsilon);
    try expectVecApproxEqAbs(im_uns[0], f32x4(0.0, 9.656854, 4.0, 1.656854), epsilon);
    try expectVecApproxEqAbs(im_uns[1], f32x4(0.0, -1.656854, -4.0, -9.656854), epsilon);
}

test "zmath.fft16" {
    const epsilon = 0.0001;
    var re = [_]F32x4{
        f32x4(1.0, 2.0, 3.0, 4.0),
        f32x4(5.0, 6.0, 7.0, 8.0),
        f32x4(9.0, 10.0, 11.0, 12.0),
        f32x4(13.0, 14.0, 15.0, 16.0),
    };
    var im = [_]F32x4{ f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0) };
    fft16(re[0..], im[0..], 1);

    var re_uns: [4]F32x4 = undefined;
    var im_uns: [4]F32x4 = undefined;
    fftUnswizzle(re[0..], re_uns[0..]);
    fftUnswizzle(im[0..], im_uns[0..]);

    try expectVecApproxEqAbs(re_uns[0], f32x4(136.0, -8.0, -8.0, -8.0), epsilon);
    try expectVecApproxEqAbs(re_uns[1], f32x4(-8.0, -8.0, -8.0, -8.0), epsilon);
    try expectVecApproxEqAbs(re_uns[2], f32x4(-8.0, -8.0, -8.0, -8.0), epsilon);
    try expectVecApproxEqAbs(re_uns[3], f32x4(-8.0, -8.0, -8.0, -8.0), epsilon);
    try expectVecApproxEqAbs(im_uns[0], f32x4(0.0, 40.218716, 19.313708, 11.972846), epsilon);
    try expectVecApproxEqAbs(im_uns[1], f32x4(8.0, 5.345429, 3.313708, 1.591299), epsilon);
    try expectVecApproxEqAbs(im_uns[2], f32x4(0.0, -1.591299, -3.313708, -5.345429), epsilon);
    try expectVecApproxEqAbs(im_uns[3], f32x4(-8.0, -11.972846, -19.313708, -40.218716), epsilon);
}

test "zmath.fftN" {
    var unity_table: [128]F32x4 = undefined;
    const epsilon = 0.0001;

    // 32 samples
    {
        var re = [_]F32x4{
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
        };
        var im = [_]F32x4{
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
        };

        fftInitUnityTable(unity_table[0..32]);
        fftRun(re[0..], im[0..], unity_table[0..32]);

        try expectVecApproxEqAbs(re[0], f32x4(528.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[1], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[2], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[3], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[4], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[5], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[6], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(re[7], f32x4(-16.0, -16.0, -16.0, -16.0), epsilon);
        try expectVecApproxEqAbs(im[0], f32x4(0.0, 162.450726, 80.437432, 52.744931), epsilon);
        try expectVecApproxEqAbs(im[1], f32x4(38.627417, 29.933895, 23.945692, 19.496056), epsilon);
        try expectVecApproxEqAbs(im[2], f32x4(16.0, 13.130861, 10.690858, 8.552178), epsilon);
        try expectVecApproxEqAbs(im[3], f32x4(6.627417, 4.853547, 3.182598, 1.575862), epsilon);
        try expectVecApproxEqAbs(im[4], f32x4(0.0, -1.575862, -3.182598, -4.853547), epsilon);
        try expectVecApproxEqAbs(im[5], f32x4(-6.627417, -8.552178, -10.690858, -13.130861), epsilon);
        try expectVecApproxEqAbs(im[6], f32x4(-16.0, -19.496056, -23.945692, -29.933895), epsilon);
        try expectVecApproxEqAbs(im[7], f32x4(-38.627417, -52.744931, -80.437432, -162.450726), epsilon);
    }

    // 64 samples
    {
        var re = [_]F32x4{
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
        };
        var im = [_]F32x4{
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
        };

        fftInitUnityTable(unity_table[0..64]);
        fftRun(re[0..], im[0..], unity_table[0..64]);

        try expectVecApproxEqAbs(re[0], f32x4(1056.0, 0.0, -32.0, 0.0), epsilon);
        var i: u32 = 1;
        while (i < 16) : (i += 1) {
            try expectVecApproxEqAbs(re[i], f32x4(-32.0, 0.0, -32.0, 0.0), epsilon);
        }

        const expected = [_]f32{
            0.0,        0.0,      324.901452,  0.000000, 160.874864,  0.0,      105.489863,  0.000000,
            77.254834,  0.0,      59.867789,   0.0,      47.891384,   0.0,      38.992113,   0.0,
            32.000000,  0.000000, 26.261721,   0.000000, 21.381716,   0.000000, 17.104356,   0.000000,
            13.254834,  0.000000, 9.707094,    0.000000, 6.365196,    0.000000, 3.151725,    0.000000,
            0.000000,   0.000000, -3.151725,   0.000000, -6.365196,   0.000000, -9.707094,   0.000000,
            -13.254834, 0.000000, -17.104356,  0.000000, -21.381716,  0.000000, -26.261721,  0.000000,
            -32.000000, 0.000000, -38.992113,  0.000000, -47.891384,  0.000000, -59.867789,  0.000000,
            -77.254834, 0.000000, -105.489863, 0.000000, -160.874864, 0.000000, -324.901452, 0.000000,
        };
        for (expected, 0..) |e, ie| {
            try expect(std.math.approxEqAbs(f32, e, im[(ie / 4)][ie % 4], epsilon));
        }
    }

    // 128 samples
    {
        var re = [_]F32x4{
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
        };
        var im = [_]F32x4{
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
        };

        fftInitUnityTable(unity_table[0..128]);
        fftRun(re[0..], im[0..], unity_table[0..128]);

        try expectVecApproxEqAbs(re[0], f32x4(2112.0, 0.0, 0.0, 0.0), epsilon);
        var i: u32 = 1;
        while (i < 32) : (i += 1) {
            try expectVecApproxEqAbs(re[i], f32x4(-64.0, 0.0, 0.0, 0.0), epsilon);
        }

        const expected = [_]f32{
            0.000000,    0.000000, 0.000000, 0.000000, 649.802905,  0.000000, 0.000000, 0.000000,
            321.749727,  0.000000, 0.000000, 0.000000, 210.979725,  0.000000, 0.000000, 0.000000,
            154.509668,  0.000000, 0.000000, 0.000000, 119.735578,  0.000000, 0.000000, 0.000000,
            95.782769,   0.000000, 0.000000, 0.000000, 77.984226,   0.000000, 0.000000, 0.000000,
            64.000000,   0.000000, 0.000000, 0.000000, 52.523443,   0.000000, 0.000000, 0.000000,
            42.763433,   0.000000, 0.000000, 0.000000, 34.208713,   0.000000, 0.000000, 0.000000,
            26.509668,   0.000000, 0.000000, 0.000000, 19.414188,   0.000000, 0.000000, 0.000000,
            12.730392,   0.000000, 0.000000, 0.000000, 6.303450,    0.000000, 0.000000, 0.000000,
            0.000000,    0.000000, 0.000000, 0.000000, -6.303450,   0.000000, 0.000000, 0.000000,
            -12.730392,  0.000000, 0.000000, 0.000000, -19.414188,  0.000000, 0.000000, 0.000000,
            -26.509668,  0.000000, 0.000000, 0.000000, -34.208713,  0.000000, 0.000000, 0.000000,
            -42.763433,  0.000000, 0.000000, 0.000000, -52.523443,  0.000000, 0.000000, 0.000000,
            -64.000000,  0.000000, 0.000000, 0.000000, -77.984226,  0.000000, 0.000000, 0.000000,
            -95.782769,  0.000000, 0.000000, 0.000000, -119.735578, 0.000000, 0.000000, 0.000000,
            -154.509668, 0.000000, 0.000000, 0.000000, -210.979725, 0.000000, 0.000000, 0.000000,
            -321.749727, 0.000000, 0.000000, 0.000000, -649.802905, 0.000000, 0.000000, 0.000000,
        };
        for (expected, 0..) |e, ie| {
            try expect(std.math.approxEqAbs(f32, e, im[(ie / 4)][ie % 4], epsilon));
        }
    }
}

test "zmath.ifft" {
    var unity_table: [512]F32x4 = undefined;
    const epsilon = 0.0001;

    // 64 samples
    {
        var re = [_]F32x4{
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
            f32x4(1.0, 2.0, 3.0, 4.0),     f32x4(5.0, 6.0, 7.0, 8.0),
            f32x4(9.0, 10.0, 11.0, 12.0),  f32x4(13.0, 14.0, 15.0, 16.0),
            f32x4(17.0, 18.0, 19.0, 20.0), f32x4(21.0, 22.0, 23.0, 24.0),
            f32x4(25.0, 26.0, 27.0, 28.0), f32x4(29.0, 30.0, 31.0, 32.0),
        };
        var im = [_]F32x4{
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
            f32x4s(0.0), f32x4s(0.0), f32x4s(0.0), f32x4s(0.0),
        };

        fftInitUnityTable(unity_table[0..64]);
        fftRun(re[0..], im[0..], unity_table[0..64]);

        try expectVecApproxEqAbs(re[0], f32x4(1056.0, 0.0, -32.0, 0.0), epsilon);
        var i: u32 = 1;
        while (i < 16) : (i += 1) {
            try expectVecApproxEqAbs(re[i], f32x4(-32.0, 0.0, -32.0, 0.0), epsilon);
        }

        ifftRun(re[0..], im[0..], unity_table[0..64]);

        try expectVecApproxEqAbs(re[0], f32x4(1.0, 2.0, 3.0, 4.0), epsilon);
        try expectVecApproxEqAbs(re[1], f32x4(5.0, 6.0, 7.0, 8.0), epsilon);
        try expectVecApproxEqAbs(re[2], f32x4(9.0, 10.0, 11.0, 12.0), epsilon);
        try expectVecApproxEqAbs(re[3], f32x4(13.0, 14.0, 15.0, 16.0), epsilon);
        try expectVecApproxEqAbs(re[4], f32x4(17.0, 18.0, 19.0, 20.0), epsilon);
        try expectVecApproxEqAbs(re[5], f32x4(21.0, 22.0, 23.0, 24.0), epsilon);
        try expectVecApproxEqAbs(re[6], f32x4(25.0, 26.0, 27.0, 28.0), epsilon);
        try expectVecApproxEqAbs(re[7], f32x4(29.0, 30.0, 31.0, 32.0), epsilon);
    }

    // 512 samples
    {
        var re: [128]F32x4 = undefined;
        var im = [_]F32x4{f32x4s(0.0)} ** 128;

        for (&re, 0..) |*v, i| {
            const f = @as(f32, @floatFromInt(i * 4));
            v.* = f32x4(f + 1.0, f + 2.0, f + 3.0, f + 4.0);
        }

        fftInitUnityTable(unity_table[0..512]);
        fftRun(re[0..], im[0..], unity_table[0..512]);

        for (re, 0..) |v, i| {
            const f = @as(f32, @floatFromInt(i * 4));
            try expect(!approxEqAbs(v, f32x4(f + 1.0, f + 2.0, f + 3.0, f + 4.0), epsilon));
        }

        ifftRun(re[0..], im[0..], unity_table[0..512]);

        for (re, 0..) |v, i| {
            const f = @as(f32, @floatFromInt(i * 4));
            try expectVecApproxEqAbs(v, f32x4(f + 1.0, f + 2.0, f + 3.0, f + 4.0), epsilon);
        }
    }
}
