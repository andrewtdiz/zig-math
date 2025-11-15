const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

const core = @import("core.zig");
const vector = @import("vector.zig");

const F32x4 = core.F32x4;
const f32x4 = core.f32x4;
const f32x4s = core.f32x4s;
const expect = std.testing.expect;
const approxEqAbs = core.approxEqAbs;

const mulAdd = vector.mulAdd;
const cmulSoa = vector.cmulSoa;
const swizzle = vector.swizzle;
const sincos = vector.sincos;

fn fftButterflyDit4_1(re0: *F32x4, im0: *F32x4) void {
    const re0l = swizzle(re0.*, .x, .x, .y, .y);
    const re0h = swizzle(re0.*, .z, .z, .w, .w);

    const im0l = swizzle(im0.*, .x, .x, .y, .y);
    const im0h = swizzle(im0.*, .z, .z, .w, .w);

    const re_temp = mulAdd(re0h, f32x4(1.0, -1.0, 1.0, -1.0), re0l);
    const im_temp = mulAdd(im0h, f32x4(1.0, -1.0, 1.0, -1.0), im0l);

    const re_shuf0 = @shuffle(f32, re_temp, im_temp, [4]i32{ 2, 3, ~@as(i32, 2), ~@as(i32, 3) });
    const re_shuf = swizzle(re_shuf0, .x, .w, .x, .w);
    const im_shuf = swizzle(re_shuf0, .z, .y, .z, .y);

    const re_templ = swizzle(re_temp, .x, .y, .x, .y);
    const im_templ = swizzle(im_temp, .x, .y, .x, .y);

    re0.* = mulAdd(re_shuf, f32x4(1.0, 1.0, -1.0, -1.0), re_templ);
    im0.* = mulAdd(im_shuf, f32x4(1.0, -1.0, -1.0, 1.0), im_templ);
}

fn fftButterflyDit4_4(
    re0: *F32x4,
    re1: *F32x4,
    re2: *F32x4,
    re3: *F32x4,
    im0: *F32x4,
    im1: *F32x4,
    im2: *F32x4,
    im3: *F32x4,
    unity_table_re: []const F32x4,
    unity_table_im: []const F32x4,
    stride: u32,
    last: bool,
) void {
    const re_temp0 = re0.* + re2.*;
    const im_temp0 = im0.* + im2.*;

    const re_temp2 = re1.* + re3.*;
    const im_temp2 = im1.* + im3.*;

    const re_temp1 = re0.* - re2.*;
    const im_temp1 = im0.* - im2.*;

    const re_temp3 = re1.* - re3.*;
    const im_temp3 = im1.* - im3.*;

    var re_temp4 = re_temp0 + re_temp2;
    var im_temp4 = im_temp0 + im_temp2;

    var re_temp5 = re_temp1 + im_temp3;
    var im_temp5 = im_temp1 - re_temp3;

    var re_temp6 = re_temp0 - re_temp2;
    var im_temp6 = im_temp0 - im_temp2;

    var re_temp7 = re_temp1 - im_temp3;
    var im_temp7 = im_temp1 + re_temp3;

    {
        const re_im = cmulSoa(re_temp5, im_temp5, unity_table_re[stride], unity_table_im[stride]);
        re_temp5 = re_im[0];
        im_temp5 = re_im[1];
    }
    {
        const re_im = cmulSoa(re_temp6, im_temp6, unity_table_re[stride * 2], unity_table_im[stride * 2]);
        re_temp6 = re_im[0];
        im_temp6 = re_im[1];
    }
    {
        const re_im = cmulSoa(re_temp7, im_temp7, unity_table_re[stride * 3], unity_table_im[stride * 3]);
        re_temp7 = re_im[0];
        im_temp7 = re_im[1];
    }

    if (last) {
        fftButterflyDit4_1(&re_temp4, &im_temp4);
        fftButterflyDit4_1(&re_temp5, &im_temp5);
        fftButterflyDit4_1(&re_temp6, &im_temp6);
        fftButterflyDit4_1(&re_temp7, &im_temp7);
    }

    re0.* = re_temp4;
    im0.* = im_temp4;

    re1.* = re_temp5;
    im1.* = im_temp5;

    re2.* = re_temp6;
    im2.* = im_temp6;

    re3.* = re_temp7;
    im3.* = im_temp7;
}

pub fn fft4(re: []F32x4, im: []F32x4, count: u32) void {
    assert(std.math.isPowerOfTwo(count));
    assert(re.len >= count);
    assert(im.len >= count);

    var index: u32 = 0;
    while (index < count) : (index += 1) {
        fftButterflyDit4_1(&re[index], &im[index]);
    }
}

pub fn fft8(re: []F32x4, im: []F32x4, count: u32) void {
    assert(std.math.isPowerOfTwo(count));
    assert(re.len >= 2 * count);
    assert(im.len >= 2 * count);

    var index: u32 = 0;
    while (index < count) : (index += 1) {
        var pre = re[index * 2 ..];
        var pim = im[index * 2 ..];

        var odds_re = @shuffle(f32, pre[0], pre[1], [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });
        var evens_re = @shuffle(f32, pre[0], pre[1], [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });
        var odds_im = @shuffle(f32, pim[0], pim[1], [4]i32{ 1, 3, ~@as(i32, 1), ~@as(i32, 3) });
        var evens_im = @shuffle(f32, pim[0], pim[1], [4]i32{ 0, 2, ~@as(i32, 0), ~@as(i32, 2) });
        fftButterflyDit4_1(&odds_re, &odds_im);
        fftButterflyDit4_1(&evens_re, &evens_im);

        {
            const re_im = cmulSoa(
                odds_re,
                odds_im,
                f32x4(1.0, 0.70710677, 0.0, -0.70710677),
                f32x4(0.0, -0.70710677, -1.0, -0.70710677),
            );
            pre[0] = evens_re + re_im[0];
            pim[0] = evens_im + re_im[1];
        }
        {
            const re_im = cmulSoa(
                odds_re,
                odds_im,
                f32x4(-1.0, -0.70710677, 0.0, 0.70710677),
                f32x4(0.0, 0.70710677, 1.0, 0.70710677),
            );
            pre[1] = evens_re + re_im[0];
            pim[1] = evens_im + re_im[1];
        }
    }
}

pub fn fft16(re: []F32x4, im: []F32x4, count: u32) void {
    assert(std.math.isPowerOfTwo(count));
    assert(re.len >= 4 * count);
    assert(im.len >= 4 * count);

    const static = struct {
        const unity_table_re = [4]F32x4{
            f32x4(1.0, 1.0, 1.0, 1.0),
            f32x4(1.0, 0.92387950, 0.70710677, 0.38268343),
            f32x4(1.0, 0.70710677, -4.3711388e-008, -0.70710677),
            f32x4(1.0, 0.38268343, -0.70710677, -0.92387950),
        };
        const unity_table_im = [4]F32x4{
            f32x4(-0.0, -0.0, -0.0, -0.0),
            f32x4(-0.0, -0.38268343, -0.70710677, -0.92387950),
            f32x4(-0.0, -0.70710677, -1.0, -0.70710677),
            f32x4(-0.0, -0.92387950, -0.70710677, 0.38268343),
        };
    };

    var index: u32 = 0;
    while (index < count) : (index += 1) {
        fftButterflyDit4_4(
            &re[index * 4],
            &re[index * 4 + 1],
            &re[index * 4 + 2],
            &re[index * 4 + 3],
            &im[index * 4],
            &im[index * 4 + 1],
            &im[index * 4 + 2],
            &im[index * 4 + 3],
            static.unity_table_re[0..],
            static.unity_table_im[0..],
            1,
            true,
        );
    }
}

fn fftN(re: []F32x4, im: []F32x4, unity_table: []const F32x4, length: u32, count: u32) void {
    assert(length > 16);
    assert(std.math.isPowerOfTwo(length));
    assert(std.math.isPowerOfTwo(count));
    assert(re.len >= length * count / 4);
    assert(re.len == im.len);

    const total = count * length;
    const total_vectors = total / 4;
    const stage_vectors = length / 4;
    const stage_vectors_mask = stage_vectors - 1;
    const stride = length / 16;
    const stride_mask = stride - 1;
    const stride_inv_mask = ~stride_mask;

    var unity_table_re = unity_table;
    var unity_table_im = unity_table[length / 4 ..];

    var index: u32 = 0;
    while (index < total_vectors / 4) : (index += 1) {
        const n = (index & stride_inv_mask) * 4 + (index & stride_mask);
        fftButterflyDit4_4(
            &re[n],
            &re[n + stride],
            &re[n + stride * 2],
            &re[n + stride * 3],
            &im[n],
            &im[n + stride],
            &im[n + stride * 2],
            &im[n + stride * 3],
            unity_table_re[(n & stage_vectors_mask)..],
            unity_table_im[(n & stage_vectors_mask)..],
            stride,
            false,
        );
    }

    if (length > 16 * 4) {
        fftN(re, im, unity_table[(length / 2)..], length / 4, count * 4);
    } else if (length == 16 * 4) {
        fft16(re, im, count * 4);
    } else if (length == 8 * 4) {
        fft8(re, im, count * 4);
    } else if (length == 4 * 4) {
        fft4(re, im, count * 4);
    }
}

pub fn fftUnswizzle(input: []const F32x4, output: []F32x4) void {
    assert(std.math.isPowerOfTwo(input.len));
    assert(input.len == output.len);
    assert(input.ptr != output.ptr);

    const log2_length = std.math.log2_int(usize, input.len * 4);
    assert(log2_length >= 2);

    const length = input.len;

    const f32_output = @as([*]f32, @ptrCast(output.ptr))[0 .. output.len * 4];

    const static = struct {
        const swizzle_table = [256]u8{
            0x00, 0x40, 0x80, 0xC0, 0x10, 0x50, 0x90, 0xD0, 0x20, 0x60, 0xA0, 0xE0, 0x30, 0x70, 0xB0, 0xF0,
            0x04, 0x44, 0x84, 0xC4, 0x14, 0x54, 0x94, 0xD4, 0x24, 0x64, 0xA4, 0xE4, 0x34, 0x74, 0xB4, 0xF4,
            0x08, 0x48, 0x88, 0xC8, 0x18, 0x58, 0x98, 0xD8, 0x28, 0x68, 0xA8, 0xE8, 0x38, 0x78, 0xB8, 0xF8,
            0x0C, 0x4C, 0x8C, 0xCC, 0x1C, 0x5C, 0x9C, 0xDC, 0x2C, 0x6C, 0xAC, 0xEC, 0x3C, 0x7C, 0xBC, 0xFC,
            0x01, 0x41, 0x81, 0xC1, 0x11, 0x51, 0x91, 0xD1, 0x21, 0x61, 0xA1, 0xE1, 0x31, 0x71, 0xB1, 0xF1,
            0x05, 0x45, 0x85, 0xC5, 0x15, 0x55, 0x95, 0xD5, 0x25, 0x65, 0xA5, 0xE5, 0x35, 0x75, 0xB5, 0xF5,
            0x09, 0x49, 0x89, 0xC9, 0x19, 0x59, 0x99, 0xD9, 0x29, 0x69, 0xA9, 0xE9, 0x39, 0x79, 0xB9, 0xF9,
            0x0D, 0x4D, 0x8D, 0xCD, 0x1D, 0x5D, 0x9D, 0xDD, 0x2D, 0x6D, 0xAD, 0xED, 0x3D, 0x7D, 0xBD, 0xFD,
            0x02, 0x42, 0x82, 0xC2, 0x12, 0x52, 0x92, 0xD2, 0x22, 0x62, 0xA2, 0xE2, 0x32, 0x72, 0xB2, 0xF2,
            0x06, 0x46, 0x86, 0xC6, 0x16, 0x56, 0x96, 0xD6, 0x26, 0x66, 0xA6, 0xE6, 0x36, 0x76, 0xB6, 0xF6,
            0x0A, 0x4A, 0x8A, 0xCA, 0x1A, 0x5A, 0x9A, 0xDA, 0x2A, 0x6A, 0xAA, 0xEA, 0x3A, 0x7A, 0xBA, 0xFA,
            0x0E, 0x4E, 0x8E, 0xCE, 0x1E, 0x5E, 0x9E, 0xDE, 0x2E, 0x6E, 0xAE, 0xEE, 0x3E, 0x7E, 0xBE, 0xFE,
            0x03, 0x43, 0x83, 0xC3, 0x13, 0x53, 0x93, 0xD3, 0x23, 0x63, 0xA3, 0xE3, 0x33, 0x73, 0xB3, 0xF3,
            0x07, 0x47, 0x87, 0xC7, 0x17, 0x57, 0x97, 0xD7, 0x27, 0x67, 0xA7, 0xE7, 0x37, 0x77, 0xB7, 0xF7,
            0x0B, 0x4B, 0x8B, 0xCB, 0x1B, 0x5B, 0x9B, 0xDB, 0x2B, 0x6B, 0xAB, 0xEB, 0x3B, 0x7B, 0xBB, 0xFB,
            0x0F, 0x4F, 0x8F, 0xCF, 0x1F, 0x5F, 0x9F, 0xDF, 0x2F, 0x6F, 0xAF, 0xEF, 0x3F, 0x7F, 0xBF, 0xFF,
        };
    };

    if ((log2_length & 1) == 0) {
        const rev32 = @as(u6, @intCast(32 - log2_length));
        var index: usize = 0;
        while (index < length) : (index += 1) {
            const n = index * 4;
            const addr =
                (@as(usize, @intCast(static.swizzle_table[n & 0xff])) << 24) |
                (@as(usize, @intCast(static.swizzle_table[(n >> 8) & 0xff])) << 16) |
                (@as(usize, @intCast(static.swizzle_table[(n >> 16) & 0xff])) << 8) |
                @as(usize, @intCast(static.swizzle_table[(n >> 24) & 0xff]));
            f32_output[addr >> rev32] = input[index][0];
            f32_output[(0x40000000 | addr) >> rev32] = input[index][1];
            f32_output[(0x80000000 | addr) >> rev32] = input[index][2];
            f32_output[(0xC0000000 | addr) >> rev32] = input[index][3];
        }
    } else {
        const rev7 = @as(usize, 1) << @as(u6, @intCast(log2_length - 3));
        const rev32 = @as(u6, @intCast(32 - (log2_length - 3)));
        var index: usize = 0;
        while (index < length) : (index += 1) {
            const n = index / 2;
            var addr =
                (((@as(usize, @intCast(static.swizzle_table[n & 0xff])) << 24) |
                    (@as(usize, @intCast(static.swizzle_table[(n >> 8) & 0xff])) << 16) |
                    (@as(usize, @intCast(static.swizzle_table[(n >> 16) & 0xff])) << 8) |
                    (@as(usize, @intCast(static.swizzle_table[(n >> 24) & 0xff])))) >> rev32) |
                ((index & 1) * rev7 * 4);
            f32_output[addr] = input[index][0];
            addr += rev7;
            f32_output[addr] = input[index][1];
            addr += rev7;
            f32_output[addr] = input[index][2];
            addr += rev7;
            f32_output[addr] = input[index][3];
        }
    }
}

pub fn fftInitUnityTable(out_unity_table: []F32x4) void {
    assert(std.math.isPowerOfTwo(out_unity_table.len));
    assert(out_unity_table.len >= 32 and out_unity_table.len <= 512);

    var unity_table = out_unity_table;

    const v0123 = f32x4(0.0, 1.0, 2.0, 3.0);
    var length = out_unity_table.len / 4;
    var vlstep = f32x4s(0.5 * math.pi / @as(f32, @floatFromInt(length)));

    while (true) {
        length /= 4;
        var vjp = v0123;

        var j: u32 = 0;
        while (j < length) : (j += 1) {
            unity_table[j] = f32x4s(1.0);
            unity_table[j + length * 4] = f32x4s(0.0);

            var vls = vjp * vlstep;
            var sin_cos = sincos(vls);
            unity_table[j + length] = sin_cos[1];
            unity_table[j + length * 5] = sin_cos[0] * f32x4s(-1.0);

            var vijp = vjp + vjp;
            vls = vijp * vlstep;
            sin_cos = sincos(vls);
            unity_table[j + length * 2] = sin_cos[1];
            unity_table[j + length * 6] = sin_cos[0] * f32x4s(-1.0);

            vijp = vijp + vjp;
            vls = vijp * vlstep;
            sin_cos = sincos(vls);
            unity_table[j + length * 3] = sin_cos[1];
            unity_table[j + length * 7] = sin_cos[0] * f32x4s(-1.0);

            vjp += f32x4s(4.0);
        }
        vlstep *= f32x4s(4.0);
        unity_table = unity_table[8 * length ..];

        if (length <= 4)
            break;
    }
}

pub fn fft(re: []F32x4, im: []F32x4, unity_table: []const F32x4) void {
    const length = @as(u32, @intCast(re.len * 4));
    assert(std.math.isPowerOfTwo(length));
    assert(length >= 4 and length <= 512);
    assert(re.len == im.len);

    var re_temp_storage: [128]F32x4 = undefined;
    var im_temp_storage: [128]F32x4 = undefined;
    const re_temp = re_temp_storage[0..re.len];
    const im_temp = im_temp_storage[0..im.len];

    @memcpy(re_temp, re);
    @memcpy(im_temp, im);

    if (length > 16) {
        assert(unity_table.len == length);
        fftN(re_temp, im_temp, unity_table, length, 1);
    } else if (length == 16) {
        fft16(re_temp, im_temp, 1);
    } else if (length == 8) {
        fft8(re_temp, im_temp, 1);
    } else if (length == 4) {
        fft4(re_temp, im_temp, 1);
    }

    fftUnswizzle(re_temp, re);
    fftUnswizzle(im_temp, im);
}

pub fn ifft(re: []F32x4, im: []const F32x4, unity_table: []const F32x4) void {
    const length = @as(u32, @intCast(re.len * 4));
    assert(std.math.isPowerOfTwo(length));
    assert(length >= 4 and length <= 512);
    assert(re.len == im.len);

    var re_temp_storage: [128]F32x4 = undefined;
    var im_temp_storage: [128]F32x4 = undefined;
    var re_temp = re_temp_storage[0..re.len];
    var im_temp = im_temp_storage[0..im.len];

    const rnp = f32x4s(1.0 / @as(f32, @floatFromInt(length)));
    const rnm = f32x4s(-1.0 / @as(f32, @floatFromInt(length)));

    for (re, 0..) |_, i| {
        re_temp[i] = re[i] * rnp;
        im_temp[i] = im[i] * rnm;
    }

    if (length > 16) {
        assert(unity_table.len == length);
        fftN(re_temp, im_temp, unity_table, length, 1);
    } else if (length == 16) {
        fft16(re_temp, im_temp, 1);
    } else if (length == 8) {
        fft8(re_temp, im_temp, 1);
    } else if (length == 4) {
        fft4(re_temp, im_temp, 1);
    }

    fftUnswizzle(re_temp, re);
}
