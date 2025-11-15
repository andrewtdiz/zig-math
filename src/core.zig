// Fundamental types
pub const F32x4 = @Vector(4, f32);
pub const F32x8 = @Vector(8, f32);
pub const F32x16 = @Vector(16, f32);
pub const Boolx4 = @Vector(4, bool);
pub const Boolx8 = @Vector(8, bool);
pub const Boolx16 = @Vector(16, bool);

// "Higher-level" aliases
pub const Vec = F32x4;
pub const Mat = [4]F32x4;
pub const Quat = F32x4;

const builtin = @import("builtin");
const std = @import("std");
const math = std.math;
const assert = std.debug.assert;
const expect = std.testing.expect;

pub const cpu_arch = builtin.cpu.arch;
pub const has_avx = if (cpu_arch == .x86_64) std.Target.x86.featureSetHas(builtin.cpu.features, .avx) else false;
pub const has_avx512f = if (cpu_arch == .x86_64) std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f) else false;
pub const has_fma = if (cpu_arch == .x86_64) std.Target.x86.featureSetHas(builtin.cpu.features, .fma) else false;
// ------------------------------------------------------------------------------
//
// 1. Initialization functions
//
// ------------------------------------------------------------------------------
pub inline fn f32x4(e0: f32, e1: f32, e2: f32, e3: f32) F32x4 {
    return .{ e0, e1, e2, e3 };
}
pub inline fn f32x8(e0: f32, e1: f32, e2: f32, e3: f32, e4: f32, e5: f32, e6: f32, e7: f32) F32x8 {
    return .{ e0, e1, e2, e3, e4, e5, e6, e7 };
}
// zig fmt: off
pub inline fn f32x16(
    e0: f32, e1: f32, e2: f32, e3: f32, e4: f32, e5: f32, e6: f32, e7: f32,
    e8: f32, e9: f32, ea: f32, eb: f32, ec: f32, ed: f32, ee: f32, ef: f32) F32x16 {
    return .{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, ea, eb, ec, ed, ee, ef };
}
// zig fmt: on

pub inline fn f32x4s(e0: f32) F32x4 {
    return splat(F32x4, e0);
}
pub inline fn f32x8s(e0: f32) F32x8 {
    return splat(F32x8, e0);
}
pub inline fn f32x16s(e0: f32) F32x16 {
    return splat(F32x16, e0);
}

pub inline fn boolx4(e0: bool, e1: bool, e2: bool, e3: bool) Boolx4 {
    return .{ e0, e1, e2, e3 };
}
pub inline fn boolx8(e0: bool, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool, e7: bool) Boolx8 {
    return .{ e0, e1, e2, e3, e4, e5, e6, e7 };
}
// zig fmt: off
pub inline fn boolx16(
    e0: bool, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool, e7: bool,
    e8: bool, e9: bool, ea: bool, eb: bool, ec: bool, ed: bool, ee: bool, ef: bool) Boolx16 {
    return .{ e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, ea, eb, ec, ed, ee, ef };
}
// zig fmt: on

pub inline fn veclen(comptime T: type) comptime_int {
    return @typeInfo(T).vector.len;
}

pub inline fn splat(comptime T: type, value: f32) T {
    return @splat(value);
}
pub inline fn splatInt(comptime T: type, value: u32) T {
    return @splat(@bitCast(value));
}

pub fn load(mem: []const f32, comptime T: type, comptime len: u32) T {
    var v = splat(T, 0.0);
    const loop_len = if (len == 0) veclen(T) else len;
    comptime var i: u32 = 0;
    inline while (i < loop_len) : (i += 1) {
        v[i] = mem[i];
    }
    return v;
}

pub fn store(mem: []f32, v: anytype, comptime len: u32) void {
    const T = @TypeOf(v);
    const loop_len = if (len == 0) veclen(T) else len;
    comptime var i: u32 = 0;
    inline while (i < loop_len) : (i += 1) {
        mem[i] = v[i];
    }
}

pub inline fn loadArr2(arr: [2]f32) F32x4 {
    return f32x4(arr[0], arr[1], 0.0, 0.0);
}
pub inline fn loadArr2zw(arr: [2]f32, z: f32, w: f32) F32x4 {
    return f32x4(arr[0], arr[1], z, w);
}
pub inline fn loadArr3(arr: [3]f32) F32x4 {
    return f32x4(arr[0], arr[1], arr[2], 0.0);
}
pub inline fn loadArr3w(arr: [3]f32, w: f32) F32x4 {
    return f32x4(arr[0], arr[1], arr[2], w);
}
pub inline fn loadArr4(arr: [4]f32) F32x4 {
    return f32x4(arr[0], arr[1], arr[2], arr[3]);
}

pub inline fn storeArr2(arr: *[2]f32, v: F32x4) void {
    arr.* = .{ v[0], v[1] };
}
pub inline fn storeArr3(arr: *[3]f32, v: F32x4) void {
    arr.* = .{ v[0], v[1], v[2] };
}
pub inline fn storeArr4(arr: *[4]f32, v: F32x4) void {
    arr.* = .{ v[0], v[1], v[2], v[3] };
}

pub inline fn arr3Ptr(ptr: anytype) *const [3]f32 {
    comptime assert(@typeInfo(@TypeOf(ptr)) == .pointer);
    const T = std.meta.Child(@TypeOf(ptr));
    comptime assert(T == F32x4);
    return @as(*const [3]f32, @ptrCast(ptr));
}

pub inline fn arrNPtr(ptr: anytype) [*]const f32 {
    comptime assert(@typeInfo(@TypeOf(ptr)) == .pointer);
    const T = std.meta.Child(@TypeOf(ptr));
    comptime assert(T == Mat or T == F32x4 or T == F32x8 or T == F32x16);
    return @as([*]const f32, @ptrCast(ptr));
}

pub inline fn vecToArr2(v: Vec) [2]f32 {
    return .{ v[0], v[1] };
}
pub inline fn vecToArr3(v: Vec) [3]f32 {
    return .{ v[0], v[1], v[2] };
}
pub inline fn vecToArr4(v: Vec) [4]f32 {
    return .{ v[0], v[1], v[2], v[3] };
}
