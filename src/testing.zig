const std = @import("std");
const math = std.math;

extern fn zigTestRunnerStoreFailureNote(ptr: [*]const u8, len: usize) void;

fn recordFailureNote(message: []const u8) void {
    zigTestRunnerStoreFailureNote(message.ptr, message.len);
}

fn recordComparisonFailure(
    comptime heading: []const u8,
    expected: anytype,
    actual: anytype,
) void {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    const writer = stream.writer();
    writer.print("{s}\nExpected: {any}\nReceived: {any}", .{ heading, expected, actual }) catch {
        recordFailureNote("message truncated");
        return;
    };
    const written = stream.getWritten();
    zigTestRunnerStoreFailureNote(written.ptr, written.len);
}

fn recordApproxFailure(
    comptime heading: []const u8,
    expected: anytype,
    actual: anytype,
    tolerance: anytype,
) void {
    var buf: [256]u8 = undefined;
    var stream = std.io.fixedBufferStream(&buf);
    const writer = stream.writer();
    writer.print("{s}\nExpected: {any}\nReceived: {any}\nTolerance: {any}", .{
        heading, expected, actual, tolerance,
    }) catch {
        recordFailureNote("message truncated");
        return;
    };
    const written = stream.getWritten();
    zigTestRunnerStoreFailureNote(written.ptr, written.len);
}

pub fn expectApproxEqAbs(expected: anytype, actual: anytype, tolerance: anytype) !void {
    const T = @TypeOf(expected, actual, tolerance);
    switch (@typeInfo(T)) {
        .float => {},
        .comptime_float => @compileError("Cannot approximately compare two comptime_float values"),
        else => @compileError("Unable to compare non floating point values"),
    }

    const exp: T = expected;
    const act: T = actual;
    const tol: T = tolerance;
    const diff = @abs(act - exp);

    if (!(diff <= tol)) {
        recordApproxFailure("expectApproxEqAbs", exp, act, tol);
        return error.TestExpectedApproxEqAbs;
    }
}

pub fn expectEqual(expected: anytype, actual: anytype) !void {
    const T = @TypeOf(expected, actual);
    const exp: T = expected;
    const act: T = actual;

    if (!std.meta.eql(exp, act)) {
        recordComparisonFailure("expectEqual", exp, act);
        return error.TestExpectedEqual;
    }
}

pub fn expect(actual: anytype) !void {
    const T = @TypeOf(actual);
    const act: T = actual;

    switch (@typeInfo(T)) {
        .bool => {
            if (!act) {
                recordComparisonFailure("expect", true, act);
                return error.TestExpect;
            }
        },
        else => @compileError("expect() requires a boolean value"),
    }
}

pub fn expectVecEqual(expected: anytype, actual: anytype) !void {
    const T = @TypeOf(expected, actual);
    inline for (0..veclen(T)) |i| {
        try std.testing.expectEqual(expected[i], actual[i]);
    }
}

pub fn expectVecApproxEqAbs(expected: anytype, actual: anytype, eps: f32) !void {
    const T = @TypeOf(expected, actual);
    inline for (0..veclen(T)) |i| {
        try std.testing.expectApproxEqAbs(expected[i], actual[i], eps);
    }
}

inline fn veclen(comptime T: type) comptime_int {
    const info = @typeInfo(T);
    return switch (info) {
        .vector => |v| v.len,
        else => @compileError("expectVec* requires vector types; got " ++ @typeName(T)),
    };
}

pub fn approxEqAbs(v0: anytype, v1: anytype, eps: f32) bool {
    const T = @TypeOf(v0, v1);
    switch (@typeInfo(T)) {
        .float => return math.approxEqAbs(T, v0, v1, eps),
        .comptime_float => {
            const a: f64 = v0;
            const b: f64 = v1;
            return math.approxEqAbs(f64, a, b, eps);
        },
        .vector => {
            comptime var i: comptime_int = 0;
            inline while (i < veclen(T)) : (i += 1) {
                if (!math.approxEqAbs(f32, v0[i], v1[i], eps)) {
                    return false;
                }
            }
            return true;
        },
        else => @compileError("approxEqAbs only supports floats or float vectors"),
    }
}
