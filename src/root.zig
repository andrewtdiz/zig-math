// ==============================================================================
//
// SIMD math library for game developers
// https://github.com/michal-z/zig-gamedev/tree/main/libs/zmath
//
// Should work on all OSes supported by Zig. Works on x86_64 and ARM.
// Provides ~140 optimized routines and ~70 extensive tests.
// Can be used with any graphics API.
//
// zmath uses row-major matrices, row vectors (each row vector is stored in a SIMD register).
// Handedness is determined by which function version is used (Rh vs. Lh),
// otherwise the function works with either left-handed or right-handed view coordinates.
//
// const va = f32x4(1.0, 2.0, 3.0, 1.0);
// const vb = f32x4(-1.0, 1.0, -1.0, 1.0);
// const v0 = va + vb - f32x4(0.0, 1.0, 0.0, 1.0) * f32x4s(3.0);
// const v1 = cross3(va, vb) + f32x4(1.0, 1.0, 1.0, 1.0);
// const v2 = va + dot3(va, vb) / v1; // dotN() returns scalar replicated on all vector components
//
// const m = rotationX(math.pi * 0.25);
// const v = f32x4(...);
// const v0 = mul(v, m); // 'v' treated as a row vector
// const v1 = mul(m, v); // 'v' treated as a column vector
// const f = m[row][column];
//
// const b = va < vb;
// if (all(b, 0)) { ... } // '0' means check all vector components; if all are 'true'
// if (all(b, 3)) { ... } // '3' means check first three vector components; if all first three are 'true'
// if (any(b, 0)) { ... } // '0' means check all vector components; if any is 'true'
// if (any(b, 3)) { ... } // '3' means check first three vector components; if any from first three is 'true'
//
// var v4 = load(mem[0..], F32x4, 0);
// var v8 = load(mem[100..], F32x8, 0);
// var v16 = load(mem[200..], F32x16, 0);
//
// var camera_position = [3]f32{ 1.0, 2.0, 3.0 };
// var cam_pos = loadArr3(camera_position);
// ...
// storeArr3(&camera_position, cam_pos);
//
// v4 = sin(v4); // SIMDx4
// v8 = cos(v8); // .x86_64 -> 2 x SIMDx4, .x86_64+avx+fma -> SIMDx8
// v16 = atan(v16); // .x86_64 -> 4 x SIMDx4, .x86_64+avx+fma -> 2 x SIMDx8, .x86_64+avx512f -> SIMDx16
//
// store(mem[0..], v4, 0);
// store(mem[100..], v8, 0);
// store(mem[200..], v16, 0);
//
// ------------------------------------------------------------------------------
// 1. Initialization functions
// ------------------------------------------------------------------------------
//
// f32x4(e0: f32, e1: f32, e2: f32, e3: f32) F32x4
// f32x8(e0: f32, e1: f32, e2: f32, e3: f32, e4: f32, e5: f32, e6: f32, e7: f32) F32x8
// f32x16(e0: f32, e1: f32, e2: f32, e3: f32, e4: f32, e5: f32, e6: f32, e7: f32,
//        e8: f32, e9: f32, ea: f32, eb: f32, ec: f32, ed: f32, ee: f32, ef: f32) F32x16
//
// f32x4s(e0: f32) F32x4
// f32x8s(e0: f32) F32x8
// f32x16s(e0: f32) F32x16
//
// boolx4(e0: bool, e1: bool, e2: bool, e3: bool) Boolx4
// boolx8(e0: bool, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool, e7: bool) Boolx8
// boolx16(e0: bool, e1: bool, e2: bool, e3: bool, e4: bool, e5: bool, e6: bool, e7: bool,
//         e8: bool, e9: bool, ea: bool, eb: bool, ec: bool, ed: bool, ee: bool, ef: bool) Boolx16
//
// load(mem: []const f32, comptime T: type, comptime len: u32) T
// store(mem: []f32, v: anytype, comptime len: u32) void
//
// loadArr2(arr: [2]f32) F32x4
// loadArr2zw(arr: [2]f32, z: f32, w: f32) F32x4
// loadArr3(arr: [3]f32) F32x4
// loadArr3w(arr: [3]f32, w: f32) F32x4
// loadArr4(arr: [4]f32) F32x4
//
// storeArr2(arr: *[2]f32, v: F32x4) void
// storeArr3(arr: *[3]f32, v: F32x4) void
// storeArr4(arr: *[4]f32, v: F32x4) void
//
// arr3Ptr(ptr: anytype) *const [3]f32
// arrNPtr(ptr: anytype) [*]const f32
//
// splat(comptime T: type, value: f32) T
// splatInt(comptime T: type, value: u32) T
//
// ------------------------------------------------------------------------------
// 2. Functions that work on all vector components (F32xN = F32x4 or F32x8 or F32x16)
// ------------------------------------------------------------------------------
//
// all(vb: anytype, comptime len: u32) bool
// any(vb: anytype, comptime len: u32) bool
//
// isNearEqual(v0: F32xN, v1: F32xN, epsilon: F32xN) BoolxN
// isNan(v: F32xN) BoolxN
// isInf(v: F32xN) BoolxN
// isInBounds(v: F32xN, bounds: F32xN) BoolxN
//
// andInt(v0: F32xN, v1: F32xN) F32xN
// andNotInt(v0: F32xN, v1: F32xN) F32xN
// orInt(v0: F32xN, v1: F32xN) F32xN
// norInt(v0: F32xN, v1: F32xN) F32xN
// xorInt(v0: F32xN, v1: F32xN) F32xN
//
// minFast(v0: F32xN, v1: F32xN) F32xN
// maxFast(v0: F32xN, v1: F32xN) F32xN
// min(v0: F32xN, v1: F32xN) F32xN
// max(v0: F32xN, v1: F32xN) F32xN
// round(v: F32xN) F32xN
// floor(v: F32xN) F32xN
// trunc(v: F32xN) F32xN
// ceil(v: F32xN) F32xN
// clamp(v0: F32xN, v1: F32xN) F32xN
// clampFast(v0: F32xN, v1: F32xN) F32xN
// saturate(v: F32xN) F32xN
// saturateFast(v: F32xN) F32xN
// lerp(v0: F32xN, v1: F32xN, t: f32) F32xN
// lerpV(v0: F32xN, v1: F32xN, t: F32xN) F32xN
// lerpInverse(v0: F32xN, v1: F32xN, t: f32) F32xN
// lerpInverseV(v0: F32xN, v1: F32xN, t: F32xN) F32xN
// mapLinear(v: F32xN, min1: f32, max1: f32, min2: f32, max2: f32) F32xN
// mapLinearV(v: F32xN, min1: F32xN, max1: F32xN, min2: F32xN, max2: F32xN) F32xN
// sqrt(v: F32xN) F32xN
// abs(v: F32xN) F32xN
// mod(v0: F32xN, v1: F32xN) F32xN
// modAngle(v: F32xN) F32xN
// mulAdd(v0: F32xN, v1: F32xN, v2: F32xN) F32xN
// select(mask: BoolxN, v0: F32xN, v1: F32xN)
// sin(v: F32xN) F32xN
// cos(v: F32xN) F32xN
// sincos(v: F32xN) [2]F32xN
// asin(v: F32xN) F32xN
// acos(v: F32xN) F32xN
// atan(v: F32xN) F32xN
// atan2(vy: F32xN, vx: F32xN) F32xN
// cmulSoa(re0: F32xN, im0: F32xN, re1: F32xN, im1: F32xN) [2]F32xN
//
// ------------------------------------------------------------------------------
// 3. 2D, 3D, 4D vector functions
// ------------------------------------------------------------------------------
//
// swizzle(v: Vec, c, c, c, c) Vec (comptime c = .x | .y | .z | .w)
// dot2(v0: Vec, v1: Vec) F32x4
// dot3(v0: Vec, v1: Vec) F32x4
// dot4(v0: Vec, v1: Vec) F32x4
// cross3(v0: Vec, v1: Vec) Vec
// lengthSq2(v: Vec) F32x4
// lengthSq3(v: Vec) F32x4
// lengthSq4(v: Vec) F32x4
// length2(v: Vec) F32x4
// length3(v: Vec) F32x4
// length4(v: Vec) F32x4
// normalize2(v: Vec) Vec
// normalize3(v: Vec) Vec
// normalize4(v: Vec) Vec
//
// vecToArr2(v: Vec) [2]f32
// vecToArr3(v: Vec) [3]f32
// vecToArr4(v: Vec) [4]f32
//
// ------------------------------------------------------------------------------
// 4. Matrix functions
// ------------------------------------------------------------------------------
//
// identity() Mat
// mul(m0: Mat, m1: Mat) Mat
// mul(s: f32, m: Mat) Mat
// mul(m: Mat, s: f32) Mat
// mul(v: Vec, m: Mat) Vec
// mul(m: Mat, v: Vec) Vec
// transpose(m: Mat) Mat
// rotationX(angle: f32) Mat
// rotationY(angle: f32) Mat
// rotationZ(angle: f32) Mat
// translation(x: f32, y: f32, z: f32) Mat
// translationV(v: Vec) Mat
// scaling(x: f32, y: f32, z: f32) Mat
// scalingV(v: Vec) Mat
// lookToLh(eyepos: Vec, eyedir: Vec, updir: Vec) Mat
// lookAtLh(eyepos: Vec, focuspos: Vec, updir: Vec) Mat
// lookToRh(eyepos: Vec, eyedir: Vec, updir: Vec) Mat
// lookAtRh(eyepos: Vec, focuspos: Vec, updir: Vec) Mat
// perspectiveFovLh(fovy: f32, aspect: f32, near: f32, far: f32) Mat
// perspectiveFovRh(fovy: f32, aspect: f32, near: f32, far: f32) Mat
// perspectiveFovLhGl(fovy: f32, aspect: f32, near: f32, far: f32) Mat
// perspectiveFovRhGl(fovy: f32, aspect: f32, near: f32, far: f32) Mat
// orthographicLh(w: f32, h: f32, near: f32, far: f32) Mat
// orthographicRh(w: f32, h: f32, near: f32, far: f32) Mat
// orthographicLhGl(w: f32, h: f32, near: f32, far: f32) Mat
// orthographicRhGl(w: f32, h: f32, near: f32, far: f32) Mat
// orthographicOffCenterLh(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat
// orthographicOffCenterRh(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat
// orthographicOffCenterLhGl(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat
// orthographicOffCenterRhGl(left: f32, right: f32, top: f32, bottom: f32, near: f32, far: f32) Mat
// determinant(m: Mat) F32x4
// inverse(m: Mat) Mat
// inverseDet(m: Mat, det: ?*F32x4) Mat
// matToQuat(m: Mat) Quat
// matFromAxisAngle(axis: Vec, angle: f32) Mat
// matFromNormAxisAngle(axis: Vec, angle: f32) Mat
// matFromQuat(quat: Quat) Mat
// matFromRollPitchYaw(pitch: f32, yaw: f32, roll: f32) Mat
// matFromRollPitchYawV(angles: Vec) Mat
// matFromArr(arr: [16]f32) Mat
//
// loadMat(mem: []const f32) Mat
// loadMat43(mem: []const f32) Mat
// loadMat34(mem: []const f32) Mat
// storeMat(mem: []f32, m: Mat) void
// storeMat43(mem: []f32, m: Mat) void
// storeMat34(mem: []f32, m: Mat) void
//
// matToArr(m: Mat) [16]f32
// matToArr43(m: Mat) [12]f32
// matToArr34(m: Mat) [12]f32
//
// ------------------------------------------------------------------------------
// 5. Quaternion functions
// ------------------------------------------------------------------------------
//
// qmul(q0: Quat, q1: Quat) Quat
// qidentity() Quat
// conjugate(quat: Quat) Quat
// inverse(q: Quat) Quat
// rotate(q: Quat, v: Vec) Vec
// slerp(q0: Quat, q1: Quat, t: f32) Quat
// slerpV(q0: Quat, q1: Quat, t: F32x4) Quat
// quatToMat(quat: Quat) Mat
// quatToAxisAngle(quat: Quat, axis: *Vec, angle: *f32) void
// quatFromMat(m: Mat) Quat
// quatFromAxisAngle(axis: Vec, angle: f32) Quat
// quatFromNormAxisAngle(axis: Vec, angle: f32) Quat
// quatFromRollPitchYaw(pitch: f32, yaw: f32, roll: f32) Quat
// quatFromRollPitchYawV(angles: Vec) Quat
//
// ------------------------------------------------------------------------------
// 6. Color functions
// ------------------------------------------------------------------------------
//
// adjustSaturation(color: F32x4, saturation: f32) F32x4
// adjustContrast(color: F32x4, contrast: f32) F32x4
// rgbToHsl(rgb: F32x4) F32x4
// hslToRgb(hsl: F32x4) F32x4
// rgbToHsv(rgb: F32x4) F32x4
// hsvToRgb(hsv: F32x4) F32x4
// rgbToSrgb(rgb: F32x4) F32x4
// srgbToRgb(srgb: F32x4) F32x4
//
// ------------------------------------------------------------------------------
// X. Misc functions
// ------------------------------------------------------------------------------
//
// linePointDistance(linept0: Vec, linept1: Vec, pt: Vec) F32x4
// sin(v: f32) f32
// cos(v: f32) f32
// sincos(v: f32) [2]f32
// asin(v: f32) f32
// acos(v: f32) f32
//
// fftInitUnityTable(unitytable: []F32x4) void
// fft(re: []F32x4, im: []F32x4, unitytable: []const F32x4) void
// ifft(re: []F32x4, im: []const F32x4, unitytable: []const F32x4) void
//
// ==============================================================================

pub const core = @import("core.zig");
pub const vector = @import("vector.zig");
pub const matrix = @import("matrix.zig");
pub const fft = @import("fft.zig");

test {
    _ = @import("test/core.zig");
    _ = @import("test/vector.zig");
    _ = @import("test/matrix.zig");
    _ = @import("test/fft.zig");
}
