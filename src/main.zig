pub const __m128i = @Vector(4, i32);

const u64x2 = @Vector(2, u64);
const i32x4 = @Vector(4, i32);
const i64x2 = @Vector(2, i64);

inline fn bitCast_u64x2(a: anytype) u64x2 {
    return @bitCast(a);
}

inline fn bitCast_i64x2(a: anytype) i64x2 {
    return @bitCast(a);
}

/// Software CRC-32C
//
// Modified from: https://github.com/DLTcollab/sse2neon/blob/4a036e60472af7dd60a31421fa01557000b5c96b/sse2neon.h#L8528C11-L8528C21
// Copyright (c) Cuda Chen <clh960524@gmail.com>
//
// which was in turn based on: https://create.stephan-brumme.com/crc32/#half-byte
// Author: unknown
//
// From: https://github.com/aqrit/sse2zig/blob/444ed8d129625ab5deec345ac5fdb06f6f9d0c6c/sse.zig#L5119-L514
inline fn crc32cSoft(crc: anytype, v: anytype) @TypeOf(crc) {
    // 4-bit-indexed table has a small memory footprint
    // while being faster than a bit-twiddling solution
    // but has a loop-carried dependence...
    const crc32c_table: [16]u32 = .{
        0x00000000, 0x105ec76f, 0x20bd8ede, 0x30e349b1,
        0x417b1dbc, 0x5125dad3, 0x61c69362, 0x7198540d,
        0x82f63b78, 0x92a8fc17, 0xa24bb5a6, 0xb21572c9,
        0xc38d26c4, 0xd3d3e1ab, 0xe330a81a, 0xf36e6f75,
    };

    // ignore bits[32..64] of crc (and validate arg type)
    var r = switch (@typeInfo(@TypeOf(crc)).int.bits) {
        32 => crc,
        64 => crc & 0x00000000FFFFFFFF,
        else => @compileError("invalid type of arg `crc`"),
    };

    // number of loop iterations (and validate arg type)
    const n = switch (@typeInfo(@TypeOf(v)).int.bits) {
        8, 16, 32, 64 => @typeInfo(@TypeOf(v)).int.bits / 4,
        else => @compileError("invalid type of arg `v`"),
    };

    r ^= v;
    for (0..n) |_| r = (r >> 4) ^ crc32c_table[@as(u4, @truncate(r))];

    return r;
}

/// Software carryless multiplication of two 64-bit integers using native 128-bit registers.
// Modified from: https://github.com/ziglang/zig/blob/8fd15c6ca8b93fa9888e2641ebec149f6d600643/lib/std/crypto/ghash_polyval.zig#L168
// Copyright (c) Zig contributors
//
// https://github.com/aqrit/sse2zig/blob/444ed8d129625ab5deec345ac5fdb06f6f9d0c6c/sse.zig#L5251-L5276
fn clmulSoft128(x: u64, y: u64) u128 {
    const x0 = x & 0x1111111111111110;
    const x1 = x & 0x2222222222222220;
    const x2 = x & 0x4444444444444440;
    const x3 = x & 0x8888888888888880;
    const y0 = y & 0x1111111111111111;
    const y1 = y & 0x2222222222222222;
    const y2 = y & 0x4444444444444444;
    const y3 = y & 0x8888888888888888;
    const z0 = (x0 * @as(u128, y0)) ^ (x1 * @as(u128, y3)) ^ (x2 * @as(u128, y2)) ^ (x3 * @as(u128, y1));
    const z1 = (x0 * @as(u128, y1)) ^ (x1 * @as(u128, y0)) ^ (x2 * @as(u128, y3)) ^ (x3 * @as(u128, y2));
    const z2 = (x0 * @as(u128, y2)) ^ (x1 * @as(u128, y1)) ^ (x2 * @as(u128, y0)) ^ (x3 * @as(u128, y3));
    const z3 = (x0 * @as(u128, y3)) ^ (x1 * @as(u128, y2)) ^ (x2 * @as(u128, y1)) ^ (x3 * @as(u128, y0));

    const x0_mask = @as(u64, 0) -% (x & 1);
    const x1_mask = @as(u64, 0) -% ((x >> 1) & 1);
    const x2_mask = @as(u64, 0) -% ((x >> 2) & 1);
    const x3_mask = @as(u64, 0) -% ((x >> 3) & 1);
    const extra = (x0_mask & y) ^ (@as(u128, x1_mask & y) << 1) ^
        (@as(u128, x2_mask & y) << 2) ^ (@as(u128, x3_mask & y) << 3);

    return (z0 & 0x11111111111111111111111111111111) ^
        (z1 & 0x22222222222222222222222222222222) ^
        (z2 & 0x44444444444444444444444444444444) ^
        (z3 & 0x88888888888888888888888888888888) ^ extra;
}

pub inline fn _mm_loadu_si128(mem_addr: *align(1) const __m128i) __m128i {
    return mem_addr.*;
}

pub inline fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) __m128i {
    const r: __m128i = .{ e3, e2, e1, e0 };
    return @bitCast(r);
}

pub inline fn _mm_set_epi64x(e1: i64, e0: i64) __m128i {
    const r: i64x2 = .{ e0, e1 };
    return @bitCast(r);
}

pub inline fn _mm_cvtsi32_si128(a: i32) __m128i {
    const r = i32x4{ a, 0, 0, 0 };
    return @bitCast(r);
}

pub inline fn _mm_cvtsi128_si64(a: __m128i) i64 {
    return bitCast_i64x2(a)[0];
}

pub inline fn _mm_xor_si128(a: __m128i, b: __m128i) __m128i {
    return a ^ b;
}

pub inline fn _mm_clmulepi64_si128(a: __m128i, b: __m128i, comptime imm8: comptime_int) __m128i {
    const x = bitCast_u64x2(a)[imm8 & 1];
    const y = bitCast_u64x2(b)[(imm8 >> 4) & 1];
    const r = clmulSoft128(x, y);
    return _mm_set_epi64x(@bitCast(@as(u64, @truncate(r >> 64))), @bitCast(@as(u64, @truncate(r))));
}

pub inline fn _mm_crc32_u64(crc: u64, v: u64) u64 {
    return crc32cSoft(crc, v);
}

pub inline fn _mm_extract_epi64(a: __m128i, comptime imm8: comptime_int) i64 {
    return bitCast_i64x2(a)[imm8];
}

inline fn indexPtr(ptr: [*]const u8) u64 {
    return std.mem.readInt(u64, ptr[0..8], .little);
}

pub fn crc32_4k_fusion(acc: u32, buf_const: []const u8) u32 {
    var acc_a = acc;
    var buf = buf_const.ptr;

    var acc_b: u32 = 0;
    var acc_c: u32 = 0;

    var buf2: [*]const u8 = buf + 2176;
    var x1 = _mm_loadu_si128(@ptrCast(buf2));
    var x2 = _mm_loadu_si128(@ptrCast(buf2 + 16));
    var x3 = _mm_loadu_si128(@ptrCast(buf2 + 32));
    var x4 = _mm_loadu_si128(@ptrCast(buf2 + 48));

    const k1k2 = _mm_setr_epi32(0x740EEF02, 0, @bitCast(@as(u32, 0x9E4ADDF8)), 0);
    const end = buf + 4096 - 64;

    while (@intFromPtr(buf2) < @intFromPtr(end)) {
        acc_a = @truncate(_mm_crc32_u64(@intCast(acc_a), indexPtr(buf)));
        var x5 = _mm_clmulepi64_si128(x1, k1k2, 0x00);
        acc_b = @truncate(_mm_crc32_u64(@intCast(acc_b), indexPtr(buf + 728)));
        x1 = _mm_clmulepi64_si128(x1, k1k2, 0x11);
        acc_c = @truncate(_mm_crc32_u64(@intCast(acc_c), indexPtr(buf + 728 * 2)));
        var x6 = _mm_clmulepi64_si128(x2, k1k2, 0x00);
        acc_a = @truncate(_mm_crc32_u64(@intCast(acc_a), indexPtr(buf + 8)));
        x2 = _mm_clmulepi64_si128(x2, k1k2, 0x11);
        acc_b = @truncate(_mm_crc32_u64(@intCast(acc_b), indexPtr(buf + 728 + 8)));
        var x7 = _mm_clmulepi64_si128(x3, k1k2, 0x00);
        acc_c = @truncate(_mm_crc32_u64(@intCast(acc_c), indexPtr(buf + 728 * 2 + 8)));
        x3 = _mm_clmulepi64_si128(x3, k1k2, 0x11);
        acc_a = @truncate(_mm_crc32_u64(@intCast(acc_a), indexPtr(buf + 16)));
        var x8 = _mm_clmulepi64_si128(x4, k1k2, 0x00);
        acc_b = @truncate(_mm_crc32_u64(@intCast(acc_b), indexPtr(buf + 728 + 16)));
        x4 = _mm_clmulepi64_si128(x4, k1k2, 0x11);
        acc_c = @truncate(_mm_crc32_u64(@intCast(acc_c), indexPtr(buf + 728 * 2 + 16)));

        x5 = _mm_xor_si128(x5, _mm_loadu_si128(@ptrCast(buf2 + 64)));
        x1 = _mm_xor_si128(x1, x5);
        x6 = _mm_xor_si128(x6, _mm_loadu_si128(@ptrCast(buf2 + 80)));
        x2 = _mm_xor_si128(x2, x6);
        x7 = _mm_xor_si128(x7, _mm_loadu_si128(@ptrCast(buf2 + 96)));
        x3 = _mm_xor_si128(x3, x7);
        x8 = _mm_xor_si128(x8, _mm_loadu_si128(@ptrCast(buf2 + 112)));
        x4 = _mm_xor_si128(x4, x8);

        buf2 += 64;
        buf += 24;
    }

    const k3k4 = _mm_setr_epi32(@bitCast(@as(u32, 0xF20C0DFE)), 0, 0x493C7D27, 0);
    acc_a = @truncate(_mm_crc32_u64(acc_a, indexPtr(buf)));
    var x5 = _mm_clmulepi64_si128(x1, k3k4, 0x00);
    acc_b = @truncate(_mm_crc32_u64(acc_b, indexPtr(buf + 728)));
    x1 = _mm_clmulepi64_si128(x1, k3k4, 0x11);
    acc_c = @truncate(_mm_crc32_u64(acc_c, indexPtr(buf + 728 * 2)));
    var x6 = _mm_clmulepi64_si128(x3, k3k4, 0x00);
    acc_a = @truncate(_mm_crc32_u64(acc_a, indexPtr(buf + 8)));
    x3 = _mm_clmulepi64_si128(x3, k3k4, 0x11);
    acc_b = @truncate(_mm_crc32_u64(acc_b, indexPtr(buf + 728 + 8)));
    acc_c = @truncate(_mm_crc32_u64(acc_c, indexPtr(buf + 728 * 2 + 8)));
    acc_a = @truncate(_mm_crc32_u64(acc_a, indexPtr(buf + 16)));
    acc_b = @truncate(_mm_crc32_u64(acc_b, indexPtr(buf + 728 + 16)));
    x5 = _mm_xor_si128(x5, x2);
    acc_c = @truncate(_mm_crc32_u64(acc_c, indexPtr(buf + 728 * 2 + 16)));
    x1 = _mm_xor_si128(x1, x5);
    acc_a = @truncate(_mm_crc32_u64(acc_a, indexPtr(buf + 24)));

    const k5k6 = _mm_setr_epi32(0x3DA6D0CB, 0, @bitCast(@as(u32, 0xBA4FC28E)), 0);
    x6 = _mm_xor_si128(x6, x4);
    x3 = _mm_xor_si128(x3, x6);
    x5 = _mm_clmulepi64_si128(x1, k5k6, 0x00);
    acc_b = @truncate(_mm_crc32_u64(acc_b, indexPtr(buf + 728 + 24)));
    x1 = _mm_clmulepi64_si128(x1, k5k6, 0x11);

    const kCk0 = _mm_setr_epi32(@bitCast(@as(u32, 0xF48642E9)), 0, 0, 0);
    const vec_c = _mm_clmulepi64_si128(_mm_cvtsi32_si128(@bitCast(acc_c)), kCk0, 0x00);

    const kAkB = _mm_setr_epi32(0x155AD968, 0, 0x2E7D11A7, 0);
    const vec_a = _mm_clmulepi64_si128(_mm_cvtsi32_si128(@bitCast(acc_a)), kAkB, 0x00);
    const vec_b = _mm_clmulepi64_si128(_mm_cvtsi32_si128(@bitCast(acc_b)), kAkB, 0x10);

    x5 = _mm_xor_si128(x5, x3);
    x1 = _mm_xor_si128(x1, x5);
    const abc: u64 = @bitCast(_mm_cvtsi128_si64(_mm_xor_si128(_mm_xor_si128(vec_c, vec_a), vec_b)));
    var crc: u32 = @truncate(_mm_crc32_u64(0, @bitCast(_mm_extract_epi64(x1, 0))));
    crc = @truncate(_mm_crc32_u64(crc, abc ^ @as(u64, @bitCast(_mm_extract_epi64(x1, 1)))));

    return crc;
}

pub fn crc32c_scalar(acc: u32, buf: []const u8) u32 {
    const poly_reflected = 0x82F63B78;
    var crc = ~acc;

    for (buf) |b| {
        crc ^= @as(u32, b);
        var i: u8 = 0;
        while (i < 8) : (i += 1) {
            if ((crc & 1) != 0)
                crc = (crc >> 1) ^ poly_reflected
            else
                crc >>= 1;
        }
    }

    return ~crc;
}

const std = @import("std");

pub fn main() void {
    var buffer: [4096]u8 = .{0} ** 4096;
    for (0..buffer.len) |i| buffer[i] = @intCast(i % 256);

    std.debug.print("crc32c_scalar: {x}\n", .{crc32c_scalar(0, &buffer)});
    std.debug.print("crc32_4k_fusion: {x}\n", .{crc32_4k_fusion(0, &buffer)});
    std.debug.print("crc32Isci: {x}\n", .{std.hash.crc.Crc32Iscsi.hash(&buffer)});
}
