//! aarch64 load/store helpers, the only `unsafe` the NEON kernels need (docs/safety.md). Each
//! takes a fixed-size array reference, so the type proves that the memory is in bounds; the
//! kernels get the arrays from slices with `chunks_exact` / `try_into` in safe code. Kernel
//! modules append the helpers they need here, one per array type.

use core::arch::aarch64::*;

macro_rules! ld_st {
    ($ld:ident, $st:ident, $t:ty, $n:literal, $v:ty, $load:ident, $store:ident) => {
        #[inline]
        #[target_feature(enable = "neon")]
        pub(crate) fn $ld(a: &[$t; $n]) -> $v {
            // SAFETY: `a` is `size_of::<$v>()` readable bytes of the element type.
            unsafe { $load(a.as_ptr()) }
        }
        #[inline]
        #[target_feature(enable = "neon")]
        pub(crate) fn $st(a: &mut [$t; $n], v: $v) {
            // SAFETY: `a` is `size_of::<$v>()` writable bytes of the element type.
            unsafe { $store(a.as_mut_ptr(), v) }
        }
    };
}

ld_st!(ld_u16x8, st_u16x8, u16, 8, uint16x8_t, vld1q_u16, vst1q_u16);
ld_st!(ld_i16x8, st_i16x8, i16, 8, int16x8_t, vld1q_s16, vst1q_s16);
ld_st!(ld_i32x4, st_i32x4, i32, 4, int32x4_t, vld1q_s32, vst1q_s32);
ld_st!(ld_u8x16, st_u8x16, u8, 16, uint8x16_t, vld1q_u8, vst1q_u8);
ld_st!(ld_u16x4, st_u16x4, u16, 4, uint16x4_t, vld1_u16, vst1_u16);
ld_st!(ld_i16x4, st_i16x4, i16, 4, int16x4_t, vld1_s16, vst1_s16);
ld_st!(ld_u8x8, st_u8x8, u8, 8, uint8x8_t, vld1_u8, vst1_u8);
