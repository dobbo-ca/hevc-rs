//! x86_64 load/store helpers, the only `unsafe` the x86 kernels need (docs/safety.md). Each takes
//! a fixed-size array reference, so the type proves that the memory is in bounds; the kernels get
//! the arrays from slices with `chunks_exact` / `try_into` in safe code. All loads and stores are
//! unaligned. Kernel modules append the helpers they need here, one per array type.

use core::arch::x86_64::*;

macro_rules! ld_st {
    ($feat:literal, $ld:ident, $st:ident, $t:ty, $n:literal, $v:ty, $load:ident, $store:ident) => {
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) fn $ld(a: &[$t; $n]) -> $v {
            // SAFETY: the intrinsic reads at most `size_of::<[$t; $n]>()` bytes, all inside `a`; it is
            // unaligned.
            unsafe { $load(a.as_ptr().cast()) }
        }
        #[inline]
        #[target_feature(enable = $feat)]
        pub(crate) fn $st(a: &mut [$t; $n], v: $v) {
            // SAFETY: the intrinsic writes at most `size_of::<[$t; $n]>()` bytes, all inside `a`; it is
            // unaligned.
            unsafe { $store(a.as_mut_ptr().cast(), v) }
        }
    };
}

// 128 bits (SSE2 is part of x86_64; the AVX2 feature lets AVX2 and AVX-512 kernels inline them).
ld_st!(
    "avx2",
    ld_u16x8,
    st_u16x8,
    u16,
    8,
    __m128i,
    _mm_loadu_si128,
    _mm_storeu_si128
);
ld_st!(
    "avx2",
    ld_i16x8,
    st_i16x8,
    i16,
    8,
    __m128i,
    _mm_loadu_si128,
    _mm_storeu_si128
);
ld_st!(
    "avx2",
    ld_i32x4,
    st_i32x4,
    i32,
    4,
    __m128i,
    _mm_loadu_si128,
    _mm_storeu_si128
);
ld_st!(
    "avx2",
    ld_u8x16,
    st_u8x16,
    u8,
    16,
    __m128i,
    _mm_loadu_si128,
    _mm_storeu_si128
);
// 64 bits, in the low half of a 128-bit register (the upper half is zero on load).
ld_st!(
    "avx2",
    ld_u16x4,
    st_u16x4,
    u16,
    4,
    __m128i,
    _mm_loadl_epi64,
    _mm_storel_epi64
);
ld_st!(
    "avx2",
    ld_i16x4,
    st_i16x4,
    i16,
    4,
    __m128i,
    _mm_loadl_epi64,
    _mm_storel_epi64
);
ld_st!(
    "avx2",
    ld_u8x8,
    st_u8x8,
    u8,
    8,
    __m128i,
    _mm_loadl_epi64,
    _mm_storel_epi64
);
// 256 bits.
ld_st!(
    "avx2",
    ld_u16x16,
    st_u16x16,
    u16,
    16,
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
ld_st!(
    "avx2",
    ld_i16x16,
    st_i16x16,
    i16,
    16,
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
ld_st!(
    "avx2",
    ld_i32x8,
    st_i32x8,
    i32,
    8,
    __m256i,
    _mm256_loadu_si256,
    _mm256_storeu_si256
);
// 512 bits, for the AVX-512 transform kernels (`src/transform/avx512.rs`).
ld_st!(
    "avx512f",
    ld_i32x16,
    st_i32x16,
    i32,
    16,
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512
);
ld_st!(
    "avx512f",
    ld_i16x32,
    st_i16x32,
    i16,
    32,
    __m512i,
    _mm512_loadu_si512,
    _mm512_storeu_si512
);
