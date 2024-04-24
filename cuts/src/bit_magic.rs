// Sarah wrote this
// She's so smart

// eager version
//             B0 B1 B2 B3 ..
// A0 * D0
// A1 * D1
// A2 * D2
// A3 * D3
// ..
//
// accumulator += LHS (f64) * RHS (bits)
//
// lazy
//
//             B0 B1 B2 B3 ..
//             --------------
//
//             --------------
//
//             --------------
// A0 |  |  |
// A1 |  |  |
// A2 |  |  |
// A3 |  |  |
// ..
//
// accumulator += LHS (bits) * D(f64) * RHS (bits)

use std::iter::zip;

use aligned_vec::CACHELINE_ALIGN;
use dyn_stack::PodStack;
use equator::{assert, debug_assert};
use faer::MatRef;
#[cfg(feature = "nightly")]
use pulp::{b8, f64x8, x86::V4};
use pulp::{f64x4, u64x4, x86::V3, Simd};
use reborrow::*;

#[derive(Copy, Clone, Debug)]
pub struct CacheParams {
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
}

#[cfg(feature = "nightly")]
pub fn cache_parameters_avx512(m: usize, n: usize, k: usize) -> CacheParams {
    let gemm_common::cache::KernelParams { kc, mc, nc } =
        gemm_common::cache::kernel_params(m, n, k, 32, 8, 1);

    CacheParams { mc, nc, kc }
}

pub fn cache_parameters_avx2(m: usize, n: usize, k: usize) -> CacheParams {
    let gemm_common::cache::KernelParams { kc, mc, nc } =
        gemm_common::cache::kernel_params(m, n, k, 8, 8, core::mem::size_of::<f64>());

    CacheParams { mc, nc, kc }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

#[inline(always)]
#[cfg(feature = "nightly")]
pub unsafe fn lazy_kernel_32x8_avx512(
    simd: V4,
    k: usize,
    dst: &mut [f64],
    dst_col_stride: usize,
    lhs: &[u8],
    lhs_col_stride: usize,
    rhs: &[u8],
    rhs_row_stride: usize,
    diag: &[f64],
) {
    let dst = dst.as_mut_ptr();
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();
    let diag = diag.as_ptr();

    let mut acc = [[simd.splat_f64x8(0.0); 4]; 8];
    let pos = simd.splat_f64x8(0.0);
    let neg = simd.splat_f64x8(-0.0);

    let pos_one = simd.splat_f64x8(1.0);
    let neg_one = simd.splat_f64x8(-1.0);

    let mut count = k / 4;
    while count > 0 {
        let depth = k / 4 - count;
        count -= 1;

        seq_macro::seq! { DEPTH_INNER in 0..4 {{
            let depth_inner = DEPTH_INNER;
            let d = simd.splat_f64x8(*diag.add(4 * depth + depth_inner));

            let lhs0 = *lhs.add(((4 * depth + depth_inner) * lhs_col_stride) / 8 + 0);
            let lhs1 = *lhs.add(((4 * depth + depth_inner) * lhs_col_stride) / 8 + 1);
            let lhs2 = *lhs.add(((4 * depth + depth_inner) * lhs_col_stride) / 8 + 2);
            let lhs3 = *lhs.add(((4 * depth + depth_inner) * lhs_col_stride) / 8 + 3);
            let all_rhs = *rhs.add(((4 * depth + depth_inner) * rhs_row_stride) / 8);

            let lhs0 = simd.select_f64x8(b8(lhs0), pos, neg);
            let lhs1 = simd.select_f64x8(b8(lhs1), pos, neg);
            let lhs2 = simd.select_f64x8(b8(lhs2), pos, neg);
            let lhs3 = simd.select_f64x8(b8(lhs3), pos, neg);

            let lhs0 = simd.xor_f64x8(lhs0, d);
            let lhs1 = simd.xor_f64x8(lhs1, d);
            let lhs2 = simd.xor_f64x8(lhs2, d);
            let lhs3 = simd.xor_f64x8(lhs3, d);

            seq_macro::seq! { J in 0..8 {{
                let j = J;
                let rhs = simd.select_f64x8(b8(((all_rhs >> j) & 1).wrapping_neg()), pos_one, neg_one);
                acc[j][0] = simd.mul_add_f64x8(rhs, lhs0, acc[j][0]);
                acc[j][1] = simd.mul_add_f64x8(rhs, lhs1, acc[j][1]);
                acc[j][2] = simd.mul_add_f64x8(rhs, lhs2, acc[j][2]);
                acc[j][3] = simd.mul_add_f64x8(rhs, lhs3, acc[j][3]);
            }}}
        }}}
    }
    count = k % 4;

    while count > 0 {
        let depth = k - count;
        count -= 1;

        let d = simd.splat_f64x8(*diag.add(depth));

        let lhs0 = *lhs.add((depth * lhs_col_stride) / 8 + 0);
        let lhs1 = *lhs.add((depth * lhs_col_stride) / 8 + 1);
        let lhs2 = *lhs.add((depth * lhs_col_stride) / 8 + 2);
        let lhs3 = *lhs.add((depth * lhs_col_stride) / 8 + 3);
        let all_rhs = *rhs.add((depth * rhs_row_stride) / 8);

        let lhs0 = simd.select_f64x8(b8(lhs0), pos, neg);
        let lhs1 = simd.select_f64x8(b8(lhs1), pos, neg);
        let lhs2 = simd.select_f64x8(b8(lhs2), pos, neg);
        let lhs3 = simd.select_f64x8(b8(lhs3), pos, neg);

        let lhs0 = simd.xor_f64x8(lhs0, d);
        let lhs1 = simd.xor_f64x8(lhs1, d);
        let lhs2 = simd.xor_f64x8(lhs2, d);
        let lhs3 = simd.xor_f64x8(lhs3, d);

        seq_macro::seq! { J in 0..8 {{
            let j = J;
            let rhs = simd.select_f64x8(b8(((all_rhs >> j) & 1).wrapping_neg()), pos_one, neg_one);
            acc[j][0] = simd.mul_add_f64x8(rhs, lhs0, acc[j][0]);
            acc[j][1] = simd.mul_add_f64x8(rhs, lhs1, acc[j][1]);
            acc[j][2] = simd.mul_add_f64x8(rhs, lhs2, acc[j][2]);
            acc[j][3] = simd.mul_add_f64x8(rhs, lhs3, acc[j][3]);
        }}}
    }

    let dst = dst as *mut f64x8;
    let dst_col_stride = dst_col_stride / 8;

    seq_macro::seq! { J in 0..8 {{
        seq_macro::seq! { I in 0..4 {{
            let i = I;
            let j = J;
            let dst = &mut *dst.add(i + j * dst_col_stride);
            *dst = simd.add_f64x8(*dst, acc[j][i]);
        }}}
    }}}
}

#[inline(always)]
pub unsafe fn kernel_8x4_avx2(
    simd: V3,
    k: usize,
    dst: &mut [f64],
    dst_col_stride: usize,
    lhs: &[f64],
    lhs_col_stride: usize,
    rhs: &[f64],
    rhs_row_stride: usize,
) {
    debug_assert!(k % 4 == 0);

    let dst = dst.as_mut_ptr();
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();

    let mut acc = [[simd.splat_f64x4(0.0); 2]; 4];

    let mut depth = k / 4;
    while depth > 0 {
        depth -= 1;

        seq_macro::seq! { DEPTH_INNER in 0..4 {{
            let depth_inner = DEPTH_INNER;

            let lhs0 = *(lhs.add(((4 * depth + depth_inner) * lhs_col_stride) + 4 * 0) as *const f64x4);
            let lhs1 = *(lhs.add(((4 * depth + depth_inner) * lhs_col_stride) + 4 * 1) as *const f64x4);

            seq_macro::seq! { J in 0..4 {{
                let j = J;
                let rhs = simd.splat_f64x4(*rhs.add(((4 * depth + depth_inner) * rhs_row_stride) + J));
                acc[j][0] = simd.mul_add_f64x4(rhs, lhs0, acc[j][0]);
                acc[j][1] = simd.mul_add_f64x4(rhs, lhs1, acc[j][1]);
            }}}
        }}}
    }

    let dst = dst as *mut f64x4;
    let dst_col_stride = dst_col_stride / 4;

    seq_macro::seq! { J in 0..4 {{
        seq_macro::seq! { I in 0..2 {{
            let i = I;
            let j = J;
            let dst = &mut *dst.add(i + j * dst_col_stride);
            *dst = simd.add_f64x4(*dst, acc[j][i]);
        }}}
    }}}
}

#[inline(always)]
pub unsafe fn half_lazy_kernel_8x4_avx2(
    simd: V3,
    k: usize,
    dst: &mut [f64],
    dst_col_stride: usize,
    lhs: &[f64],
    lhs_col_stride: usize,
    rhs: &[u8],
    rhs_offset: u32,
    rhs_row_stride: usize,
) {
    debug_assert!(k % 4 == 0);

    let dst = dst.as_mut_ptr();
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();

    let mut acc = [[simd.splat_f64x4(0.0); 2]; 4];

    let pos_one = simd.splat_f64x4(1.0);
    let neg_one = simd.splat_f64x4(-1.0);

    let mut depth = k / 4;
    while depth > 0 {
        depth -= 1;

        seq_macro::seq! { DEPTH_INNER in 0..4 {{
            let depth_inner = DEPTH_INNER;

            let lhs0 = *(lhs.add(((4 * depth + depth_inner) * lhs_col_stride) + 4 * 0) as *const f64x4);
            let lhs1 = *(lhs.add(((4 * depth + depth_inner) * lhs_col_stride) + 4 * 1) as *const f64x4);
            let all_rhs = *rhs.add(((4 * depth + depth_inner) * rhs_row_stride) / 8);

            seq_macro::seq! { J in 0..4 {{
                let j = J;
                let rhs = core::mem::transmute(simd.splat_u64x4((((all_rhs >> (J + rhs_offset)) & 1) as u64).wrapping_neg()));
                let rhs = simd.select_f64x4(rhs, pos_one, neg_one);
                acc[j][0] = simd.mul_add_f64x4(rhs, lhs0, acc[j][0]);
                acc[j][1] = simd.mul_add_f64x4(rhs, lhs1, acc[j][1]);
            }}}
        }}}
    }

    let dst = dst as *mut f64x4;
    let dst_col_stride = dst_col_stride / 4;

    seq_macro::seq! { J in 0..4 {{
        seq_macro::seq! { I in 0..2 {{
            let i = I;
            let j = J;
            let dst = &mut *dst.add(i + j * dst_col_stride);
            *dst = simd.add_f64x4(*dst, acc[j][i]);
        }}}
    }}}
}

#[inline(always)]
pub unsafe fn lazy_kernel_8x4_avx2(
    simd: V3,
    k: usize,
    dst: &mut [f64],
    dst_col_stride: usize,
    lhs: &[u8],
    lhs_col_stride: usize,
    rhs: &[u8],
    rhs_offset: u32,
    rhs_row_stride: usize,
    diag: &[f64],
) {
    debug_assert!(k % 4 == 0);

    let dst = dst.as_mut_ptr();
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();
    let diag = diag.as_ptr();

    let mut acc = [[simd.splat_f64x4(0.0); 2]; 4];
    let pos = simd.splat_f64x4(0.0);
    let neg = simd.splat_f64x4(-0.0);

    let pos_one = simd.splat_f64x4(1.0);
    let neg_one = simd.splat_f64x4(-1.0);

    let mut depth = k / 4;
    while depth > 0 {
        depth -= 1;

        seq_macro::seq! { DEPTH_INNER in 0..4 {{
            let depth_inner = DEPTH_INNER;
            let d = simd.splat_f64x4(*diag.add(4 * depth + depth_inner));

            let lhs01 = *lhs.add(((4 * depth + depth_inner) * lhs_col_stride) / 8 + 0);
            let all_rhs = *rhs.add(((4 * depth + depth_inner) * rhs_row_stride) / 8);

            let mask0 = u64x4(1, 2, 4, 8);
            let mask1 = u64x4(16, 32, 64, 128);

            let lhs0 = simd.cmp_eq_u64x4(simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0), mask0);
            let lhs1 = simd.cmp_eq_u64x4(simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1), mask1);

            let lhs0 = simd.select_f64x4(lhs0, pos, neg);
            let lhs1 = simd.select_f64x4(lhs1, pos, neg);

            let lhs0 = simd.xor_f64x4(lhs0, d);
            let lhs1 = simd.xor_f64x4(lhs1, d);

            seq_macro::seq! { J in 0..4 {{
                let j = J;
                let rhs = core::mem::transmute(simd.splat_u64x4((((all_rhs >> (J + rhs_offset)) & 1) as u64).wrapping_neg()));
                let rhs = simd.select_f64x4(rhs, pos_one, neg_one);
                acc[j][0] = simd.mul_add_f64x4(rhs, lhs0, acc[j][0]);
                acc[j][1] = simd.mul_add_f64x4(rhs, lhs1, acc[j][1]);
            }}}
        }}}
    }

    let dst = dst as *mut f64x4;
    let dst_col_stride = dst_col_stride / 4;

    seq_macro::seq! { J in 0..4 {{
        seq_macro::seq! { I in 0..2 {{
            let i = I;
            let j = J;
            let dst = &mut *dst.add(i + j * dst_col_stride);
            *dst = simd.add_f64x4(*dst, acc[j][i]);
        }}}
    }}}
}

#[cfg(feature = "nightly")]
pub fn lazy_matmul_avx512(
    cache_params: CacheParams,
    m: usize,
    n: usize,
    k: usize,
    dst: &mut [f64],
    lhs: &[u8],
    diag: &[f64],
    rhs: &[u8],
    rhs_layout: Layout,
    stack: PodStack<'_>,
) {
    struct Impl<'a> {
        cache_params: CacheParams,
        m: usize,
        n: usize,
        k: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        diag: &'a [f64],
        rhs: &'a [u8],
        rhs_layout: Layout,
        stack: PodStack<'a>,
        simd: V4,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                cache_params,
                m,
                n,
                k,
                dst,
                lhs,
                diag,
                rhs,
                rhs_layout,
                stack,
                simd,
            } = self;

            assert!(all(
                dst.len() == m * n,
                lhs.len() * 8 == m * k,
                rhs.len() * 8 == n * k,
                diag.len() == k,
                m % 32 == 0,
                n % 8 == 0,
            ));

            let mut stack = stack;
            _ = &mut stack;

            // lhs should be stored in column major mode
            // rhs can be column major or row major

            assert!(rhs_layout == Layout::RowMajor);

            let CacheParams { mc, nc, kc } = cache_params;

            let mut col = 0;
            while col < n {
                let nb = Ord::min(n - col, nc);

                let mut depth = 0;
                while depth < k {
                    let kb = Ord::min(k - depth, kc);

                    let mut row = 0;
                    while row < m {
                        let mb = Ord::min(m - row, mc);

                        let mut col_inner = 0;
                        while col_inner < nb {
                            let mut row_inner = 0;
                            while row_inner < mb {
                                let row = row + row_inner;
                                let col = col + col_inner;

                                unsafe {
                                    lazy_kernel_32x8_avx512(
                                        simd,
                                        kb,
                                        &mut dst[row + m * col..],
                                        m,
                                        &lhs[(row + m * depth) / 8..],
                                        m,
                                        &rhs[(col + n * depth) / 8..],
                                        n,
                                        &diag[depth..],
                                    )
                                };

                                row_inner += 32;
                            }
                            col_inner += 8;
                        }

                        row += mb;
                    }
                    depth += kb;
                }
                col += nb;
            }
        }
    }

    let simd = V4::try_new().unwrap();

    simd.vectorize(Impl {
        cache_params,
        m,
        n,
        k,
        dst,
        lhs,
        diag,
        rhs,
        rhs_layout,
        stack,
        simd,
    });
}

pub fn lazy_matmul_avx2(
    cache_params: CacheParams,
    m: usize,
    n: usize,
    k: usize,
    dst: &mut [f64],
    lhs: &[u8],
    diag: &[f64],
    rhs: &[u8],
    rhs_layout: Layout,
    stack: PodStack<'_>,
) {
    struct Impl<'a> {
        cache_params: CacheParams,
        m: usize,
        n: usize,
        k: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        diag: &'a [f64],
        rhs: &'a [u8],
        rhs_layout: Layout,
        stack: PodStack<'a>,
        simd: V3,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                cache_params,
                m,
                n,
                k,
                dst,
                lhs,
                diag,
                rhs,
                rhs_layout,
                stack,
                simd,
            } = self;

            assert!(all(
                dst.len() == m * n,
                lhs.len() * 8 == m * k,
                rhs.len() * 8 == n * k,
                diag.len() == k,
                m % 8 == 0,
                n % 8 == 0,
            ));

            let mut stack = stack;
            _ = &mut stack;

            // lhs should be stored in column major mode
            // rhs can be column major or row major

            assert!(rhs_layout == Layout::RowMajor);

            let CacheParams { mc, nc, kc } = cache_params;

            let mut col = 0;
            while col < n {
                let nb = Ord::min(n - col, nc);

                let mut depth = 0;
                while depth < k {
                    let kb = Ord::min(k - depth, kc);

                    let mut row = 0;
                    while row < m {
                        let mb = Ord::min(m - row, mc);

                        let mut col_inner = 0;
                        while col_inner < nb {
                            let mut row_inner = 0;
                            while row_inner < mb {
                                let row = row + row_inner;
                                let col = col + col_inner;

                                unsafe {
                                    lazy_kernel_8x4_avx2(
                                        simd,
                                        kb,
                                        &mut dst[row + m * col..],
                                        m,
                                        &lhs[(row + m * depth) / 8..],
                                        m,
                                        &rhs[(col + n * depth) / 8..],
                                        0,
                                        n,
                                        &diag[depth..],
                                    );

                                    lazy_kernel_8x4_avx2(
                                        simd,
                                        kb,
                                        &mut dst[row + m * (col + 4)..],
                                        m,
                                        &lhs[(row + m * depth) / 8..],
                                        m,
                                        &rhs[(col + 4 + n * depth) / 8..],
                                        4,
                                        n,
                                        &diag[depth..],
                                    );
                                }

                                row_inner += 8;
                            }
                            col_inner += 8;
                        }

                        row += mb;
                    }
                    depth += kb;
                }
                col += nb;
            }
        }
    }

    let simd = V3::try_new().unwrap();

    simd.vectorize(Impl {
        cache_params,
        m,
        n,
        k,
        dst,
        lhs,
        diag,
        rhs,
        rhs_layout,
        stack,
        simd,
    });
}

pub fn half_lazy_matmul_avx2(
    cache_params: CacheParams,
    m: usize,
    n: usize,
    k: usize,
    dst: &mut [f64],
    lhs: &[u8],
    diag: &[f64],
    rhs: &[u8],
    rhs_layout: Layout,
    stack: PodStack<'_>,
) {
    struct Impl<'a> {
        cache_params: CacheParams,
        m: usize,
        n: usize,
        k: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        diag: &'a [f64],
        rhs: &'a [u8],
        rhs_layout: Layout,
        stack: PodStack<'a>,
        simd: V3,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                cache_params,
                m,
                n,
                k,
                dst,
                lhs,
                diag,
                rhs,
                rhs_layout,
                stack,
                simd,
            } = self;

            assert!(all(
                dst.len() == m * n,
                lhs.len() * 8 == m * k,
                rhs.len() * 8 == n * k,
                diag.len() == k,
                m % 8 == 0,
                n % 8 == 0,
            ));

            let mut stack = stack;

            // lhs should be stored in column major mode
            // rhs can be column major or row major

            assert!(rhs_layout == Layout::RowMajor);

            let CacheParams { mc, nc, kc } = cache_params;

            let (packed_lhs, _) = stack
                .rb_mut()
                .make_aligned_raw::<f64>(mc * kc, CACHELINE_ALIGN);

            let mut col = 0;
            while col < n {
                let nb = Ord::min(n - col, nc);

                let mut depth = 0;
                while depth < k {
                    let kb = Ord::min(k - depth, kc);

                    let mut row = 0;
                    while row < m {
                        let mb = Ord::min(m - row, mc);

                        // pack the LHS, and multiply on the right by diag
                        let packed_lhs = &mut packed_lhs[..mb * kb];
                        let lhs = &lhs[(row + depth * m) / 8..];
                        let diag = &diag[depth..];

                        for j in 0..kb {
                            let d = diag[j];
                            let src_col = &lhs[(j * mb) / 8..][..mb / 8];
                            let dst_col =
                                pulp::as_arrays_mut::<8, _>(&mut packed_lhs[j * mb..][..mb]).0;

                            for (dst, &src) in core::iter::zip(dst_col, src_col) {
                                for (idx, dst) in dst.iter_mut().enumerate() {
                                    let bit = ((src >> idx) & 1) == 1;
                                    let src = if bit { d } else { -d };
                                    *dst = src;
                                }
                            }
                        }

                        let mut col_inner = 0;
                        while col_inner < nb {
                            let mut row_inner = 0;
                            while row_inner < mb {
                                let row = row + row_inner;
                                let col = col + col_inner;

                                unsafe {
                                    half_lazy_kernel_8x4_avx2(
                                        simd,
                                        kb,
                                        &mut dst[row + m * col..],
                                        m,
                                        &packed_lhs[row_inner..],
                                        mb,
                                        &rhs[(col + n * depth) / 8..],
                                        0,
                                        n,
                                    );

                                    half_lazy_kernel_8x4_avx2(
                                        simd,
                                        kb,
                                        &mut dst[row + m * (col + 4)..],
                                        mb,
                                        &packed_lhs[row_inner..],
                                        mb,
                                        &rhs[(col + 4 + n * depth) / 8..],
                                        4,
                                        n,
                                    );
                                }

                                row_inner += 8;
                            }
                            col_inner += 8;
                        }

                        row += mb;
                    }
                    depth += kb;
                }
                col += nb;
            }
        }
    }

    let simd = V3::try_new().unwrap();

    simd.vectorize(Impl {
        cache_params,
        m,
        n,
        k,
        dst,
        lhs,
        diag,
        rhs,
        rhs_layout,
        stack,
        simd,
    });
}

#[allow(dead_code)]
static BYTE_TO_VAL: [[f64; 8]; 256] = {
    let mut table = [[0.0f64; 8]; 256];

    let mut i = 0;
    while i < 256 {
        let ii = i as u8;
        let mut val = [0.0f64; 8];

        let mut idx = 0;
        while idx < 8 {
            let bit = (ii >> idx) & 1 == 1;
            val[idx] = if bit { 1.0 } else { -1.0 };
            idx += 1;
        }

        table[i] = val;

        i += 1;
    }

    table
};

pub fn matmul_avx2(
    cache_params: CacheParams,
    m: usize,
    n: usize,
    k: usize,
    dst: &mut [f64],
    lhs: &[u8],
    diag: &[f64],
    rhs: &[u8],
    rhs_layout: Layout,
    stack: PodStack<'_>,
) {
    struct Impl<'a> {
        cache_params: CacheParams,
        m: usize,
        n: usize,
        k: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        diag: &'a [f64],
        rhs: &'a [u8],
        rhs_layout: Layout,
        stack: PodStack<'a>,
        simd: V3,
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                cache_params,
                m,
                n,
                k,
                dst,
                lhs,
                diag,
                rhs,
                rhs_layout,
                stack,
                simd,
            } = self;

            assert!(all(
                dst.len() == m * n,
                lhs.len() * 8 == m * k,
                rhs.len() * 8 == n * k,
                diag.len() == k,
                m % 8 == 0,
                n % 8 == 0,
            ));

            let mut stack = stack;

            // lhs should be stored in column major mode
            // rhs can be column major or row major

            assert!(rhs_layout == Layout::RowMajor);

            let CacheParams { mc, nc, kc } = cache_params;

            let (packed_lhs, mut stack) = stack
                .rb_mut()
                .make_aligned_raw::<f64>(mc * kc, CACHELINE_ALIGN);
            let (packed_rhs, _) = stack
                .rb_mut()
                .make_aligned_raw::<f64>(kc * 8, CACHELINE_ALIGN);

            let mut col = 0;
            while col < n {
                let nb = Ord::min(n - col, nc);

                let mut depth = 0;
                while depth < k {
                    let kb = Ord::min(k - depth, kc);

                    let mut row = 0;
                    while row < m {
                        let mb = Ord::min(m - row, mc);

                        // pack the LHS, and multiply on the right by diag
                        let packed_lhs = &mut packed_lhs[..mb * kb];
                        {
                            let lhs = &lhs[(row + depth * m) / 8..];
                            let diag = &diag[depth..];

                            for j in 0..kb {
                                let d = diag[j];
                                let src_col = &lhs[(j * m) / 8..][..mb / 8];
                                let dst_col =
                                    pulp::as_arrays_mut::<8, _>(&mut packed_lhs[j * mb..][..mb]).0;

                                for (dst, &src) in core::iter::zip(dst_col, src_col) {
                                    for (idx, dst) in dst.iter_mut().enumerate() {
                                        let bit = ((src >> idx) & 1) == 1;
                                        let src = if bit { d } else { -d };
                                        *dst = src;
                                    }
                                }
                            }
                        }

                        let mut col_inner = 0;
                        while col_inner < nb {
                            let col = col + col_inner;

                            let packed_rhs = &mut packed_rhs[..8 * kb];

                            {
                                let packed_rhs = pulp::as_arrays_mut::<8, _>(packed_rhs).0;
                                let rhs = &rhs[(col + depth * n) / 8..];

                                for i in 0..kb {
                                    let src = rhs[(i * n) / 8];
                                    let dst = &mut packed_rhs[i];

                                    for (idx, dst) in dst.iter_mut().enumerate() {
                                        let bit = ((src >> idx) & 1) == 1;
                                        let src = if bit { 1.0 } else { -1.0 };
                                        *dst = src;
                                    }
                                }
                            }

                            let mut row_inner = 0;
                            while row_inner < mb {
                                let row = row + row_inner;

                                unsafe {
                                    kernel_8x4_avx2(
                                        simd,
                                        kb,
                                        &mut dst[row + m * col..],
                                        m,
                                        &packed_lhs[row_inner..],
                                        mb,
                                        packed_rhs,
                                        8,
                                    );

                                    kernel_8x4_avx2(
                                        simd,
                                        kb,
                                        &mut dst[row + m * (col + 4)..],
                                        m,
                                        &packed_lhs[row_inner..],
                                        mb,
                                        &packed_rhs[4..],
                                        8,
                                    );
                                }

                                row_inner += 8;
                            }
                            col_inner += 8;
                        }

                        row += mb;
                    }
                    depth += kb;
                }
                col += nb;
            }
        }
    }

    let simd = V3::try_new().unwrap();

    simd.vectorize(Impl {
        cache_params,
        m,
        n,
        k,
        dst,
        lhs,
        diag,
        rhs,
        rhs_layout,
        stack,
        simd,
    });
}

#[cfg(feature = "nightly")]
pub fn matvec_avx512(m: usize, n: usize, dst: &mut [f64], lhs: &[u8], rhs: &[f64]) {
    struct Impl<'a> {
        simd: V4,
        m: usize,
        n: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        rhs: &'a [f64],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                m % 8 == 0,
                lhs.len() == m * n / 8, //
                rhs.len() == n,         //
                dst.len() == m,         //
            ));

            let dst = pulp::as_arrays_mut::<8, _>(dst).0;

            for (lhs, &rhs) in zip(lhs.chunks_exact(m / 8), rhs) {
                let pos_rhs = simd.splat_f64x8(rhs);
                let neg_rhs = simd.splat_f64x8(-rhs);

                for (dst, &lhs) in zip(&mut *dst, lhs) {
                    let acc = simd.select_f64x8(b8(lhs), pos_rhs, neg_rhs);
                    *dst = pulp::cast(simd.add_f64x8(pulp::cast(*dst), acc));
                }
            }
        }
    }

    let simd = V4::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        m,
        n,
        dst,
        lhs,
        rhs,
    });
}

pub fn matvec_avx2(m: usize, n: usize, dst: &mut [f64], lhs: &[u8], rhs: &[f64]) {
    struct Impl<'a> {
        simd: V3,
        m: usize,
        n: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        rhs: &'a [f64],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                m % 8 == 0,
                lhs.len() == m * n / 8, //
                rhs.len() == n,         //
                dst.len() == m,         //
            ));

            let mask0 = u64x4(1, 2, 4, 8);
            let mask1 = u64x4(16, 32, 64, 128);

            let dst = pulp::as_arrays_mut::<8, _>(dst).0;

            for (lhs, &rhs) in zip(lhs.chunks_exact(m / 8), rhs) {
                let pos_rhs = simd.splat_f64x4(rhs);
                let neg_rhs = simd.splat_f64x4(-rhs);

                for (dst, &lhs01) in zip(&mut *dst, lhs) {
                    let lhs0 = simd
                        .cmp_eq_u64x4(simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0), mask0);
                    let lhs1 = simd
                        .cmp_eq_u64x4(simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1), mask1);

                    let acc0 = simd.select_f64x4(lhs0, pos_rhs, neg_rhs);
                    let acc1 = simd.select_f64x4(lhs1, pos_rhs, neg_rhs);

                    let [dst0, dst1]: &mut [f64x4; 2] = bytemuck::cast_mut(dst);
                    *dst0 = simd.add_f64x4(*dst0, acc0);
                    *dst1 = simd.add_f64x4(*dst1, acc1);
                }
            }
        }
    }

    let simd = V3::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        m,
        n,
        dst,
        lhs,
        rhs,
    });
}

#[cfg(feature = "nightly")]
pub fn tmatvec_avx512(m: usize, n: usize, dst: &mut [f64], lhs: &[u8], rhs: &[f64]) {
    struct Impl<'a> {
        simd: V4,
        m: usize,
        n: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        rhs: &'a [f64],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                m % 8 == 0,
                lhs.len() == m * n / 8, //
                rhs.len() == m,         //
                dst.len() == n,         //
            ));

            let pos_one = simd.splat_f64x8(1.0);
            let neg_one = simd.splat_f64x8(-1.0);

            let rhs = pulp::as_arrays::<8, _>(rhs).0;

            for (dst, lhs) in zip(dst, lhs.chunks_exact(m / 8)) {
                let mut acc0 = simd.splat_f64x8(0.0);
                let mut acc1 = simd.splat_f64x8(0.0);
                let mut acc2 = simd.splat_f64x8(0.0);
                let mut acc3 = simd.splat_f64x8(0.0);
                let mut acc4 = simd.splat_f64x8(0.0);
                let mut acc5 = simd.splat_f64x8(0.0);
                let mut acc6 = simd.splat_f64x8(0.0);
                let mut acc7 = simd.splat_f64x8(0.0);

                let (lhs_head, lhs_tail) = pulp::as_arrays::<8, _>(lhs);
                let (rhs_head, rhs_tail) = pulp::as_arrays::<8, _>(rhs);

                for (lhs, rhs) in zip(lhs_head, rhs_head) {
                    {
                        let lhs = lhs[0];
                        let rhs = rhs[0];
                        let a = &mut acc0;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[1];
                        let rhs = rhs[1];
                        let a = &mut acc1;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[2];
                        let rhs = rhs[2];
                        let a = &mut acc2;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[3];
                        let rhs = rhs[3];
                        let a = &mut acc3;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[4];
                        let rhs = rhs[4];
                        let a = &mut acc4;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[5];
                        let rhs = rhs[5];
                        let a = &mut acc5;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[6];
                        let rhs = rhs[6];
                        let a = &mut acc6;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                    {
                        let lhs = lhs[7];
                        let rhs = rhs[7];
                        let a = &mut acc7;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                }

                for (lhs, rhs) in zip(lhs_tail, rhs_tail) {
                    {
                        let lhs = *lhs;
                        let rhs = *rhs;
                        let a = &mut acc0;

                        let l = simd.select_f64x8(b8(lhs), pos_one, neg_one);
                        let r = pulp::cast(rhs);
                        *a = simd.mul_add_f64x8(l, r, *a);
                    }
                }

                acc0 = simd.add_f64x8(acc0, acc1);
                acc2 = simd.add_f64x8(acc2, acc3);
                acc4 = simd.add_f64x8(acc4, acc5);
                acc6 = simd.add_f64x8(acc6, acc7);

                acc0 = simd.add_f64x8(acc0, acc2);
                acc4 = simd.add_f64x8(acc4, acc6);

                acc0 = simd.add_f64x8(acc0, acc4);
                *dst += simd.f64s_reduce_sum(acc0);
            }
        }
    }

    let simd = V4::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        m,
        n,
        dst,
        lhs,
        rhs,
    });
}

pub fn tmatvec_avx2(m: usize, n: usize, dst: &mut [f64], lhs: &[u8], rhs: &[f64]) {
    struct Impl<'a> {
        simd: V3,
        m: usize,
        n: usize,
        dst: &'a mut [f64],
        lhs: &'a [u8],
        rhs: &'a [f64],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                m % 8 == 0,
                lhs.len() == m * n / 8, //
                rhs.len() == m,         //
                dst.len() == n,         //
            ));

            let mask0 = u64x4(1, 2, 4, 8);
            let mask1 = u64x4(16, 32, 64, 128);
            let pos_one = simd.splat_f64x4(1.0);
            let neg_one = simd.splat_f64x4(-1.0);

            let rhs = pulp::as_arrays::<8, _>(rhs).0;

            for (dst, lhs) in zip(dst, lhs.chunks_exact(m / 8)) {
                let mut acc0 = simd.splat_f64x4(0.0);
                let mut acc1 = simd.splat_f64x4(0.0);
                let mut acc2 = simd.splat_f64x4(0.0);
                let mut acc3 = simd.splat_f64x4(0.0);
                let mut acc4 = simd.splat_f64x4(0.0);
                let mut acc5 = simd.splat_f64x4(0.0);
                let mut acc6 = simd.splat_f64x4(0.0);
                let mut acc7 = simd.splat_f64x4(0.0);

                let (lhs_head, lhs_tail) = pulp::as_arrays::<4, _>(lhs);
                let (rhs_head, rhs_tail) = pulp::as_arrays::<4, _>(rhs);

                for (lhs, rhs) in zip(lhs_head, rhs_head) {
                    {
                        let lhs01 = lhs[0];
                        let rhs01 = rhs[0];
                        let a0 = &mut acc0;
                        let a1 = &mut acc1;

                        let l0 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0),
                            mask0,
                        );
                        let l1 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1),
                            mask1,
                        );
                        let [r0, r1]: [f64x4; 2] = pulp::cast(rhs01);
                        *a0 = simd.mul_add_f64x4(simd.select_f64x4(l0, pos_one, neg_one), r0, *a0);
                        *a1 = simd.mul_add_f64x4(simd.select_f64x4(l1, pos_one, neg_one), r1, *a1);
                    }
                    {
                        let lhs01 = lhs[1];
                        let rhs01 = rhs[1];
                        let a0 = &mut acc2;
                        let a1 = &mut acc3;

                        let l0 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0),
                            mask0,
                        );
                        let l1 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1),
                            mask1,
                        );
                        let [r0, r1]: [f64x4; 2] = pulp::cast(rhs01);
                        *a0 = simd.mul_add_f64x4(simd.select_f64x4(l0, pos_one, neg_one), r0, *a0);
                        *a1 = simd.mul_add_f64x4(simd.select_f64x4(l1, pos_one, neg_one), r1, *a1);
                    }
                    {
                        let lhs01 = lhs[2];
                        let rhs01 = rhs[2];
                        let a0 = &mut acc4;
                        let a1 = &mut acc5;

                        let l0 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0),
                            mask0,
                        );
                        let l1 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1),
                            mask1,
                        );
                        let [r0, r1]: [f64x4; 2] = pulp::cast(rhs01);
                        *a0 = simd.mul_add_f64x4(simd.select_f64x4(l0, pos_one, neg_one), r0, *a0);
                        *a1 = simd.mul_add_f64x4(simd.select_f64x4(l1, pos_one, neg_one), r1, *a1);
                    }
                    {
                        let lhs01 = lhs[3];
                        let rhs01 = rhs[3];
                        let a0 = &mut acc6;
                        let a1 = &mut acc7;

                        let l0 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0),
                            mask0,
                        );
                        let l1 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1),
                            mask1,
                        );
                        let [r0, r1]: [f64x4; 2] = pulp::cast(rhs01);
                        *a0 = simd.mul_add_f64x4(simd.select_f64x4(l0, pos_one, neg_one), r0, *a0);
                        *a1 = simd.mul_add_f64x4(simd.select_f64x4(l1, pos_one, neg_one), r1, *a1);
                    }
                }

                for (lhs, rhs) in zip(lhs_tail, rhs_tail) {
                    {
                        let lhs01 = *lhs;
                        let rhs01 = *rhs;
                        let a0 = &mut acc0;
                        let a1 = &mut acc1;

                        let l0 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask0),
                            mask0,
                        );
                        let l1 = simd.cmp_eq_u64x4(
                            simd.and_u64x4(simd.splat_u64x4(lhs01 as u64), mask1),
                            mask1,
                        );
                        let [r0, r1]: [f64x4; 2] = pulp::cast(rhs01);
                        *a0 = simd.mul_add_f64x4(simd.select_f64x4(l0, pos_one, neg_one), r0, *a0);
                        *a1 = simd.mul_add_f64x4(simd.select_f64x4(l1, pos_one, neg_one), r1, *a1);
                    }
                }

                acc0 = simd.add_f64x4(acc0, acc1);
                acc2 = simd.add_f64x4(acc2, acc3);
                acc4 = simd.add_f64x4(acc4, acc5);
                acc6 = simd.add_f64x4(acc6, acc7);

                acc0 = simd.add_f64x4(acc0, acc2);
                acc4 = simd.add_f64x4(acc4, acc6);

                acc0 = simd.add_f64x4(acc0, acc4);
                *dst += simd.f64s_reduce_sum(acc0);
            }
        }
    }

    let simd = V3::try_new().unwrap();
    simd.vectorize(Impl {
        simd,
        m,
        n,
        dst,
        lhs,
        rhs,
    });
}

#[cfg(feature = "nightly")]
pub fn tmatvec_bit_avx512(
    simd: V4,
    m: usize,
    n: usize,
    dst: &mut [f64],
    lhs: MatRef<'_, f64>,
    rhs: &[u8],
) {
    struct Impl<'a> {
        simd: V4,
        m: usize,
        n: usize,
        dst: &'a mut [f64],
        lhs: MatRef<'a, f64>,
        rhs: &'a [u8],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                simd,
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                m % 8 == 0,
                lhs.nrows() == m,
                lhs.ncols() == n,
                rhs.len() == m / 8,
                dst.len() == n,
            ));

            let (rhs4, rhs1) = pulp::as_arrays::<4, _>(rhs);
            let pos = simd.splat_f64x8(1.0);
            let neg = simd.splat_f64x8(-1.0);
            for j in 0..n {
                let lhs = pulp::as_arrays::<8, _>(lhs.col(j).try_as_slice().unwrap()).0;
                let lhs: &[f64x8] = bytemuck::cast_slice(lhs);
                let mut acc0 = simd.splat_f64x8(0.0);
                let mut acc1 = simd.splat_f64x8(0.0);
                let mut acc2 = simd.splat_f64x8(0.0);
                let mut acc3 = simd.splat_f64x8(0.0);

                let (lhs4, lhs1) = pulp::as_arrays::<4, _>(lhs);
                for (&[l0, l1, l2, l3], &[r0, r1, r2, r3]) in zip(lhs4, rhs4) {
                    acc0 = simd.mul_add_f64x8(l0, simd.select_f64x8(b8(r0), pos, neg), acc0);
                    acc1 = simd.mul_add_f64x8(l1, simd.select_f64x8(b8(r1), pos, neg), acc1);
                    acc2 = simd.mul_add_f64x8(l2, simd.select_f64x8(b8(r2), pos, neg), acc2);
                    acc3 = simd.mul_add_f64x8(l3, simd.select_f64x8(b8(r3), pos, neg), acc3);
                }
                for (&l0, &r0) in zip(lhs1, rhs1) {
                    acc0 = simd.mul_add_f64x8(l0, simd.select_f64x8(b8(r0), pos, neg), acc0);
                }
                acc0 = simd.add_f64x8(acc0, acc1);
                acc2 = simd.add_f64x8(acc2, acc3);
                acc0 = simd.add_f64x8(acc0, acc2);
                dst[j] += simd.f64s_reduce_sum(acc0);
            }
        }
    }

    simd.vectorize(Impl {
        simd,
        m,
        n,
        dst,
        lhs,
        rhs,
    })
}

pub fn matvec_bit(m: usize, n: usize, dst: &mut [f64], lhs: MatRef<'_, f64>, rhs: &[u8]) {
    struct Impl<'a> {
        m: usize,
        n: usize,
        dst: &'a mut [f64],
        lhs: MatRef<'a, f64>,
        rhs: &'a [u8],
    }

    impl pulp::NullaryFnOnce for Impl<'_> {
        type Output = ();

        #[inline(always)]
        fn call(self) -> Self::Output {
            let Self {
                m,
                n,
                dst,
                lhs,
                rhs,
            } = self;

            assert!(all(
                n % 8 == 0,
                lhs.nrows() == m,
                lhs.ncols() == n,
                rhs.len() == n / 8,
                dst.len() == m,
            ));

            for j in 0..n {
                let lhs = lhs.col(j).try_as_slice().unwrap();
                let pos = (rhs[j / 8] >> (j % 8)) & 1 == 1;
                if pos {
                    for (dst, &lhs) in zip(&mut *dst, lhs) {
                        *dst += lhs;
                    }
                } else {
                    for (dst, &lhs) in zip(&mut *dst, lhs) {
                        *dst -= lhs;
                    }
                }
            }
        }
    }

    pulp::Arch::new().dispatch(Impl {
        m,
        n,
        dst,
        lhs,
        rhs,
    })
}

pub fn tmatvec_bit(m: usize, n: usize, dst: &mut [f64], lhs: MatRef<'_, f64>, rhs: &[u8]) {
    #[cfg(feature = "nightly")]
    if let Some(simd) = V4::try_new() {
        return tmatvec_bit_avx512(simd, m, n, dst, lhs, rhs);
    }
    todo!()
}

pub fn tmatvec(m: usize, n: usize, dst: &mut [f64], lhs: &[u8], rhs: &[f64]) {
    #[cfg(feature = "nightly")]
    if V4::try_new().is_some() {
        return tmatvec_avx512(m, n, dst, lhs, rhs);
    }
    tmatvec_avx2(m, n, dst, lhs, rhs);
}

pub fn matvec(m: usize, n: usize, dst: &mut [f64], lhs: &[u8], rhs: &[f64]) {
    #[cfg(feature = "nightly")]
    if V4::try_new().is_some() {
        return matvec_avx512(m, n, dst, lhs, rhs);
    }
    matvec_avx2(m, n, dst, lhs, rhs)
}

pub fn cache_parameters(m: usize, n: usize, k: usize) -> CacheParams {
    #[cfg(feature = "nightly")]
    if V4::try_new().is_some() {
        return cache_parameters_avx512(m, n, k);
    }
    cache_parameters_avx2(m, n, k)
}

pub fn matmul(
    cache_params: CacheParams,
    m: usize,
    n: usize,
    k: usize,
    dst: &mut [f64],
    lhs: &[u8],
    diag: &[f64],
    rhs: &[u8],
    rhs_layout: Layout,
    stack: PodStack<'_>,
) {
    #[cfg(feature = "nightly")]
    if V4::try_new().is_some() {
        return lazy_matmul_avx512(
            cache_params,
            m,
            n,
            k,
            dst,
            lhs,
            diag,
            rhs,
            rhs_layout,
            stack,
        );
    }
    matmul_avx2(
        cache_params,
        m,
        n,
        k,
        dst,
        lhs,
        diag,
        rhs,
        rhs_layout,
        stack,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use equator::assert;

    fn naive_matmul(
        m: usize,
        n: usize,
        k: usize,
        dst: &mut [f64],
        lhs: &[u8],
        diag: &[f64],
        rhs: &[u8],
        rhs_layout: Layout,
    ) {
        assert!(rhs_layout == Layout::RowMajor);
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for depth in 0..k {
                    let lhs = ((lhs[(i + depth * m) / 8] >> (i % 8)) & 1) == 1;
                    let rhs = ((rhs[(j + depth * n) / 8] >> (j % 8)) & 1) == 1;

                    let lhs = if lhs { 1.0 } else { -1.0 };
                    let rhs = if rhs { 1.0 } else { -1.0 };
                    let diag = diag[depth];

                    acc += lhs * rhs * diag;
                }
                dst[i + j * m] += acc;
            }
        }
    }

    #[cfg(feature = "nightly")]
    #[test]
    fn test_lazy_matmul() {
        let m = 128;
        let n = 128;
        let k = 16;
        let mut a = vec![0u8; m * k / 8];
        let mut b = vec![0u8; k * n / 8];
        let mut c = vec![0.0; m * n];
        let mut diag = vec![0.0; k];

        for x in &mut a {
            *x = rand::random();
        }
        for x in &mut b {
            *x = rand::random();
        }
        for x in &mut c {
            *x = rand::random();
        }
        for x in &mut diag {
            *x = rand::random();
        }

        let mut d = c.clone();
        let mut d_target = c.clone();

        let cache_params = cache_parameters_avx512(m, n, k);

        naive_matmul(m, n, k, &mut d_target, &a, &diag, &b, Layout::RowMajor);
        lazy_matmul_avx512(
            cache_params,
            m,
            n,
            k,
            &mut d,
            &a,
            &diag,
            &b,
            Layout::RowMajor,
            PodStack::new(&mut []),
        );

        for (&actual, &target) in core::iter::zip(&d, &d_target) {
            assert!((actual - target).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lazy_matmul_avx2() {
        let m = 128;
        let n = 128;
        let k = 16;
        let mut a = vec![0u8; m * k / 8];
        let mut b = vec![0u8; k * n / 8];
        let mut c = vec![0.0; m * n];
        let mut diag = vec![0.0; k];

        for x in &mut a {
            *x = rand::random();
        }
        for x in &mut b {
            *x = rand::random();
        }
        for x in &mut c {
            *x = rand::random();
        }
        for x in &mut diag {
            *x = rand::random();
        }

        let mut d = c.clone();
        let mut d_target = c.clone();

        let cache_params = cache_parameters_avx2(m, n, k);

        naive_matmul(m, n, k, &mut d_target, &a, &diag, &b, Layout::RowMajor);
        lazy_matmul_avx2(
            cache_params,
            m,
            n,
            k,
            &mut d,
            &a,
            &diag,
            &b,
            Layout::RowMajor,
            PodStack::new(&mut []),
        );

        for (&actual, &target) in core::iter::zip(&d, &d_target) {
            assert!((actual - target).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matmul_avx2() {
        let m = 256;
        let n = 256;
        let k = 16;
        let mut a = vec![0u8; m * k / 8];
        let mut b = vec![0u8; k * n / 8];
        let mut c = vec![0.0; m * n];
        let mut diag = vec![0.0; k];

        for x in &mut a {
            *x = rand::random();
        }
        for x in &mut b {
            *x = rand::random();
        }
        for x in &mut c {
            *x = rand::random();
        }
        for x in &mut diag {
            *x = rand::random();
        }

        let mut d = c.clone();
        let mut d_target = c.clone();

        let cache_params = cache_parameters_avx2(m, n, k);

        naive_matmul(m, n, k, &mut d_target, &a, &diag, &b, Layout::RowMajor);
        matmul_avx2(
            cache_params,
            m,
            n,
            k,
            &mut d,
            &a,
            &diag,
            &b,
            Layout::RowMajor,
            PodStack::new(&mut vec![
                0u8;
                (cache_params.mc * cache_params.kc
                    + 8 * cache_params.kc)
                    * core::mem::size_of::<f64>()
                    + CACHELINE_ALIGN
            ]),
        );

        for (&actual, &target) in core::iter::zip(&d, &d_target) {
            assert!((actual - target).abs() < 1e-10);
        }
    }
}
