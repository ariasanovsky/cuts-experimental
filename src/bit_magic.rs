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

use dyn_stack::PodStack;
use equator::{assert, debug_assert};
#[cfg(not(feature = "nightly"))]
use pulp::{f64x4, u64x4, x86::V3};

#[cfg(feature = "nightly")]
use pulp::{
    b8, f64x4, f64x8, u64x4,
    x86::{V3, V4},
};

#[derive(Copy, Clone, Debug)]
pub(crate) struct CacheParams {
    pub mc: usize,
    pub nc: usize,
    pub kc: usize,
}

#[cfg(feature = "nightly")]
pub(crate) fn cache_parameters_avx512(m: usize, n: usize, k: usize) -> CacheParams {
    let gemm_common::cache::KernelParams { kc, mc, nc } =
        gemm_common::cache::kernel_params(m, n, k, 32, 8, 1);

    CacheParams { mc, nc, kc }
}

pub(crate) fn cache_parameters_avx2(m: usize, n: usize, k: usize) -> CacheParams {
    let gemm_common::cache::KernelParams { kc, mc, nc } =
        gemm_common::cache::kernel_params(m, n, k, 8, 8, 1);

    CacheParams { mc, nc, kc }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    RowMajor,
    ColMajor,
}

#[cfg(feature = "nightly")]
#[inline(always)]
pub(crate) unsafe fn lazy_kernel_32x8_avx512(
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
    debug_assert!(k % 4 == 0);

    let dst = dst.as_mut_ptr();
    let lhs = lhs.as_ptr();
    let rhs = rhs.as_ptr();
    let diag = diag.as_ptr();

    let mut acc = [[simd.splat_f64x8(0.0); 4]; 8];
    let pos = simd.splat_f64x8(0.0);
    let neg = simd.splat_f64x8(-0.0);

    let pos_one = simd.splat_f64x8(1.0);
    let neg_one = simd.splat_f64x8(-1.0);

    let mut depth = k / 4;
    while depth > 0 {
        depth -= 1;

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
pub(crate) unsafe fn lazy_kernel_8x4_avx2(
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
pub(crate) fn lazy_matmul_avx512(
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

pub(crate) fn lazy_matmul_avx2(
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
}
