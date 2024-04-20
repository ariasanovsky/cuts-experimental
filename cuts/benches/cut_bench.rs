use aligned_vec::{avec, CACHELINE_ALIGN};
use cuts::bit_magic::*;
use diol::prelude::*;
use dyn_stack::PodStack;

mod mat_tmat {
    use super::*;
    pub fn params() -> Vec<[usize; 3]> {
        vec![[4096; 3]]
    }

    pub fn bit_mat_tmat_avx2(bencher: Bencher, [m, n, k]: [usize; 3]) {
        let a = avec![1u8; m * k / 8];
        let b = avec![1u8; k * n / 8];
        let diag = avec![1.0; k];
        let mut c = avec![1.0; m * n];

        let cache_params = cache_parameters_avx2(m, n, k);
        let mut mem = vec![
            0u8;
            (8 + cache_params.mc) * cache_params.kc * core::mem::size_of::<f64>()
                + CACHELINE_ALIGN
        ];

        bencher.bench(|| {
            matmul_avx2(
                cache_params,
                m,
                n,
                k,
                &mut c,
                &a,
                &diag,
                &b,
                Layout::RowMajor,
                PodStack::new(&mut mem),
            );
        })
    }

    pub fn bit_mat_tmat_avx512(bencher: Bencher, [m, n, k]: [usize; 3]) {
        if pulp::x86::V4::try_new().is_none() {
            return;
        }

        let a = avec![1u8; m * k / 8];
        let b = avec![1u8; k * n / 8];
        let diag = avec![1.0; k];
        let mut c = avec![1.0; m * n];

        let cache_params = cache_parameters_avx512(m, n, k);

        bencher.bench(|| {
            lazy_matmul_avx512(
                cache_params,
                m,
                n,
                k,
                &mut c,
                &a,
                &diag,
                &b,
                Layout::RowMajor,
                PodStack::new(&mut []),
            );
        })
    }

    pub fn gemm_mat_tmat<T: From<f32> + Copy + 'static>(bencher: Bencher, [m, n, k]: [usize; 3]) {
        let one: T = T::from(1.0);
        let a = avec![one; m * k];
        let b = avec![one; k * n];
        let mut c = avec![one; m * n];
        let mut ad = a.clone();

        bencher.bench(|| unsafe {
            ad.copy_from_slice(&a);
            gemm::gemm(
                m,
                n,
                k,
                c.as_mut_ptr(),
                m as _,
                1,
                true,
                ad.as_ptr(),
                m as _,
                1,
                b.as_ptr(),
                1,
                n as _,
                one,
                one,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        });
    }
}

mod matvec {
    use super::*;
    pub fn params() -> Vec<[usize; 2]> {
        vec![[4096; 2], [4096, 16384]]
    }

    pub fn bit_matvec_avx2(bencher: Bencher, [m, n]: [usize; 2]) {
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; n];
        let mut c = avec![1.0; m];

        bencher.bench(|| matvec_avx2(m, n, &mut c, &a, &b))
    }

    pub fn bit_matvec_avx512(bencher: Bencher, [m, n]: [usize; 2]) {
        if pulp::x86::V4::try_new().is_none() {
            return;
        }
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; n];
        let mut c = avec![1.0; m];

        bencher.bench(|| matvec_avx512(m, n, &mut c, &a, &b))
    }

    pub fn gemm_matvec<T: From<f32> + Copy + 'static>(bencher: Bencher, [m, n]: [usize; 2]) {
        let one = T::from(1.0);
        let a = avec![one; m * n];
        let b = avec![one; n];
        let mut c = avec![one; m];

        bencher.bench(|| unsafe {
            gemm::gemm(
                m,
                1,
                n,
                c.as_mut_ptr(),
                m as _,
                1,
                true,
                a.as_ptr(),
                m as _,
                1,
                b.as_ptr(),
                n as _,
                1,
                one,
                one,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        });
    }
}

mod tmatvec {
    use super::*;
    pub fn params() -> Vec<[usize; 2]> {
        vec![[4096; 2], [4096, 16384]]
    }

    pub fn bit_tmatvec_avx2(bencher: Bencher, [m, n]: [usize; 2]) {
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; m];
        let mut c = avec![1.0; n];

        bencher.bench(|| tmatvec_avx2(m, n, &mut c, &a, &b))
    }

    pub fn bit_tmatvec_avx512(bencher: Bencher, [m, n]: [usize; 2]) {
        if pulp::x86::V4::try_new().is_none() {
            return;
        }
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; m];
        let mut c = avec![1.0; n];

        bencher.bench(|| tmatvec_avx512(m, n, &mut c, &a, &b))
    }

    pub fn gemm_tmatvec<T: From<f32> + Copy + 'static>(bencher: Bencher, [m, n]: [usize; 2]) {
        let one = T::from(1.0);
        let a = avec![one; m * n];
        let b = avec![one; n];
        let mut c = avec![one; m];

        bencher.bench(|| unsafe {
            gemm::gemm(
                m,
                1,
                n,
                c.as_mut_ptr(),
                m as _,
                1,
                true,
                a.as_ptr(),
                1,
                n as _,
                b.as_ptr(),
                n as _,
                1,
                one,
                one,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        });
    }
}

mod sctvec {
    use super::*;
    pub fn params() -> Vec<PlotArg> {
        (1..17).map(|k| k * 1024).map(PlotArg).collect()
    }

    pub fn bit_sctvec_avx2(bencher: Bencher, PlotArg(k): PlotArg) {
        let m = 4096;
        let n = 4096;
        
        let s = avec![1u8; m * k / 8];
        let t = avec![1u8; k * n / 8];
        let c = avec![1.0; k];

        let x = avec![1.0; n];
        let mut y = avec![1.0; k];
        let mut z = avec![1.0; m];

        bencher.bench(|| {
            y.fill(0.0);
            z.fill(0.0);
            tmatvec_avx2(n, k, &mut y, &t, &x);
            for (y, &c) in y.iter_mut().zip(&*c) {
                *y = *y * c;
            }
            matvec_avx2(m, k, &mut z, &s, &y);
        })
    }

    pub fn bit_sctvec_avx512(bencher: Bencher, PlotArg(k): PlotArg) {
        let m = 4096;
        let n = 4096;
        if pulp::x86::V4::try_new().is_none() {
            return;
        }
        let s = avec![1u8; m * k / 8];
        let t = avec![1u8; k * n / 8];
        let c = avec![1.0; k];

        let x = avec![1.0; n];
        let mut y = avec![1.0; k];
        let mut z = avec![1.0; m];

        bencher.bench(|| {
            y.fill(0.0);
            z.fill(0.0);
            tmatvec_avx512(n, k, &mut y, &t, &x);
            for (y, &c) in y.iter_mut().zip(&*c) {
                *y = *y * c;
            }
            matvec_avx512(m, k, &mut z, &s, &y);
        })
    }

    pub fn gemm_sctvec<T: From<f32> + Copy + 'static>(bencher: Bencher, _: PlotArg) {
        let one = T::from(1.0);
        let m = 4096;
        let n = 4096;
        let a = avec![one; m * n];
        let x = avec![one; n];
        let mut z = avec![one; m];

        bencher.bench(|| unsafe {
            gemm::gemm(
                m,
                1,
                n,
                z.as_mut_ptr(),
                m as _,
                1,
                true,
                a.as_ptr(),
                1,
                n as _,
                x.as_ptr(),
                n as _,
                1,
                one,
                one,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        });
    }
}

fn main() {
    let config = BenchConfig::from_args();
    let mut bench = Bench::new(config);
    bench.register_many(
        list![
            mat_tmat::bit_mat_tmat_avx2,
            mat_tmat::bit_mat_tmat_avx512,
            mat_tmat::gemm_mat_tmat::<f32>,
            mat_tmat::gemm_mat_tmat::<f64>,
        ],
        mat_tmat::params(),
    );
    bench.register_many(
        list![
            matvec::bit_matvec_avx2,
            matvec::bit_matvec_avx512,
            matvec::gemm_matvec::<f32>,
            matvec::gemm_matvec::<f64>,
        ],
        matvec::params(),
    );
    bench.register_many(
        list![
            tmatvec::bit_tmatvec_avx2,
            tmatvec::bit_tmatvec_avx512,
            tmatvec::gemm_tmatvec::<f32>,
            tmatvec::gemm_tmatvec::<f64>,
        ],
        tmatvec::params(),
    );
    bench.register_many(
        list![
            sctvec::bit_sctvec_avx2,
            sctvec::bit_sctvec_avx512,
            sctvec::gemm_sctvec::<f32>,
            sctvec::gemm_sctvec::<f64>,
        ],
        sctvec::params(),
    );
    bench.run().unwrap();
}
