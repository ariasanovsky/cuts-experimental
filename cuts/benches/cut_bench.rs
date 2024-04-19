use aligned_vec::{avec, CACHELINE_ALIGN};
use cuts::bit_magic::*;
use diol::prelude::*;
use dyn_stack::PodStack;

mod mat_tmat {
    use super::*;
    pub fn params() -> Vec<[usize; 3]> {
        vec![[4096; 3]]
    }

    pub fn bitmul_avx2(bencher: Bencher, [m, n, k]: [usize; 3]) {
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

    pub fn bitmul_avx512(bencher: Bencher, [m, n, k]: [usize; 3]) {
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

    pub fn gemm(bencher: Bencher, [m, n, k]: [usize; 3]) {
        let a = avec![1.0; m * k];
        let b = avec![1.0; k * n];
        let mut c = avec![1.0; m * n];
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
                1.0,
                1.0,
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

    pub fn bitmul_avx2(bencher: Bencher, [m, n]: [usize; 2]) {
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; n];
        let mut c = avec![1.0; m];

        bencher.bench(|| matvec_avx2(m, n, &mut c, &a, &b))
    }

    pub fn bitmul_avx512(bencher: Bencher, [m, n]: [usize; 2]) {
        if pulp::x86::V4::try_new().is_none() {
            return;
        }
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; n];
        let mut c = avec![1.0; m];

        bencher.bench(|| matvec_avx512(m, n, &mut c, &a, &b))
    }

    pub fn gemm(bencher: Bencher, [m, n]: [usize; 2]) {
        let a = avec![1.0; m * n];
        let b = avec![1.0; n];
        let mut c = avec![1.0; m];

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
                1.0,
                1.0,
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

    pub fn bitmul_avx2(bencher: Bencher, [m, n]: [usize; 2]) {
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; m];
        let mut c = avec![1.0; n];

        bencher.bench(|| tmatvec_avx2(m, n, &mut c, &a, &b))
    }

    pub fn bitmul_avx512(bencher: Bencher, [m, n]: [usize; 2]) {
        if pulp::x86::V4::try_new().is_none() {
            return;
        }
        let a = avec![1u8; m * n / 8];
        let b = avec![1.0; m];
        let mut c = avec![1.0; n];

        bencher.bench(|| tmatvec_avx512(m, n, &mut c, &a, &b))
    }

    pub fn gemm(bencher: Bencher, [m, n]: [usize; 2]) {
        let a = avec![1.0; m * n];
        let b = avec![1.0; n];
        let mut c = avec![1.0; m];

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
                1.0,
                1.0,
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
    pub fn params() -> Vec<[usize; 3]> {
        (1..17).map(|k| k * 1024).map(|k| [4096, 4096, k]).collect()
    }

    pub fn bitmul_avx2(bencher: Bencher, [m, n, k]: [usize; 3]) {
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

    pub fn bitmul_avx512(bencher: Bencher, [m, n, k]: [usize; 3]) {
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

    pub fn gemm(bencher: Bencher, [m, n, _]: [usize; 3]) {
        let a = avec![1.0; m * n];
        let x = avec![1.0; n];
        let mut z = avec![1.0; m];

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
                1.0,
                1.0,
                false,
                false,
                false,
                gemm::Parallelism::None,
            );
        });
    }
}

fn main() {
    let mut config = BenchConfig::from_args();
    let mut bench = Bench::new(config.clone());
    bench.register_many(
        list![
            mat_tmat::bitmul_avx2,
            mat_tmat::bitmul_avx512,
            mat_tmat::gemm
        ],
        mat_tmat::params(),
    );
    bench.run();

    let mut bench = Bench::new(config.clone());
    bench.register_many(
        list![matvec::bitmul_avx2, matvec::bitmul_avx512, matvec::gemm],
        matvec::params(),
    );
    bench.run();

    let mut bench = Bench::new(config.clone());
    bench.register_many(
        list![tmatvec::bitmul_avx2, tmatvec::bitmul_avx512, tmatvec::gemm],
        tmatvec::params(),
    );
    bench.run();

    let mut bench = Bench::new(config.clone());
    bench.register_many(
        list![sctvec::bitmul_avx2, sctvec::bitmul_avx512, sctvec::gemm],
        sctvec::params(),
    );
    bench.run();
}
