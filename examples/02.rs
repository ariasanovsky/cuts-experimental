use cuts::sct::SctApproximation;
use faer::{stats::UnitaryMat, Mat};
use rand_distr::Distribution;

fn main() {
    let unitary_mats = UnitaryMat {
        dimension: 8,
    };
    let mut rng = rand::thread_rng();
    let mat: Mat<f64> = unitary_mats.sample(&mut rng);
    dbg!(mat.squared_norm_l2());
    println!("{mat:?}");
    let mut skt = SctApproximation::new(mat.nrows(), mat.ncols());
    for i in 0..50 {
        skt.extend(&mut rng, mat.as_ref(), 1_000, true);
        // println!("s = {:?}", skt.last_s());
        // println!("t = {:?}", skt.last_t());
        println!("l2, cut{{{i}}} = {}, {:?}", skt.remainder(mat.as_ref()).squared_norm_l2(), skt.last_cut());
    }
}