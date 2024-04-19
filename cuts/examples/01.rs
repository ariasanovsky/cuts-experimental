use cuts::sct_old::CutDecomposition;
use faer::{stats::UnitaryMat, Mat};
use rand_distr::Distribution;

fn main() {
    // let normal = Normal::new(0.0, 1.0).unwrap();
    // let normal_mat = NormalMat {
    //     nrows: 64,
    //     ncols: 256,
    //     normal,
    // };
    let unitary_mats = UnitaryMat { dimension: 8 };
    let mut rng = rand::thread_rng();
    let mat: Mat<f64> = unitary_mats.sample(&mut rng);
    dbg!(mat.squared_norm_l2());
    // let mat: Mat<f64> = Mat::from_fn(3, 2, |i, j| (i + j) as f64);
    println!("{:?}", mat);

    // let mut s: Vec<[bool; 256]> = vec![];
    // let mut t: Vec<[bool; 256]> = vec![];
    // let mut tas: Vec<f64> = vec![];

    let mut decomp = CutDecomposition::empty_decomposition(mat);
    for i in 0..50 {
        let res = decomp.extend(&mut rng);
        dbg!(res.tas);
        let rem = decomp.remainder();
        dbg!(i, rem.squared_norm_l2());
        println!("{:?}", rem);
    }
}
