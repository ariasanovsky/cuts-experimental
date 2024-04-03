use std::collections::HashMap;

use faer::{mat, Col, Mat, MatRef};
use rand::thread_rng;

use crate::{faer::FlatMat, sct_helper::Sct};

use super::{CutSetBuffers, CutSetLdl};

use equator::assert;

#[test]
fn ldl_sct_with_identity_matrix_of_size_8() {
    let mut remainder = identity_matrix(8);
    // let required_projections
    // let required_cuts = [
    //     [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
    //     [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,],
    // ].into_iter().map(|col| {
    //     Col::from_fn(8, |row| col[row])
    // }).enumerate().collect();
    // let required_temp_columns: &[&[f64]] = &[
    //     // &[1.0],
    //     // &[36.0, 1.0],
    // ];
    // let required_temp_columns = required_temp_columns.into_iter().map(|col| {
    //     Col::from_fn(col.len(), |row| col[row])
    // }).enumerate().collect::<HashMap<usize, _>>();
    // let required_projections: &[&[f64]] = &[
    //     // &[],
    //     // &[36.0],
    // ];
    // let required_projections = required_projections.into_iter().map(|col| {
    //     Col::from_fn(col.len(), |row| col[row])
    // }).enumerate().collect::<HashMap<usize, _>>();
    // let required_remainder_improvements = [
    //     1.0,
    // ].into_iter().enumerate().collect::<HashMap<usize, f64>>();
    // let new_columns: &[&[f64]] = &[
    //     &[1.0],
    //     &[-0.5625, 1.0],
    // ];
    // let required_new_columns = new_columns.into_iter().map(|col| {
    //     Col::from_fn(col.len(), |row| col[row])
    // }).enumerate().collect::<HashMap<usize, _>>();
    // let mut required_remainders = vec![];
    // let required_remainder = Mat::from_fn(8, 8, |row, col| {
    //     if row == col {
    //         0.875
    //     } else {
    //         -0.125
    //     }
    // });
    // required_remainders.push(required_remainder);
    // let required_remainders = required_remainders.into_iter().enumerate().collect::<HashMap<_, _>>();

    test_decompose_matrix(
        &mut remainder,
        100,
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
        Default::default(),
    );
    // return;
    // test_decompose_matrix(
    //     &mut remainder,
    //     9,
    //     required_cuts,
    //     required_temp_columns,
    //     required_projections,
    //     required_remainder_improvements,
    //     required_new_columns,
    //     required_remainders,
    // );
}


struct SignMatrices {
    s_signs: Vec<Col<f64>>,
    t_signs: Vec<Col<f64>>,
}

impl SignMatrices {
    fn new() -> Self {
        Self {
            s_signs: vec![],
            t_signs: vec![],
        }
    }

    fn push(&mut self, cutset: &CutSetBuffers) {
        self.s_signs.push(cutset.s_signs().clone());
        self.t_signs.push(cutset.t_signs().clone());
    }

    fn inner_products(&self) -> Mat<f64> {
        let Self { s_signs, t_signs } = self;
        let rank = self.s_signs.len();
        Mat::from_fn(rank, rank, |row, col| {
            let s_prod = s_signs[row].transpose() * &s_signs[col];
            let t_prod = t_signs[row].transpose() * &t_signs[col];
            s_prod * t_prod
        })
    }

    fn iter(&self) -> impl Iterator<Item = (&Col<f64>, &Col<f64>)> + '_ {
        let s_signs = self.s_signs.iter();
        let t_signs = self.t_signs.iter();
        s_signs.zip(t_signs)
    }
}

fn identity_matrix(size: usize) -> FlatMat {
    let mut remainder = FlatMat::zeros(size, size);
    for i in 0..size {
        remainder.write(i, i, 1.0)
    }
    remainder
}

fn test_decompose_matrix(
    remainder: &mut FlatMat,
    rank: usize,
    required_cuts: HashMap<usize, Col<f64>>,
    required_temp_columns: HashMap<usize, Col<f64>>,
    required_projections: HashMap<usize, Col<f64>>,
    required_remainder_improvements: HashMap<usize, f64>,
    required_new_columns: HashMap<usize, Col<f64>>,
    required_remainders: HashMap<usize, Mat<f64>>,
) {
    let nrows = remainder.as_ref().nrows();
    let ncols = remainder.as_ref().ncols();
    let mut ldl = CutSetLdl::new(nrows, ncols, rank);
    let mut sct = Sct::new(nrows, ncols, rank);
    let mut cutset = CutSetBuffers::new(nrows, ncols);
    let mut rng = thread_rng();
    let mut signs = SignMatrices::new();
    let s = (nrows * ncols) as f64;
    for r in 0..rank {
        println!("r = {r}");
        if remainder.as_ref().squared_norm_l2() <= 1e-10 {
            return
        }
        // println!("{:?}", remainder.as_ref());
        let mut cut = cutset.write_cut(remainder.as_ref(), &mut rng, 1_000);
        if let Some(required_cut) = required_cuts.get(&r) {
            while cutset.s_signs().ne(required_cut) {
                cut = cutset.write_cut(remainder.as_ref(), &mut rng, 1_000);
            }
        }
        let mut pos_s = vec![];
        let mut neg_s = vec![];
        for (i, &s) in cutset.s_signs().as_slice().iter().enumerate() {
            if s == 1.0 {
                pos_s.push(i)
            } else if s == -1.0 {
                neg_s.push(i)
            } else {
                unreachable!("s_{r}[{i}] != +/- 1.0")
            }
        }
        let mut pos_t = vec![];
        let mut neg_t = vec![];
        for (i, &t) in cutset.t_signs().as_slice().iter().enumerate() {
            if t == 1.0 {
                pos_t.push(i)
            } else if t == -1.0 {
                neg_t.push(i)
            } else {
                unreachable!("t_{r}[{i}] != +/- 1.0")
            }
        }
        // println!("s+, s- = {pos_s:?}, {neg_s:?}");
        // println!("t+, t- = {pos_t:?}, {neg_t:?}");
        signs.push(&cutset);
        // let sllt = signs.inner_products();
        // println!("{sllt:?}");
        sct.extend_with(cutset.s_signs(), cutset.t_signs(), cut.value);
        let mut projection_buffer = Col::zeros(r);
        // unsafe { ldl.write_projection(&mut projection_buffer) };
        let new_diagonal =  unsafe { ldl.write_temp_columns(&sct, &mut projection_buffer) };
        // dbg!(ldl.m_t.as_ref());
        // if let Some(temp) = required_temp_columns.get(&r) {
        //     assert!(temp.as_slice().eq(ldl.m_t.col_as_slice(r)))
        // }
        // if let Some(proj) = required_projections.get(&r) {
        //     assert_eq!(proj.as_ref(), projection_buffer.as_ref())
        // }
        // let projection_contribution = projection_buffer.squared_norm_l2();
        // let remainder_improvement = CutSetLdl::remainder_improvement(s.recip(), cut.value, projection_contribution);
        // if let Some(improvement) = required_remainder_improvements.get(&r) {
        //     assert!(*improvement == remainder_improvement)
        // }
        unsafe { ldl.fill_column(&mut projection_buffer) };
        // dbg!(ldl.m_t.as_ref());
        let new_col = ldl.m_t.col_as_slice(r);
        if let Some(col) = required_new_columns.get(&r) {
            assert_eq!(col.as_slice(), new_col)
        }
        unsafe { ldl.adjust_remainder(remainder, &sct, cut.value, new_diagonal) };
        // dbg!(remainder.as_ref().selfadjoint_eigenvalues(faer::Side::Upper));
        if let Some(rem) = required_remainders.get(&r) {
            assert_eq!(rem.as_ref(), remainder.as_ref())
        }
        for (i, (s, t)) in signs.iter().enumerate() {
            let prod = s.transpose() * remainder.as_ref() * t;
            assert!(prod <= 1e-10, "i = {i}")
        }
    }
}

fn test_is_identity_scaled(mat: MatRef<f64>, scale: f64) {
    assert!(mat.nrows() == mat.ncols());
    for row in 0..mat.nrows() {
        for col in 0..mat.ncols() {
            let value = if row == col {
                scale
            } else {
                0.0
            };
            assert!(mat[(row, col)] == value);
        }
    }
}

impl CutSetLdl {
    fn m(&self) -> MatRef<f64> {
        todo!();
        // let Self { s_recip, m_t, d_recip } = self;
        // m_t.as_ref()
    }
}