use faer::Mat;
use rand::{prelude::SliceRandom, Rng};

#[cfg(feature = "dfdx")]
pub mod dfdx;
#[cfg(feature = "old_stuff_that_didnt_work")]
pub mod sct;
pub mod sct_helper;
#[cfg(feature = "old_stuff_that_didnt_work")]
pub mod inplace_sct;
pub mod inplace_sct_signed;
pub mod tensorboard;

pub struct CutDecomposition {
    mat: Mat<f64>,
    rows: Vec<Vec<bool>>,
    cols: Vec<Vec<bool>>,
    cuts: Vec<f64>,
}

pub struct CutRef<'a> {
    pub s: &'a [bool],
    pub t: &'a [bool],
    pub tas: f64,
}

impl CutDecomposition {
    pub fn empty_decomposition(mat: Mat<f64>) -> Self {
        Self {
            mat,
            rows: vec![],
            cols: vec![],
            cuts: vec![],
        }
    }

    pub fn extend(&mut self, rng: &mut impl Rng) -> CutRef {
        let nrows = self.mat.ncols();
        let col = Self::generate_col(nrows, rng);
        let row = vec![false; self.mat.nrows()];
        // dbg!(nrows, &col, &row);
        self.rows.push(row);
        self.cols.push(col);
        self.cuts.push(0.0);
        for _i in 0..(nrows * self.mat.ncols()) {
            let _ = unsafe { self.optimize_last_row() };
            // let improved = unsafe { self.optimize_last_row() };
            let improved = unsafe { self.optimize_last_col() };
            if !improved {
                break;
            }
        }
        unsafe {
            CutRef {
                s: self.rows.last().unwrap_unchecked(),
                t: self.cols.last().unwrap_unchecked(),
                tas: *self.cuts.last().unwrap_unchecked(),
            }
        }
    }

    fn generate_col(nrows: usize, rng: &mut impl Rng) -> Vec<bool> {
        if nrows == 1 {
            return vec![true];
        }
        let num_ones = rng.gen_range(1..nrows);
        let indices: Vec<usize> = (0..nrows).collect();
        // dbg!(num_ones, &indices);
        let indices = indices.choose_multiple(rng, num_ones);
        let mut col = vec![false; nrows];
        for &i in indices {
            col[i] = true;
        }
        col
    }

    unsafe fn optimize_last_row(&mut self) -> bool {
        let (cols, prev_cols) = unsafe { self.cols.split_last().unwrap_unchecked() };
        let (rows, prev_rows) = unsafe { self.rows.split_last_mut().unwrap_unchecked() };
        let (cut, prev_cuts) = unsafe { self.cuts.split_last_mut().unwrap_unchecked() };
        // dbg!(&col, &row, &cut, cols.len(), rows.len(), cuts.len());
        let mut prod = vec![0.0; self.mat.nrows()];
        /* calculating `prod`
        * let `A` be the associated matrix of `self.mat`
        * let `B = A - C`
        * here `C = S * diag(cut) * T^t`
        * let `t` be the `{0, 1}`-valued column vector `col`, also treated as a set of row indices
        * then `prod = B * t`
        */
        let mut pos = vec![false; self.mat.nrows()];
        let mut neg = vec![false; self.mat.nrows()];
        let mut pos_sum = 0.0;
        let mut neg_sum = 0.0;
        for i in 0..self.mat.nrows() {
            /* calculating `prod[i]`
            * by careful inspection, `(Bt)_{i} =
                sum_{j in t} A_{ij} -
                sum_{m: i\in s_{m}} cut_{m} * |t \cap t_{m}|`
            */
            for j in 0..self.mat.ncols() {
                if cols[j] {
                    prod[i] += self.mat[(i, j)];
                }
            }
            for m in 0..prev_rows.len() {
                if prev_rows[m][i] {
                    prod[i] -= prev_cuts[m] * cols.iter().zip(&prev_cols[m]).filter(|(&a, &b)| a && b).count() as f64;
                }
            }
            if prod[i] > 0.0 {
                pos[i] = true;
                pos_sum += prod[i];
            } else if prod[i] < 0.0 {
                neg[i] = true;
                neg_sum += prod[i];
            }
        }
        let (best_row, best_sum) = if pos_sum > -neg_sum {
            (pos, pos_sum)
        } else {
            (neg, neg_sum)
        };
        let old_s_cardinality = rows.iter().filter(|&&b| b).count();
        let old_t_cardinality = cols.iter().filter(|&&b| b).count();
        let old_cut = *cut * (old_s_cardinality * old_t_cardinality) as f64;
        if best_sum.abs() > old_cut.abs() {
            *rows = best_row;
            let s_cardinality = rows.iter().filter(|&&b| b).count();
            let t_cardinality = cols.iter().filter(|&&b| b).count();
            *cut = best_sum / (s_cardinality * t_cardinality) as f64;
            true
        } else {
            false
        }
    }

    unsafe fn optimize_last_col(&mut self) -> bool {
        let (cols, prev_cols) = unsafe { self.cols.split_last_mut().unwrap_unchecked() };
        let (rows, prev_rows) = unsafe { self.rows.split_last().unwrap_unchecked() };
        let (cut, prev_cuts) = unsafe { self.cuts.split_last_mut().unwrap_unchecked() };
        // dbg!(&col, &row, &cut, cols.len(), rows.len(), cuts.len());
        let mut prod = vec![0.0; self.mat.ncols()];
        /* calculating `prod`
        * let `A` be the associated matrix of `self.mat`
        * let `B = A - C`
        * here `C = S * diag(cut) * T^t`
        * let `s` be the `{0, 1}`-valued row vector `row`, also treated as a set of column indices
        * then `prod = s^t * B`
        */
        let mut pos = vec![false; self.mat.ncols()];
        let mut neg = vec![false; self.mat.ncols()];
        let mut pos_sum = 0.0;
        let mut neg_sum = 0.0;
        for j in 0..self.mat.ncols() {
            /* calculating `prod[j]`
            * by careful inspection, `(sB)_{j} =
                sum_{i in s} A_{ij} -
                sum_{m: j\in t_{m}} cut_{m} * |s \cap s_{m}|`
            */
            for i in 0..self.mat.nrows() {
                if rows[i] {
                    prod[j] += self.mat[(i, j)];
                }
            }
            for m in 0..prev_cols.len() {
                if prev_cols[m][j] {
                    prod[j] -= prev_cuts[m] * rows.iter().zip(&prev_rows[m]).filter(|(&a, &b)| a && b).count() as f64;
                }
            }
            if prod[j] > 0.0 {
                pos[j] = true;
                pos_sum += prod[j];
            } else if prod[j] < 0.0 {
                neg[j] = true;
                neg_sum += prod[j];
            }
        }
        let (best_col, best_sum) = if pos_sum > -neg_sum {
            (pos, pos_sum)
        } else {
            (neg, neg_sum)
        };
        let old_s_cardinality = rows.iter().filter(|&&b| b).count();
        let old_t_cardinality = cols.iter().filter(|&&b| b).count();
        let old_cut = *cut * (old_s_cardinality * old_t_cardinality) as f64;
        if best_sum.abs() > old_cut.abs() {
            *cols = best_col;
            let s_cardinality = rows.iter().filter(|&&b| b).count();
            let t_cardinality = cols.iter().filter(|&&b| b).count();
            *cut = best_sum / (s_cardinality * t_cardinality) as f64;
            true
        } else {
            false
        }
    }

    pub fn remainder(&self) -> Mat<f64> {
        let mut remainder = self.mat.clone();
        for i in 0..self.mat.nrows() {
            for j in 0..self.mat.ncols() {
                for m in 0..self.rows.len() {
                    if self.rows[m][i] && self.cols[m][j] {
                        remainder[(i, j)] -= self.cuts[m];
                    }
                }
            }
        }
        remainder
    }
}