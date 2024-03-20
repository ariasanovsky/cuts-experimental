use faer::{Mat, MatRef};
use rand::{prelude::SliceRandom, Rng};

pub struct SctApproximation<E> {
    // `k`
    rank: usize,
    // `m`
    nrows: usize,
    // `n`
    ncols: usize,
    // `m x k`
    s: Vec<bool>,
    // `k x k` diagonal matrix
    c: Vec<Cut<E>>,
    // `n x k` transposed
    t: Vec<bool>,
}

#[derive(Clone, Debug)]
pub struct Cut<E> {
    pub s_size: usize,
    pub t_size: usize,
    pub cut: E,
}

impl SctApproximation<f64> {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            rank: 0,
            nrows,
            ncols,
            s: vec![],
            c: vec![],
            t: vec![],
        }
    }

    pub fn last_s(&self) -> &[bool] {
        &self.s[(self.rank - 1) * self.nrows..self.rank * self.nrows]
    }

    pub fn last_t(&self) -> &[bool] {
        &self.t[(self.rank - 1) * self.ncols..self.rank * self.ncols]
    }

    pub fn last_cut(&self) -> &Cut<f64> {
        &self.c[self.rank - 1]
    }

    pub fn extend(
        &mut self,
        rng: &mut impl Rng,
        mat: MatRef<f64>,
        trials: usize,
        at_least_half: bool,
        halved_cut: bool,
    ) {
        assert_eq!(self.nrows, mat.nrows());
        assert_eq!(self.ncols, mat.ncols());

        let mut s_opt = vec![false; self.nrows];
        let mut t_opt = vec![false; self.ncols];
        let mut c_opt: Cut<f64> = Cut {
            s_size: 0,
            t_size: 0,
            cut: 0.0,
        };
        for trial in 0..trials {
            println!("trial {trial}");
            let (mut t, t_size) = Self::generate_col(self.nrows, rng, at_least_half);
            let mut s = vec![false; self.ncols];
            let mut c: Cut<f64> = Cut {
                s_size: 0,
                t_size,
                cut: 0.0,
            };
            for improvements in 0..(3 * self.nrows * self.ncols) {
                let improved =
                    self.optimize_cut(&mut s, &mut c, &mut t, mat, at_least_half, halved_cut);
                if !improved {
                    break;
                }
                if c.cut.abs() > c_opt.cut.abs() {
                    s_opt.copy_from_slice(&s);
                    t_opt.copy_from_slice(&t);
                    c_opt.clone_from(&c);
                    println!("\tat {trial}, {improvements}\tc = {c:?}");
                }
            }
        }
        let Self {
            rank,
            nrows: _,
            ncols: _,
            s,
            c,
            t,
        } = self;

        if at_least_half {
            assert!(
                2 * c_opt.s_size >= self.nrows && 2 * c_opt.t_size >= self.ncols,
                "{}, {}",
                c_opt.s_size,
                c_opt.t_size,
            );
        }

        *rank += 1;

        assert_ne!(c_opt.s_size, 0);
        assert_ne!(c_opt.t_size, 0);

        s.extend(s_opt.into_iter());
        c.push(c_opt);
        t.extend(t_opt.into_iter());
    }

    fn generate_col(nrows: usize, rng: &mut impl Rng, _at_least_half: bool) -> (Vec<bool>, usize) {
        assert!(nrows > 0);
        if nrows == 1 {
            return (vec![true], 1);
        }
        let num_ones = rng.gen_range(1..nrows);
        let indices: Vec<usize> = (0..nrows).collect();
        let mut col = vec![false; nrows];
        for i in indices.choose_multiple(rng, num_ones) {
            col[*i] = true;
        }
        (col, num_ones)
    }

    fn optimize_cut(
        &self,
        s_new: &mut [bool],
        c_new: &mut Cut<f64>,
        t_new: &mut [bool],
        mat: MatRef<f64>,
        at_least_half: bool,
        halved_cut: bool,
    ) -> bool {
        let improved_s = self.optimize_s(s_new, c_new, t_new, mat, at_least_half, halved_cut);
        // return improved_s;
        if !improved_s {
            return false;
        }
        self.optimize_t(s_new, c_new, t_new, mat, at_least_half, halved_cut)
    }

    fn optimize_s(
        &self,
        s_new: &mut [bool],
        c_new: &mut Cut<f64>,
        t_new: &[bool],
        mat: MatRef<f64>,
        _at_least_half: bool,
        halved_cut: bool,
    ) -> bool {
        // let num_s_bools = self.s.len();
        // let num_t_bools = self.t.len();
        // assert_eq!(num_s_bools, self.rank * self.nrows);
        // assert_eq!(num_t_bools, self.rank * self.ncols);
        // assert_eq!(self.c.len(), self.rank);

        let mut prod = vec![0.0; self.nrows];
        let mut pos = vec![false; self.nrows];
        let mut neg = vec![false; self.nrows];
        let mut pos_cut = 0.0;
        let mut neg_cut = 0.0;
        let mut num_pos = 0;
        let mut num_neg = 0;
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                if t_new[j] {
                    prod[i] += mat[(i, j)];
                }
            }
            for (m, c_m) in self.c.iter().enumerate() {
                let s_m = &self.s[m * self.nrows..(m + 1) * self.nrows];
                if s_m[i] {
                    let t_m = &self.t[m * self.ncols..(m + 1) * self.ncols];
                    let Cut {
                        s_size,
                        t_size,
                        cut,
                    } = c_m;
                    let intersection = t_m.iter().zip(t_new).filter(|(a, b)| **a && **b).count();
                    let rescale = if halved_cut {
                        (intersection as f64) / (*s_size * *t_size * 2) as f64
                    } else {
                        (intersection as f64) / (*s_size * *t_size) as f64
                    };
                    prod[i] -= rescale * cut;
                }
            }
            if prod[i] > 0.0 {
                pos[i] = true;
                pos_cut += prod[i];
                num_pos += 1;
            } else if prod[i] < 0.0 {
                neg[i] = true;
                neg_cut += prod[i];
                num_neg += 1;
            }
        }
        let (best_s, best_cut, best_size): (_, f64, _) = if pos_cut > -neg_cut {
            (pos, pos_cut, num_pos)
        } else {
            (neg, neg_cut, num_neg)
        };
        let Cut {
            s_size,
            t_size: _,
            cut,
        } = c_new;
        if best_cut.abs() > cut.abs() {
            s_new.copy_from_slice(&best_s);
            *s_size = best_size;
            *cut = best_cut;
            true
        } else {
            false
        }
    }

    fn optimize_t(
        &self,
        s_new: &[bool],
        c_new: &mut Cut<f64>,
        t_new: &mut [bool],
        mat: MatRef<f64>,
        _at_least_half: bool,
        halved_cut: bool,
    ) -> bool {
        let mut prod = vec![0.0; self.ncols];
        let mut pos = vec![false; self.ncols];
        let mut neg = vec![false; self.ncols];
        let mut pos_cut = 0.0;
        let mut neg_cut = 0.0;
        let mut num_pos = 0;
        let mut num_neg = 0;
        for j in 0..self.ncols {
            for i in 0..self.nrows {
                if s_new[i] {
                    prod[j] += mat[(i, j)];
                }
            }
            for (m, c_m) in self.c.iter().enumerate() {
                let t_m = &self.t[m * self.ncols..(m + 1) * self.ncols];
                if t_m[j] {
                    let s_m = &self.s[m * self.nrows..(m + 1) * self.nrows];
                    let Cut {
                        s_size,
                        t_size,
                        cut,
                    } = c_m;
                    let intersection = s_m.iter().zip(s_new).filter(|(a, b)| **a && **b).count();
                    let rescale = if halved_cut {
                        (intersection as f64) / (*s_size * *t_size * 2) as f64
                    } else {
                        (intersection as f64) / (*s_size * *t_size) as f64
                    };
                    prod[j] -= rescale * cut;
                }
            }
            if prod[j] > 0.0 {
                pos[j] = true;
                pos_cut += prod[j];
                num_pos += 1;
            } else if prod[j] < 0.0 {
                neg[j] = true;
                neg_cut += prod[j];
                num_neg += 1;
            }
        }
        let (best_t, best_cut, best_size): (_, f64, _) = if pos_cut > -neg_cut {
            (pos, pos_cut, num_pos)
        } else {
            (neg, neg_cut, num_neg)
        };
        let Cut {
            s_size: _,
            t_size,
            cut,
        } = c_new;
        if best_cut.abs() > cut.abs() {
            t_new.copy_from_slice(&best_t);
            *t_size = best_size;
            *cut = best_cut;
            true
        } else {
            false
        }
    }

    pub fn remainder(&self, mat: MatRef<f64>, halved_cut: bool) -> Mat<f64> {
        let mut remainder = mat.to_owned();
        for m in 0..self.rank {
            // let k = self.c[m].cut / (self.c[m].s_size * self.c[m].t_size) as f64;
            let k = if halved_cut {
                self.c[m].cut / (self.c[m].s_size * self.c[m].t_size * 2) as f64
            } else {
                self.c[m].cut / (self.c[m].s_size * self.c[m].t_size) as f64
            };
            for i in 0..self.nrows {
                for j in 0..self.ncols {
                    if self.s[m * self.nrows + i] && self.t[m * self.ncols + j] {
                        remainder[(i, j)] -= k;
                    }
                }
            }
        }
        remainder
    }

    pub fn approximation(&self, halved_cut: bool) -> Mat<f64> {
        let mut approx = Mat::zeros(self.nrows, self.ncols);
        for m in 0..self.rank {
            // let k = self.c[m].cut / (self.c[m].s_size * self.c[m].t_size) as f64;
            let k = if halved_cut {
                self.c[m].cut / (self.c[m].s_size * self.c[m].t_size * 2) as f64
            } else {
                self.c[m].cut / (self.c[m].s_size * self.c[m].t_size) as f64
            };
            for i in 0..self.nrows {
                for j in 0..self.ncols {
                    if self.s[m * self.nrows + i] && self.t[m * self.ncols + j] {
                        approx[(i, j)] += k;
                    }
                }
            }
        }
        approx
    }
}
