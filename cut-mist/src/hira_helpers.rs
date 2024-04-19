use cuts::{inplace_sct_signed::SignedCut, ls_sct::LdlCut, sparse_cut::SparseCut};
use faer::MatRef;
use tensorboard_writer::{proto::tensorboard::Summary, SummaryBuilder};

#[derive(Debug)]
pub struct HiraParameters {
    pub max_iters: usize,
    pub rank: usize,
    pub parallelism: faer::Parallelism,
}

impl std::fmt::Display for HiraParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "iters.{}.rank.{}", self.max_iters, self.rank)
    }
}

#[derive(Debug)]
pub struct HiraLogger {
    nrows: usize,
    ncols: usize,
    mat_frob_squared: f64,
    rem_frob_squared: f64,
    max_cut: f64,
    rank: usize,
}

impl HiraLogger {
    pub fn new(mat: MatRef<f64>) -> (Self, Summary) {
        let frob_squared = mat.squared_norm_l2();
        let logger = Self {
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            mat_frob_squared: frob_squared,
            rem_frob_squared: frob_squared,
            max_cut: f64::NEG_INFINITY,
            rank: 0,
        };
        let summary = SummaryBuilder::new()
            .scalar("remainder_norm/value", frob_squared.sqrt() as f32)
            .scalar("remainder_norm/value_over_init", 1.0)
            .scalar("remainder_norm/value_over_init_negative_log", 0.0)
            .build();
        (logger, summary)
    }

    pub fn log_cut(&mut self, cut: &SignedCut) -> Summary {
        let SignedCut {
            s_sizes: (s_pos, s_neg),
            t_sizes: (t_pos, t_neg),
            value,
        } = cut;
        let Self {
            nrows,
            ncols,
            mat_frob_squared,
            rem_frob_squared,
            max_cut,
            rank,
        } = self;
        // update values
        *rank += 1;
        *rem_frob_squared -= (*value * *value) / (*nrows * *ncols) as f64;
        *max_cut = max_cut.max(*value);
        let frob_value = rem_frob_squared.sqrt();
        let frob_value_over_init_squared = *rem_frob_squared / *mat_frob_squared;
        let frob_value_over_init = frob_value_over_init_squared.sqrt();
        let logged_frob = -frob_value_over_init_squared.ln(); // * LOG_SCALE;
        let logged_frob_over_rank = logged_frob / *rank as f64;
        let summary = SummaryBuilder::new()
            .scalar("remainder_norm/value", frob_value as _)
            .scalar("remainder_norm/value_over_init", frob_value_over_init as _)
            .scalar(
                "remainder_norm/value_over_init_negative_log",
                logged_frob as _,
            )
            .scalar(
                "remainder_norm/value_over_init_negative_log_over_rank",
                logged_frob_over_rank as _,
            );
        // summarize cut values
        let cut_value = *value;
        let cut_value_over_max = *value / *max_cut;
        let log_cut = -cut_value_over_max.ln();
        let log_cut_over_rank = log_cut / *rank as f64;
        let summary = summary
            .scalar("cut/value", cut_value as _)
            .scalar("cut/value_over_max", cut_value_over_max as _)
            .scalar("cut/value_over_max_negative_log", log_cut as _)
            .scalar(
                "cut/value_over_max_negative_log_over_rank",
                log_cut_over_rank as _,
            );
        // summarize sizes
        let s_size = *s_pos.max(s_neg);
        let s_size_over_m_over_two = s_size as f64 / *nrows as f64;
        let t_size = *t_pos.max(t_neg);
        let t_size_over_n_over_two = t_size as f64 / *ncols as f64;
        let summary = summary
            .scalar("size/s_value", s_size as _)
            .scalar("size/s_value_over_m", s_size_over_m_over_two as _)
            .scalar("size/t_value", t_size as _)
            .scalar("size/t_value_over_n", t_size_over_n_over_two as _);
        summary.build()
    }
    
    pub fn remainder_norm_squared(&self) -> f64 {
        self.rem_frob_squared
    }
}

#[derive(Debug)]
pub struct LdlHiraLogger {
    nrows: usize,
    ncols: usize,
    mat_frob_squared: f64,
    rem_frob_squared: f64,
    max_cut: f64,
    max_ldl: f64,
    rank: usize,
}

impl LdlHiraLogger {
    pub fn new(mat: MatRef<f64>) -> (Self, Summary) {
        let frob_squared = mat.squared_norm_l2();
        let logger = Self {
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            mat_frob_squared: frob_squared,
            rem_frob_squared: frob_squared,
            max_cut: f64::NEG_INFINITY,
            max_ldl: f64::NEG_INFINITY,
            rank: 0,
        };
        let summary = SummaryBuilder::new()
            .scalar("remainder_norm/value", frob_squared.sqrt() as f32)
            .scalar("remainder_norm/value_over_init", 1.0)
            .scalar("remainder_norm/value_over_init_negative_log", 0.0)
            .build();
        (logger, summary)
    }

    pub fn log_cut(&mut self, cut: &SignedCut, improved_cut: &LdlCut) -> Summary {
        let SignedCut {
            s_sizes: (s_pos, s_neg),
            t_sizes: (t_pos, t_neg),
            value,
        } = cut;
        let LdlCut { squared_frob_decrease } = *improved_cut;
        let Self {
            nrows,
            ncols,
            mat_frob_squared,
            rem_frob_squared,
            max_cut,
            max_ldl,
            rank,
        } = self;
        // update values
        *rank += 1;
        *rem_frob_squared -= squared_frob_decrease;
        *max_cut = max_cut.max(*value);
        *max_ldl = max_ldl.max(squared_frob_decrease.sqrt());
        let frob_value = rem_frob_squared.sqrt();
        let frob_value_over_init_squared = *rem_frob_squared / *mat_frob_squared;
        let frob_value_over_init = frob_value_over_init_squared.sqrt();
        let logged_frob = -frob_value_over_init_squared.ln(); // * LOG_SCALE;
        let logged_frob_over_rank = logged_frob / *rank as f64;
        let summary = SummaryBuilder::new()
            .scalar("remainder_norm/value", frob_value as _)
            .scalar("remainder_norm/value_over_init", frob_value_over_init as _)
            .scalar(
                "remainder_norm/value_over_init_negative_log",
                logged_frob as _,
            )
            .scalar(
                "remainder_norm/value_over_init_negative_log_over_rank",
                logged_frob_over_rank as _,
            );
        // summarize cut values
        let cut_value = *value;
        let cut_value_over_max = *value / *max_cut;
        let log_cut = -cut_value_over_max.ln();
        let log_cut_over_rank = log_cut / *rank as f64;
        let summary = summary
            .scalar("cut/value", cut_value as _)
            .scalar("cut/value_over_max", cut_value_over_max as _)
            .scalar("cut/value_over_max_negative_log", log_cut as _)
            .scalar(
                "cut/value_over_max_negative_log_over_rank",
                log_cut_over_rank as _,
            );
        // summarize ldl cut values
        let cut_ldl = squared_frob_decrease.sqrt();
        let cut_ldl_over_max = cut_ldl / *max_ldl;
        let log_ldl = -cut_ldl_over_max.ln();
        let log_ldl_over_rank = log_ldl / *rank as f64;
        let summary = summary
            .scalar("cut/ldl", cut_ldl as _)
            .scalar("cut/ldl_over_max", cut_ldl_over_max as _)
            .scalar("cut/ldl_over_max_negative_log", log_ldl as _)
            .scalar(
                "cut/ldl_over_max_negative_log_over_rank",
                log_ldl_over_rank as _,
            );
        // summarize sizes
        let s_size = *s_pos.max(s_neg);
        let s_size_over_m_over_two = s_size as f64 / *nrows as f64;
        let t_size = *t_pos.max(t_neg);
        let t_size_over_n_over_two = t_size as f64 / *ncols as f64;
        let summary = summary
            .scalar("size/s_value", s_size as _)
            .scalar("size/s_value_over_m", s_size_over_m_over_two as _)
            .scalar("size/t_value", t_size as _)
            .scalar("size/t_value_over_n", t_size_over_n_over_two as _);
        summary.build()
    }

    pub fn remainder_norm_squared(&self) -> f64 {
        self.rem_frob_squared
    }
}

#[derive(Debug)]
pub struct SparseHiraLogger {
    nrows: usize,
    ncols: usize,
    mat_frob_squared: f64,
    rem_frob_squared: f64,
    // max_cut: f64,
    num_bits: usize,
}

impl SparseHiraLogger {
    pub fn new(mat: MatRef<f64>) -> (Self, Summary) {
        let frob_squared = mat.squared_norm_l2();
        // todo!();
        let logger = Self {
            mat_frob_squared: frob_squared,
            rem_frob_squared: frob_squared,
            num_bits: 0,
            nrows: mat.nrows(),
            ncols: mat.ncols(),
        //     mat_frob_squared: frob_squared,
        //     rem_frob_squared: frob_squared,
        //     max_cut: f64::NEG_INFINITY,
        //     rank: 0,
        };
        let summary = SummaryBuilder::new()
            .scalar("remainder_norm/value", frob_squared.sqrt() as f32)
            .scalar("remainder_norm/value_over_init", 1.0)
            .scalar("remainder_norm/value_over_init_negative_log", 0.0)
            .build();
        (logger, summary)
    }

    pub fn log_cut(&mut self, cut: &SparseCut) -> Summary {
        let SparseCut {
            s_sizes: (s_pos, s_neg),
            t_sizes: (t_pos, t_neg),
            value,
        } = *cut;
        let Self {
            mat_frob_squared,
            rem_frob_squared,
            num_bits,
            nrows,
            ncols,
        //     mat_frob_squared,
        //     rem_frob_squared,
        //     max_cut,
        //     rank,
        } = self;
        // TODO! hard-coded for `u16`
        // update values
        let s_cardinality = s_pos + s_neg;
        let t_cardinality = t_pos + t_neg;
        *num_bits += (s_cardinality + t_cardinality) * 16;
        *rem_frob_squared -= (value * value) / (s_cardinality * t_cardinality) as f64;
        // println!("logger l2^2 = {rem_frob_squared}");
        // todo!();
        // *max_cut = max_cut.max(*value);
        let frob_value = rem_frob_squared.sqrt();
        let frob_value_over_init_squared = *rem_frob_squared / *mat_frob_squared;
        let frob_value_over_init = frob_value_over_init_squared.sqrt();
        // let logged_frob = -frob_value_over_init_squared.ln(); // * LOG_SCALE;
        // let logged_frob_over_rank = logged_frob / *rank as f64;
        let summary = SummaryBuilder::new()
            .scalar("remainder_norm/value", frob_value as _)
            .scalar("remainder_norm/value_over_init", frob_value_over_init as _);
        // // summarize cut values
        let cut_value = value;
        // let cut_value_over_max = *value / *max_cut;
        // let log_cut = -cut_value_over_max.ln();
        // let log_cut_over_rank = log_cut / *rank as f64;
        let summary = summary
            .scalar("cut/value", cut_value as _);
        //     .scalar("cut/value_over_max", cut_value_over_max as _)
        //     .scalar("cut/value_over_max_negative_log", log_cut as _)
        //     .scalar(
        //         "cut/value_over_max_negative_log_over_rank",
        //         log_cut_over_rank as _,
        //     );
        // // summarize sizes
        let s_size = s_cardinality;
        let s_size_over_m = s_size as f64 / *nrows as f64;
        let t_size = t_cardinality;
        let t_size_over_n = t_size as f64 / *ncols as f64;
        let summary = summary
            .scalar("size/s_value", s_size as _)
            .scalar("size/s_value_over_m", s_size_over_m as _)
            .scalar("size/t_value", t_size as _)
            .scalar("size/t_value_over_n", t_size_over_n as _);
        summary.build()
    }
    
    pub fn remainder_norm_squared(&self) -> f64 {
        todo!();
        // self.rem_frob_squared
    }
}
