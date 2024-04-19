use std::path::Path;

use candle_core::{Device, Tensor};
use cuts::sct_helper::Sct;
use faer::{reborrow::Reborrow, MatRef};
use half::bf16;
use safetensors::serialize_to_file;

fn entries(mat: MatRef<f64>) -> impl Iterator<Item = f64> + '_ {
    (0..mat.ncols()).flat_map(move |col| {
        let col = mat.col(col);
        (0..mat.nrows()).map(move |row| col.read(row))
    })
}

// TODO! no `candle`
// TODO! impl `View` correctly so it's copyfree
pub fn serialize_mats<P: AsRef<Path>>(
    outdir: P,
    approximat: MatRef<f64>,
    remainder: MatRef<f64>,
    big_mats: bool,
    small_mats: bool,
) -> candle_core::Result<()> {
    let mut tensors: Vec<(&'static str, Tensor)> = Vec::with_capacity(2);
    let device = Device::Cpu;
    let nrows = approximat.nrows();
    let ncols = approximat.ncols();
    if big_mats {
        let t = Tensor::from_iter(entries(approximat.rb()), &device)?
            .reshape(&[ncols, nrows])?
            .transpose(0, 1)?;
        tensors.push(("a_f64", t));
        // let filename = outdir.as_ref().join("a_f64.safetensors");
        // t.save_safetensors("a", filename)?;
    }
    if small_mats {
        let t = Tensor::from_iter(
            entries(approximat).map(bf16::from_f64),
            // (0..ncols)
            //     .flat_map(|col| approximat.col_as_slice(col))
            //     .map(|a| bf16::from_f64(*a)),
            &device,
        )?
        .reshape(&[ncols, nrows])?
        .transpose(0, 1)?;
        tensors.push(("a_bf16", t));
        // let filename = outdir.as_ref().join("a_bf16.safetensors");
        // t.save_safetensors("a", filename)?;
    }
    let nrows = remainder.nrows();
    let ncols = remainder.ncols();
    if big_mats {
        let t = Tensor::from_iter(
            // (0..ncols)
            //     .flat_map(|col| remainder.col_as_slice(col))
            //     .copied(),
            entries(remainder),
            &device,
        )?
        .reshape(&[ncols, nrows])?
        .transpose(0, 1)?;
        tensors.push(("r_f64", t));
        // let filename = outdir.as_ref().join("r_f64.safetensors");
        // t.save_safetensors("r", filename)?;
    }
    if small_mats {
        let t = Tensor::from_iter(
            // (0..ncols)
            //     .flat_map(|col| remainder.col_as_slice(col))
            //     .map(|r| bf16::from_f64(*r)),
            entries(remainder).map(bf16::from_f64),
            &device,
        )?
        .reshape(&[ncols, nrows])?
        .transpose(0, 1)?;
        tensors.push(("r_bf16", t));
        // let filename = outdir.as_ref().join("r_bf16.safetensors");
        // t.save_safetensors("r", filename)?;
    }

    // TODO! fork `safetensors` & write `serialize_to_file_with_capacity`
    // TODO! also `&Option<T>` -> `Option<T>` or `Option<&T>`
    serialize_to_file(
        tensors,
        &None,
        outdir.as_ref().join("mats.safetensors").as_path(),
    )?;
    Ok(())
}

pub trait SerializeSct {
    fn serialize<P: AsRef<Path>>(&self, out_dir: P, big_mats: bool) -> candle_core::Result<()>;
}

impl SerializeSct for Sct {
    fn serialize<P: AsRef<Path>>(&self, out_dir: P, big_mats: bool) -> candle_core::Result<()> {
        todo!()
    }
}

pub fn serialize_sct<P: AsRef<Path>>(
    outdir: P,
    sct_helper: &Sct,
    big_mats: bool,
) -> candle_core::Result<()> {
    let mut tensors: Vec<(&'static str, Tensor)> = Vec::with_capacity(6);
    let device = Device::Cpu;
    let dims = sct_helper.dimensions();
    // serialize `s` with `u8` (bitsets)
    let s_shape_u8 = [dims.rank(), dims.num_s_bytes()];
    let t = Tensor::from_iter(sct_helper.s().iter().copied(), &device)?.reshape(&s_shape_u8)?;
    tensors.push(("s_signs", t));
    // let filename = outdir.as_ref().join("s_u8.safetensors");
    // t.save_safetensors("s", filename)?;
    // serialize `s` with `f64` (signs)
    if big_mats {
        let s_shape_f64 = [dims.rank(), dims.nrows()];
        let t = Tensor::from_iter(
            bits_to_signs(sct_helper.s(), dims.nrows() * dims.rank()),
            &device,
        )?
        .reshape(&s_shape_f64)?;
        tensors.push(("s_f64", t));
        // let filename = outdir.as_ref().join("s_f64.safetensors");
        // t.save_safetensors("s", filename)?;
    }
    // serialize `t` with `u8` (bitsets)
    let t_shape_u8 = [dims.rank(), dims.num_t_bytes()];
    let t = Tensor::from_iter(sct_helper.t().iter().copied(), &device)?.reshape(&t_shape_u8)?;
    tensors.push(("t_signs", t));
    // let filename = outdir.as_ref().join("t_u8.safetensors");
    // t.save_safetensors("t", filename)?;
    // serialize `t` with `f64` (signs)
    if big_mats {
        let t_shape_f64 = [dims.rank(), dims.ncols()];
        let t = Tensor::from_iter(
            bits_to_signs(sct_helper.t(), dims.ncols() * dims.rank()),
            &device,
        )?
        .reshape(&t_shape_f64)?;
        tensors.push(("t_f64", t));
        // let filename = outdir.as_ref().join("t_f64.safetensors");
        // t.save_safetensors("t", filename)?;
    }
    // serialize `c` with `f64`
    let t = Tensor::from_iter(sct_helper.c().iter().copied(), &device)?;
    tensors.push(("c_f64", t));
    // let filename = outdir.as_ref().join("c_f64.safetensors");
    // t.save_safetensors("c", filename)?;
    // serialize `c` with `bf16`
    let t = Tensor::from_iter(sct_helper.c().iter().map(|c| bf16::from_f64(*c)), &device)?;
    tensors.push(("c_bf16", t));
    // let filename = outdir.as_ref().join("c_bf16.safetensors");
    // t.save_safetensors("c", filename)
    serialize_to_file(
        tensors,
        &None,
        outdir.as_ref().join("sct.safetensors").as_path(),
    )?;
    Ok(())
}

pub fn bits_to_signs(bits: &[u8], max: usize) -> impl Iterator<Item = f64> + '_ {
    bits.iter()
        .flat_map(|byte| (0..8).map(|i| if (1 << i) & *byte != 0 { -1.0 } else { 1.0 }))
        .take(max)
}
