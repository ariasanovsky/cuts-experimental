use cuts::sparse_cut::helpers::SparseSct;

use crate::safetensors::SerializeSct;

impl SerializeSct for SparseSct {
    fn serialize<P: AsRef<std::path::Path>>(&self, out_dir: P, big_mats: bool) -> candle_core::Result<()> {
        todo!()
    }
}
