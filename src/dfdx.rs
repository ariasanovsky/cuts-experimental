use dfdx::{nn::Module, shapes::{Dtype, Rank0, Rank1, Rank2}, tensor::{HasErr, SplitTape, Tape, Tensor}, tensor_ops::{BroadcastTo, ChooseFrom, Device, PermuteTo, TryMatMul}};

pub struct SctLinear<const I: usize, const K: usize, const O: usize, E: Dtype, D: Device<E>> {
    pub s: Tensor<Rank2<O, K>, bool, D>,
    pub k: Tensor<Rank1<K>, E, D>,
    pub t: Tensor<Rank2<K, I>, bool, D>,
    pub b: Tensor<Rank1<O>, E, D>,
    zero: Tensor<Rank0, E, D>,
    one: Tensor<Rank0, E, D>,
}

impl<const I: usize, const K: usize, const O: usize, E: Dtype, D: Device<E>, T> Module<T> for SctLinear<I, K, O, E, D>
where
    T: SplitTape + HasErr<Err = D::Err> + TryMatMul<Tensor<Rank2<I, K>, E, D, T::Tape>>,
    T::Tape: Tape<E, D>,
    T::Output: TryMatMul<Tensor<Rank1<K>, E, D, T::Tape>, Err = D::Err, Output = T::Output>,
    T::Output: TryMatMul<Tensor<Rank2<K, O>, E, D, T::Tape>, Err = D::Err>,
{
    type Output = <T::Output as TryMatMul<Tensor<Rank2<K, O>, E, D, T::Tape>>>::Output;

    type Error = D::Err;

    fn try_forward(&self, x: T) -> Result<Self::Output, Self::Error> {
        // `x` is `I`-dim'l
        let one = self.one.clone().try_broadcast()?;
        let zero = self.zero.clone().try_broadcast()?;
        let t = self.t.clone().try_choose(one, zero)?;
        let t = t.retaped::<T::Tape>().try_permute()?;
        let o = x.try_matmul(t)?;
        // `o` is `K`-dim'l
        let k = self.k.clone();
        let o = o.try_matmul(k.retaped::<T::Tape>())?;
        // `o` is `K`-dim'l
        let one = self.one.clone().try_broadcast()?;
        let zero = self.zero.clone().try_broadcast()?;
        let s = self.s.clone().try_choose(one, zero)?;
        let s = s.retaped::<T::Tape>().try_permute()?;
        let o = o.try_matmul(s)?;
        // `o` is `O`-dim'l
        // let shape = o.shape();
        // o.try_add(self.b.retaped::<T>().try_broadcast_like(&shape)?)
        Ok(o)
    }
    
    fn forward(&self, input: T) -> Self::Output {
        self.try_forward(input).unwrap()
    }
}

// where <
//     T as TryMatMul<
//         Tensor<
//             (Const<I>, Const<K>),
//             E,
//             D,
//             <
//                 T as SplitTape
//             >::Tape
//         >
//     >
// >::Output:
//     TryMatMul<
//         Tensor<
//             (Const<K>, Const<O>), E, D, <T as SplitTape>::Tape>>