use std::{iter, marker::PhantomData};

use ff_ext::{ff::PrimeField, ExtensionField};
use itertools::Itertools;

use crate::util::arithmetic::inner_product;

use super::{LassoSubtable, SubtableStategy};

pub struct FullLimbSubtable<F, E, const LIMB_SIZE: usize>(PhantomData<(F, E)>);

impl<F: PrimeField, E: ExtensionField<F>, const LIMB_SIZE: usize> LassoSubtable<F, E>
    for FullLimbSubtable<F, E, LIMB_SIZE>
{
    fn materialize(&self, M: usize) -> Vec<F> {
        assert_eq!(M, 1 << LIMB_SIZE);
        (0..M).map(|x| F::from(x as u64)).collect_vec()
    }

    fn evaluate_mle(&self, point: &[E]) -> E {
        let b = point.len();
        let mut result = E::ZERO;
        for i in 0..b {
            result += point[b] * F::from(1u64 << (i));
        }
        result
    }
}

pub struct ReminderSubtable<F, E, const NUM_BITS: usize, const LIMB_SIZE: usize>(
    PhantomData<(F, E)>,
);

impl<F: PrimeField, E: ExtensionField<F>, const NUM_BITS: usize, const LIMB_SIZE: usize>
    LassoSubtable<F, E> for ReminderSubtable<F, E, NUM_BITS, LIMB_SIZE>
{
    fn materialize(&self, M: usize) -> Vec<F> {
        assert_eq!(M, 1 << LIMB_SIZE);
        let remainder = NUM_BITS % LIMB_SIZE;
        let mut evals = vec![];
        (0..1 << remainder).for_each(|i| {
            evals.push(F::from(i));
        });
        evals
    }

    fn evaluate_mle(&self, point: &[E]) -> E {
        let b = point.len();
        let remainder = NUM_BITS % LIMB_SIZE;
        let mut result = E::ZERO;
        for i in 0..b {
            if i < remainder {
                result += point[b] * F::from(1u64 << (i));
            } else {
                result *= E::ONE - point[b];
            }
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct RangeStategy<const NUM_BITS: usize, const LIMB_BITS: usize>;

impl<const NUM_BITS: usize, const LIMB_BITS: usize> SubtableStategy
    for RangeStategy<NUM_BITS, LIMB_BITS>
{
    fn combine_lookups<F: PrimeField>(&self, operands: &[F]) -> F {
        let weight = F::from(1 << LIMB_BITS);
        inner_product(
            operands,
            iter::successors(Some(F::ONE), |power_of_weight| {
                Some(*power_of_weight * weight)
            })
            .take(operands.len())
            .collect_vec()
            .iter(),
        )
    }

    fn subtables<F: PrimeField, E: ExtensionField<F>>(&self) -> Vec<Box<dyn LassoSubtable<F, E>>> {
        let full = Box::new(FullLimbSubtable::<F, E, LIMB_BITS>(PhantomData));
        if NUM_BITS % LIMB_BITS == 0 {
            vec![full]
        } else {
            let rem = Box::new(ReminderSubtable::<F, E, NUM_BITS, LIMB_BITS>(PhantomData));
            vec![full, rem]
        }
    }
}
