use ff_ext::ff::PrimeField;

use crate::{
    poly::{
        evaluate, merge, merge_last, merge_last_in_place, BoxMultilinearPoly,
        BoxMultilinearPolyOwned, MultilinearPoly, MultilinearPolyExt, MultilinearPolyOwned,
    },
    util::arithmetic::{ExtensionField, Field},
};
use std::{fmt::Debug, marker::PhantomData, ops::Index};

#[derive(Clone, Debug)]
pub struct DensePolynomial<F, S: AsRef<[F]>> {
    evals: S,
    num_vars: usize,
    _marker: PhantomData<F>,
}

impl<F, S: AsRef<[F]>> DensePolynomial<F, S> {
    pub fn new(evals: S) -> Self {
        let num_vars = evals.as_ref().len().ilog2() as usize;
        assert_eq!(evals.as_ref().len(), 1 << num_vars);

        Self {
            evals,
            num_vars,
            _marker: PhantomData,
        }
    }
}

impl<F: PrimeField> DensePolynomial<F, Vec<F>> {
    pub fn from_usize<E: ExtensionField<F>>(
        z: &[usize],
    ) -> BoxMultilinearPoly<'static, F, E> {
        box_dense_poly(
            (0..z.len())
                .map(|i| F::from(z[i] as u64))
                .collect::<Vec<F>>(),
        )
    }
}

impl<F: Field> DensePolynomial<F, Vec<F>> {
    pub(crate) fn box_owned<'a>(self) -> BoxMultilinearPolyOwned<'a, F> {
        Box::new(self)
    }
}

impl<F, S: AsRef<[F]>> Index<usize> for DensePolynomial<F, S> {
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        &self.evals.as_ref()[index]
    }
}

impl<F: Field, E: ExtensionField<F>, S: Clone + Debug + AsRef<[F]> + Send + Sync>
    MultilinearPoly<F, E> for DensePolynomial<F, S>
{
    fn clone_box(&self) -> BoxMultilinearPoly<F, E> {
        self.clone().boxed()
    }

    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn fix_var(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E> {
        let evals = merge(self.evals.as_ref(), x_i);
        box_owned_dense_poly(evals)
    }

    fn fix_var_last(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E> {
        let evals = merge_last(self.evals.as_ref(), x_i);
        box_owned_dense_poly(evals)
    }

    fn evaluate(&self, x: &[E]) -> E {
        evaluate(self.evals.as_ref(), x)
    }

    fn as_dense(&self) -> Option<&[F]> {
        Some(self.evals.as_ref())
    }
}

impl<F: Field> MultilinearPolyOwned<F> for DensePolynomial<F, Vec<F>> {
    fn fix_var_in_place(&mut self, x_i: &F) {
        self.num_vars -= 1;
        self.evals = merge(&self.evals, x_i);
    }

    fn fix_var_last_in_place(&mut self, x_i: &F) {
        self.num_vars -= 1;
        merge_last_in_place(&mut self.evals, x_i);
    }
}

pub fn box_dense_poly<'a, F, E, S>(evals: S) -> BoxMultilinearPoly<'a, F, E>
where
    F: Field,
    E: ExtensionField<F>,
    S: 'a + Clone + Debug + AsRef<[F]> + Send + Sync,
{
    DensePolynomial::new(evals).boxed()
}

pub fn box_owned_dense_poly<'a, F>(evals: Vec<F>) -> BoxMultilinearPolyOwned<'a, F>
where
    F: Field,
{
    DensePolynomial::new(evals).box_owned()
}

pub fn repeated_dense_poly<'a, F, E, S>(evals: S, log2_reps: usize) -> BoxMultilinearPoly<'a, F, E>
where
    F: Field,
    E: ExtensionField<F>,
    S: 'a + Clone + Debug + AsRef<[F]> + Send + Sync,
{
    DensePolynomial::new(evals).repeated(log2_reps).boxed()
}
