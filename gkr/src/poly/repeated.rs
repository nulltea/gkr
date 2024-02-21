use crate::{
    poly::{
        BoxMultilinearPoly, BoxMultilinearPolyOwned, DenseMultilinearPoly, MultilinearPoly,
        MultilinearPolyOwned,
    },
    util::arithmetic::{ExtensionField, Field},
};
use std::{fmt::Debug, marker::PhantomData, ops::Index};

#[derive(Clone, Debug)]
pub struct RepeatedMultilinearPoly<T, F, E = F> {
    inner: T,
    log2_reps: usize,
    _marker: PhantomData<(F, E)>,
}

impl<F, E, T: MultilinearPoly<F, E>> RepeatedMultilinearPoly<T, F, E> {
    pub fn new(inner: T, log2_reps: usize) -> Self {
        Self {
            inner,
            log2_reps,
            _marker: PhantomData,
        }
    }
}

impl<F, E, T: MultilinearPoly<F, E>> RepeatedMultilinearPoly<T, F, E> {
    pub(crate) fn box_owned<'a>(self) -> BoxMultilinearPolyOwned<'a, F>
    where
        F: 'a,
        E: 'a,
        T: 'a,
        Self: MultilinearPolyOwned<F>,
    {
        Box::new(self)
    }
}

impl<F, E, T> Index<usize> for RepeatedMultilinearPoly<T, F, E>
where
    F: Field,
    E: ExtensionField<F>,
    T: MultilinearPoly<F, E>,
{
    type Output = F;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len());

        &self.inner[index % self.inner.len()]
    }
}

impl<F, E, T> MultilinearPoly<F, E> for RepeatedMultilinearPoly<T, F, E>
where
    F: Field,
    E: ExtensionField<F>,
    T: MultilinearPoly<F, E>,
{
    fn clone_box(&self) -> BoxMultilinearPoly<F, E> {
        RepeatedMultilinearPoly::new(self.inner.clone_box(), self.log2_reps).boxed()
    }

    fn num_vars(&self) -> usize {
        self.inner.num_vars() + self.log2_reps
    }

    fn fix_var(&self, x_i: &E) -> BoxMultilinearPolyOwned<'static, E> {
        if self.inner.num_vars() > 0 {
            RepeatedMultilinearPoly::new(self.inner.fix_var(x_i), self.log2_reps).box_owned()
        } else {
            assert!(self.log2_reps > 0);

            let inner = DenseMultilinearPoly::new(vec![E::from(self.inner[0])]).box_owned();
            RepeatedMultilinearPoly::new(inner, self.log2_reps - 1).box_owned()
        }
    }

    fn evaluate(&self, x: &[E]) -> E {
        self.inner.evaluate(&x[..self.inner.num_vars()])
    }

    fn as_dense(&self) -> Option<&[F]> {
        None
    }
}

impl<F: Field> MultilinearPolyOwned<F>
    for RepeatedMultilinearPoly<BoxMultilinearPolyOwned<'static, F>, F>
{
    fn fix_var_in_place(&mut self, x_i: &F) {
        if self.inner.num_vars() > 0 {
            self.inner.fix_var_in_place(x_i);
        } else {
            assert!(self.log2_reps > 0);

            self.log2_reps -= 1;
        }
    }
}
