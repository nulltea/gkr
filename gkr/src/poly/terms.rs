use ff_ext::{ff::Field, ExtensionField};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

/// Multilinear polynomials are represented as expressions
#[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct MultilinearPolyTerms<F: Field> {
    num_vars: usize,
    expression: PolyExpr<F>,
}

impl<F: Field> MultilinearPolyTerms<F> {
    pub fn new(num_vars: usize, expression: PolyExpr<F>) -> Self {
        Self {
            num_vars,
            expression,
        }
    }
}

impl<F: Field> MultilinearPolyTerms<F> {
    pub fn evaluate<E: ExtensionField<F>>(&self, x: &[E]) -> E {
        assert_eq!(x.len(), self.num_vars);
        self.expression.evaluate(x)
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PolyExpr<F> {
    Const(F),
    Var(usize),
    Sum(Vec<PolyExpr<F>>),
    Prod(Vec<PolyExpr<F>>),
    Pow(Box<PolyExpr<F>>, u32),
}

impl<F: Field> PolyExpr<F> {
    fn evaluate<E: ExtensionField<F>>(&self, x: &[E]) -> E {
        match self {
            PolyExpr::Const(c) => E::ONE * c.clone(),
            PolyExpr::Var(i) => x[*i],
            PolyExpr::Sum(v) => v
                .par_iter()
                .map(|t| t.evaluate(x))
                .reduce(|| E::ZERO, |acc, f| acc + f),
            PolyExpr::Prod(v) => v
                .par_iter()
                .map(|t| t.evaluate(x))
                .reduce(|| E::ONE, |acc, f| acc * f),
            PolyExpr::Pow(inner, e) => inner.evaluate(x).pow([*e as u64]),
        }
    }
}

impl<F: Default> Default for PolyExpr<F> {
    fn default() -> Self {
        PolyExpr::Const(F::default())
    }
}
