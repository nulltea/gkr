use std::any::TypeId;

use enum_dispatch::enum_dispatch;
use ff_ext::{ff::PrimeField, ExtensionField};
use plonkish_backend::poly::multilinear::MultilinearPolynomialTerms;

use crate::{
    poly::{BoxMultilinearPoly, BoxMultilinearPolyOwned, MultilinearPolyTerms},
    util::expression::Expression,
};

mod range;

pub type SubtableId = TypeId;

#[enum_dispatch]
pub trait LassoSubtable<F: PrimeField, E: ExtensionField<F>>: 'static + Sync {
    /// Returns the TypeId of this subtable.
    /// The `Jolt` trait has associated enum types `InstructionSet` and `Subtables`.
    /// This function is used to resolve the many-to-many mapping between `InstructionSet` variants
    /// and `Subtables` variants,
    fn subtable_id(&self) -> SubtableId {
        TypeId::of::<Self>()
    }
    /// Fully materializes a subtable of size `M`, reprensented as a Vec of length `M`.
    fn materialize(&self, M: usize) -> Vec<F>;
    /// Evaluates the multilinear extension polynomial for this subtable at the given `point`,
    /// interpreted to be of size log_2(M), where M is the size of the subtable.
    fn evaluate_mle(&self, point: &[E]) -> E;
}

pub trait SubtableStategy {
    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups<F: PrimeField>(&self, operands: &[F]) -> F;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables<F: PrimeField, E: ExtensionField<F>>(
        &self,
        // C: usize,
        // M: usize,
    ) -> Vec<Box<dyn LassoSubtable<F, E>>>;
}

/// This is a trait that contains information about decomposable table to which
/// backend prover and verifier can ask
pub trait DecomposableTable<F: PrimeField, E: ExtensionField<F>>:
    std::fmt::Debug + Sync + DecomposableTableClone<F, E>
{
    fn num_memories(&self) -> usize;

    /// Returns multilinear extension polynomials of each subtable
    fn subtable_polys(&self) -> Vec<BoxMultilinearPoly<'static, F, E>>;

    fn subtable_polys_terms(&self) -> Vec<MultilinearPolyTerms<F>>;

    fn combine_lookup_expressions(
        &self,
        expressions: Vec<Expression<E, usize>>,
    ) -> Expression<E, usize>;

    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups(&self, operands: &[F]) -> F;

    /// Returns the size of bits for each chunk.
    /// Each chunk can have different bits.
    fn chunk_bits(&self) -> Vec<usize>;

    /// Returns the indices of each subtable lookups
    /// The length of `index_bits` is same as actual bit length of table index
    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>>;

    fn memory_to_subtable_index(&self, memory_index: usize) -> usize;

    fn memory_to_chunk_index(&self, memory_index: usize) -> usize;
}

pub trait DecomposableTableClone<F, E> {
    fn clone_box(&self) -> Box<dyn DecomposableTable<F, E>>;
}

impl<T, F: PrimeField, E: ExtensionField<F>> DecomposableTableClone<F, E> for T
where
    T: DecomposableTable<F, E> + Clone + 'static,
{
    fn clone_box(&self) -> Box<dyn DecomposableTable<F, E>> {
        Box::new(self.clone())
    }
}

impl<F, E> Clone for Box<dyn DecomposableTable<F, E>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
