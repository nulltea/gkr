use std::any::TypeId;

use enum_dispatch::enum_dispatch;
use ff_ext::{ff::PrimeField, ExtensionField};
use fixedbitset::FixedBitSet;
use plonkish_backend::poly::multilinear::MultilinearPolynomialTerms;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
    poly::{BoxMultilinearPoly, BoxMultilinearPolyOwned, MultilinearPolyTerms},
    util::expression::Expression,
};

pub mod range;

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

pub trait SubtableSet<F: PrimeField, E: ExtensionField<F>>:
    LassoSubtable<F, E> + IntoEnumIterator + EnumCount + From<SubtableId> + Into<usize> + Send + Sync
{
    fn enum_index(subtable: Box<dyn LassoSubtable<F, E>>) -> usize {
        Self::from(subtable.subtable_id()).into()
    }
}

pub trait CircuitLookups: LookupType + IntoEnumIterator + EnumCount + Send + Sync + std::fmt::Debug + Copy {
    fn enum_index(lookup_type: &Self) -> usize {
        // Discriminant: https://doc.rust-lang.org/reference/items/enumerations.html#pointer-casting
        let byte = unsafe { *(lookup_type as *const Self as *const u8) };
        byte as usize
    }
}

#[enum_dispatch]
pub trait LookupType: Clone + Send + Sync {
    /// The `g` function that computes T[r] = g(T_1[r_1], ..., T_k[r_1], T_{k+1}[r_2], ..., T_{\alpha}[r_c])
    fn combine_lookups<F: PrimeField>(&self, operands: &[F]) -> F;

    /// Returns a Vec of the unique subtable types used by this instruction. For some instructions,
    /// e.g. SLL, the list of subtables depends on the dimension `C`.
    fn subtables<F: PrimeField, E: ExtensionField<F>>(
        &self,
        // C: usize,
        // M: usize,
    ) -> Vec<(Box<dyn LassoSubtable<F, E>>, SubtableIndices)>;

    // fn to_indices<F: PrimeField>(&self, value: &F) -> Vec<usize>;

    fn output<F: PrimeField>(
        &self,
        index: &F
    ) -> F;

    fn chunk_bits(&self) -> Vec<usize>;

    /// Returns the indices of each subtable lookups
    /// The length of `index_bits` is same as actual bit length of table index
    fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>>;
}

#[derive(Clone)]
pub struct SubtableIndices {
    bitset: FixedBitSet,
}

impl SubtableIndices {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bitset: FixedBitSet::with_capacity(capacity),
        }
    }

    pub fn union_with(&mut self, other: &Self) {
        self.bitset.union_with(&other.bitset);
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bitset.ones()
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.bitset.count_ones(..)
    }

    pub fn contains(&self, index: usize) -> bool {
        self.bitset.contains(index)
    }
}

impl From<usize> for SubtableIndices {
    fn from(index: usize) -> Self {
        let mut bitset = FixedBitSet::new();
        bitset.grow_and_insert(index);
        Self { bitset }
    }
}

// impl From<Range<usize>> for SubtableIndices {
//     fn from(range: Range<usize>) -> Self {
//         let bitset = FixedBitSet::from_iter(range);
//         Self { bitset }
//     }
// }

/// This is a trait that contains information about decomposable table to which
/// backend prover and verifier can ask
pub trait DecomposableTable<F: PrimeField, E: ExtensionField<F>>:
    std::fmt::Debug + Sync + DecomposableTableClone<F, E>
{
    fn num_memories(&self) -> usize;

    fn subtables(
        &self,
        // C: usize,
        // M: usize,
    ) -> Vec<Box<dyn LassoSubtable<F, E>>>;

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
