use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{box_dense_poly, merge, BoxMultilinearPoly, MultilinearPoly},
    sum_check::{err_unmatched_evaluation, verify_sum_check, SumCheckFunction},
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{div_ceil, ExtensionField, Field},
        chain, izip, Itertools,
    },
    Error,
};
use ff_ext::ff::PrimeField;
use memory_checking::{Chunk, Memory, MemoryCheckingProver};
use plonkish_backend::{
    backend::lookup,
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::{MultilinearPolynomial, MultilinearPolynomialTerms},
    util::parallel::{num_threads, parallelize_iter},
};
// use prover::LassoProver;
use rayon::prelude::*;
use std::{
    array::{self, from_fn},
    cmp::Ordering::*,
    collections::HashMap,
    iter,
    marker::PhantomData,
    ops::Index,
    slice::Chunks,
};
use strum::{EnumCount, IntoEnumIterator};
use surge::Surge;
use table::DecomposableTable;
// use verifier::LassoVerifier;

pub mod memory_checking;
pub mod surge;
// pub mod test;
pub mod table;
// pub mod verifier;

#[derive(Clone, Debug)]
pub struct GeneralizedLasso<F: Field, Pcs: PolynomialCommitmentScheme<F>>(
    PhantomData<F>,
    PhantomData<Pcs>,
);

#[derive(Clone, Debug)]
pub struct LassoNode<F, E> {
    num_vars: usize,
    final_poly_log2_size: usize,
    table: Box<dyn DecomposableTable<F, E>>,
}

impl<F: PrimeField, E: ExtensionField<F>> Node<F, E> for LassoNode<F, E> {
    fn is_input(&self) -> bool {
        false
    }

    fn log2_input_size(&self) -> usize {
        self.num_vars.max(self.final_poly_log2_size)
    }

    fn log2_output_size(&self) -> usize {
        0
    }

    fn evaluate(&self, _: Vec<&BoxMultilinearPoly<F, E>>) -> BoxMultilinearPoly<'static, F, E> {
        box_dense_poly([F::ZERO])
    }

    fn prove_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        inputs: Vec<&BoxMultilinearPoly<F, E>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let table = self.table.clone();

        // get subtable_polys
        let subtable_polys = table.subtable_polys();
        let subtable_polys = subtable_polys.iter().collect_vec();
        let subtable_polys = subtable_polys.as_slice();

        // println!("combined claim {} {:?}", claim.points.len(), claim.value);

        let num_chunks = table.chunk_bits().len();

        // let (lookup_output_poly, inputs) = inputs.split_first().unwrap();
        // let (e_polys, inputs) = inputs.split_at(table.num_memories());
        // let (dims, inputs) = inputs.split_at(num_chunks);
        // let (read_ts_polys, inputs) = inputs.split_at(num_chunks);
        // let (final_cts_polys, inputs) = inputs.split_at(num_chunks);

        // let [input, output] = inputs.try_into().unwrap();
        let lookup_output_poly = inputs[0];
        let lookup_index_poly = lookup_output_poly;

        let (polys, _) = Self::commit(
            &table,
            subtable_polys,
            lookup_output_poly,
            lookup_index_poly,
        )?;

        let [e_polys, dims, read_ts_polys, final_cts_polys] = polys;

        let num_vars = lookup_output_poly.num_vars();
        assert!(num_vars == self.num_vars);

        assert_eq!(final_cts_polys[0].num_vars(), self.final_poly_log2_size);

        // should this be passed from CombinedEvalClaim?
        let r = transcript.squeeze_challenges(num_vars);

        // let mut lookup_opening_points = vec![];
        // let mut lookup_opening_evals = vec![];

        let res = Surge::<F, E>::prove_sum_check(
            &table,
            lookup_output_poly,
            &e_polys,
            &r,
            num_vars,
            // points_offset,
            // lookup_opening_points,
            // lookup_opening_evals,
            transcript,
        )?;

        let lookup_output_eval_claim = res.into_iter().take(1).collect_vec();

        println!("Proved Surge sum check");

        let [gamma, tau] = transcript.squeeze_challenges(2).try_into().unwrap();

        // memory_checking
        let mut memory_checking = Self::prepare_memory_checking(
            &table,
            subtable_polys,
            &dims,
            &read_ts_polys,
            &final_cts_polys,
            &e_polys,
            &gamma,
            &tau,
        );

        memory_checking
            .iter_mut()
            .map(|memory_checking| memory_checking.prove(transcript))
            .collect::<Result<Vec<()>, Error>>()?;

        Ok(lookup_output_eval_claim)
    }

    fn verify_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let table = self.table.clone();
        let num_vars = self.num_vars;
        let r = transcript.squeeze_challenges(num_vars);

        let g = Surge::<F, E>::sum_check_function(&table, num_vars);
        let claimed_sum = transcript.read_felt_ext()?;

        let (sub_claim, r_x_prime) = verify_sum_check(&g, claimed_sum, transcript)?;

        // let e_polys_eval = transcript.read_felt_exts(table.num_memories())?;

        // // // Round n+1
        let [gamma, tau] = transcript.squeeze_challenges(2).try_into().unwrap();

        // // memory checking
        let memory_checking = memory_checking::verifier::prepare_memory_checking(&table);

        memory_checking
            .iter()
            .map(|memory_checking| memory_checking.verify(num_vars, &gamma, &tau, transcript))
            .collect::<Result<Vec<()>, Error>>()?;

        Ok(chain![iter::once(vec![EvalClaim::new(r.to_vec(), claimed_sum)]),].collect_vec())
    }
}

impl<F: PrimeField, E: ExtensionField<F>> LassoNode<F, E> {
    pub fn new(
        table: Box<dyn DecomposableTable<F, E>>,
        num_vars: usize,
        final_poly_log2_size: usize,
    ) -> Self {
        Self {
            num_vars,
            table,
            final_poly_log2_size,
        }
    }

    pub fn commit<'a>(
        // pp: &Pcs::ProverParam,
        // lookup_polys_offset: usize,
        table: &Box<dyn DecomposableTable<F, E>>,
        subtable_polys: &[&BoxMultilinearPoly<'static, F, E>],
        lookup_output_poly: &BoxMultilinearPoly<'a, F, E>,
        lookup_index_poly: &BoxMultilinearPoly<'a, F, E>,
        // transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<
        (
            [Vec<BoxMultilinearPoly<'a, F, E>>; 4],
            (), // Vec<Vec<Pcs::Commitment>>,
        ),
        Error,
    > {
        let num_chunks = table.chunk_bits().len();

        // commit to lookup_output_poly
        // let lookup_output_comm = Pcs::commit_and_write(&pp, &lookup_output_poly, transcript)?;

        // get surge and dims
        let mut surge = Surge::<F, E>::new();

        // commit to dims
        let dims = surge.commit(&table, &lookup_index_poly);
        // let dim_comms = Pcs::batch_commit_and_write(pp, &dims, transcript)?;

        // get e_polys & read_ts_polys & final_cts_polys
        let e_polys = {
            let indices = surge.indices();
            Self::e_polys(table, subtable_polys, &indices)
        };
        let (read_ts_polys, final_cts_polys) = surge.counter_polys(&table);

        // commit to read_ts_polys & final_cts_polys & e_polys
        // let read_ts_comms = Pcs::batch_commit_and_write(&pp, &read_ts_polys, transcript)?;
        // let final_cts_comms = Pcs::batch_commit_and_write(&pp, &final_cts_polys, transcript)?;
        // let e_comms = Pcs::batch_commit_and_write(&pp, e_polys.as_slice(), transcript)?;

        // let dims = dims
        //     .into_iter()
        //     .enumerate()
        //     .map(|(chunk_index, dim)| Poly {
        //         offset: lookup_polys_offset + 1 + chunk_index,
        //         poly: dim,
        //     })
        //     .collect_vec();

        // let read_ts_polys = read_ts_polys
        //     .into_iter()
        //     .enumerate()
        //     .map(|(chunk_index, read_ts_poly)| Poly {
        //         offset: lookup_polys_offset + 1 + num_chunks + chunk_index,
        //         poly: read_ts_poly,
        //     })
        //     .collect_vec();

        // let final_cts_polys = final_cts_polys
        //     .into_iter()
        //     .enumerate()
        //     .map(|(chunk_index, final_cts_poly)| Poly {
        //         offset: lookup_polys_offset + 1 + 2 * num_chunks + chunk_index,
        //         poly: final_cts_poly,
        //     })
        //     .collect_vec();

        // let e_polys = e_polys
        //     .into_iter()
        //     .enumerate()
        //     .map(|(memory_index, e_poly)| Poly {
        //         offset: lookup_polys_offset + 1 + 3 * num_chunks + memory_index,
        //         poly: e_poly,
        //     })
        //     .collect_vec();

        Ok((
            [e_polys, dims, read_ts_polys, final_cts_polys],
            (), // vec![
                //     // vec![lookup_output_comm],
                //     // dim_comms,
                //     // read_ts_comms,
                //     // final_cts_comms,
                //     // e_comms,
                // ],
        ))
    }

    fn e_polys<'a>(
        table: &Box<dyn DecomposableTable<F, E>>,
        subtable_polys: &[&BoxMultilinearPoly<F, E>],
        indices: &Vec<&[usize]>,
    ) -> Vec<BoxMultilinearPoly<'a, F, E>> {
        let num_chunks = table.chunk_bits().len();
        let num_memories = table.num_memories();
        assert_eq!(indices.len(), num_chunks);
        let num_reads = indices[0].len();
        (0..num_memories)
            .map(|i| {
                let mut e_poly = Vec::with_capacity(num_reads);
                let subtable_poly = subtable_polys[table.memory_to_subtable_index(i)];
                let index = indices[table.memory_to_chunk_index(i)];
                (0..num_reads).for_each(|j| {
                    e_poly.push(subtable_poly[index[j]].clone());
                });
                box_dense_poly(e_poly)
            })
            .collect_vec()
    }

    fn chunks<'a>(
        table: &Box<dyn DecomposableTable<F, E>>,
        subtable_polys: &'a [&BoxMultilinearPoly<F, E>],
        dims: &'a [BoxMultilinearPoly<F, E>],
        read_ts_polys: &'a [BoxMultilinearPoly<F, E>],
        final_cts_polys: &'a [BoxMultilinearPoly<F, E>],
        e_polys: &'a [BoxMultilinearPoly<F, E>],
    ) -> Vec<Chunk<'a, F, E>> {
        // key: chunk index, value: chunk
        let mut chunk_map: HashMap<usize, Chunk<F, E>> = HashMap::new();

        let num_memories = table.num_memories();
        let memories = (0..num_memories).map(|memory_index| {
            let subtable_poly = subtable_polys[table.memory_to_subtable_index(memory_index)];
            Memory::<F, E>::new(subtable_poly, &e_polys[memory_index])
        });
        memories.enumerate().for_each(|(memory_index, memory)| {
            let chunk_index = table.memory_to_chunk_index(memory_index);
            if chunk_map.get(&chunk_index).is_some() {
                chunk_map.entry(chunk_index).and_modify(|chunk| {
                    chunk.add_memory(memory);
                });
            } else {
                let dim = &dims[chunk_index];
                let read_ts_poly = &read_ts_polys[chunk_index];
                let final_cts_poly = &final_cts_polys[chunk_index];
                chunk_map.insert(
                    chunk_index,
                    Chunk::new(chunk_index, dim, read_ts_poly, final_cts_poly, memory),
                );
            }
        });

        // sanity check
        {
            let num_chunks = table.chunk_bits().len();
            assert_eq!(chunk_map.len(), num_chunks);
        }

        let mut chunks = chunk_map.into_iter().collect_vec();
        chunks.sort_by_key(|(chunk_index, _)| *chunk_index);
        chunks.into_iter().map(|(_, chunk)| chunk).collect_vec()
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare_memory_checking<'a>(
        table: &Box<dyn DecomposableTable<F, E>>,
        subtable_polys: &'a [&BoxMultilinearPoly<'static, F, E>],
        dims: &'a [BoxMultilinearPoly<'a, F, E>],
        read_ts_polys: &'a [BoxMultilinearPoly<'a, F, E>],
        final_cts_polys: &'a [BoxMultilinearPoly<'a, F, E>],
        e_polys: &'a [BoxMultilinearPoly<'a, F, E>],
        gamma: &E,
        tau: &E,
    ) -> Vec<MemoryCheckingProver<'a, F, E>> {
        let chunks = Self::chunks(
            table,
            subtable_polys,
            dims,
            read_ts_polys,
            final_cts_polys,
            e_polys,
        );
        let chunk_bits = table.chunk_bits();
        // key: chunk bits, value: chunks
        let mut chunk_map: HashMap<usize, Vec<Chunk<F, E>>> = HashMap::new();

        chunks.iter().enumerate().for_each(|(chunk_index, chunk)| {
            let chunk_bits = chunk_bits[chunk_index];
            if let Some(_) = chunk_map.get(&chunk_bits) {
                chunk_map.entry(chunk_bits).and_modify(|chunks| {
                    chunks.push(chunk.clone());
                });
            } else {
                chunk_map.insert(chunk_bits, vec![chunk.clone()]);
            }
        });

        chunk_map
            .into_iter()
            .enumerate()
            .map(|(index, (_, chunks))| MemoryCheckingProver::new(chunks, tau, gamma))
            .collect_vec()
    }
}

pub trait SubtableSet<F: PrimeField, E: ExtensionField<F>>:
    DecomposableTable<F, E> + IntoEnumIterator + EnumCount + Send + Sync
{
    // fn enum_index(subtable: Box<dyn DecomposableTable<F, E>>) -> usize {
    //     Self::from(subtable.subtable_id()).into()
    // }
}

#[derive(Clone)]
pub struct InstructionLookupsPreprocessing<F> {
    subtable_to_memory_indices: Vec<Vec<usize>>, // Vec<Range<usize>>?
    instruction_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<BoxMultilinearPoly<'static, F, E>>,
    num_memories: usize,
}

impl<F: PrimeField> InstructionLookupsPreprocessing<F> {
    pub fn preprocess<
        const C: usize,
        const M: usize,
        InstructionSet,
        Subtables: SubtableSet<F, E>,
    >() -> Self {
        let materialized_subtables = Self::materialize_subtables::<M, Subtables>();

        // Build a mapping from subtable type => chunk indices that access that subtable type
        let mut subtable_indices: Vec<SubtableIndices> =
            vec![SubtableIndices::with_capacity(C); Subtables::COUNT];
        for instruction in InstructionSet::iter() {
            for (subtable, indices) in instruction.subtables::<F>(C, M) {
                subtable_indices[Subtables::enum_index(subtable)].union_with(&indices);
            }
        }

        let mut subtable_to_memory_indices = Vec::with_capacity(Subtables::COUNT);
        let mut memory_to_subtable_index = vec![];
        let mut memory_to_dimension_index = vec![];

        let mut memory_index = 0;
        for (subtable_index, dimension_indices) in subtable_indices.iter().enumerate() {
            subtable_to_memory_indices
                .push((memory_index..memory_index + dimension_indices.len()).collect_vec());
            memory_to_subtable_index.extend(vec![subtable_index; dimension_indices.len()]);
            memory_to_dimension_index.extend(dimension_indices.iter());
            memory_index += dimension_indices.len();
        }
        let num_memories = memory_index;

        let mut instruction_to_memory_indices = vec![vec![]; InstructionSet::COUNT];
        for instruction in InstructionSet::iter() {
            for (subtable, dimension_indices) in instruction.subtables::<F>(C, M) {
                let memory_indices: Vec<_> = subtable_to_memory_indices
                    [Subtables::enum_index(subtable)]
                .iter()
                .filter(|memory_index| {
                    dimension_indices.contains(memory_to_dimension_index[**memory_index])
                })
                .collect();
                instruction_to_memory_indices[InstructionSet::enum_index(&instruction)]
                    .extend(memory_indices);
            }
        }

        Self {
            num_memories,
            materialized_subtables,
            subtable_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            instruction_to_memory_indices,
        }
    }

    fn materialize_subtables<const M: usize, Subtables>() -> Vec<Vec<F>>
    where
        Subtables: SubtableSet<F, E>,
    {
        let mut subtables = Vec::with_capacity(Subtables::COUNT);
        for subtable in Subtables::iter() {
            subtables.push(subtable.subtable_polys());
        }
        subtables
    }
}
#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{input::InputNode, log_up::LogUpNode, VanillaGate, VanillaNode},
            test::{run_circuit, TestData},
            Circuit,
        },
        poly::{
            box_dense_poly, box_owned_dense_poly, BoxMultilinearPoly, BoxMultilinearPolyOwned,
            MultilinearPolyTerms, PolyExpr,
        },
        util::{
            arithmetic::{div_ceil, inner_product, ExtensionField, Field},
            chain,
            dev::{rand_range, rand_vec, seeded_std_rng},
            expression::Expression,
            Itertools, RngCore,
        },
    };
    use core::num;
    use ff_ext::ff::PrimeField;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use num_integer::Integer;
    use plonkish_backend::{
        pcs::multilinear::MultilinearBrakedown,
        util::{code::BrakedownSpec6, Deserialize, DeserializeOwned, Serialize},
    };
    use rayon::vec;
    use std::{iter, marker::PhantomData};

    use super::{table::DecomposableTable, Lasso, LassoNode};
    use halo2_curves::bn256;
    use rand::Rng;
    use strum_macros::{EnumCount, EnumIter};

    pub type Brakedown<F> =
        MultilinearBrakedown<F, plonkish_backend::util::hash::Keccak256, BrakedownSpec6>;

    #[derive(Clone, Debug)]
    pub struct RangeTable<F, E, const NUM_BITS: usize, const LIMB_BITS: usize>(
        PhantomData<F>,
        PhantomData<E>,
    );

    impl<F, E, const NUM_BITS: usize, const LIMB_BITS: usize> RangeTable<F, E, NUM_BITS, LIMB_BITS> {
        pub fn new() -> Self {
            Self(PhantomData, PhantomData)
        }
    }

    impl<F: PrimeField, E: ExtensionField<F>, const NUM_BITS: usize, const LIMB_BITS: usize>
        DecomposableTable<F, E> for RangeTable<F, E, NUM_BITS, LIMB_BITS>
    {
        fn chunk_bits(&self) -> Vec<usize> {
            let remainder_bits = if NUM_BITS % LIMB_BITS != 0 {
                vec![NUM_BITS % LIMB_BITS]
            } else {
                vec![]
            };
            iter::repeat(LIMB_BITS)
                .take(NUM_BITS / LIMB_BITS)
                .chain(remainder_bits)
                .collect_vec()
        }

        fn combine_lookup_expressions(
            &self,
            expressions: Vec<Expression<E, usize>>,
        ) -> Expression<E, usize> {
            Expression::distribute_powers(expressions, E::from_bases(&[F::from(1 << LIMB_BITS)]))
        }

        fn combine_lookups(&self, operands: &[F]) -> F {
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

        fn num_memories(&self) -> usize {
            div_ceil(NUM_BITS, LIMB_BITS)
        }

        fn subtable_indices(&self, index_bits: Vec<bool>) -> Vec<Vec<bool>> {
            index_bits.chunks(LIMB_BITS).map(Vec::from).collect_vec()
        }

        fn subtable_polys(&self) -> Vec<BoxMultilinearPoly<'static, F, E>> {
            let mut evals = vec![];
            (0..1 << LIMB_BITS).for_each(|i| evals.push(F::from(i)));
            let limb_subtable_poly = box_dense_poly(evals);
            if NUM_BITS % LIMB_BITS != 0 {
                let remainder = NUM_BITS % LIMB_BITS;
                let mut evals = vec![];
                (0..1 << remainder).for_each(|i| {
                    evals.push(F::from(i));
                });
                let rem_subtable_poly = box_dense_poly(evals);
                vec![limb_subtable_poly, rem_subtable_poly]
            } else {
                vec![limb_subtable_poly]
            }
        }

        fn subtable_polys_terms(&self) -> Vec<MultilinearPolyTerms<F>> {
            let limb_init = PolyExpr::Var(0);
            let mut limb_terms = vec![limb_init];
            (1..LIMB_BITS).for_each(|i| {
                let coeff = PolyExpr::Pow(Box::new(PolyExpr::Const(F::from(2))), i as u32);
                let x = PolyExpr::Var(i);
                let term = PolyExpr::Prod(vec![coeff, x]);
                limb_terms.push(term);
            });
            let limb_subtable_poly =
                MultilinearPolyTerms::new(LIMB_BITS, PolyExpr::Sum(limb_terms));
            if NUM_BITS % LIMB_BITS == 0 {
                vec![limb_subtable_poly]
            } else {
                let remainder = NUM_BITS % LIMB_BITS;
                let rem_init = PolyExpr::Var(0);
                let mut rem_terms = vec![rem_init];
                (1..remainder).for_each(|i| {
                    let coeff = PolyExpr::Pow(Box::new(PolyExpr::Const(F::from(2))), i as u32);
                    let x = PolyExpr::Var(i);
                    let term = PolyExpr::Prod(vec![coeff, x]);
                    rem_terms.push(term);
                });
                vec![
                    limb_subtable_poly,
                    MultilinearPolyTerms::new(remainder, PolyExpr::Sum(rem_terms)),
                ]
            }
        }

        fn memory_to_chunk_index(&self, memory_index: usize) -> usize {
            memory_index
        }

        fn memory_to_subtable_index(&self, memory_index: usize) -> usize {
            if NUM_BITS % LIMB_BITS != 0 && memory_index == NUM_BITS / LIMB_BITS {
                1
            } else {
                0
            }
        }
    }

    #[derive(EnumCount, EnumIter)]
    enum RangeTables<F, E, const LIMB_BITS: usize> {
        RangeTable16(RangeTable<F, E, 16, LIMB_BITS>),
        RangeTable55(RangeTable<F, E, 55, LIMB_BITS>),
        RangeTable128(RangeTable<F, E, 128, LIMB_BITS>),
    }

    #[test]
    fn lasso_single() {
        run_circuit::<Goldilocks, Goldilocks>(lasso_circuit::<_, _, 1>);
    }

    fn lasso_circuit<
        F: PrimeField + Serialize + DeserializeOwned,
        E: ExtensionField<F>,
        const N: usize,
    >(
        num_vars: usize,
        mut rng: &mut impl Rng,
    ) -> TestData<F, E> {
        // let log2_t_size = rand_range(0..2 * num_vars, &mut rng);

        println!("num_vars: {}", num_vars);
        let size = 1 << num_vars;

        const LIMB_BITS: usize = 16;

        let table: Box<dyn DecomposableTable<F, E>> =
            Box::new(RangeTable::<F, E, 128, LIMB_BITS>::new());

        let evals = iter::repeat_with(|| F::from_u128(rng.gen_range(0..(1 << 64))))
            .take(size)
            .collect_vec();
        // let input = box_dense_poly(evals.clone());
        let output = box_dense_poly(evals);

        let circuit = {
            let mut circuit = Circuit::default();

            let lookup_output = circuit.insert(InputNode::new(num_vars, 1));
            let lasso = circuit.insert(LassoNode::new(table, num_vars, LIMB_BITS));
            circuit.connect(lookup_output, lasso);

            circuit
        };

        let inputs = vec![output];

        // let inputs = layers[0]
        //     .iter()
        //     .flat_map(|layer| {
        //         [
        //             box_dense_poly(layer.v_l.to_dense()),
        //             box_dense_poly(layer.v_r.to_dense()),
        //         ]
        //     })
        //     .collect_vec();

        // let values = layers
        //     .into_iter()
        //     .flat_map(|layers| layers.into_iter().flat_map(|layer| [layer.v_l, layer.v_r]))
        //     .collect_vec();

        // let values = vec![];

        (circuit, inputs, None)
    }
}
