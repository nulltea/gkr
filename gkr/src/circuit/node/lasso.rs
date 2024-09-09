use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{
        box_dense_poly, merge, BoxMultilinearPoly, DensePolynomial, MultilinearPoly,
        MultilinearPolyTerms,
    },
    sum_check::{
        err_unmatched_evaluation, generic::Generic, prove_sum_check, verify_sum_check,
        SumCheckFunction, SumCheckPoly,
    },
    transcript::{Transcript, TranscriptRead, TranscriptWrite},
    util::{
        arithmetic::{div_ceil, ExtensionField, Field},
        chain,
        expression::Expression,
        izip, Itertools,
    },
    Error,
};
use ark_std::log2;
use ff_ext::ff::PrimeField;
use memory_checking::{Chunk, Memory, MemoryCheckingProver};
use plonkish_backend::{
    backend::lookup,
    pcs::PolynomialCommitmentScheme,
    poly::multilinear::{MultilinearPolynomial, MultilinearPolynomialTerms},
    util::{
        arithmetic::{fe_to_bits_le, usize_from_bits_le},
        parallel::{num_threads, parallelize_iter},
    },
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
use table::{
    CircuitLookups, DecomposableTable, LassoSubtable, LookupType, SubtableId, SubtableIndices,
    SubtableSet,
};
// use verifier::LassoVerifier;

pub mod memory_checking;
// pub mod test;
pub mod table;
// pub mod verifier;

#[derive(Clone, Debug)]
pub struct GeneralizedLasso<F: Field, Pcs: PolynomialCommitmentScheme<F>>(
    PhantomData<F>,
    PhantomData<Pcs>,
);

#[derive(Debug)]
pub struct LassoNode<F, E, Lookups: CircuitLookups, const C: usize, const M: usize> {
    num_vars: usize,
    final_poly_log2_size: usize,
    preprocessing: LassoLookupsPreprocessing<F, E>,
    _marker: PhantomData<Lookups>,
}

impl<
        F: PrimeField,
        E: ExtensionField<F>,
        Lookups: CircuitLookups,
        const C: usize,
        const M: usize,
    > Node<F, E> for LassoNode<F, E, Lookups, C, M>
{
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
        // println!("combined claim {} {:?}", claim.points.len(), claim.value);

        // let [input, output] = inputs.try_into().unwrap();
        let lookup_output_poly = inputs[0];
        let lookup_index_poly = lookup_output_poly;
        let polys = Self::polynomialize(&self.preprocessing, lookup_index_poly);
        let mock_lookup = Lookups::iter().collect_vec()[0];

        let LassoPolynomials {
            dims,
            read_cts: read_ts_polys,
            final_cts: final_cts_polys,
            e_polys,
            lookup_outputs,
        } = polys;

        // assert_eq!(lookup_output_poly.to_dense(), lookup_outputs.to_dense());

        // let [e_polys, dims, read_ts_polys, final_cts_polys] = polys;

        let num_vars = lookup_output_poly.num_vars();
        assert!(num_vars == self.num_vars);

        assert_eq!(final_cts_polys[0].num_vars(), self.final_poly_log2_size);

        // should this be passed from CombinedEvalClaim?
        let r = transcript.squeeze_challenges(num_vars);

        let res = self.prove_collation_sum_check(
            &lookup_outputs,
            &mock_lookup,
            &e_polys,
            &r,
            num_vars,
            transcript,
        )?;

        let lookup_output_eval_claim = res.into_iter().take(1).collect_vec();

        println!("Proved Surge sum check");

        let [gamma, tau] = transcript.squeeze_challenges(2).try_into().unwrap();

        // memory_checking
        Self::prove_memory_checking(
            &self.preprocessing,
            &dims,
            &read_ts_polys,
            &final_cts_polys,
            &e_polys,
            &gamma,
            &tau,
            transcript,
        )?;

        Ok(lookup_output_eval_claim)
    }

    fn verify_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let mock_lookup = Lookups::iter().collect_vec()[0];
        let num_vars = self.num_vars;
        let r = transcript.squeeze_challenges(num_vars);

        let g = self.collation_sum_check_function(&mock_lookup, num_vars);
        let claimed_sum = transcript.read_felt_ext()?;

        let (sub_claim, r_x_prime) = verify_sum_check(&g, claimed_sum, transcript)?;

        // let e_polys_eval = transcript.read_felt_exts(table.num_memories())?;

        // Round n+1
        let [gamma, tau] = transcript.squeeze_challenges(2).try_into().unwrap();

        // memory checking
        Self::verify_memory_checking(&self.preprocessing, num_vars, &gamma, &tau, transcript)?;

        Ok(chain![iter::once(vec![EvalClaim::new(r.to_vec(), claimed_sum)]),].collect_vec())
    }
}

impl<
        F: PrimeField,
        E: ExtensionField<F>,
        Lookups: CircuitLookups,
        const C: usize,
        const M: usize,
    > LassoNode<F, E, Lookups, C, M>
{
    pub fn new(
        // table: Box<dyn DecomposableTable<F, E>>,
        preprocessing: LassoLookupsPreprocessing<F, E>,
        num_vars: usize,
        final_poly_log2_size: usize,
    ) -> Self {
        Self {
            num_vars,
            final_poly_log2_size,
            preprocessing,
            _marker: PhantomData,
        }
    }

    fn chunks<'a>(
        preprocessing: &'a LassoLookupsPreprocessing<F, E>,
        dims: &'a [BoxMultilinearPoly<F, E>],
        read_ts_polys: &'a [BoxMultilinearPoly<F, E>],
        final_cts_polys: &'a [BoxMultilinearPoly<F, E>],
        e_polys: &'a [BoxMultilinearPoly<F, E>],
    ) -> Vec<Chunk<'a, F, E>> {
        // key: chunk index, value: chunk
        let mut chunk_map: HashMap<usize, Chunk<F, E>> = HashMap::new();

        let num_memories = preprocessing.num_memories;
        let memories = (0..num_memories).map(|memory_index| {
            let subtable_poly = &preprocessing.materialized_subtables
                [preprocessing.memory_to_subtable_index[memory_index]];
            Memory::<F, E>::new(subtable_poly, &e_polys[memory_index])
        });
        memories.enumerate().for_each(|(memory_index, memory)| {
            let chunk_index = preprocessing.memory_to_dimension_index[memory_index];
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
        // {
        //     let num_chunks = table.chunk_bits().len();
        //     assert_eq!(chunk_map.len(), num_chunks);
        // }

        let mut chunks = chunk_map.into_iter().collect_vec();
        chunks.sort_by_key(|(chunk_index, _)| *chunk_index);
        chunks.into_iter().map(|(_, chunk)| chunk).collect_vec()
    }

    fn verify_memory_checking(
        preprocessing: &LassoLookupsPreprocessing<F, E>,
        num_vars: usize,
        gamma: &E,
        tau: &E,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<(), Error> {

        let num_memories = preprocessing.num_memories;
        // let mock_lookup = Lookups::iter().collect_vec()[0];
        // let chunk_bits = mock_lookup.chunk_bits(log2(M) as usize);
        // key: chunk index, value: chunk
        let mut chunk_map: HashMap<usize, memory_checking::verifier::Chunk<F>> = HashMap::new();
        (0..num_memories).for_each(|memory_index| {
            let chunk_index = preprocessing.memory_to_dimension_index[memory_index];
            // let chunk_bits = chunk_bits[chunk_index];
            let subtable_poly = &preprocessing.subtable_evaluate_mle_exprs
                [preprocessing.memory_to_subtable_index[memory_index]];
            let memory =
                memory_checking::verifier::Memory::new(memory_index, subtable_poly.clone());
            if chunk_map.get(&chunk_index).is_some() {
                chunk_map.entry(chunk_index).and_modify(|chunk| {
                    chunk.add_memory(memory);
                });
            } else {
                chunk_map.insert(
                    chunk_index,
                    memory_checking::verifier::Chunk::new(chunk_index, log2(M) as usize, memory),
                );
            }
        });

        // sanity check
        // {
        //     let num_chunks = preprocessing.chunk_bits().len();
        //     assert_eq!(chunk_map.len(), num_chunks);
        // }

        let mut chunks = chunk_map.into_iter().collect_vec();
        chunks.sort_by_key(|(chunk_index, _)| *chunk_index);
        let chunks = chunks.into_iter().map(|(_, chunk)| chunk).collect_vec();

        let mem_check = memory_checking::verifier::MemoryCheckingVerifier::new(chunks);
        mem_check.verify(num_vars, gamma, tau, transcript)
    }

    #[allow(clippy::too_many_arguments)]
    fn prove_memory_checking<'a>(
        preprocessing: &'a LassoLookupsPreprocessing<F, E>,
        dims: &'a [BoxMultilinearPoly<'a, F, E>],
        read_ts_polys: &'a [BoxMultilinearPoly<'a, F, E>],
        final_cts_polys: &'a [BoxMultilinearPoly<'a, F, E>],
        e_polys: &'a [BoxMultilinearPoly<'a, F, E>],
        gamma: &E,
        tau: &E,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<(), Error> {
        let chunks = Self::chunks(preprocessing, dims, read_ts_polys, final_cts_polys, e_polys);
        MemoryCheckingProver::new(chunks, tau, gamma).prove(transcript)
    }

    pub fn polynomialize<'a>(
        preprocessing: &LassoLookupsPreprocessing<F, E>,
        // ops: &Vec<JoltTraceStep<InstructionSet>>,
        lookup_index_poly: &BoxMultilinearPoly<F, E>,
    ) -> LassoPolynomials<'a, F, E> {
        let num_memories = preprocessing.num_memories;
        let num_reads: usize = lookup_index_poly.len();

        let lookup = Lookups::iter().collect_vec()[0];

        // assert_eq!(num_memories, C);

        // subtable_lookup_indices : [[usize; num_rows]; num_chunks]
        let lookup_indexes = lookup_index_poly.as_dense().unwrap();
        let subtable_lookup_indices: Vec<Vec<usize>> =
            Self::subtable_lookup_indices(lookup_indexes, lookup);

        println!("num memories: {}", num_memories);

        let polys: Vec<_> = (0..preprocessing.num_memories)
            // .into_par_iter()
            .into_iter()
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[dim_index];

                let mut final_cts_i = vec![0usize; M]; // TODO: or should be lookup s
                let mut read_cts_i = vec![0usize; num_reads];
                let mut subtable_lookups = vec![F::ZERO; num_reads];

                println!("memory_index {} dim {} access_sequence: {:?}", memory_index, dim_index, access_sequence);
                for (j, op) in lookup_indexes.iter().enumerate() {
                    let memories_used =
                        &preprocessing.lookup_to_memory_indices[Lookups::enum_index(&lookup)];
                    if memories_used.contains(&memory_index) {
                        let memory_address = access_sequence[j];
                        debug_assert!(memory_address < M);

                        let counter = final_cts_i[memory_address];
                        read_cts_i[j] = counter;
                        final_cts_i[memory_address] = counter + 1;
                        subtable_lookups[j] =
                            preprocessing.materialized_subtables[subtable_index][memory_address];

                        println!("lookup_indexes[j]: {:?}", op);
                        println!("memory_address: {}", memory_address);
                        println!("subtable_lookups[j]: {:?}", subtable_lookups[j]);
                    }
                }

                (
                    DensePolynomial::from_usize::<E>(&read_cts_i),
                    DensePolynomial::from_usize::<E>(&final_cts_i),
                    box_dense_poly(subtable_lookups),
                )
            })
            .collect();

        // Vec<(DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>)> -> (Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>, Vec<DensePolynomial<F>>)
        let (read_cts, final_cts, e_polys) = polys.into_iter().fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |(mut read_acc, mut final_acc, mut e_acc), (read, f, e)| {
                read_acc.push(read);
                final_acc.push(f);
                e_acc.push(e);
                (read_acc, final_acc, e_acc)
            },
        );

        let dims: Vec<_> = (0..C)
            .into_par_iter()
            .map(|i| {
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[i];
                DensePolynomial::from_usize(access_sequence)
            })
            .collect();

        // let mut instruction_flag_bitvectors: Vec<Vec<u64>> =
        //     vec![vec![0u64; m]; Self::NUM_INSTRUCTIONS];
        // for (j, op) in ops.iter().enumerate() {
        //     if let Some(instr) = &op.instruction_lookup {
        //         instruction_flag_bitvectors[InstructionSet::enum_index(instr)][j] = 1;
        //     }
        // }

        // let instruction_flag_polys: Vec<DensePolynomial<F>> = instruction_flag_bitvectors
        //     .par_iter()
        //     .map(|flag_bitvector| DensePolynomial::from_u64(flag_bitvector))
        //     .collect();

        let mut lookup_outputs = Self::compute_lookup_outputs(&lookup_indexes);
        lookup_outputs.resize(num_reads, F::ZERO);
        let lookup_outputs = box_dense_poly(lookup_outputs);

        LassoPolynomials {
            dims,
            read_cts,
            final_cts,
            // instruction_flag_polys,
            // instruction_flag_bitvectors,
            e_polys,
            lookup_outputs,
        }
    }

    fn subtable_lookup_indices(index_poly: &[F], lookup: impl LookupType) -> Vec<Vec<usize>> {
        let num_rows: usize = index_poly.len();
        let num_chunks = C;
        println!("num_chunks: {}", num_chunks);

        let indices = (0..num_rows)
            .map(|i| {
                let mut index_bits = fe_to_bits_le(index_poly[i]);
                index_bits.truncate(lookup.chunk_bits(log2(M) as usize).iter().sum());
                // index_bits.truncate()
                assert_eq!(
                    usize_from_bits_le(&fe_to_bits_le(index_poly[i])),
                    usize_from_bits_le(&index_bits)
                );

                let mut chunked_index = iter::repeat(0).take(num_chunks).collect_vec();
                let chunked_index_bits = lookup.subtable_indices(index_bits, M.ilog2() as usize);
                chunked_index
                    .iter_mut()
                    .zip(chunked_index_bits)
                    .map(|(chunked_index, index_bits)| {
                        *chunked_index = usize_from_bits_le(&index_bits);
                    })
                    .collect_vec();
                chunked_index
            })
            .collect_vec();
        let mut lookup_indices = vec![vec![]; num_chunks];
        lookup_indices
            .iter_mut()
            .enumerate()
            .for_each(|(i, lookup_indices)| {
                let indices = indices
                    .iter()
                    .map(|indices| {
                        lookup_indices.push(indices[i]);
                        indices[i]
                    })
                    .collect_vec();
            });
        lookup_indices
    }

    #[tracing::instrument(skip_all, name = "LassoNode::compute_lookup_outputs")]
    fn compute_lookup_outputs(lookup_index_poly: &[F]) -> Vec<F> {
        let lookup = &Lookups::iter().collect_vec()[0];
        lookup_index_poly
            .par_iter()
            .map(|i| lookup.output(i))
            .collect()
    }

    pub fn prove_collation_sum_check(
        // table: &Box<dyn DecomposableTable<F, E>>,
        &self,
        lookup_output_poly: &BoxMultilinearPoly<F, E>,
        lookup: &impl LookupType,
        e_polys: &[BoxMultilinearPoly<F, E>],
        r: &[E],
        num_vars: usize,
        // points_offset: usize,
        // lookup_opening_points: &mut Vec<Vec<F>>,
        // lookup_opening_evals: &mut Vec<Evaluation<F>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let claimed_sum = self.sum_check_claim(r, lookup, e_polys);
        assert_eq!(claimed_sum, lookup_output_poly.evaluate(r));

        transcript.write_felt_ext(&claimed_sum)?;

        let g = self.collation_sum_check_function(lookup, num_vars);

        let polys = e_polys
            .iter()
            .map(|e_poly| SumCheckPoly::Base::<_, _, _, BoxMultilinearPoly<E, E>>(e_poly))
            .collect_vec();

        let (claim, r_x_prime, e_polys_evals) =
            prove_sum_check(&g, claimed_sum, polys, transcript)?;

        println!("r_x_prime: {:?}", r_x_prime.len());
        println!("evals: {:?}", e_polys_evals.len());

        // transcript.write_felt_exts(&e_polys_evals)?;

        // lookup_opening_points.extend_from_slice(&[r.to_vec(), x]);
        // let evals = expression
        //     .used_query()
        //     .into_iter()
        //     .map(|query| {
        //         transcript
        //             .write_felt(&evals[query.poly()])
        //             .unwrap();
        //         Evaluation::new(
        //             e_polys[query.poly()].offset,
        //             points_offset + 1,
        //             evals[query.poly()],
        //         )
        //     })
        //     .chain([Evaluation::new(
        //         lookup_output_poly.offset,
        //         points_offset,
        //         claimed_sum,
        //     )])
        //     .collect_vec();
        // lookup_opening_evals.extend_from_slice(&evals);

        Ok(chain![
            iter::once(vec![EvalClaim::new(r.to_vec(), claimed_sum)]),
            e_polys_evals
                .into_iter()
                .map(|e| vec![EvalClaim::new(r_x_prime.clone(), e)])
        ]
        .collect_vec())
    }

    pub fn sum_check_claim(
        &self,
        r: &[E], // claim: CombinedEvalClaim<E>,
        lookup: &impl LookupType,
        e_polys: &[BoxMultilinearPoly<F, E>],
    ) -> E {
        let num_memories = self.preprocessing.num_memories;
        assert_eq!(e_polys.len(), num_memories);
        let num_vars = e_polys[0].num_vars();
        let bh_size = 1 << num_vars;
        let eq = MultilinearPolynomial::eq_xy(r);
        // \sum_{k \in \{0, 1\}^{\log m}} (\tilde{eq}(r, k) * g(E_1(k), ..., E_{\alpha}(k)))
        let claim = (0..bh_size)
            .into_par_iter()
            .map(|k| {
                let operands = e_polys.iter().map(|e_poly| e_poly[k]).collect_vec();
                eq[k] * lookup.combine_lookups(&operands, C, M)
            })
            .sum();

        claim
    }

    // (\tilde{eq}(r, k) * g(E_1(k), ..., E_{\alpha}(k)))
    pub fn collation_sum_check_function(
        &self,
        lookup: &impl LookupType,
        num_vars: usize,
    ) -> Generic<F, E> {
        let num_memories = self.preprocessing.num_memories;
        let exprs = lookup.combine_lookup_expressions(
            (0..num_memories)
                .map(|idx| Expression::poly(idx))
                .collect_vec(),
            C,
            M,
        );

        let eq_r_x = &Expression::poly(0);

        Generic::new(num_vars, &(eq_r_x * exprs))
    }
}

/// All polynomials associated with Jolt instruction lookups.
pub struct LassoPolynomials<'a, F: PrimeField, E: ExtensionField<F>> {
    /// `C` sized vector of `DensePolynomials` whose evaluations correspond to
    /// indices at which the memories will be evaluated. Each `DensePolynomial` has size
    /// `m` (# lookups).
    pub dims: Vec<BoxMultilinearPoly<'a, F, E>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// read access counts to the memory. Each `DensePolynomial` has size `m` (# lookups).
    pub read_cts: Vec<BoxMultilinearPoly<'a, F, E>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// final access counts to the memory. Each `DensePolynomial` has size M, AKA subtable size.
    pub final_cts: Vec<BoxMultilinearPoly<'a, F, E>>,

    /// `NUM_MEMORIES` sized vector of `DensePolynomials` whose evaluations correspond to
    /// the evaluation of memory accessed at each step of the CPU. Each `DensePolynomial` has
    /// size `m` (# lookups).
    pub e_polys: Vec<BoxMultilinearPoly<'a, F, E>>,

    /// Polynomial encodings for flag polynomials for each instruction.
    /// If using a single instruction this will be empty.
    /// NUM_INSTRUCTIONS sized, each polynomial of length 'm' (# lookups).
    ///
    /// Stored independently for use in sumcheck, combined into single DensePolynomial for commitment.
    // pub instruction_flag_polys: Vec<BoxMultilinearPoly<'a, F, E>>,

    /// Instruction flag polynomials as bitvectors, kept in this struct for more efficient
    /// construction of the memory flag polynomials in `read_write_grand_product`.
    // instruction_flag_bitvectors: Vec<Vec<u64>>,
    /// The lookup output for each instruction of the execution trace.
    pub lookup_outputs: BoxMultilinearPoly<'a, F, E>,
}

#[derive(Debug)]
pub struct LassoLookupsPreprocessing<F, E> {
    subtable_to_memory_indices: Vec<Vec<usize>>, // Vec<Range<usize>>?
    lookup_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<BoxMultilinearPoly<'static, F, E>>,
    subtable_ids: Vec<SubtableId>,
    subtable_evaluate_mle_exprs: Vec<MultilinearPolyTerms<F>>,
    // subtables: Vec<Box<dyn LassoSubtable<F, E>>>,
    num_memories: usize, // C
    _marker: PhantomData<E>,
}

impl<F: PrimeField, E: ExtensionField<F>> LassoLookupsPreprocessing<F, E> {
    pub fn preprocess<
        const C: usize,
        const M: usize,
        Lookups: CircuitLookups,
        Subtables: SubtableSet<F, E>,
    >() -> Self {
        let materialized_subtables = Self::materialize_subtables::<M, Subtables>()
            .into_iter()
            .map(box_dense_poly)
            .collect_vec();

        // Build a mapping from subtable type => chunk indices that access that subtable type
        let mut subtable_indices: Vec<SubtableIndices> =
            vec![SubtableIndices::with_capacity(C); Subtables::COUNT];
        let mut subtable_ids: Vec<std::any::TypeId> = Vec::with_capacity(Subtables::COUNT);
        let mut subtable_evaluate_mle_exprs = Vec::with_capacity(Subtables::COUNT);
        for lookup in Lookups::iter() {
            for (subtable, indices) in lookup.subtables::<F, E>(C, M).into_iter() {
                subtable_ids.push(subtable.subtable_id());
                subtable_evaluate_mle_exprs.push(subtable.evaluate_mle_expr(log2(M) as usize));
                let subtable_idx = Subtables::enum_index(subtable);
                subtable_indices[subtable_idx].union_with(&indices);
            }
        }
        println!(
            "subtable_indices: {:?}",
            subtable_indices
                .iter()
                .enumerate()
                .map(|(i, e)| (i, e.len()))
                .collect_vec()
        );

        let mut subtable_to_memory_indices = Vec::with_capacity(Subtables::COUNT);
        let mut memory_to_subtable_index = vec![];
        let mut memory_to_dimension_index = vec![];

        let mut memory_index = 0;
        for (subtable_index, dimension_indices) in subtable_indices.iter().enumerate() {
            subtable_to_memory_indices
                .push((memory_index..memory_index + dimension_indices.len()).collect_vec());
            memory_to_subtable_index.extend(vec![subtable_index; dimension_indices.len()]);
            memory_to_dimension_index.extend(dimension_indices.iter());
            println!(
                "memory_to_dimension_index: {:?}",
                dimension_indices.iter().collect_vec()
            );
            memory_index += dimension_indices.len();
        }
        let num_memories = memory_index;

        // instruction is a type of lookup
        // assume all instreuctions are the same first
        let mut lookup_to_memory_indices = vec![vec![]; Lookups::COUNT];
        for lookup_type in Lookups::iter() {
            for (subtable, dimension_indices) in lookup_type.subtables::<F, E>(C, M) {
                let memory_indices: Vec<_> = subtable_to_memory_indices
                    [Subtables::enum_index(subtable)]
                .iter()
                .filter(|memory_index| {
                    dimension_indices.contains(memory_to_dimension_index[**memory_index])
                })
                .collect();
                lookup_to_memory_indices[Lookups::enum_index(&lookup_type)].extend(memory_indices);
            }
        }

        Self {
            num_memories,
            materialized_subtables,
            subtable_to_memory_indices,
            memory_to_subtable_index,
            memory_to_dimension_index,
            lookup_to_memory_indices,
            subtable_ids,
            subtable_evaluate_mle_exprs,
            _marker: PhantomData,
        }
    }

    fn materialize_subtables<const M: usize, Subtables>() -> Vec<Vec<F>>
    where
        Subtables: SubtableSet<F, E>,
    {
        let mut subtables = Vec::with_capacity(Subtables::COUNT);
        for subtable in Subtables::iter() {
            subtables.push(subtable.materialize(M));
        }
        subtables
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{
                input::InputNode, lasso::LassoLookupsPreprocessing, log_up::LogUpNode,
                range::RangeTable, VanillaGate, VanillaNode,
            },
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
            dev::{rand_range, rand_vec, seeded_std_rng, std_rng},
            expression::Expression,
            Itertools, RngCore,
        },
    };
    use core::num;
    use enum_dispatch::enum_dispatch;
    use ff_ext::ff::PrimeField;
    use goldilocks::{Goldilocks, GoldilocksExt2};
    use num_integer::Integer;
    use plonkish_backend::{
        pcs::multilinear::MultilinearBrakedown,
        util::{code::BrakedownSpec6, Deserialize, DeserializeOwned, Serialize},
    };
    use rayon::vec;
    use std::{iter, marker::PhantomData};

    use super::{
        table::{
            range::{FullLimbSubtable, RangeStategy, ReminderSubtable},
            CircuitLookups, DecomposableTable, LassoSubtable, LookupType, SubtableIndices,
            SubtableSet,
        },
        LassoNode,
    };
    use crate::circuit::node::SubtableId;
    use halo2_curves::bn256;
    use rand::Rng;
    use std::any::TypeId;
    use strum_macros::{EnumCount, EnumIter};

    pub type Brakedown<F> =
        MultilinearBrakedown<F, plonkish_backend::util::hash::Keccak256, BrakedownSpec6>;

    /// Generates an enum out of a list of LassoSubtable types. All LassoSubtable methods
    /// are callable on the enum type via enum_dispatch.
    // #[macro_export]
    macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[enum_dispatch(LassoSubtable<F, E>)]
        #[derive(EnumCount, EnumIter, Debug)]
        pub enum $enum_name<F: PrimeField, E: ExtensionField<F>> { $($alias($struct)),+ }
        impl<F: PrimeField, E: ExtensionField<F>> From<SubtableId> for $enum_name<F, E> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) }
          }
        }

        impl<F: PrimeField, E: ExtensionField<F>> SubtableSet<F, E> for $enum_name<F, E> {}
    };
}

    const TEST_BITS: usize = 55;

    subtable_enum!(
        RangeSubtables,
        Full: FullLimbSubtable<F, E>,
        Reminder: ReminderSubtable<F, E, TEST_BITS>
    );

    #[derive(Copy, Clone, Debug, EnumCount, EnumIter)]
    #[enum_dispatch(LookupType)]
    enum RangeLookups<const LIMB_BITS: usize> {
        // Range128(RangeStategy<128>),
        // Range64(RangeStategy<64>),
        RangeTest(RangeStategy<TEST_BITS>),
    }
    impl<const LIMB_BITS: usize> CircuitLookups for RangeLookups<LIMB_BITS> {}

    // #[derive(Clone, Debug, EnumCount, EnumIter)]
    // #[enum_dispatch(LassoSubtable<F, E>)]
    // enum RangeSubtables2<F: PrimeField,E: ExtensionField<F>, const LIMB_BITS: usize> {
    //    Full(FullLimbSubtable<F, E, LIMB_BITS>),
    // }
    // impl<F: PrimeField, E: ExtensionField<F>> SubtableSet<F, E> for RangeSubtables2<F, E, 16> {}

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
        const C: usize = 8;
        const M: usize = 1 << 16;
        // let log2_t_size = rand_range(0..2 * num_vars, &mut rng);

        println!("num_vars: {}", num_vars);
        let size = 1 << num_vars;

        const LIMB_BITS: usize = 16;

        let rng = &mut std_rng();
        let evals = iter::repeat_with(|| F::from_u128(rng.gen_range(0..(1 << TEST_BITS))))
            .take(size)
            .collect_vec();
        // let input = box_dense_poly(evals.clone());
        let output = box_dense_poly(evals);

        let preprocessing = LassoLookupsPreprocessing::<F, E>::preprocess::<
            C,
            M,
            RangeLookups<LIMB_BITS>,
            RangeSubtables<F, E>,
        >();

        let circuit = {
            let mut circuit = Circuit::default();

            let lookup_output = circuit.insert(InputNode::new(num_vars, 1));
            let lasso = circuit.insert(LassoNode::<_, _, RangeLookups<LIMB_BITS>, C, M>::new(
                preprocessing,
                num_vars,
                LIMB_BITS,
            ));
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
