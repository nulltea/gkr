use crate::{
    circuit::node::{CombinedEvalClaim, EvalClaim, Node},
    poly::{box_dense_poly, merge, BoxMultilinearPoly, DensePolynomial, MultilinearPoly},
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
use surge::Surge;
use table::{
    CircuitLookups, DecomposableTable, LassoSubtable, LookupType, SubtableIndices, SubtableSet,
};
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
pub struct LassoNode<F, E, Lookups: CircuitLookups, const C: usize> {
    num_vars: usize,
    final_poly_log2_size: usize,
    table: Box<dyn DecomposableTable<F, E>>,
    preprocessing: LassoLookupsPreprocessing<F, E>,
    _marker: PhantomData<Lookups>,
}

impl<F: PrimeField, E: ExtensionField<F>, Lookups: CircuitLookups, const C: usize> Node<F, E>
    for LassoNode<F, E, Lookups, C>
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

        // let (polys, _) = Self::commit(
        //     &table,
        //     subtable_polys,
        //     lookup_output_poly,
        //     lookup_index_poly,
        // )?;

        let polys = Self::polynomialize(&self.preprocessing, lookup_index_poly);

        let e_polys = polys.e_polys;
        let dims = polys.dims;
        let read_ts_polys = polys.read_cts;
        let final_cts_polys = polys.final_cts;

        // let [e_polys, dims, read_ts_polys, final_cts_polys] = polys;

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

impl<F: PrimeField, E: ExtensionField<F>, Lookups: CircuitLookups, const C: usize>
    LassoNode<F, E, Lookups, C>
{
    pub fn new(
        table: Box<dyn DecomposableTable<F, E>>,
        preprocessing: LassoLookupsPreprocessing<F, E>,
        num_vars: usize,
        final_poly_log2_size: usize,
    ) -> Self {
        Self {
            num_vars,
            table,
            final_poly_log2_size,
            preprocessing,
            _marker: PhantomData,
        }
    }

    // pub fn commit<'a>(
    //     // pp: &Pcs::ProverParam,
    //     // lookup_polys_offset: usize,
    //     table: &Box<dyn DecomposableTable<F, E>>,
    //     subtable_polys: &[&BoxMultilinearPoly<'static, F, E>],
    //     lookup_output_poly: &BoxMultilinearPoly<'a, F, E>,
    //     lookup_index_poly: &BoxMultilinearPoly<'a, F, E>,
    //     // transcript: &mut dyn TranscriptWrite<F, E>,
    // ) -> Result<
    //     (
    //         [Vec<BoxMultilinearPoly<'a, F, E>>; 4],
    //         (), // Vec<Vec<Pcs::Commitment>>,
    //     ),
    //     Error,
    // > {
    //     let num_chunks = table.chunk_bits().len();

    //     // commit to lookup_output_poly
    //     // let lookup_output_comm = Pcs::commit_and_write(&pp, &lookup_output_poly, transcript)?;

    //     // get surge and dims
    //     let mut surge = Surge::<F, E>::new();

    //     // commit to dims
    //     let dims = surge.commit(&table, &lookup_index_poly);
    //     // let dim_comms = Pcs::batch_commit_and_write(pp, &dims, transcript)?;

    //     // get e_polys & read_ts_polys & final_cts_polys
    //     let e_polys = {
    //         let indices = surge.indices();
    //         Self::e_polys(table, subtable_polys, &indices)
    //     };
    //     let (read_ts_polys, final_cts_polys) = surge.counter_polys(&table);

    //     // commit to read_ts_polys & final_cts_polys & e_polys
    //     // let read_ts_comms = Pcs::batch_commit_and_write(&pp, &read_ts_polys, transcript)?;
    //     // let final_cts_comms = Pcs::batch_commit_and_write(&pp, &final_cts_polys, transcript)?;
    //     // let e_comms = Pcs::batch_commit_and_write(&pp, e_polys.as_slice(), transcript)?;

    //     // let dims = dims
    //     //     .into_iter()
    //     //     .enumerate()
    //     //     .map(|(chunk_index, dim)| Poly {
    //     //         offset: lookup_polys_offset + 1 + chunk_index,
    //     //         poly: dim,
    //     //     })
    //     //     .collect_vec();

    //     // let read_ts_polys = read_ts_polys
    //     //     .into_iter()
    //     //     .enumerate()
    //     //     .map(|(chunk_index, read_ts_poly)| Poly {
    //     //         offset: lookup_polys_offset + 1 + num_chunks + chunk_index,
    //     //         poly: read_ts_poly,
    //     //     })
    //     //     .collect_vec();

    //     // let final_cts_polys = final_cts_polys
    //     //     .into_iter()
    //     //     .enumerate()
    //     //     .map(|(chunk_index, final_cts_poly)| Poly {
    //     //         offset: lookup_polys_offset + 1 + 2 * num_chunks + chunk_index,
    //     //         poly: final_cts_poly,
    //     //     })
    //     //     .collect_vec();

    //     // let e_polys = e_polys
    //     //     .into_iter()
    //     //     .enumerate()
    //     //     .map(|(memory_index, e_poly)| Poly {
    //     //         offset: lookup_polys_offset + 1 + 3 * num_chunks + memory_index,
    //     //         poly: e_poly,
    //     //     })
    //     //     .collect_vec();

    //     Ok((
    //         [e_polys, dims, read_ts_polys, final_cts_polys],
    //         (), // vec![
    //             //     // vec![lookup_output_comm],
    //             //     // dim_comms,
    //             //     // read_ts_comms,
    //             //     // final_cts_comms,
    //             //     // e_comms,
    //             // ],
    //     ))
    // }

    // fn e_polys<'a>(
    //     table: &Box<dyn DecomposableTable<F, E>>,
    //     subtable_polys: &[&BoxMultilinearPoly<F, E>],
    //     indices: &Vec<&[usize]>,
    // ) -> Vec<BoxMultilinearPoly<'a, F, E>> {
    //     let num_chunks = table.chunk_bits().len();
    //     let num_memories = table.num_memories();
    //     assert_eq!(indices.len(), num_chunks);
    //     let num_reads = indices[0].len();
    //     (0..num_memories)
    //         .map(|i| {
    //             let mut e_poly = Vec::with_capacity(num_reads);
    //             let subtable_poly = subtable_polys[table.memory_to_subtable_index(i)];
    //             let index = indices[table.memory_to_chunk_index(i)];
    //             (0..num_reads).for_each(|j| {
    //                 e_poly.push(subtable_poly[index[j]].clone());
    //             });
    //             box_dense_poly(e_poly)
    //         })
    //         .collect_vec()
    // }

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

    pub fn polynomialize<'a>(
        preprocessing: &LassoLookupsPreprocessing<F, E>,
        // ops: &Vec<JoltTraceStep<InstructionSet>>,
        lookup_index_poly: &BoxMultilinearPoly<F, E>,
    ) -> LassoPolynomials<'a, F, E> {
        let num_chunks = preprocessing.num_chunks;
        let num_memories = preprocessing.num_memories;
        let num_rows: usize = lookup_index_poly.len();

        let lookup = Lookups::iter().collect_vec()[0];
        let chunk_bits = lookup.chunk_bits(); // log2(M)

        assert_eq!(preprocessing.num_chunks, C);

        // subtable_lookup_indices : [[usize; num_rows]; num_chunks]
        let lookup_indexes = lookup_index_poly.as_dense().unwrap();
        let subtable_lookup_indices: Vec<Vec<usize>> =
            Self::subtable_lookup_indices(lookup_indexes, lookup);

        println!("num memories: {}", num_memories);

        let polys: Vec<_> = (0..preprocessing.num_memories)
            .into_par_iter()
            .map(|memory_index| {
                let dim_index = preprocessing.memory_to_dimension_index[memory_index];
                let subtable_index = preprocessing.memory_to_subtable_index[memory_index];
                let access_sequence: &Vec<usize> = &subtable_lookup_indices[dim_index];

                let memory_size = 1 << chunk_bits[memory_index];

                let mut final_cts_i = vec![0usize; memory_size]; // TODO: or should be lookup s
                let mut read_cts_i = vec![0usize; num_rows];
                let mut subtable_lookups = vec![F::ZERO; num_rows];

                for (j, op) in lookup_indexes.iter().enumerate() {
                    let memories_used =
                        &preprocessing.lookup_to_memory_indices[Lookups::enum_index(&lookup)];
                    if memories_used.contains(&memory_index) {
                        let memory_address = access_sequence[j];
                        println!("memory_address: {}", memory_address);
                        // debug_assert!(memory_address < M);

                        let counter = final_cts_i[memory_address];
                        read_cts_i[j] = counter;
                        final_cts_i[memory_address] = counter + 1;
                        // dims?
                        subtable_lookups[j] =
                            preprocessing.materialized_subtables[subtable_index][memory_address];
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

        let dims: Vec<_> = (0..num_chunks)
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
        lookup_outputs.resize(num_rows, F::ZERO);
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

    // fn subtable_lookup_indices(
    //     lookup_index_poly: &[F],
    //     lookup: impl LookupType,
    // ) -> Vec<Vec<usize>> {
    //     let m = lookup_index_poly.len().next_power_of_two();
    //     // let log_M = M.log_2();
    //     let chunked_indices: Vec<Vec<usize>> = lookup_index_poly
    //         .iter()
    //         .map(|e| lookup.to_indices(e))
    //         .collect();

    //     let mut subtable_lookup_indices: Vec<Vec<usize>> = Vec::with_capacity(C);
    //     for i in 0..C {
    //         let mut access_sequence: Vec<usize> =
    //             chunked_indices.iter().map(|chunks| chunks[i]).collect();
    //         access_sequence.resize(m, 0);
    //         subtable_lookup_indices.push(access_sequence);
    //     }
    //     subtable_lookup_indices
    // }

    fn subtable_lookup_indices(index_poly: &[F], lookup: impl LookupType) -> Vec<Vec<usize>> {
        let num_rows: usize = index_poly.len();
        let num_chunks = lookup.chunk_bits().len();
        println!("num_chunks: {}", num_chunks);

        let indices = (0..num_rows)
            .map(|i| {
                let mut index_bits = fe_to_bits_le(index_poly[i]);
                index_bits.truncate(lookup.chunk_bits().iter().sum());
                assert_eq!(
                    usize_from_bits_le(&fe_to_bits_le(index_poly[i])),
                    usize_from_bits_le(&index_bits)
                );

                let mut chunked_index = iter::repeat(0).take(num_chunks).collect_vec();
                let chunked_index_bits = lookup.subtable_indices(index_bits);
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

#[derive(Clone, Debug)]
pub struct LassoLookupsPreprocessing<F, E> {
    subtable_to_memory_indices: Vec<Vec<usize>>, // Vec<Range<usize>>?
    lookup_to_memory_indices: Vec<Vec<usize>>,
    memory_to_subtable_index: Vec<usize>,
    memory_to_dimension_index: Vec<usize>,
    materialized_subtables: Vec<Vec<F>>,
    num_memories: usize,
    num_chunks: usize, // C
    _marker: PhantomData<E>,
}

impl<F: PrimeField, E: ExtensionField<F>> LassoLookupsPreprocessing<F, E> {
    pub fn preprocess<
        const C: usize,
        const M: usize,
        Lookups: CircuitLookups,
        Subtables: SubtableSet<F, E>,
    >() -> Self {
        let materialized_subtables = Self::materialize_subtables::<M, Subtables>();

        // Build a mapping from subtable type => chunk indices that access that subtable type
        let mut subtable_indices: Vec<SubtableIndices> =
            vec![SubtableIndices::with_capacity(C); Subtables::COUNT];
            // println!("subtable_indices: {:?}", subtable_indices);
        for lookup in Lookups::iter() {
            for (subtable, indices) in lookup.subtables::<F, E>() {
                let subtable_idx = Subtables::enum_index(subtable);
                subtable_indices[subtable_idx].union_with(&indices);
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
            println!("memory_to_dimension_index: {:?}", dimension_indices.iter().collect_vec());
            memory_index += dimension_indices.len();
        }
        let num_memories = memory_index;

        // instruction is a type of lookup
        // assume all instreuctions are the same first
        let mut lookup_to_memory_indices = vec![vec![]; Lookups::COUNT];
        for lookup_type in Lookups::iter() {
            for (subtable, dimension_indices) in lookup_type.subtables::<F, E>() {
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
            num_chunks: C,
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
            dev::{rand_range, rand_vec, seeded_std_rng},
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

    subtable_enum!(
        RangeSubtables,
        Full: FullLimbSubtable<F, E, 16>
    );

    #[derive(Copy, Clone, Debug, EnumCount, EnumIter)]
    #[enum_dispatch(LookupType)]
    enum RangeLookups<const LIMB_BITS: usize> {
        Range128(RangeStategy<128, LIMB_BITS>),
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

        let preprocessing = LassoLookupsPreprocessing::<F, E>::preprocess::<
            8,
            { 1 << LIMB_BITS },
            RangeLookups<LIMB_BITS>,
            RangeSubtables<F, E>,
        >();

        let circuit = {
            let mut circuit = Circuit::default();

            let lookup_output = circuit.insert(InputNode::new(num_vars, 1));
            let lasso = circuit.insert(LassoNode::<_, _, RangeLookups<LIMB_BITS>, 8>::new(
                table, preprocessing, num_vars, LIMB_BITS,
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
