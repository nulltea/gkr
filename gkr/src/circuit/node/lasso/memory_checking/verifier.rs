use std::{collections::HashMap, iter, marker::PhantomData};

use ff_ext::{ff::PrimeField, ExtensionField};
use itertools::{chain, izip, Itertools};

use plonkish_backend::{pcs::Evaluation, poly::multilinear::MultilinearPolynomialTerms};

use crate::{
    circuit::node::{lasso::memory_checking::MemoryCheckingProver, DecomposableTable}, poly::{BoxMultilinearPoly, MultilinearPolyTerms}, sum_check::verify_sum_check, transcript::TranscriptRead, util::arithmetic::inner_product, Error
};

#[derive(Debug)]
pub struct Chunk<F> {
    chunk_index: usize,
    chunk_bits: usize,
    pub(crate) memory: Vec<Memory<F>>,
}

impl<F: PrimeField> Chunk<F> {
    pub fn chunk_polys_index(&self, offset: usize, num_chunks: usize) -> Vec<usize> {
        let dim_poly_index = offset + 1 + self.chunk_index;
        let read_ts_poly_index = offset + 1 + num_chunks + self.chunk_index;
        let final_cts_poly_index = offset + 1 + 2 * num_chunks + self.chunk_index;
        vec![dim_poly_index, read_ts_poly_index, final_cts_poly_index]
    }

    pub fn new(chunk_index: usize, chunk_bits: usize, memory: Memory<F>) -> Self {
        Self {
            chunk_index,
            chunk_bits,
            memory: vec![memory],
        }
    }

    pub fn num_memories(&self) -> usize {
        self.memory.len()
    }

    pub fn chunk_bits(&self) -> usize {
        self.chunk_bits
    }

    pub fn add_memory(&mut self, memory: Memory<F>) {
        self.memory.push(memory);
    }

    pub fn memory_indices(&self) -> Vec<usize> {
        self.memory
            .iter()
            .map(|memory| memory.memory_index)
            .collect_vec()
    }

    pub fn verify_memories<E: ExtensionField<F>>(
        &self,
        read_xs: &[E],
        write_xs: &[E],
        init_ys: &[E],
        final_read_ys: &[E],
        y: &[E],
        hash: impl Fn(&E, &E, &E) -> E,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<(E, E, E, Vec<E>), Error> {
        let [dim_x, read_ts_poly_x, final_cts_poly_y] =
            transcript.read_felts_as_exts(3)?.try_into().unwrap();
        let e_poly_xs = transcript.read_felts_as_exts(self.num_memories())?;
        let id_poly_y = inner_product(
            iter::successors(Some(E::ONE), |power_of_two| Some(power_of_two.double()))
                .take(y.len())
                .collect_vec()
                .into_iter(),
            y.to_vec(),
        );
        self.memory.iter().enumerate().for_each(|(i, memory)| {
            assert_eq!(read_xs[i], hash(&dim_x, &e_poly_xs[i], &read_ts_poly_x));
            assert_eq!(
                write_xs[i],
                hash(&dim_x, &e_poly_xs[i], &(read_ts_poly_x + F::ONE))
            );
            let subtable_poly_y = memory.subtable_poly.evaluate(y);
            assert_eq!(init_ys[i], hash(&id_poly_y, &subtable_poly_y, &E::ZERO));
            assert_eq!(
                final_read_ys[i],
                hash(&id_poly_y, &subtable_poly_y, &final_cts_poly_y)
            );
        });
        Ok((dim_x, read_ts_poly_x, final_cts_poly_y, e_poly_xs))
    }
}

#[derive(Debug)]
pub struct Memory<F> {
    memory_index: usize,
    subtable_poly: MultilinearPolyTerms<F>,
}

impl<F> Memory<F> {
    pub fn new(memory_index: usize, subtable_poly: MultilinearPolyTerms<F>) -> Self {
        Self {
            memory_index,
            subtable_poly,
        }
    }
}

#[derive(Debug)]
pub struct MemoryCheckingVerifier<F: PrimeField, E: ExtensionField<F>> {
    /// chunks with the same bits size
    chunks: Vec<Chunk<F>>,
    _marker: PhantomData<F>,
    _marker_e: PhantomData<E>,
}

impl<'a, F: PrimeField, E: ExtensionField<F>> MemoryCheckingVerifier<F, E> {
    pub fn new(chunks: Vec<Chunk<F>>) -> Self {
        Self {
            chunks,
            _marker: PhantomData,
            _marker_e: PhantomData,
        }
    }

    pub fn verify(
        &self,
        // num_chunks: usize,
        num_reads: usize,
        // polys_offset: usize,
        // points_offset: usize,
        gamma: &E,
        tau: &E,
        // lookup_opening_points: &mut Vec<Vec<F>>,
        // lookup_opening_evals: &mut Vec<Evaluation<F>>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<(), Error> {
        let num_memories: usize = self.chunks.iter().map(|chunk| chunk.num_memories()).sum();
        let memory_bits = self.chunks[0].chunk_bits();
        let (read_write_xs, x) = Self::verify_grand_product(
            num_reads,
            iter::repeat(None).take(2 * num_memories),
            transcript,
        )?;
        let (read_xs, write_xs) = read_write_xs.split_at(num_memories);

        let (init_final_read_ys, y) = Self::verify_grand_product(
            memory_bits,
            iter::repeat(None).take(2 * num_memories),
            transcript,
        )?;
        let (init_ys, final_read_ys) = init_final_read_ys.split_at(num_memories);

        let hash = |a: &E, v: &E, t: &E| -> E { *a + *v * gamma + *t * gamma.square() - tau };
        let mut offset = 0;
        let (dim_xs, read_ts_poly_xs, final_cts_poly_ys, e_poly_xs) = self
            .chunks
            .iter()
            .map(|chunk| {
                let num_memories = chunk.num_memories();
                let result = chunk.verify_memories(
                    &read_xs[offset..offset + num_memories],
                    &write_xs[offset..offset + num_memories],
                    &init_ys[offset..offset + num_memories],
                    &final_read_ys[offset..offset + num_memories],
                    &y,
                    hash,
                    transcript,
                );
                offset += num_memories;
                result
            })
            .collect::<Result<Vec<(E, E, E, Vec<E>)>, Error>>()?
            .into_iter()
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<Vec<_>>)>();

        // self.opening_evals(
        //     num_chunks,
        //     polys_offset,
        //     points_offset,
        //     &lookup_opening_points,
        //     lookup_opening_evals,
        //     &dim_xs,
        //     &read_ts_poly_xs,
        //     &final_cts_poly_ys,
        //     &e_poly_xs.concat(),
        // );
        // lookup_opening_points.extend_from_slice(&[x, y]);

        Ok(())
    }

    fn verify_grand_product(
        num_vars: usize,
        claimed_v_0s: impl IntoIterator<Item = Option<E>>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<(Vec<E>, Vec<E>), Error> {
        let claimed_v_0s = claimed_v_0s.into_iter().collect_vec();
        let num_batching = claimed_v_0s.len();

        assert!(num_batching != 0);
        let claimed_v_0s = {
            claimed_v_0s
                .into_iter()
                .map(|claimed| match claimed {
                    Some(claimed) => {
                        transcript.common_felts(&claimed.as_bases());
                        Ok(claimed)
                    }
                    None => transcript.read_felt_ext(),
                })
                .try_collect::<_, Vec<_>, _>()?
        };

        (0..num_vars).try_fold((claimed_v_0s, Vec::new()), |result, num_vars| {
            let (claimed_v_ys, y) = result;

            let (mut x, evals) = if num_vars == 0 {
                let evals = transcript.read_felt_exts(2 * num_batching)?;
                for (claimed_v, (&v_l, &v_r)) in izip!(claimed_v_ys, evals.iter().tuples()) {
                    if claimed_v != v_l * v_r {
                        return Err(Error::InvalidSumCheck(
                            "unmatched sum check output".to_string(),
                        ));
                    }
                }

                (Vec::new(), evals)
            } else {
                let gamma = transcript.squeeze_challenge();
                let g = MemoryCheckingProver::sum_check_function(num_vars, num_batching, gamma);

                let (_x_eval, x) = {
                    let claim = MemoryCheckingProver::sum_check_claim(&claimed_v_ys, gamma);
                    verify_sum_check(&g, claim, transcript)?
                };

                let evals = transcript.read_felt_exts(2 * num_batching)?;

                // let eval_by_query = eval_by_query(&evals);
                // if x_eval != evaluate(&expression, num_vars, &eval_by_query, &[gamma], &[&y], &x) {
                //     return Err(Error::InvalidSumCheck("unmatched sum check output".to_string()));
                // }

                (x, evals)
            };

            let mu = transcript.squeeze_challenge();

            let v_xs = MemoryCheckingProver::layer_down_claim(&evals, mu);
            x.push(mu);

            Ok((v_xs, x))
        })
    }

    // fn opening_evals(
    //     &self,
    //     num_chunks: usize,
    //     polys_offset: usize,
    //     points_offset: usize,
    //     lookup_opening_points: &Vec<Vec<F>>,
    //     lookup_opening_evals: &mut Vec<Evaluation<F>>,
    //     dim_xs: &[F],
    //     read_ts_poly_xs: &[F],
    //     final_cts_poly_ys: &[F],
    //     e_poly_xs: &[F],
    // ) {
    //     let x_offset = points_offset + lookup_opening_points.len();
    //     let y_offset = x_offset + 1;
    //     let (dim_xs, read_ts_poly_xs, final_cts_poly_ys) = self
    //         .chunks
    //         .iter()
    //         .enumerate()
    //         .map(|(i, chunk)| {
    //             let chunk_polys_index = chunk.chunk_polys_index(polys_offset, num_chunks);
    //             (
    //                 Evaluation::new(chunk_polys_index[0], x_offset, dim_xs[i]),
    //                 Evaluation::new(chunk_polys_index[1], x_offset, read_ts_poly_xs[i]),
    //                 Evaluation::new(chunk_polys_index[2], y_offset, final_cts_poly_ys[i]),
    //             )
    //         })
    //         .multiunzip::<(Vec<Evaluation<F>>, Vec<Evaluation<F>>, Vec<Evaluation<F>>)>();

    //     let e_poly_offset = polys_offset + 1 + 3 * num_chunks;
    //     let e_poly_xs = self
    //         .chunks
    //         .iter()
    //         .flat_map(|chunk| chunk.memory_indices())
    //         .zip(e_poly_xs)
    //         .map(|(memory_index, &e_poly_x)| {
    //             Evaluation::new(e_poly_offset + memory_index, x_offset, e_poly_x)
    //         })
    //         .collect_vec();
    //     lookup_opening_evals.extend_from_slice(
    //         &chain!(dim_xs, read_ts_poly_xs, final_cts_poly_ys, e_poly_xs).collect_vec(),
    //     );
    // }
}

fn chunks<F: PrimeField, E: ExtensionField<F>>(
    table: &Box<dyn DecomposableTable<F, E>>,
) -> Vec<Chunk<F>> {
    let num_memories = table.num_memories();
    let chunk_bits = table.chunk_bits();
    let subtable_polys = table.subtable_polys_terms();
    // key: chunk index, value: chunk
    let mut chunk_map: HashMap<usize, Chunk<F>> = HashMap::new();
    (0..num_memories).for_each(|memory_index| {
        let chunk_index = table.memory_to_chunk_index(memory_index);
        let chunk_bits = chunk_bits[chunk_index];
        let subtable_poly = &subtable_polys[table.memory_to_subtable_index(memory_index)];
        let memory = Memory::new(memory_index, subtable_poly.clone());
        if chunk_map.get(&chunk_index).is_some() {
            chunk_map.entry(chunk_index).and_modify(|chunk| {
                chunk.add_memory(memory);
            });
        } else {
            chunk_map.insert(chunk_index, Chunk::new(chunk_index, chunk_bits, memory));
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

pub fn prepare_memory_checking<F: PrimeField, E: ExtensionField<F>>(
    table: &Box<dyn DecomposableTable<F, E>>,
) -> Vec<MemoryCheckingVerifier<F, E>> {
    let chunks = chunks(table);
    let chunk_bits = table.chunk_bits();
    // key: chunk_bits, value: chunks
    let mut chunk_map = HashMap::<usize, Vec<Chunk<F>>>::new();
    chunks
        .into_iter()
        .enumerate()
        .for_each(|(chunk_index, chunk)| {
            let chunk_bits = chunk_bits[chunk_index];
            if chunk_map.get(&chunk_bits).is_some() {
                chunk_map.entry(chunk_bits).and_modify(|chunks| {
                    chunks.push(chunk);
                });
            } else {
                chunk_map.insert(chunk_bits, vec![chunk]);
            }
        });
    chunk_map
        .into_iter()
        .enumerate()
        .map(|(_, (_, chunks))| MemoryCheckingVerifier::new(chunks))
        .collect_vec()
}
