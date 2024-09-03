use std::iter;

use ff_ext::{ff::PrimeField, ExtensionField};
use itertools::{chain, Itertools};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use crate::{poly::{box_dense_poly, BoxMultilinearPoly}, transcript::TranscriptWrite, Error};

use super::{Chunk, MemoryGKR};

pub struct MemoryCheckingProver<'a, F: PrimeField, E: ExtensionField<F>> {
    /// chunks with the same bits size
    chunks: Vec<Chunk<'a, F, E>>,
    /// GKR initial polynomials for each memory
    pub memories: Vec<MemoryGKR<'a, F, E>>,
}

impl<'a, F: PrimeField, E: ExtensionField<F>> MemoryCheckingProver<'a, F, E> {
    // T_1[dim_1(x)], ..., T_k[dim_1(x)],
    // ...
    // T_{\alpha-k+1}[dim_c(x)], ..., T_{\alpha}[dim_c(x)]
    pub fn new(chunks: Vec<Chunk<'a, F, E>>, tau: &F, gamma: &F) -> Self {
        let num_reads = chunks[0].num_reads();
        println!("num_reads: {}", num_reads);
        let memory_size = 1 << chunks[0].chunk_bits();
        println!("memory_size: {}", memory_size);

        let hash = |a: &F, v: &F, t: &F| -> F { *a + *v * gamma + *t * gamma.square() - tau };

        let memories_gkr: Vec<MemoryGKR<F, E>> = (0..chunks.len())
            .into_par_iter()
            .flat_map(|i| {
                let chunk = &chunks[i];
                let chunk_polys = chunk.chunk_polys().collect_vec();
                let (dim, read_ts_poly, final_cts_poly) =
                    (chunk_polys[0], chunk_polys[1], chunk_polys[2]);
                chunk
                    .memories()
                    .map(|memory| {
                        let memory_polys = memory.polys().collect_vec();
                        let (subtable_poly, e_poly) = (memory_polys[0], memory_polys[1]);
                        let mut init = vec![];
                        let mut read = vec![];
                        let mut write = vec![];
                        let mut final_read = vec![];
                        (0..memory_size).for_each(|i| {
                            init.push(hash(&F::from(i as u64), &subtable_poly[i], &F::ZERO));
                            final_read.push(hash(
                                &F::from(i as u64),
                                &subtable_poly[i],
                                &final_cts_poly[i],
                            ));
                        });
                        (0..num_reads).for_each(|i| {
                            read.push(hash(&dim[i], &e_poly[i], &read_ts_poly[i]));
                            write.push(hash(&dim[i], &e_poly[i], &(read_ts_poly[i] + F::ONE)));
                        });
                        MemoryGKR::new(
                            box_dense_poly(init),
                            box_dense_poly(read),
                            box_dense_poly(write),
                            box_dense_poly(final_read),
                        )
                    })
                    .collect_vec()
            })
            .collect();

        Self {
            chunks,
            memories: memories_gkr,
        }
    }

    pub fn inits(&self) -> impl Iterator<Item = &BoxMultilinearPoly<'a, F, E>> {
        self.memories.iter().map(|memory| &memory.init)
    }

    pub fn reads(&self) -> impl Iterator<Item = &BoxMultilinearPoly<'a, F, E>> {
        self.memories.iter().map(|memory| &memory.read)
    }

    pub fn writes(&self) -> impl Iterator<Item = &BoxMultilinearPoly<'a, F, E>> {
        self.memories.iter().map(|memory| &memory.write)
    }

    pub fn final_reads(&self) -> impl Iterator<Item = &BoxMultilinearPoly<'a, F, E>> {
        self.memories.iter().map(|memory| &memory.final_read)
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            &BoxMultilinearPoly<F, E>,
            &BoxMultilinearPoly<F, E>,
            &BoxMultilinearPoly<F, E>,
            &BoxMultilinearPoly<F, E>,
        ),
    > {
        self.memories.iter().map(|memory| {
            (
                &memory.init,
                &memory.read,
                &memory.write,
                &memory.final_read,
            )
        })
    }

    pub fn claimed_v_0s(&self) -> impl IntoIterator<Item = Vec<Option<F>>> {
        let (claimed_read_0s, claimed_write_0s, claimed_init_0s, claimed_final_read_0s) = self
            .iter()
            .map(|(init, read, write, final_read)| {
                let claimed_init_0 = init.to_dense().iter().product();
                let claimed_read_0 = read.to_dense().iter().product();
                let claimed_write_0 = write.to_dense().iter().product();
                let claimed_final_read_0 = final_read.to_dense().iter().product();

                // sanity check
                debug_assert_eq!(
                    claimed_init_0 * claimed_write_0,
                    claimed_read_0 * claimed_final_read_0,
                    "Multiset hashes don't match",
                );
                (
                    Some(claimed_read_0),
                    Some(claimed_write_0),
                    Some(claimed_init_0),
                    Some(claimed_final_read_0),
                )
            })
            .multiunzip::<(Vec<_>, Vec<_>, Vec<_>, Vec<_>)>();
        chain!([
            claimed_read_0s,
            claimed_write_0s,
            claimed_init_0s,
            claimed_final_read_0s
        ])
    }

    pub fn prove(
        &mut self,
        points_offset: usize,
        // lookup_opening_points: &mut Vec<Vec<F>>,
        // lookup_opening_evals: &mut Vec<Evaluation<F>>,
        transcript: &mut impl TranscriptWrite<F, E>,
    ) -> Result<(), Error> {

        // let (_, x) = prove_grand_product(
        //     iter::repeat(None).take(self.memories.len() * 2),
        //     chain!(self.reads(), self.writes()),
        //     transcript,
        // )?;

        // let (_, y) = prove_grand_product(
        //     iter::repeat(None).take(self.memories.len() * 2),
        //     chain!(self.inits(), self.final_reads()),
        //     transcript,
        // )?;

        // assert_eq!(
        //     points_offset + lookup_opening_points.len(),
        //     self.points_offset
        // );
        // let x_offset = points_offset + lookup_opening_points.len();
        // let y_offset = x_offset + 1;
        // let (dim_xs, read_ts_poly_xs, final_cts_poly_ys, e_poly_xs) = self
        //     .chunks
        //     .iter()
        //     .map(|chunk| {
        //         let chunk_poly_evals = chunk.chunk_poly_evals(&x, &y);
        //         let e_poly_xs = chunk.e_poly_evals(&x);
        //         transcript.write_felt_exts(&chunk_poly_evals).unwrap();
        //         transcript.write_felt_exts(&e_poly_xs).unwrap();

        //         (
        //             Evaluation::new(chunk.dim.offset, x_offset, chunk_poly_evals[0]),
        //             Evaluation::new(chunk.read_ts_poly.offset, x_offset, chunk_poly_evals[1]),
        //             Evaluation::new(chunk.final_cts_poly.offset, y_offset, chunk_poly_evals[2]),
        //             chunk
        //                 .memories()
        //                 .enumerate()
        //                 .map(|(i, memory)| {
        //                     Evaluation::new(memory.e_poly.offset, x_offset, e_poly_xs[i])
        //                 })
        //                 .collect_vec(),
        //         )
        //     })
        //     .multiunzip::<(
        //         Vec<Evaluation<F>>,
        //         Vec<Evaluation<F>>,
        //         Vec<Evaluation<F>>,
        //         Vec<Vec<Evaluation<F>>>,
        //     )>();

        // lookup_opening_points.extend_from_slice(&[x, y]);
        // let opening_evals = chain!(
        //     dim_xs,
        //     read_ts_poly_xs,
        //     final_cts_poly_ys,
        //     e_poly_xs.concat()
        // )
        // .collect_vec();
        // lookup_opening_evals.extend_from_slice(&opening_evals);

        Ok(())
    }
}
