use std::{
    collections::BTreeSet,
    iter::{self, repeat},
    marker::PhantomData,
};

use ff_ext::{
    ff::{Field, PrimeField},
    ExtensionField,
};
use itertools::{chain, Itertools};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};

use plonkish_backend::{
    pcs::{CommitmentChunk, Evaluation, PolynomialCommitmentScheme},
    piop::sum_check::{
        classic::{ClassicSumCheck, EvaluationsProver},
        SumCheck as _, VirtualPolynomial,
    },
    poly::multilinear::MultilinearPolynomial,
    util::arithmetic::{fe_to_bits_le, usize_from_bits_le},
};

use crate::{
    circuit::node::{lasso::table::DecomposableTable, CombinedEvalClaim, EvalClaim},
    poly::{box_dense_poly, BoxMultilinearPoly},
    sum_check::{generic::Generic, prove_sum_check, quadratic::Quadratic, SumCheckPoly},
    transcript::TranscriptWrite,
    util::expression::Expression,
    Error,
};

type SumCheck<F> = ClassicSumCheck<EvaluationsProver<F>>;

pub struct Surge<
    F: Field + PrimeField,
    E: ExtensionField<F>,
    // Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
> {
    lookup_indices: Vec<Vec<usize>>,
    _marker: PhantomData<F>,
    // _marker2: PhantomData<Pcs>,
    _marker3: PhantomData<E>,
}

impl<
        F: Field + PrimeField,
        E: ExtensionField<F>,
        // Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    > Surge<F, E>
{
    pub fn new() -> Self {
        Self {
            lookup_indices: vec![vec![]],
            _marker: PhantomData,
            // _marker2: PhantomData,
            _marker3: PhantomData,
        }
    }

    pub fn indices(&'_ self) -> Vec<&[usize]> {
        self.lookup_indices
            .iter()
            .map(|lookup_indices| lookup_indices.as_slice())
            .collect_vec()
    }

    /// computes dim_1, ..., dim_c where c == DecomposableTable::C
    pub fn commit<'a>(
        &mut self,
        table: &Box<dyn DecomposableTable<F, E>>,
        index_poly: &BoxMultilinearPoly<'a, F, E>,
    ) -> Vec<BoxMultilinearPoly<'a, F, E>> {
        let num_rows: usize = 1 << index_poly.num_vars();
        let num_chunks = table.chunk_bits().len();
        // get indices of non-zero columns of all rows where each index is chunked
        let indices = (0..num_rows)
            .map(|i| {
                let mut index_bits = fe_to_bits_le(index_poly[i]);
                index_bits.truncate(table.chunk_bits().iter().sum());
                assert_eq!(
                    usize_from_bits_le(&fe_to_bits_le(index_poly[i])),
                    usize_from_bits_le(&index_bits)
                );

                let mut chunked_index = repeat(0).take(num_chunks).collect_vec();
                let chunked_index_bits = table.subtable_indices(index_bits);
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
        let mut dims = Vec::with_capacity(num_chunks);
        self.lookup_indices.resize(num_chunks, vec![]);
        self.lookup_indices
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
                dims.push(box_dense_poly(indices.into_iter().map(|i| F::from_u128(i as u128)).collect_vec()));
            });

        dims
    }

    pub fn counter_polys<'a>(
        &self,
        table: &Box<dyn DecomposableTable<F, E>>,
    ) -> (Vec<BoxMultilinearPoly<'a, F, E>>, Vec<BoxMultilinearPoly<'a, F, E>>) {
        let num_chunks = table.chunk_bits().len();
        let mut read_ts_polys = Vec::with_capacity(num_chunks);
        let mut final_cts_polys = Vec::with_capacity(num_chunks);
        let chunk_bits = table.chunk_bits();
        self.lookup_indices
            .iter()
            .enumerate()
            .for_each(|(i, lookup_indices)| {
                let num_reads = lookup_indices.len();
                let memory_size = 1 << chunk_bits[i];
                let mut final_timestamps = vec![0usize; memory_size];
                let mut read_timestamps = vec![0usize; num_reads];
                (0..num_reads).for_each(|i| {
                    let memory_address = lookup_indices[i];
                    let ts = final_timestamps[memory_address];
                    read_timestamps[i] = ts;
                    let write_timestamp = ts + 1;
                    final_timestamps[memory_address] = write_timestamp;
                });
                read_ts_polys.push(box_dense_poly(read_timestamps.into_iter().map(|t| F::from_u128(t as u128)).collect_vec()));
                final_cts_polys.push(box_dense_poly(final_timestamps.into_iter().map(|t| F::from_u128(t as u128)).collect_vec()));
            });

        (read_ts_polys, final_cts_polys)
    }

    pub fn prove_sum_check(
        table: &Box<dyn DecomposableTable<F, E>>,
        lookup_output_poly: &BoxMultilinearPoly<F, E>,
        e_polys: &[BoxMultilinearPoly<F, E>],
        r: &[E],
        num_vars: usize,
        // points_offset: usize,
        // lookup_opening_points: &mut Vec<Vec<F>>,
        // lookup_opening_evals: &mut Vec<Evaluation<F>>,
        transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let claimed_sum = Self::sum_check_claim(r, table, e_polys);
        assert_eq!(claimed_sum, lookup_output_poly.evaluate(r));

        transcript.write_felt_ext(&claimed_sum)?;

        // proceed sumcheck
        // let (x, evals) = SumCheck::prove(
        //     &(),
        //     num_vars,
        //     VirtualPolynomial::new(
        //         &expression,
        //         e_polys.iter().map(|e_poly| &e_poly.poly),
        //         &[],
        //         &[r.to_vec()],
        //     ),
        //     claimed_sum,
        //     transcript,
        // )?;

        let g = Self::sum_check_function(table, num_vars);

        // let polys = {
        //     let eq_r_x = box_dense_poly(eq_poly(&r_x, E::ONE));
        //     let omegas = box_dense_poly(self.omegas(layer).copied().collect::<Vec<_>>());
        //     let w_interms = w_interms
        //         .iter()
        //         .map(|w_interms| w_interms.iter().nth_back(layer).unwrap())
        //         .map(|w| repeated_dense_poly(w, 1));
        //     chain![
        //         [SumCheckPoly::Extension(eq_r_x), SumCheckPoly::Base(omegas)],
        //         w_interms.map(SumCheckPoly::Extension)
        //     ]
        //     .collect_vec()
        // };

        println!("e_polys: {}", e_polys.len());

        let polys = e_polys
            .iter()
            .map(|e_poly| SumCheckPoly::Base::<_, _, _, BoxMultilinearPoly<E, E>>(e_poly))
            .collect_vec();

        let (claim, r_x_prime, e_polys_evals) = prove_sum_check(&g, claimed_sum, polys, transcript)?;

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
        r: &[E], // claim: CombinedEvalClaim<E>,
        table: &Box<dyn DecomposableTable<F, E>>,
        e_polys: &[BoxMultilinearPoly<F, E>],
    ) -> E {
        let num_memories = table.num_memories();
        assert_eq!(e_polys.len(), num_memories);
        let num_vars = e_polys[0].num_vars();
        let bh_size = 1 << num_vars;
        let eq = MultilinearPolynomial::eq_xy(r);
        // \sum_{k \in \{0, 1\}^{\log m}} (\tilde{eq}(r, k) * g(E_1(k), ..., E_{\alpha}(k)))
        let claim = (0..bh_size)
            .into_par_iter()
            .map(|k| {
                let operands = e_polys.iter().map(|e_poly| e_poly[k]).collect_vec();
                eq[k] * table.combine_lookups(&operands)
            })
            .sum();

        claim
    }

    // (\tilde{eq}(r, k) * g(E_1(k), ..., E_{\alpha}(k)))
    pub fn sum_check_function(
        table: &Box<dyn DecomposableTable<F, E>>,
        num_vars: usize,
    ) -> Generic<F, E> {
        let num_memories = table.num_memories();
        let exprs = table.combine_lookup_expressions(
            (0..num_memories)
                .map(|idx| Expression::poly(idx))
                .collect_vec(),
        );

        let eq_r_x = &Expression::poly(0);

        Generic::new(num_vars, &(eq_r_x * exprs))
    }
}
