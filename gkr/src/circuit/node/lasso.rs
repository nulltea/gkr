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
    log2_t_size: usize,
    table: Box<dyn DecomposableTable<F, E>>,
}

impl<F: PrimeField, E: ExtensionField<F>> Node<F, E> for LassoNode<F, E> {
    fn is_input(&self) -> bool {
        false
    }

    fn log2_input_size(&self) -> usize {
        self.log2_t_size
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
        // let subtable_polys = table.subtable_polys();
        // let subtable_polys = subtable_polys.iter().collect_vec();
        // let subtable_polys = subtable_polys.as_slice();

        // println!("combined claim {} {:?}", claim.points.len(), claim.value);

        let num_chunks = table.chunk_bits().len();

        let (lookup_output_poly, inputs) = inputs.split_first().unwrap();
        // let (dims, inputs) = inputs.split_at(num_chunks);
        // let (read_ts_polys, inputs) = inputs.split_at(num_chunks);
        // let (final_cts_polys, inputs) = inputs.split_at(num_chunks);
        let e_polys = inputs;
        assert!(e_polys.len() == table.num_memories());

        let num_vars = lookup_output_poly.num_vars();
        assert!(num_vars == self.log2_t_size);

        // should this be passed from CombinedEvalClaim?
        let r = transcript.squeeze_challenges(num_vars);

        // let mut lookup_opening_points = vec![];
        // let mut lookup_opening_evals = vec![];

        let res = Surge::<F, E>::prove_sum_check(
            &table,
            lookup_output_poly,
            e_polys,
            &r,
            num_vars,
            // points_offset,
            // lookup_opening_points,
            // lookup_opening_evals,
            transcript,
        )?;

        println!("Proved Surge sum check");

        // let [beta, gamma] = transcript.squeeze_challenges(2).try_into().unwrap();

        // // memory_checking
        // LassoProver::<F, Pcs>::memory_checking(
        //     0, //pp.lookup_points_offset,
        //     &mut lookup_opening_points,
        //     &mut lookup_opening_evals,
        //     table,
        //     subtable_polys,
        //     dims,
        //     read_ts_polys,
        //     final_cts_polys,
        //     e_polys,
        //     &beta,
        //     &gamma,
        //     transcript,
        // ).unwrap();

        Ok(res)
    }

    fn verify_claim_reduction(
        &self,
        _: CombinedEvalClaim<E>,
        transcript: &mut dyn TranscriptRead<F, E>,
    ) -> Result<Vec<Vec<EvalClaim<E>>>, Error> {
        let table = self.table.clone();
        let num_vars = self.log2_t_size;
        let r = transcript.squeeze_challenges(num_vars);

        let g = Surge::<F, E>::sum_check_function(&table, num_vars);
        let claimed_sum = transcript.read_felt_ext()?;

        let (sub_claim, r_x_prime) = verify_sum_check(&g, claimed_sum, transcript)?;

        let e_polys_eval = transcript.read_felt_exts(table.num_memories())?;

        // // Round n+1
        // let [beta, gamma] = transcript.squeeze_challenges(2).try_into().unwrap();

        // // memory checking
        // LassoVerifier::<F, Pcs>::memory_checking(
        //     num_vars,
        //     0, //vp.lookup_polys_offset,
        //     0, //vp.lookup_points_offset,
        //     lookup_opening_points,
        //     lookup_opening_evals,
        //     table,
        //     &beta,
        //     &gamma,
        //     transcript,
        // )?;

        Ok(chain![
            iter::once(vec![EvalClaim::new(r.to_vec(), claimed_sum)]),
            e_polys_eval
                .into_iter()
                .map(|e| vec![EvalClaim::new(r_x_prime.clone(), e)])
        ]
        .collect_vec())
    }
}

impl<F, E> LassoNode<F, E> {
    pub fn new(table: Box<dyn DecomposableTable<F, E>>, log2_t_size: usize) -> Self {
        Self { log2_t_size, table }
    }
}

pub struct Lasso<F, E, Pcs>(PhantomData<F>, PhantomData<E>, PhantomData<Pcs>);

impl<
        F: Field + PrimeField,
        E: ExtensionField<F>,
        Pcs: PolynomialCommitmentScheme<F, Polynomial = MultilinearPolynomial<F>>,
    > Lasso<F, E, Pcs>
{
    pub fn commit<'a>(
        // pp: &Pcs::ProverParam,
        // lookup_polys_offset: usize,
        table: &Box<dyn DecomposableTable<F, E>>,
        subtable_polys: &[&BoxMultilinearPoly<'static, F, E>],
        lookup_output_poly: BoxMultilinearPoly<'a, F, E>,
        lookup_index_poly: BoxMultilinearPoly<'a, F, E>,
        // transcript: &mut dyn TranscriptWrite<F, E>,
    ) -> Result<
        (
            [Vec<BoxMultilinearPoly<'a, F, E>>; 5],
            Vec<Vec<Pcs::Commitment>>,
        ),
        plonkish_backend::Error,
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
            [
                vec![lookup_output_poly],
                e_polys,
                dims,
                read_ts_polys,
                final_cts_polys,
            ],
            vec![
                // vec![lookup_output_comm],
                // dim_comms,
                // read_ts_comms,
                // final_cts_comms,
                // e_comms,
            ],
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
        gamma: &F,
        tau: &F,
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

struct Layer<'a, F, E> {
    v_l: BoxMultilinearPoly<'a, F, E>,
    v_r: BoxMultilinearPoly<'a, F, E>,
}

impl<'a, F: PrimeField, E: ExtensionField<F>> From<[Vec<F>; 2]> for Layer<'a, F, E> {
    fn from(values: [Vec<F>; 2]) -> Self {
        let [v_l, v_r] = values.map(box_dense_poly);
        Self { v_l, v_r }
    }
}

impl<'a, F: PrimeField, E: ExtensionField<F>> Layer<'a, F, E> {
    fn bottom<'b>(v: &BoxMultilinearPoly<'a, F, E>) -> Layer<'b, F, E> {
        let mid = v.to_dense().len() >> 1;
        [&v.to_dense()[..mid], &v.to_dense()[mid..]]
            .map(ToOwned::to_owned)
            .into()
    }

    fn num_vars(&self) -> usize {
        self.v_l.num_vars()
    }

    fn polys(&self) -> [&BoxMultilinearPoly<'a, F, E>; 2] {
        [&self.v_l, &self.v_r]
    }

    fn poly_chunks(&self, chunk_size: usize) -> impl Iterator<Item = (&[F], &[F])> {
        let [v_l, v_r] = self
            .polys()
            .map(|poly| poly.as_dense().unwrap().chunks(chunk_size));
        izip!(v_l, v_r)
    }

    fn up<'b>(&self) -> Layer<'b, F, E> {
        assert!(self.num_vars() != 0);

        let len = 1 << self.num_vars();
        let chunk_size = div_ceil(len, num_threads()).next_power_of_two();

        let mut outputs: [_; 2] = array::from_fn(|_| vec![F::ZERO; len >> 1]);
        let (v_up_l, v_up_r) = outputs.split_at_mut(1);

        parallelize_iter(
            izip!(
                chain![v_up_l, v_up_r].flat_map(|v_up| v_up.chunks_mut(chunk_size)),
                self.poly_chunks(chunk_size),
            ),
            |(v_up, (v_l, v_r))| {
                izip!(v_up, v_l, v_r).for_each(|(v_up, v_l, v_r)| {
                    *v_up = *v_l * *v_r;
                })
            },
        );

        outputs.into()
    }
}

#[cfg(test)]
pub mod test {
    use crate::{
        circuit::{
            node::{input::InputNode, lasso::Layer, log_up::LogUpNode, VanillaGate, VanillaNode},
            test::{run_circuit, TestData},
            Circuit,
        },
        poly::{box_dense_poly, box_owned_dense_poly, BoxMultilinearPoly, BoxMultilinearPolyOwned},
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
    use std::{iter, marker::PhantomData};

    use super::{table::DecomposableTable, Lasso, LassoNode};
    use halo2_curves::bn256;
    use rand::Rng;

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

        let table: Box<dyn DecomposableTable<F, E>> = Box::new(RangeTable::<F, E, 128, 16>::new());

        let subtable_polys = table.subtable_polys();
        let subtable_polys = subtable_polys.iter().collect_vec();
        let subtable_polys = subtable_polys.as_slice();

        let evals = iter::repeat_with(|| F::from_u128(rng.gen_range(0..(1 << 64))))
            .take(size)
            .collect_vec();
        let input = box_dense_poly(evals.clone());
        let output = box_dense_poly(evals);

        let (polys, _) =
            Lasso::<F, E, Brakedown<F>>::commit(&table, subtable_polys, input, output).unwrap();

        let [lookup_output_poly, e_polys, dims, read_ts_polys, final_cts_polys] = polys;

        let [beta, gamma] = [F::random(&mut rng), F::random(&mut rng)];

        let mut memory_checking = Lasso::<F, E, Brakedown<F>>::prepare_memory_checking(
            &table,
            subtable_polys,
            &dims,
            &read_ts_polys,
            &final_cts_polys,
            &e_polys,
            &beta,
            &gamma,
        );

        let memory_checking = memory_checking.pop().unwrap();
        let num_batching = memory_checking.memories.len() * 2;

        let vs = chain!(memory_checking.reads(), memory_checking.writes());

        let bottom_layers = vs.map(Layer::bottom).collect_vec();
        let layers = iter::successors(bottom_layers.into(), |layers| {
            (layers[0].num_vars() > 0).then(|| layers.iter().map(Layer::up).collect())
        })
        .collect_vec();

        println!(
            "layers: {:?}",
            layers
                .iter()
                .map(|l| l
                    .iter()
                    .map(|e| [e.v_l.num_vars(), e.v_r.num_vars()])
                    .collect_vec())
                .collect_vec()
        );

        let circuit = {
            let mut circuit = Circuit::default();

            // let inputs = chain![&lookup_output_poly, &e_polys]
            //     .map(|poly| circuit.insert(InputNode::new(num_vars, 1)))
            //     .collect_vec();
            // println!("inputs: {:?}", inputs.len());
            // let lasso = circuit.insert(LassoNode::new(table, num_vars));
            // inputs
            //     .into_iter()
            //     .for_each(|from| circuit.connect(from, lasso));

            let mut prev_layer = None;
            #[allow(clippy::never_loop)]
            for (i, layers) in layers.iter().enumerate() {
                let mut new_layer = vec![];
                for layer in layers.iter() {
                    let gates = vec![VanillaGate::mul((0, 0), (1, 0))];
                    let [left, right] = prev_layer.unwrap_or_else(|| {
                        println!("input node vars {:?}", layer.polys().map(|p| p.num_vars()));
                        layer
                            .polys()
                            .map(|p| circuit.insert(InputNode::new(p.num_vars(), 1)))
                    });
                    println!("vanilla node reps {}", num_vars - 2 - i);
                    let product =
                        circuit.insert(VanillaNode::new(2, 1, gates, 1 << (num_vars - 2 - i)));
                    circuit.connect(left, product);
                    circuit.connect(right, product);
                    new_layer.push(product);
                }
                prev_layer = Some(new_layer.try_into().unwrap());
                // break;
            }

            layers
            .iter()
            .fold(Ok(Vec::new()), |result, layers| {
                let polys = layers.iter().flat_map(|layer| layer.polys());

            });

            circuit
        };

        // let values = chain![lookup_output_poly, e_polys].collect_vec();

        let inputs = layers[0]
            .iter()
            .flat_map(|layer| {
                [
                    box_dense_poly(layer.v_l.to_dense()),
                    box_dense_poly(layer.v_r.to_dense()),
                ]
            })
            .collect_vec();

        let values = layers
            .into_iter()
            .flat_map(|layers| layers.into_iter().flat_map(|layer| [layer.v_l, layer.v_r]))
            .collect_vec();

        // let values = vec![];

        (circuit, inputs, None)
    }
}
