use crate::{
    sum_check::{
        eq_poly, err_unmatched_sum_check_evaluation, evaluate_multilinear_poly,
        prove_quadratic_sum_check, verify_quadratic_sum_check,
    },
    transcript::{TranscriptRead, TranscriptWrite},
    Error,
};
use ff::Field;
use itertools::{izip, Itertools};

#[derive(Clone, Debug)]
pub struct Layer<F> {
    input_num_vars: usize,
    output_num_vars: usize,
    adds: Vec<(F, usize, usize)>,
    muls: Vec<(F, usize, usize, usize)>,
}

impl<F: Field> Layer<F> {
    pub fn new(
        input_num_vars: usize,
        output_num_vars: usize,
        adds: Vec<(F, usize, usize)>,
        muls: Vec<(F, usize, usize, usize)>,
    ) -> Self {
        Self {
            input_num_vars,
            output_num_vars,
            adds,
            muls,
        }
    }

    pub fn input_num_vars(&self) -> usize {
        self.input_num_vars
    }

    pub fn output_num_vars(&self) -> usize {
        self.output_num_vars
    }

    fn phase1_wiring(&self, eq_r: &[F], input: &[F]) -> Vec<F> {
        let mut wiring = vec![F::ZERO; 1 << self.input_num_vars];
        self.adds.iter().for_each(|(scalar, lhs, out)| {
            wiring[*lhs] += *scalar * eq_r[*out];
        });
        self.muls.iter().for_each(|(scalar, lhs, rhs, out)| {
            wiring[*lhs] += *scalar * eq_r[*out] * input[*rhs];
        });
        wiring
    }

    fn phase1_eval(&self, eq_r: &[F], eq_x: &[F]) -> F {
        self.adds
            .iter()
            .map(|(scalar, lhs, out)| *scalar * eq_r[*out] * eq_x[*lhs])
            .sum()
    }

    fn phase2_wiring(&self, eq_r: &[F], eq_x: &[F]) -> Vec<F> {
        let mut wiring = vec![F::ZERO; 1 << self.input_num_vars];
        self.muls.iter().for_each(|(scalar, lhs, rhs, out)| {
            wiring[*rhs] += *scalar * eq_r[*out] * eq_x[*lhs];
        });
        wiring
    }

    fn phase2_eval(&self, eq_r: &[F], eq_x: &[F], eq_y: &[F]) -> F {
        self.muls
            .iter()
            .map(|(scalar, lhs, rhs, out)| *scalar * eq_r[*out] * eq_x[*lhs] * eq_y[*rhs])
            .sum()
    }
}

#[derive(Clone, Debug)]
pub struct Circuit<F> {
    layers: Vec<Layer<F>>,
}

impl<F: Field> Circuit<F> {
    pub fn new(layers: Vec<Layer<F>>) -> Self {
        Self { layers }
    }

    pub fn layers(&self) -> &[Layer<F>] {
        &self.layers
    }

    pub fn input_num_vars(&self) -> usize {
        self.layers.first().unwrap().input_num_vars()
    }

    pub fn output_num_vars(&self) -> usize {
        self.layers.last().unwrap().output_num_vars()
    }

    pub fn evaluate(&self, input: Vec<F>) -> Vec<Vec<F>> {
        self.layers.iter().fold(vec![input], |mut polys, layer| {
            let input = polys.last().unwrap();
            let mut output = vec![F::ZERO; 1 << layer.output_num_vars];
            layer.adds.iter().for_each(|(scalar, lhs, out)| {
                output[*out] += *scalar * input[*lhs];
            });
            layer.muls.iter().for_each(|(scalar, lhs, rhs, out)| {
                output[*out] += *scalar * input[*lhs] * input[*rhs];
            });
            polys.push(output);
            polys
        })
    }
}

#[derive(Clone, Debug)]
pub struct EvalClaim<F> {
    r: Vec<F>,
    eval: F,
}

impl<F: Field> EvalClaim<F> {
    pub fn new(r: Vec<F>, eval: F) -> Self {
        Self { r, eval }
    }

    pub fn r(&self) -> &[F] {
        &self.r
    }

    pub fn eval(&self) -> F {
        self.eval
    }
}

pub fn prove_libra<F: Field>(
    circuit: &Circuit<F>,
    output_claim: &EvalClaim<F>,
    polys: Vec<Vec<F>>,
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<Vec<EvalClaim<F>>, Error> {
    debug_assert_eq!(circuit.layers.len() + 1, polys.len());

    let (output, _) = polys.split_last().unwrap();
    debug_assert_eq!(
        evaluate_multilinear_poly(output, &output_claim.r),
        output_claim.eval
    );

    let mut claims = vec![output_claim.clone()];
    for (layer, input) in izip!(&circuit.layers, polys).rev() {
        let num_vars = layer.input_num_vars;

        let (claim, eq_r) = if claims.len() == 1 {
            (claims[0].eval, eq_poly(&claims[0].r, F::ONE))
        } else {
            let alpha = transcript.squeeze_challenge();
            let beta = transcript.squeeze_challenge();
            (
                claims[0].eval * alpha + claims[1].eval * beta,
                izip!(eq_poly(&claims[0].r, alpha), eq_poly(&claims[1].r, beta))
                    .map(|(a, b)| a + b)
                    .collect_vec(),
            )
        };

        let wiring = layer.phase1_wiring(&eq_r, &input);
        let (claim, x) = prove_quadratic_sum_check(num_vars, claim, [&input, &wiring], transcript)?;
        let input_x = evaluate_multilinear_poly(&input, &x);
        transcript.write_felt(&input_x)?;
        let eq_x = eq_poly(&x, input_x);

        let claim = claim - layer.phase1_eval(&eq_r, &eq_x);
        let wiring = layer.phase2_wiring(&eq_r, &eq_x);
        let (_, y) = prove_quadratic_sum_check(num_vars, claim, [&input, &wiring], transcript)?;
        let input_y = evaluate_multilinear_poly(&input, &y);
        transcript.write_felt(&input_y)?;

        claims = vec![EvalClaim::new(x, input_x), EvalClaim::new(y, input_y)];
    }

    Ok(claims)
}

pub fn verify_libra<F: Field>(
    circuit: &Circuit<F>,
    output_claim: &EvalClaim<F>,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<Vec<EvalClaim<F>>, Error> {
    let mut claims = vec![output_claim.clone()];
    for layer in circuit.layers.iter().rev() {
        let num_vars = layer.input_num_vars;

        let (claim, eq_r) = if claims.len() == 1 {
            (claims[0].eval, eq_poly(&claims[0].r, F::ONE))
        } else {
            let alpha = transcript.squeeze_challenge();
            let beta = transcript.squeeze_challenge();
            (
                claims[0].eval * alpha + claims[1].eval * beta,
                izip!(eq_poly(&claims[0].r, alpha), eq_poly(&claims[1].r, beta))
                    .map(|(a, b)| a + b)
                    .collect_vec(),
            )
        };

        let (claim, x) = verify_quadratic_sum_check(num_vars, claim, transcript)?;
        let input_x = transcript.read_felt()?;
        let eq_x = eq_poly(&x, input_x);

        let claim = claim - layer.phase1_eval(&eq_r, &eq_x);
        let (claim, y) = verify_quadratic_sum_check(num_vars, claim, transcript)?;
        let input_y = transcript.read_felt()?;
        let eq_y = eq_poly(&y, input_y);

        if claim != layer.phase2_eval(&eq_r, &eq_x, &eq_y) {
            return Err(err_unmatched_sum_check_evaluation());
        }

        claims = vec![EvalClaim::new(x, input_x), EvalClaim::new(y, input_y)];
    }

    Ok(claims)
}

#[cfg(test)]
mod test {
    use crate::{
        dev::rand_felts,
        libra::{prove_libra, verify_libra, Circuit, EvalClaim, Layer},
        sum_check::evaluate_multilinear_poly,
        transcript::StdRngTranscript,
    };
    use ff::PrimeField;
    use halo2curves::bn256;
    use rand::{
        rngs::{OsRng, StdRng},
        Rng, RngCore, SeedableRng,
    };
    use std::io::Cursor;

    fn run_libra<F: PrimeField>() {
        let mut rng = StdRng::seed_from_u64(OsRng.next_u64());
        for _ in 0..10 {
            let input_num_vars = rng.gen_range(1..10);
            let output_num_vars = rng.gen_range(1..10);
            let layers = vec![Layer {
                input_num_vars,
                output_num_vars,
                adds: (0..1 << input_num_vars)
                    .map(|_| {
                        (
                            F::random(&mut rng),
                            rng.gen_range(0..1 << input_num_vars),
                            rng.gen_range(0..1 << output_num_vars),
                        )
                    })
                    .collect(),
                muls: (0..1 << input_num_vars)
                    .map(|_| {
                        (
                            F::random(&mut rng),
                            rng.gen_range(0..1 << input_num_vars),
                            rng.gen_range(0..1 << input_num_vars),
                            rng.gen_range(0..1 << output_num_vars),
                        )
                    })
                    .collect(),
            }];
            let circuit = Circuit::new(layers);

            let input = rand_felts(1 << input_num_vars, &mut rng);
            let polys = circuit.evaluate(input.clone());
            let output_claim = {
                let r = rand_felts(output_num_vars, &mut rng);
                let eval = evaluate_multilinear_poly(polys.last().unwrap(), &r);
                EvalClaim::new(r, eval)
            };

            let mut transcript = StdRngTranscript::new(Vec::new());
            prove_libra(&circuit, &output_claim, polys, &mut transcript).unwrap();

            let mut transcript = StdRngTranscript::new(Cursor::new(transcript.into_proof()));
            let input_claims = verify_libra(&circuit, &output_claim, &mut transcript).unwrap();

            input_claims.iter().for_each(|claim| {
                assert_eq!(evaluate_multilinear_poly(&input, &claim.r), claim.eval)
            });
        }
    }

    #[test]
    fn libra() {
        run_libra::<bn256::Fr>();
    }
}
