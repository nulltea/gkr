use crate::{
    transcript::{TranscriptRead, TranscriptWrite},
    Error,
};
use ff::Field;
use itertools::izip;

pub fn prove_quadratic_sum_check<F: Field>(
    num_vars: usize,
    claim: F,
    polys: [&Vec<F>; 2],
    transcript: &mut impl TranscriptWrite<F>,
) -> Result<(F, Vec<F>), Error>
where
{
    debug_assert!(num_vars > 0);

    let mut claim = claim;
    let mut polys = polys.map(Clone::clone);
    let mut r = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let (f, g) = (&polys[0], &polys[1]);
        debug_assert_eq!(
            izip!(f.iter(), g.iter()).map(|(f, g)| *f * g).sum::<F>(),
            claim,
        );

        let mut uni_poly = [F::ZERO; 3];
        for b in 0..f.len() / 2 {
            uni_poly[0] += f[2 * b] * g[2 * b];
            uni_poly[2] += (f[2 * b + 1] - f[2 * b]) * (g[2 * b + 1] - g[2 * b]);
        }
        uni_poly[1] = claim - uni_poly[0].double() - uni_poly[2];

        transcript.write_felt(&uni_poly[0])?;
        transcript.write_felt(&uni_poly[2])?;

        let r_i = transcript.squeeze_challenge();

        claim = evaluate_univariate_poly(&uni_poly, &r_i);
        polys.iter_mut().for_each(|poly| fix_var(poly, &r_i));
        r.push(r_i);
    }

    Ok((claim, r))
}

pub fn verify_quadratic_sum_check<F: Field>(
    num_vars: usize,
    claim: F,
    transcript: &mut impl TranscriptRead<F>,
) -> Result<(F, Vec<F>), Error> {
    debug_assert!(num_vars > 0);

    let mut claim = claim;
    let mut r = Vec::with_capacity(num_vars);
    for _ in 0..num_vars {
        let mut uni_poly = vec![F::ZERO; 3];
        uni_poly[0] = transcript.read_felt()?;
        uni_poly[2] = transcript.read_felt()?;
        uni_poly[1] = claim - uni_poly[0].double() - uni_poly[2];

        let r_i = transcript.squeeze_challenge();

        claim = evaluate_univariate_poly(&uni_poly, &r_i);
        r.push(r_i);
    }

    Ok((claim, r))
}

fn evaluate_univariate_poly<F: Field>(poly: &[F], x: &F) -> F {
    poly.iter()
        .rev()
        .fold(F::ZERO, |acc, coeff| acc * x + coeff)
}

fn fix_var<F: Field>(poly: &mut Vec<F>, r_i: &F) {
    for b in 0..poly.len() / 2 {
        poly[b] = (poly[2 * b + 1] - poly[2 * b]) * r_i + poly[2 * b]
    }
    poly.truncate(poly.len() / 2);
}

pub fn evaluate_multilinear_poly<F: Field>(poly: &[F], r: &[F]) -> F {
    debug_assert_eq!(poly.len(), 1 << r.len());
    let mut poly = poly.to_vec();
    r.iter().for_each(|r_i| fix_var(&mut poly, r_i));
    poly[0]
}

pub fn eq_poly<F: Field>(r: &[F], scalar: F) -> Vec<F> {
    let mut eq = vec![F::ZERO; 1 << r.len()];
    eq[0] = scalar;
    for (i, r) in izip!(0.., r) {
        let (lo, hi) = eq.split_at_mut(1 << i);
        izip!(&mut *hi, &*lo).for_each(|(hi, lo)| *hi = *r * lo);
        izip!(&mut *lo, &*hi).for_each(|(lo, hi)| *lo -= hi as &_);
    }
    eq
}

pub fn err_unmatched_sum_check_evaluation() -> Error {
    Error::InvalidSumCheck("Unmatched evaluation from SumCheck subclaim".to_string())
}

#[cfg(test)]
mod test {
    use crate::{
        dev::rand_felts,
        sum_check::{
            evaluate_multilinear_poly, prove_quadratic_sum_check, verify_quadratic_sum_check,
        },
        transcript::StdRngTranscript,
    };
    use ff::PrimeField;
    use halo2curves::bn256;
    use itertools::izip;
    use rand::{
        rngs::{OsRng, StdRng},
        RngCore, SeedableRng,
    };
    use std::{array::from_fn, io::Cursor};

    fn run_qudratic<F: PrimeField>() {
        let rng = &mut StdRng::seed_from_u64(OsRng.next_u64());
        for num_vars in 1..10 {
            let [f, g] = &from_fn(|_| rand_felts(1 << num_vars, rng));
            let claim = F::sum(izip!(f, g).map(|(f, g)| *f * g));

            let mut transcript = StdRngTranscript::new(Vec::new());
            prove_quadratic_sum_check(num_vars, claim, [f, g], &mut transcript).unwrap();

            let mut transcript = StdRngTranscript::new(Cursor::new(transcript.into_proof()));
            let (claim, r) = verify_quadratic_sum_check(num_vars, claim, &mut transcript).unwrap();

            let [f_eval, g_eval] = [f, g].map(|poly| evaluate_multilinear_poly(poly, &r));
            assert_eq!(f_eval * g_eval, claim);
        }
    }

    #[test]
    fn qudratic() {
        run_qudratic::<bn256::Fr>();
    }
}
