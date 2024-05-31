pub mod libra;
pub mod sum_check;
pub mod transcript;

#[derive(Debug)]
pub enum Error {
    InvalidSumCheck(String),
    Transcript(std::io::ErrorKind, String),
}

#[cfg(test)]
pub mod dev {
    use ff::Field;
    use itertools::Itertools;
    use rand::RngCore;
    use std::iter;

    pub fn rand_felts<F: Field>(n: usize, rng: &mut impl RngCore) -> Vec<F> {
        iter::repeat_with(|| F::random(&mut *rng))
            .take(n)
            .collect_vec()
    }
}
