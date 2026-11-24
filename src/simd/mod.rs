//! SIMD tier selection (docs/safety.md). The scalar code is always compiled; it is the oracle of
//! the cross-tier tests and the fallback. The tier is detected once per process.
//!
//! `HEVC_SIMD=scalar|avx2|avx512|neon` (read once) forces a tier. A value that names a tier the
//! CPU does not have, or an unknown value, selects the best detected tier.

// Per-tier load/store helpers: the only `unsafe` besides the dispatch of each kernel family
// (docs/safety.md). Unused until the kernels that call them land.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
pub(crate) mod arm;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, dead_code)]
pub(crate) mod x86;

use std::sync::OnceLock;

/// A SIMD tier. `Avx512` means AVX-512 F, BW and VL (and AVX2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code, reason = "each target constructs only its own tiers")]
pub(crate) enum Tier {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl Tier {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Tier::Scalar => "scalar",
            Tier::Avx2 => "avx2",
            Tier::Avx512 => "avx512",
            Tier::Neon => "neon",
        }
    }
}

/// True when this CPU has AVX-512 F, BW and VL, the features of the `Avx512` kernels. The
/// dispatch guards use it, so a tier value that was not taken from `available()` stays safe.
#[cfg(target_arch = "x86_64")]
pub(crate) fn has_avx512() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512bw")
        && std::arch::is_x86_feature_detected!("avx512vl")
}

/// The tiers this CPU can run, scalar first.
pub(crate) fn available() -> Vec<Tier> {
    let mut v = vec![Tier::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            v.push(Tier::Avx2);
            if has_avx512() {
                v.push(Tier::Avx512);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            v.push(Tier::Neon);
        }
    }
    v
}

fn select(forced: Option<&str>) -> Tier {
    let avail = available();
    let best = avail.last().copied().unwrap_or(Tier::Scalar);
    match forced {
        Some(name) => avail
            .iter()
            .copied()
            .find(|t| t.name() == name)
            .unwrap_or(best),
        None => best,
    }
}

/// The tier of this process: the best detected one, or the one `HEVC_SIMD` forces.
pub(crate) fn tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| select(std::env::var("HEVC_SIMD").ok().as_deref()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection() {
        let avail = available();
        assert_eq!(avail.first(), Some(&Tier::Scalar));
        let best = *avail.last().unwrap();
        assert_eq!(select(None), best);
        assert_eq!(select(Some("scalar")), Tier::Scalar);
        assert_eq!(select(Some("no-such-tier")), best);
        for t in &avail {
            assert_eq!(select(Some(t.name())), *t);
        }
        // The CI hosts have these tiers; a missing one would leave its kernels untested.
        #[cfg(target_arch = "x86_64")]
        assert!(avail.contains(&Tier::Avx2), "AVX2 not detected");
        #[cfg(target_arch = "aarch64")]
        assert!(avail.contains(&Tier::Neon), "NEON not detected");
    }
}
