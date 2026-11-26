//! SIMD tier selection (docs/safety.md). The scalar code is always compiled; it is the oracle of
//! the cross-tier tests and the fallback. The tier is detected once per process.
//!
//! `HEVC_SIMD=scalar|avx2|avx512|neon` (read once) forces a tier. A value that names a tier the
//! CPU does not have, or an unknown value, selects the best detected tier. The exception is
//! `avx512`, which is a requirement (for CI under an emulator): it panics on a CPU without AVX-512
//! F, BW and VL (and on a non-x86_64 target), so a test run cannot fall back to AVX2 without a word.

// Per-tier load/store helpers: the only `unsafe` besides the dispatch of each kernel family
// (docs/safety.md). dead_code: each kernel family uses only some helpers.
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code, dead_code)]
pub(crate) mod arm;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code, dead_code)]
pub(crate) mod x86;

use std::sync::OnceLock;

/// A SIMD tier. `Avx512` means AVX-512 F, BW and VL (and AVX2, FMA and F16C, which rustc's
/// avx512f implies).
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

/// The tiers this CPU can run, scalar first.
pub(crate) fn available() -> Vec<Tier> {
    let mut v = vec![Tier::Scalar];
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            v.push(Tier::Avx2);
            // rustc's avx512f implies avx2, fma and f16c, so the AVX-512 kernels may use them:
            // list the tier only when the CPU reports them too.
            if std::arch::is_x86_feature_detected!("avx512f")
                && std::arch::is_x86_feature_detected!("avx512bw")
                && std::arch::is_x86_feature_detected!("avx512vl")
                && std::arch::is_x86_feature_detected!("fma")
                && std::arch::is_x86_feature_detected!("f16c")
            {
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

/// Whether the AVX-512 tier is available here: false on a target without it.
pub(crate) fn avx512_kernels() -> bool {
    available().contains(&Tier::Avx512)
}

/// The tier of this process: the best detected one, or the one `HEVC_SIMD` forces.
/// `HEVC_SIMD=avx512` panics when the AVX-512 tier is not available.
pub(crate) fn tier() -> Tier {
    static TIER: OnceLock<Tier> = OnceLock::new();
    *TIER.get_or_init(|| {
        let forced = std::env::var("HEVC_SIMD").ok();
        if forced.as_deref() == Some("avx512") {
            assert!(
                avx512_kernels(),
                "HEVC_SIMD=avx512, but this CPU lacks AVX-512 F, BW and VL"
            );
        }
        select(forced.as_deref())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Also logs the tier in use (`--nocapture`), so a CI log shows what the run tested.
    /// `HEVC_EXPECT_VNNI=1|0` (the SDE steps of CI) also checks AVX-512 VNNI, which selects the
    /// AVX-512 transform residual kernel over the AVX2 one.
    #[test]
    fn tier_is_detected_or_forced() {
        #[cfg(target_arch = "x86_64")]
        let vnni = std::arch::is_x86_feature_detected!("avx512vnni");
        #[cfg(not(target_arch = "x86_64"))]
        let vnni = false;
        eprintln!(
            "SIMD tier: {}, AVX-512 kernels: {}, VNNI: {vnni}",
            tier().name(),
            avx512_kernels()
        );
        match std::env::var("HEVC_EXPECT_VNNI").as_deref() {
            Ok("1") => assert!(vnni, "HEVC_EXPECT_VNNI=1, but VNNI is not detected"),
            Ok("0") => assert!(!vnni, "HEVC_EXPECT_VNNI=0, but VNNI is detected"),
            _ => {}
        }
        match std::env::var("HEVC_SIMD").as_deref() {
            Ok("avx512") => assert_eq!(tier(), Tier::Avx512),
            Ok("scalar") => assert_eq!(tier(), Tier::Scalar),
            _ => {}
        }
    }

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
