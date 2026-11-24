//! residual_coding() (7.3.8.11) for I slices, ChromaArrayType 0 or 1, no range-extension tools.
//! The context selection follows 9.3.4.2.3 to 9.3.4.2.7 and the binarization of
//! coeff_abs_level_remaining follows 9.3.3.11 with persistent_rice_adaptation off.

use crate::cabac::{Cabac, Contexts, Ctx};
use crate::error::{Error, Result};

const BAD: Error = Error::InvalidData("residual_coding index out of range");

fn get<T: Copy>(a: &[T], i: usize) -> Result<T> {
    a.get(i).copied().ok_or(BAD)
}

fn put<T>(a: &mut [T], i: usize, v: T) -> Result<()> {
    *a.get_mut(i).ok_or(BAD)? = v;
    Ok(())
}

fn ctx_mut(a: &mut [Ctx], i: usize) -> Result<&mut Ctx> {
    a.get_mut(i).ok_or(BAD)
}

/// Scan order arrays 6.5.3 (kind 0, up-right diagonal), 6.5.4 (1, horizontal), 6.5.5 (2,
/// vertical) for blkSize = 1 << `log2`; entries are (x, y), the rest of the array is unused.
// Evaluated at compile time: an out-of-range index would be a build error, not a runtime panic.
#[allow(clippy::indexing_slicing)]
const fn scan_array(kind: usize, log2: usize) -> [(u8, u8); 64] {
    let size = 1usize << log2;
    let mut out = [(0u8, 0u8); 64];
    let mut i = 0;
    if kind == 0 {
        let (mut x, mut y) = (0i32, 0i32);
        while i < size * size {
            while y >= 0 {
                if (x as usize) < size && (y as usize) < size {
                    out[i] = (x as u8, y as u8);
                    i += 1;
                }
                y -= 1;
                x += 1;
            }
            y = x;
            x = 0;
        }
    } else {
        let mut a = 0;
        while a < size {
            let mut b = 0;
            while b < size {
                out[i] = if kind == 1 {
                    (b as u8, a as u8)
                } else {
                    (a as u8, b as u8)
                };
                i += 1;
                b += 1;
            }
            a += 1;
        }
    }
    out
}

// Evaluated at compile time like `scan_array`.
#[allow(clippy::indexing_slicing)]
const fn build_scans() -> [[[(u8, u8); 64]; 3]; 4] {
    let mut t = [[[(0u8, 0u8); 64]; 3]; 4];
    let mut l = 0;
    while l < 4 {
        let mut k = 0;
        while k < 3 {
            t[l][k] = scan_array(k, l);
            k += 1;
        }
        l += 1;
    }
    t
}

/// SCANS[log2 blkSize][scanIdx]: blkSize 1, 2, 4, 8 (sub-block arrays and the 4x4 inside).
static SCANS: [[[(u8, u8); 64]; 3]; 4] = build_scans();

/// ScanOrder[log2_blk][scan_idx] as a slice of (x, y).
fn scan(log2_blk: u32, scan_idx: u32) -> Result<&'static [(u8, u8)]> {
    let t = SCANS
        .get(log2_blk as usize)
        .and_then(|a| a.get(scan_idx as usize))
        .ok_or(BAD)?;
    t.get(..1usize << (2 * log2_blk)).ok_or(BAD)
}

/// Table 9-50.
const CTX_IDX_MAP: [u8; 16] = [0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8, 8];

/// scanIdx of 7.4.9.11 for an intra CU with ChromaArrayType 0 or 1 (`pred_mode_intra` is the
/// luma mode for cIdx 0, else the chroma mode).
pub(crate) fn scan_idx(log2_size: u32, c_idx: usize, pred_mode_intra: u32) -> u32 {
    if log2_size == 2 || (log2_size == 3 && c_idx == 0) {
        match pred_mode_intra {
            6..=14 => 2,
            22..=30 => 1,
            _ => 0,
        }
    } else {
        0
    }
}

pub(crate) struct ResidualParams {
    /// log2TrafoSize of this block (2..=5), for chroma the chroma block size.
    pub log2_size: u32,
    pub c_idx: usize,
    pub scan_idx: u32,
    /// transform_skip_enabled_flag && !cu_transquant_bypass_flag && log2_size == 2
    /// (Log2MaxTransformSkipSize is 2 without range extensions).
    pub transform_skip_allowed: bool,
    pub bypass: bool,
    /// sign_data_hiding_enabled_flag.
    pub sign_hiding: bool,
}

pub(crate) struct ResidualOut {
    pub transform_skip: bool,
    /// (rows, cols): every non-zero coefficient is in the first `rows` rows and the first
    /// `cols` columns (the largest y and x written, plus one).
    pub ext: (usize, usize),
}

/// last_sig_coeff_{x,y}_prefix: TR with cMax = (log2 << 1) - 1, ctxInc by 9.3.4.2.3.
fn last_prefix(cabac: &mut Cabac, ctxs: &mut [Ctx], log2: u32, c_idx: usize) -> Result<u32> {
    let (offset, shift) = if c_idx == 0 {
        (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2)
    } else {
        (15, log2 - 2)
    };
    let c_max = (log2 << 1) - 1;
    let mut v = 0;
    while v < c_max && cabac.decode(ctx_mut(ctxs, ((v >> shift) + offset) as usize)?)? {
        v += 1;
    }
    Ok(v)
}

/// Position from prefix and suffix (7-74 .. 7-77).
fn last_pos(cabac: &mut Cabac, prefix: u32) -> Result<usize> {
    if prefix <= 3 {
        return Ok(prefix as usize);
    }
    let n = (prefix >> 1) - 1;
    let suffix = cabac.bypass_bits(n)?;
    Ok(((1 << n) * (2 + (prefix & 1)) + suffix) as usize)
}

/// sigCtx before the sub-block offset (9.3.4.2.5) for blocks over 4x4, by prevCsbf and the
/// position inside the sub-block (`yP * 4 + xP`).
#[allow(clippy::indexing_slicing)] // evaluated at compile time
const fn build_sig_pat() -> [[u8; 16]; 4] {
    let mut t = [[0u8; 16]; 4];
    let mut p = 0;
    while p < 4 {
        let mut i = 0;
        while i < 16 {
            let (xp, yp) = (i & 3, i >> 2);
            t[p][i] = match p {
                0 => {
                    let d = xp + yp;
                    if d == 0 {
                        2
                    } else if d <= 2 {
                        1
                    } else {
                        0
                    }
                }
                1 => 2usize.saturating_sub(yp) as u8,
                2 => 2usize.saturating_sub(xp) as u8,
                _ => 2,
            };
            i += 1;
        }
        p += 1;
    }
    t
}

static SIG_PAT: [[u8; 16]; 4] = build_sig_pat();

/// The `yP * 4 + xP` table of sigCtx for a sub-block: Table 9-50 for a 4x4 block, else by
/// prevCsbf.
fn sig_pattern(log2: u32, prev_csbf: usize) -> Result<&'static [u8; 16]> {
    if log2 == 2 {
        Ok(&CTX_IDX_MAP)
    } else {
        SIG_PAT.get(prev_csbf).ok_or(BAD)
    }
}

/// What 9.3.4.2.5 adds to the pattern value for the sub-block (xS, yS), chroma offset 27
/// included. The DC of the whole block (sub-block 0, position 0) is `27 * (cIdx > 0)` instead.
fn sig_offset(log2: u32, c_idx: usize, scan_idx: u32, xs: usize, ys: usize) -> usize {
    if log2 == 2 {
        if c_idx == 0 { 0 } else { 27 }
    } else if c_idx == 0 {
        usize::from(xs + ys > 0) * 3
            + if log2 == 3 {
                if scan_idx == 0 { 9 } else { 15 }
            } else {
                21
            }
    } else {
        27 + if log2 == 3 { 9 } else { 12 }
    }
}

/// coeff_abs_level_remaining, 9.3.3.11 (extended_precision off): TR prefix of up to four ones,
/// then an EG(rice + 1) suffix, whose unary part continues the run of ones.
fn abs_level_remaining(cabac: &mut Cabac, rice: u32) -> Result<i32> {
    let prefix = cabac.bypass_ones(32)?.ok_or(Error::InvalidData(
        "coeff_abs_level_remaining prefix too long",
    ))?;
    let v: u64 = if prefix <= 3 {
        (u64::from(prefix) << rice) + u64::from(cabac.bypass_bits(rice)?)
    } else {
        let m = prefix - 4; // ones inside the EG suffix
        let n = m + 1 + rice;
        let low = cabac.bypass_wide(n)?;
        (((1u64 << (m + 1)) + 2) << rice) + low
    };
    Ok(i32::try_from(v).unwrap_or(i32::MAX))
}

/// Decodes one residual block into `coeffs` (raster, nTbS * nTbS, row = y), which must be all
/// zero on entry: only the coded coefficients are written (the transform clears them again,
/// `transform::reconstruct`). On error the buffer is zeroed. Values are saturated to i32;
/// conformance keeps them within 16 bits.
pub(crate) fn residual_coding(
    cabac: &mut Cabac,
    ctx: &mut Contexts,
    p: &ResidualParams,
    coeffs: &mut [i32],
) -> Result<ResidualOut> {
    let out = coding(cabac, ctx, p, coeffs);
    if out.is_err() {
        coeffs.fill(0);
    }
    out
}

/// [`residual_coding`] without the clear on error.
fn coding(
    cabac: &mut Cabac,
    ctx: &mut Contexts,
    p: &ResidualParams,
    coeffs: &mut [i32],
) -> Result<ResidualOut> {
    let log2 = p.log2_size;
    if !(2..=5).contains(&log2) || p.scan_idx > 2 {
        return Err(Error::InvalidData("residual_coding block size or scanIdx"));
    }
    let n_tb = 1usize << log2;
    if coeffs.len() < n_tb * n_tb {
        return Err(Error::InvalidData("residual buffer too small"));
    }
    let mut ext = (0, 0);
    let chroma = p.c_idx > 0;

    let transform_skip = p.transform_skip_allowed
        && cabac.decode(ctx_mut(&mut ctx.transform_skip_flag, usize::from(chroma))?)?;

    let px = last_prefix(cabac, &mut ctx.last_x_prefix, log2, p.c_idx)?;
    let py = last_prefix(cabac, &mut ctx.last_y_prefix, log2, p.c_idx)?;
    let mut lx = last_pos(cabac, px)?;
    let mut ly = last_pos(cabac, py)?;
    if p.scan_idx == 2 {
        std::mem::swap(&mut lx, &mut ly);
    }

    let sb_scan = scan(log2 - 2, p.scan_idx)?;
    let pos_scan = scan(2, p.scan_idx)?;
    let sb_dim = 1usize << (log2 - 2);
    let last_sb = sb_scan
        .iter()
        .position(|&(x, y)| usize::from(x) == lx >> 2 && usize::from(y) == ly >> 2)
        .ok_or(Error::InvalidData(
            "last significant position outside block",
        ))?;
    let last_scan_pos = pos_scan
        .iter()
        .position(|&(x, y)| usize::from(x) == lx & 3 && usize::from(y) == ly & 3)
        .ok_or(BAD)?;

    let mut csbf = 0u64; // coded_sub_block_flag[yS][xS] as bit yS * 8 + xS
    let mut g1_carry = 1u32; // greater1Ctx left by the previous sub-block (1 at the start)
    for i in (0..=last_sb).rev() {
        let (xs, ys) = {
            let (x, y) = get(sb_scan, i)?;
            (usize::from(x), usize::from(y))
        };
        let sb_bit = ys * 8 + xs;
        let right = if xs + 1 < sb_dim {
            (csbf >> (sb_bit + 1)) & 1
        } else {
            0
        };
        let below = if ys + 1 < sb_dim {
            (csbf >> (sb_bit + 8)) & 1
        } else {
            0
        };
        let prev_csbf = (right + 2 * below) as usize;

        let mut infer_dc = false;
        let coded = if i < last_sb && i > 0 {
            // 9.3.4.2.4
            let inc = usize::from(right + below > 0) + if chroma { 2 } else { 0 };
            infer_dc = true;
            cabac.decode(ctx_mut(&mut ctx.coded_sub_block_flag, inc)?)?
        } else {
            true
        };
        csbf |= u64::from(coded) << sb_bit;
        if !coded {
            continue;
        }

        // 9.3.4.2.5: sig_coeff_flag ctxInc = offset of the sub-block + pattern of the position.
        let pat = sig_pattern(log2, prev_csbf)?;
        let sig_off = sig_offset(log2, p.c_idx, p.scan_idx, xs, ys);
        let dc_inc = if chroma { 27 } else { 0 };

        let mut sig = 0u16; // bit n: sig_coeff_flag at scan position n of the sub-block
        let mut n = 15;
        if i == last_sb {
            sig |= 1 << last_scan_pos;
            n = last_scan_pos as i32 - 1;
        }
        while n >= 0 {
            let nu = n as usize;
            if nu > 0 || !infer_dc {
                let inc = if i == 0 && nu == 0 {
                    dc_inc
                } else {
                    let (xp, yp) = get(pos_scan, nu)?;
                    sig_off + usize::from(get(pat, (usize::from(yp) << 2) + usize::from(xp))?)
                };
                if cabac.decode(ctx_mut(&mut ctx.sig_coeff_flag, inc)?)? {
                    sig |= 1 << nu;
                    infer_dc = false;
                }
            } else {
                // n == 0 with inferSbDcSigCoeffFlag still set: inferred 1.
                sig |= 1;
            }
            n -= 1;
        }

        if sig == 0 {
            continue;
        }
        let n_sig = sig.count_ones() as usize;
        let hi = 15 - sig.leading_zeros() as usize; // highest scan position with a coefficient
        let first_sig = sig.trailing_zeros() as usize;

        // 9.3.4.2.6: ctxSet for this sub-block.
        let mut ctx_set = if i == 0 || chroma { 0 } else { 2 };
        if g1_carry == 0 {
            ctx_set += 1;
        }
        let mut g1_ctx = 1u32;
        let mut base = [1u8; 16];
        let mut num_g1 = 0;
        let mut last_g1: Option<usize> = None;
        let mut m = sig;
        while m != 0 && num_g1 < 8 {
            let nu = 15 - m.leading_zeros() as usize;
            m &= !(1u16 << nu);
            let inc = ctx_set * 4 + g1_ctx.min(3) as usize + if chroma { 16 } else { 0 };
            let flag = cabac.decode(ctx_mut(&mut ctx.greater1, inc)?)?;
            num_g1 += 1;
            if flag {
                put(&mut base, nu, 2)?;
                if last_g1.is_none() {
                    last_g1 = Some(nu);
                }
            }
            if g1_ctx > 0 {
                g1_ctx = if flag { 0 } else { g1_ctx + 1 };
            }
        }
        g1_carry = g1_ctx;

        let sign_hidden = p.sign_hiding && !p.bypass && hi - first_sig > 3;

        // 9.3.4.2.7
        if let Some(n2) = last_g1 {
            let inc = ctx_set + if chroma { 4 } else { 0 };
            if cabac.decode(ctx_mut(&mut ctx.greater2, inc)?)? {
                put(&mut base, n2, 3)?;
            }
        }

        let n_signs = n_sig - usize::from(sign_hidden);
        let signs = cabac.bypass_bits(n_signs as u32)?;

        let mut num_sig = 0usize;
        let mut sign_idx = 0usize;
        let mut rice = 0u32;
        let mut sum_abs = 0u64;
        let mut m = sig;
        while m != 0 {
            let nu = 15 - m.leading_zeros() as usize;
            m &= !(1u16 << nu);
            let b = get(&base, nu)?;
            let thr = if num_sig < 8 {
                if last_g1 == Some(nu) { 3 } else { 2 }
            } else {
                1
            };
            let mut abs = i64::from(b);
            if b == thr {
                let rem = abs_level_remaining(cabac, rice)?;
                abs += i64::from(rem);
                if abs > 3 * (1i64 << rice) {
                    rice = (rice + 1).min(4);
                }
            }
            sum_abs += abs as u64;
            let negative = if sign_hidden && nu == first_sig {
                sum_abs & 1 == 1
            } else {
                let bit = (signs >> (n_signs - 1 - sign_idx)) & 1;
                sign_idx += 1;
                bit == 1
            };
            let mag = i32::try_from(abs).unwrap_or(i32::MAX);
            let (xp, yp) = get(pos_scan, nu)?;
            let x = (xs << 2) + usize::from(xp);
            let y = (ys << 2) + usize::from(yp);
            put(coeffs, y * n_tb + x, if negative { -mag } else { mag })?;
            ext = (ext.0.max(y + 1), ext.1.max(x + 1));
            num_sig += 1;
        }
    }
    Ok(ResidualOut {
        transform_skip,
        ext,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bits::BitReader;
    /// coeff_abs_level_remaining, 9.3.3.11 (extended_precision off): TR prefix of up to four ones,
    /// then an EG(rice + 1) suffix, whose unary part continues the run of ones.
    fn ref_abs_level_remaining(cabac: &mut Cabac, rice: u32) -> Result<i32> {
        let mut prefix = 0u32;
        while cabac.bypass()? {
            prefix += 1;
            if prefix > 32 {
                return Err(Error::InvalidData(
                    "coeff_abs_level_remaining prefix too long",
                ));
            }
        }
        let v: u64 = if prefix <= 3 {
            (u64::from(prefix) << rice) + u64::from(cabac.bypass_bits(rice)?)
        } else {
            let m = prefix - 4; // ones inside the EG suffix
            let n = m + 1 + rice;
            let mut low = 0u64;
            for _ in 0..n {
                low = (low << 1) | u64::from(cabac.bypass()?);
            }
            (((1u64 << (m + 1)) + 2) << rice) + low
        };
        Ok(i32::try_from(v).unwrap_or(i32::MAX))
    }

    /// Decodes one residual block into `coeffs` (raster, nTbS * nTbS, row = y). The buffer is
    /// zeroed first. Values are saturated to i32; conformance keeps them within 16 bits.
    fn ref_residual_coding(
        cabac: &mut Cabac,
        ctx: &mut Contexts,
        p: &ResidualParams,
        coeffs: &mut [i32],
    ) -> Result<ResidualOut> {
        let log2 = p.log2_size;
        if !(2..=5).contains(&log2) || p.scan_idx > 2 {
            return Err(Error::InvalidData("residual_coding block size or scanIdx"));
        }
        let n_tb = 1usize << log2;
        coeffs
            .get_mut(..n_tb * n_tb)
            .ok_or(Error::InvalidData("residual buffer too small"))?
            .fill(0);
        let chroma = p.c_idx > 0;

        let transform_skip = p.transform_skip_allowed
            && cabac.decode(ctx_mut(&mut ctx.transform_skip_flag, usize::from(chroma))?)?;

        let px = last_prefix(cabac, &mut ctx.last_x_prefix, log2, p.c_idx)?;
        let py = last_prefix(cabac, &mut ctx.last_y_prefix, log2, p.c_idx)?;
        let mut lx = last_pos(cabac, px)?;
        let mut ly = last_pos(cabac, py)?;
        if p.scan_idx == 2 {
            std::mem::swap(&mut lx, &mut ly);
        }

        let sb_scan = scan(log2 - 2, p.scan_idx)?;
        let pos_scan = scan(2, p.scan_idx)?;
        let sb_dim = 1usize << (log2 - 2);
        let last_sb = sb_scan
            .iter()
            .position(|&(x, y)| usize::from(x) == lx >> 2 && usize::from(y) == ly >> 2)
            .ok_or(Error::InvalidData(
                "last significant position outside block",
            ))?;
        let last_scan_pos = pos_scan
            .iter()
            .position(|&(x, y)| usize::from(x) == lx & 3 && usize::from(y) == ly & 3)
            .ok_or(BAD)?;

        let mut csbf = [0u8; 64]; // coded_sub_block_flag[yS * 8 + xS]
        let mut g1_carry = 1u32; // greater1Ctx left by the previous sub-block (1 at the start)
        for i in (0..=last_sb).rev() {
            let (xs, ys) = {
                let (x, y) = get(sb_scan, i)?;
                (usize::from(x), usize::from(y))
            };
            let right = if xs + 1 < sb_dim {
                get(&csbf, ys * 8 + xs + 1)?
            } else {
                0
            };
            let below = if ys + 1 < sb_dim {
                get(&csbf, (ys + 1) * 8 + xs)?
            } else {
                0
            };
            let prev_csbf = usize::from(right) + 2 * usize::from(below);

            let mut infer_dc = false;
            let coded = if i < last_sb && i > 0 {
                // 9.3.4.2.4
                let inc = usize::from(right + below > 0) + if chroma { 2 } else { 0 };
                infer_dc = true;
                cabac.decode(ctx_mut(&mut ctx.coded_sub_block_flag, inc)?)?
            } else {
                true
            };
            put(&mut csbf, ys * 8 + xs, u8::from(coded))?;
            if !coded {
                continue;
            }

            let mut sig = [false; 16];
            let mut n = 15;
            if i == last_sb {
                put(&mut sig, last_scan_pos, true)?;
                n = last_scan_pos as i32 - 1;
            }
            while n >= 0 {
                let nu = n as usize;
                if nu > 0 || !infer_dc {
                    let (xp, yp) = get(pos_scan, nu)?;
                    let inc = sig_ctx_inc(
                        log2,
                        p.c_idx,
                        p.scan_idx,
                        (xs, ys),
                        (usize::from(xp), usize::from(yp)),
                        prev_csbf,
                    );
                    if cabac.decode(ctx_mut(&mut ctx.sig_coeff_flag, inc)?)? {
                        put(&mut sig, nu, true)?;
                        infer_dc = false;
                    }
                } else {
                    // n == 0 with inferSbDcSigCoeffFlag still set: inferred 1.
                    put(&mut sig, 0, true)?;
                }
                n -= 1;
            }

            let n_sig = sig.iter().filter(|&&s| s).count();
            if n_sig == 0 {
                continue;
            }

            // 9.3.4.2.6: ctxSet for this sub-block.
            let mut ctx_set = if i == 0 || chroma { 0 } else { 2 };
            if g1_carry == 0 {
                ctx_set += 1;
            }
            let mut g1_ctx = 1u32;
            let mut base = [1u32; 16];
            let mut num_g1 = 0;
            let mut last_g1: Option<usize> = None;
            let mut first_sig = 16;
            let mut last_sig: Option<usize> = None;
            for nu in (0..16).rev() {
                if !get(&sig, nu)? {
                    continue;
                }
                if num_g1 < 8 {
                    let inc = ctx_set * 4 + g1_ctx.min(3) as usize + if chroma { 16 } else { 0 };
                    let flag = cabac.decode(ctx_mut(&mut ctx.greater1, inc)?)?;
                    num_g1 += 1;
                    if flag {
                        put(&mut base, nu, 2)?;
                        if last_g1.is_none() {
                            last_g1 = Some(nu);
                        }
                    }
                    if g1_ctx > 0 {
                        g1_ctx = if flag { 0 } else { g1_ctx + 1 };
                    }
                }
                last_sig.get_or_insert(nu);
                first_sig = nu;
            }
            g1_carry = g1_ctx;

            let sign_hidden = p.sign_hiding
                && !p.bypass
                && last_sig.is_some_and(|l| l as i32 - first_sig as i32 > 3);

            // 9.3.4.2.7
            if let Some(n2) = last_g1 {
                let inc = ctx_set + if chroma { 4 } else { 0 };
                if cabac.decode(ctx_mut(&mut ctx.greater2, inc)?)? {
                    put(&mut base, n2, 3)?;
                }
            }

            let n_signs = n_sig - usize::from(sign_hidden);
            let signs = cabac.bypass_bits(n_signs as u32)?;

            let mut num_sig = 0usize;
            let mut sign_idx = 0usize;
            let mut rice = 0u32;
            let mut sum_abs = 0u64;
            for nu in (0..16).rev() {
                if !get(&sig, nu)? {
                    continue;
                }
                let b = get(&base, nu)?;
                let thr = if num_sig < 8 {
                    if last_g1 == Some(nu) { 3 } else { 2 }
                } else {
                    1
                };
                let mut abs = i64::from(b);
                if b == thr {
                    let rem = ref_abs_level_remaining(cabac, rice)?;
                    abs += i64::from(rem);
                    if abs > 3 * (1i64 << rice) {
                        rice = (rice + 1).min(4);
                    }
                }
                sum_abs += abs as u64;
                let negative = if sign_hidden && nu == first_sig {
                    sum_abs & 1 == 1
                } else {
                    let bit = (signs >> (n_signs - 1 - sign_idx)) & 1;
                    sign_idx += 1;
                    bit == 1
                };
                let mag = i32::try_from(abs).unwrap_or(i32::MAX);
                let (xp, yp) = get(pos_scan, nu)?;
                let x = (xs << 2) + usize::from(xp);
                let y = (ys << 2) + usize::from(yp);
                put(coeffs, y * n_tb + x, if negative { -mag } else { mag })?;
                num_sig += 1;
            }
        }
        Ok(ResidualOut {
            transform_skip,
            ext: crate::transform::extent(coeffs, n_tb),
        })
    }

    /// ctxInc of sig_coeff_flag, 9.3.4.2.5 (transform_skip_context_enabled_flag is 0).
    fn sig_ctx_inc(
        log2: u32,
        c_idx: usize,
        scan_idx: u32,
        (xs, ys): (usize, usize),
        (xp, yp): (usize, usize),
        prev_csbf: usize,
    ) -> usize {
        let sig_ctx = if log2 == 2 {
            usize::from(CTX_IDX_MAP.get((yp << 2) + xp).copied().unwrap_or(0))
        } else if xs + ys + xp + yp == 0 {
            0
        } else {
            let mut s = match prev_csbf {
                0 => match xp + yp {
                    0 => 2,
                    1..=2 => 1,
                    _ => 0,
                },
                1 => 2usize.saturating_sub(yp),
                2 => 2usize.saturating_sub(xp),
                _ => 2,
            };
            if c_idx == 0 {
                if xs + ys > 0 {
                    s += 3;
                }
                s += if log2 == 3 {
                    if scan_idx == 0 { 9 } else { 15 }
                } else {
                    21
                };
            } else {
                s += if log2 == 3 { 9 } else { 12 };
            }
            s
        };
        if c_idx == 0 { sig_ctx } else { 27 + sig_ctx }
    }

    #[test]
    fn sig_tables_match_the_spec_derivation() {
        for log2 in 2..=5u32 {
            let sb = 1usize << (log2 - 2);
            for c_idx in 0..2usize {
                for scan_idx in 0..3u32 {
                    for ys in 0..sb {
                        for xs in 0..sb {
                            for prev in 0..4usize {
                                let pat = sig_pattern(log2, prev).unwrap();
                                let off = sig_offset(log2, c_idx, scan_idx, xs, ys);
                                for yp in 0..4usize {
                                    for xp in 0..4usize {
                                        let want = sig_ctx_inc(
                                            log2,
                                            c_idx,
                                            scan_idx,
                                            (xs, ys),
                                            (xp, yp),
                                            prev,
                                        );
                                        let got = if xs + ys + xp + yp == 0 {
                                            if c_idx == 0 { 0 } else { 27 }
                                        } else {
                                            off + usize::from(pat[(yp << 2) + xp])
                                        };
                                        assert_eq!(
                                            got, want,
                                            "log2 {log2} c {c_idx} scan {scan_idx} sb ({xs},{ys}) prev {prev} p ({xp},{yp})"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn matches_reference_residual_coding() {
        let mut seed = 99u32;
        for round in 0..600u32 {
            let len = 8 + (round % 40) as usize * 8;
            let mut data: Vec<u8> = (0..len)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                })
                .collect();
            // Some rounds start with 0xFF: ivlOffset 510/511 is rejected (9.3.2.5).
            if let Some(b) = data.first_mut().filter(|_| round.is_multiple_of(16)) {
                *b = 0xFF;
            }
            for log2 in 2..=5u32 {
                for scan_idx in 0..3u32 {
                    let p = ResidualParams {
                        log2_size: log2,
                        c_idx: round as usize % 3,
                        scan_idx,
                        transform_skip_allowed: log2 == 2 && !round.is_multiple_of(3),
                        bypass: round.is_multiple_of(7),
                        sign_hiding: round.is_multiple_of(2),
                    };
                    let (Ok(mut c1), Ok(mut c2)) = (
                        Cabac::new(BitReader::new(&data)),
                        Cabac::new(BitReader::new(&data)),
                    ) else {
                        assert_eq!(data.first(), Some(&0xFF));
                        continue;
                    };
                    let mut x1 = Contexts::init(20 + (round % 20) as i32);
                    let mut x2 = x1.clone();
                    let mut b1 = vec![0; 1 << (2 * log2)];
                    let mut b2 = vec![0; 1 << (2 * log2)];
                    let r1 = ref_residual_coding(&mut c1, &mut x1, &p, &mut b1);
                    let r2 = residual_coding(&mut c2, &mut x2, &p, &mut b2);
                    let what = format!("round {round} log2 {log2} scan {scan_idx}");
                    assert_eq!(
                        r1.as_ref().map(|o| (o.transform_skip, o.ext)),
                        r2.as_ref().map(|o| (o.transform_skip, o.ext)),
                        "{what}"
                    );
                    if r1.is_ok() {
                        assert_eq!(b1, b2, "{what}");
                        assert_eq!(format!("{x1:?}"), format!("{x2:?}"), "{what}");
                        assert_eq!(c1.spec_pos(), c2.spec_pos(), "{what}");
                    }
                }
            }
        }
    }

    /// 6.5.3 to 6.5.5 written out as in the spec.
    fn reference(kind: usize, size: i32) -> Vec<(u8, u8)> {
        let mut v = Vec::new();
        match kind {
            0 => {
                let (mut x, mut y) = (0, 0);
                while v.len() < (size * size) as usize {
                    while y >= 0 {
                        if x < size && y < size {
                            v.push((x as u8, y as u8));
                        }
                        y -= 1;
                        x += 1;
                    }
                    y = x;
                    x = 0;
                }
            }
            1 => {
                for y in 0..size {
                    for x in 0..size {
                        v.push((x as u8, y as u8));
                    }
                }
            }
            _ => {
                for x in 0..size {
                    for y in 0..size {
                        v.push((x as u8, y as u8));
                    }
                }
            }
        }
        v
    }

    #[test]
    fn scans_match_spec() {
        for kind in 0..3 {
            for log2 in 0..4u32 {
                let got = scan(log2, kind as u32).unwrap();
                assert_eq!(
                    got,
                    &reference(kind, 1 << log2)[..],
                    "kind {kind} log2 {log2}"
                );
            }
        }
        let d = scan(2, 0).unwrap();
        assert_eq!(&d[..6], &[(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]);
        assert_eq!(d[15], (3, 3));
    }

    #[test]
    fn ctx_idx_map_table_9_50() {
        assert_eq!(
            &CTX_IDX_MAP[..15],
            &[0, 1, 4, 5, 2, 3, 4, 5, 6, 6, 8, 8, 7, 7, 8]
        );
    }

    #[test]
    fn sig_ctx_examples() {
        // 4x4 luma, chroma offset 27.
        assert_eq!(sig_ctx_inc(2, 0, 0, (0, 0), (2, 0), 0), 4);
        assert_eq!(sig_ctx_inc(2, 1, 0, (0, 0), (2, 0), 0), 31);
        // DC of a larger block.
        assert_eq!(sig_ctx_inc(4, 0, 0, (0, 0), (0, 0), 0), 0);
        // 8x8 luma diagonal, sub-block (1,0), prevCsbf 0, xP+yP == 0: 2 + 3 + 9.
        assert_eq!(sig_ctx_inc(3, 0, 0, (1, 0), (0, 0), 0), 14);
        // 8x8 luma non-diagonal scan: +15.
        assert_eq!(sig_ctx_inc(3, 0, 1, (0, 0), (1, 0), 0), 1 + 15);
        // 16x16 chroma, prevCsbf 3: 2 + 12 + 27.
        assert_eq!(sig_ctx_inc(4, 1, 0, (1, 1), (3, 3), 3), 41);
        // prevCsbf 1 uses yP, 2 uses xP.
        assert_eq!(sig_ctx_inc(5, 0, 0, (0, 1), (3, 1), 1), 1 + 3 + 21);
        assert_eq!(sig_ctx_inc(5, 0, 0, (0, 1), (2, 3), 2), 3 + 21);
    }

    #[test]
    fn scan_idx_rules() {
        assert_eq!(scan_idx(2, 0, 10), 2);
        assert_eq!(scan_idx(2, 1, 26), 1);
        assert_eq!(scan_idx(3, 0, 6), 2);
        assert_eq!(scan_idx(3, 1, 10), 0);
        assert_eq!(scan_idx(4, 0, 10), 0);
        assert_eq!(scan_idx(3, 0, 15), 0);
        assert_eq!(scan_idx(3, 0, 21), 0);
        assert_eq!(scan_idx(3, 0, 14), 2);
        assert_eq!(scan_idx(3, 0, 22), 1);
    }

    #[test]
    fn garbage_input_never_panics() {
        let mut seed = 12345u32;
        for round in 0..200 {
            let data: Vec<u8> = (0..64)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    (seed >> 24) as u8
                })
                .collect();
            for log2 in 2..=5 {
                for scan_idx in 0..3 {
                    let Ok(mut cabac) = Cabac::new(BitReader::new(&data)) else {
                        continue;
                    };
                    let mut ctx = Contexts::init(26 + (round % 10));
                    let p = ResidualParams {
                        log2_size: log2,
                        c_idx: round as usize % 3,
                        scan_idx,
                        transform_skip_allowed: log2 == 2,
                        bypass: round % 7 == 0,
                        sign_hiding: round % 2 == 0,
                    };
                    let mut buf = vec![0; 1 << (2 * log2)];
                    if residual_coding(&mut cabac, &mut ctx, &p, &mut buf).is_err() {
                        assert!(buf.iter().all(|&c| c == 0), "not cleared on error");
                    }
                }
            }
        }
    }

    #[test]
    fn buffer_too_small_is_error() {
        let data = [0u8; 8];
        let mut cabac = Cabac::new(BitReader::new(&data)).unwrap();
        let mut ctx = Contexts::init(26);
        let p = ResidualParams {
            log2_size: 3,
            c_idx: 0,
            scan_idx: 0,
            transform_skip_allowed: false,
            bypass: false,
            sign_hiding: false,
        };
        assert!(residual_coding(&mut cabac, &mut ctx, &p, &mut [0; 16]).is_err());
    }
}
