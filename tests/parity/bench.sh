#!/bin/sh
# bench.sh <report-dir>: decode time of hevc-rs against libde265 (dec265) and ffmpeg on the intra
# prefix of every corpus stream, and against libheif (heic-oracle) on the HEIC corpus. Best of
# REPS runs (default 3) per tool and stream. Information only, no gate: the status is 0 unless a
# tool fails to run or no stream was timed.
#
# Columns: 1 thread (dec265 -t1, ffmpeg -threads 1, hevc-bench --threads 1) and default threads
# (dec265 -t$(nproc), ffmpeg with no -threads, hevc-bench with no --threads), for the default build
# of hevc-rs (runtime SIMD dispatch: the best tier of the host) and the x86-64-v4 build
# (hevc-bench-v4, docs/benchmarks.md). The 1-thread hevc-rs time is also given for the forced
# tiers HEVC_SIMD=scalar and avx2, so a kernel's gain is visible.
#
# Classes (from `hevc-oracle probe`): main (8-bit 4:2:0), main10 (10-bit 4:2:0), rext (the rest:
# 12-bit, 4:0:0), heic (the HEIC corpus; examples/heic and heic-oracle, HEIC_THREADS=1 for the
# 1-thread columns).
set -u
export LC_ALL=C
REP=$1 reps=${REPS:-3} W=/work NP=$(nproc)
rm -rf $W; mkdir -p "$REP" $W/in $W/prefix $W/probe
find /data/x -type f \( -iname '*.bit' -o -iname '*.bin' -o -iname '*.hevc' -o -iname '*.265' \) | sort |
while IFS= read -r f; do
  rel=$(printf '%s' "${f#/data/x/}" | sed 's#/#__#g')
  e=$(printf '%s' "${rel##*.}" | tr 'A-Z' 'a-z')
  cp "$f" "$W/in/${rel%.*}.$e"
done
bad=0
hevc-oracle prefix $W/in $W/prefix || bad=1
hevc-oracle probe $W/prefix $W/probe || bad=1

# The v4 binaries need AVX-512 (some hosted runners are AMD EPYC without it): without it they are
# not run and their columns are 0.
V4=0
grep -q avx512bw /proc/cpuinfo && V4=1
bestv4() { if [ "$V4" = 1 ]; then best "$@"; else echo 0; fi; }

# A/B mode (bench.yml input `base`): hevc-bench-base and hevc-heic-base, the default build of the
# base source, are timed at 1 thread right after the same build of this source, so both run on
# the same host and the same warm files; their column is 0 otherwise. The head runs first, which
# gives a small bias against it.
AB=0
command -v hevc-bench-base > /dev/null 2>&1 && AB=1
bestab() { if [ "$AB" = 1 ]; then best "$@"; else echo 0; fi; }

# best <cmd...>: the shortest wall time in ns of REPS runs; fails when a run fails.
best() {
  b= i=0
  while [ $i -lt "$reps" ]; do
    s=$(date +%s%N)
    "$@" > /dev/null 2>&1 || return 1
    t=$(($(date +%s%N) - s))
    if [ -z "$b" ] || [ "$t" -lt "$b" ]; then b=$t; fi
    i=$((i + 1))
  done
  echo "$b"
}

# class <probe-file>: main, main10 or rext.
class() {
  set -- $(cat "$1" 2>/dev/null) x x x x x
  case "$3:$4" in
    420:8) echo main ;;
    420:10) echo main10 ;;
    *) echo rext ;;
  esac
}

# One line per stream: name class bytes, then the times in ns:
#  4 dec265-1t 5 ffmpeg-1t 6 hevc-1t 7 hevc-v4-1t 8 hevc-scalar-1t 9 hevc-avx2-1t
#  10 dec265-nt 11 ffmpeg-nt 12 hevc-nt 13 hevc-v4-nt 14 hevc-base-1t (A/B mode, else 0)
: > $W/all.tsv; : > $W/skipped
for p in $W/prefix/*.bit; do
  [ -f "$p" ] || continue
  b=${p##*/}; s=${b%.bit}
  if c1=$(best dec265 -q -t1 "$p") \
    && f1=$(best ffmpeg -v error -nostdin -threads 1 -i "$p" -f null -) \
    && r1=$(best hevc-bench --threads 1 "$p") \
    && b1=$(bestab hevc-bench-base --threads 1 "$p") \
    && v1=$(bestv4 hevc-bench-v4 --threads 1 "$p") \
    && s1=$(HEVC_SIMD=scalar best hevc-bench --threads 1 "$p") \
    && a1=$(HEVC_SIMD=avx2 best hevc-bench --threads 1 "$p") \
    && cn=$(best dec265 -q -t"$NP" "$p") \
    && fn=$(best ffmpeg -v error -nostdin -i "$p" -f null -) \
    && rn=$(best hevc-bench "$p") \
    && vn=$(bestv4 hevc-bench-v4 "$p"); then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$(class "$W/probe/$s.probe")" \
      "$(wc -c < "$p")" "$c1" "$f1" "$r1" "$v1" "$s1" "$a1" "$cn" "$fn" "$rn" "$vn" "$b1" >> $W/all.tsv
  else
    echo "$s" >> $W/skipped
  fi
done

# HEIC: one directory per file, so each process decodes one image. Columns as above, with
# heic-oracle (libheif) in the dec265 place and no ffmpeg (0).
: > $W/heic.tsv
if [ -f /heic.list ]; then
  export HEIC_BENCH=1  # decode only: no .yuv writes, no cross-check (examples/heic.rs, heic_oracle.c)
  i=0
  while IFS= read -r f; do
    i=$((i + 1)); d=$W/h/$i; mkdir -p $d/in
    cp "/heic/x/${f#./}" $d/in/
    n=$(printf '%s' "${f#./}" | sed 's#/#__#g')
    if c1=$(HEIC_THREADS=1 best heic-oracle $d/in $d/o) && ls $d/o | grep -q '\.info$' \
      && r1=$(HEIC_THREADS=1 best hevc-heic $d/in $d/r) && ls $d/r | grep -q '\.info$' \
      && b1=$(HEIC_THREADS=1 bestab hevc-heic-base $d/in $d/r) \
      && v1=$(HEIC_THREADS=1 bestv4 hevc-heic-v4 $d/in $d/r) \
      && s1=$(HEVC_SIMD=scalar HEIC_THREADS=1 best hevc-heic $d/in $d/r) \
      && a1=$(HEVC_SIMD=avx2 HEIC_THREADS=1 best hevc-heic $d/in $d/r) \
      && cn=$(best heic-oracle $d/in $d/o) \
      && rn=$(best hevc-heic $d/in $d/r) \
      && vn=$(bestv4 hevc-heic-v4 $d/in $d/r); then
      printf '%s\theic\t%s\t%s\t0\t%s\t%s\t%s\t%s\t%s\t0\t%s\t%s\t%s\n' "$n" "$(wc -c < "/heic/x/${f#./}")" \
        "$c1" "$r1" "$v1" "$s1" "$a1" "$cn" "$rn" "$vn" "$b1" >> $W/heic.tsv
    else
      echo "heic:$n" >> $W/skipped
    fi
    rm -rf $d
  done < /heic.list
fi

files=$(cat $W/all.tsv $W/heic.tsv | wc -l)
[ "$files" -gt 0 ] || bad=1
{
  echo "## hevc-rs vs libde265, ffmpeg and libheif: decode time, ms (lower is better)"; echo
  echo "Best of $reps runs, $NP CPUs, SIMD tier of the host: $(hevc-bench --tier). Information only, no gate."
  echo "$(wc -l < $W/skipped) inputs skipped (a tool failed or gave no picture)."
  [ "$V4" = 1 ] || echo "v4: not run, the host has no AVX-512"
  echo
  echo "### Totals per class"; echo
  echo "Reference: dec265 for main/main10/rext, libheif (with libde265) for heic. v4: hevc-rs built"
  echo "with -C target-cpu=x86-64-v4 and the false-deps features. Ratio: hevc-rs / reference."; echo
  echo "| class | n | ref 1t | ffmpeg 1t | hevc 1t | v4 1t | scalar 1t | avx2 1t | ratio 1t | ratio v4 1t | ref nt | ffmpeg nt | hevc nt | v4 nt | ratio nt | ratio v4 nt |"
  echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
  cat $W/all.tsv $W/heic.tsv | awk -F'\t' '
    function row(k) {
      printf "| %s | %d | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f | %.1f | %.1f | %.1f | %.1f | %.2f | %.2f |\n",
        k, N[k], T[k,4]/1e6, T[k,5]/1e6, T[k,6]/1e6, T[k,7]/1e6, T[k,8]/1e6, T[k,9]/1e6,
        T[k,6]/T[k,4], T[k,7]/T[k,4], T[k,10]/1e6, T[k,11]/1e6, T[k,12]/1e6, T[k,13]/1e6,
        T[k,12]/T[k,10], T[k,13]/T[k,10]
    }
    { N[$2]++; for (i = 4; i <= 13; i++) T[$2,i] += $i
      if ($2 != "heic") { N["all"]++; for (i = 4; i <= 13; i++) T["all",i] += $i } }
    END { split("main main10 rext all heic", K, " ")
      for (j = 1; j <= 5; j++) if (N[K[j]] > 0) row(K[j]) }'
  if [ "$AB" = 1 ]; then
    echo; echo "### A/B: base build vs this build (default build and tier, 1 thread, same host)"; echo
    echo "Change: (this - base) / base. Faster: inputs where this build took less time."; echo
    echo "| class | n | base 1t | this 1t | change | faster |"
    echo "|---|---|---|---|---|---|"
    cat $W/all.tsv $W/heic.tsv | awk -F'\t' '
      function row(k) {
        printf "| %s | %d | %.1f | %.1f | %+.2f%% | %d |\n", k, N[k], B[k]/1e6, H[k]/1e6,
          (B[k] > 0 ? 100 * (H[k] - B[k]) / B[k] : 0), F[k]
      }
      function add(k) { N[k]++; B[k] += $14; H[k] += $6; if ($6 < $14) F[k]++ }
      { add($2); if ($2 != "heic") add("all") }
      END { split("main main10 rext all heic", K, " ")
        for (j = 1; j <= 5; j++) if (N[K[j]] > 0) row(K[j]) }'
  fi
  echo; echo "### Per input (1 thread; ms)"; echo
  echo "| input | class | bytes | ref | ffmpeg | hevc | v4 | scalar | avx2 | hevc / ref |"
  echo "|---|---|---|---|---|---|---|---|---|---|"
  cat $W/all.tsv $W/heic.tsv | awk -F'\t' '{ printf "| %s | %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |\n",
    $1, $2, $3, $4/1e6, $5/1e6, $6/1e6, $7/1e6, $8/1e6, $9/1e6, $6/$4 }'
  if [ -s $W/skipped ]; then echo; echo "Skipped:"; echo; sed 's/^/- /' $W/skipped; fi
} > "$REP/summary.md"
cp $W/all.tsv "$REP/all.tsv"; cp $W/heic.tsv "$REP/heic.tsv"
rm -rf $W
exit $bad
