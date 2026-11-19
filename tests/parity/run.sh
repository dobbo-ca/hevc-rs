#!/bin/sh
# Parity gates of hevc-rs (docs/design.md, "Oracles and parity"). CI only, called by
# .github/ci/Dockerfile; needs the oracle binaries on PATH.
#   run.sh full <report-dir>   the gates on /data/x + tests/data/fuzz-grown
#   run.sh ref <out-dir>       hevc-oracle on the first 20 corpus streams, one thread
#   run.sh arm <ref-dir>       the aarch64 oracle (qemu) on the same prefixes vs <ref-dir>
# Binaries: hevc-oracle (examples/oracle), dec265 (libde265), ffmpeg, ffprobe, and for `arm`
# hevc-oracle-arm64 run by $ARM_RUN. Exit status 0 only when every gate is an empty diff.
# PARITY_DECODE=1 turns the picture gates (dec265, ffmpeg, hash) on; otherwise they only report.
set -u
export LC_ALL=C
W=/work

# flat <dst> <root>: copy the files named on stdin (paths under <root>) to <dst>, names flattened
# (/ becomes __) and the extension lowercased.
flat() {
  while IFS= read -r f; do
    rel=$(printf '%s' "${f#"$2"/}" | sed 's#/#__#g')
    e=$(printf '%s' "${rel##*.}" | tr 'A-Z' 'a-z')
    cp "$f" "$1/${rel%.*}.$e"
  done
}

# gather <dst>: every bitstream of /data/x, and the fuzz-grown files (no extension: .bit added).
gather() {
  mkdir -p "$1"
  find /data/x -type f \( -iname '*.bit' -o -iname '*.bin' -o -iname '*.hevc' -o -iname '*.265' \) | sort | flat "$1" /data/x
  if [ -d tests/data/fuzz-grown ]; then
    find tests/data/fuzz-grown -type f ! -name '.*' | sort | while IFS= read -r f; do
      cp "$f" "$1/fuzz-grown__${f##*/}.bit"
    done
  fi
}

count() { find "$1" -type f -name "$2" | wc -l; }

# norm <chroma>: the chroma token of the oracle or of ffprobe, as mono or 420.
norm() {
  case $(printf '%s' "$1" | tr 'A-Z' 'a-z') in
    mono | gray | 400 | 0) echo mono ;;
    420 | yuv420 | yuv420p | 1) echo 420 ;;
    *) echo "$1" ;;
  esac
}

# gate <label> <diff-file> [soft]: an empty diff passes. A soft gate only fails when
# PARITY_DECODE=1.
gate() {
  lines=$(wc -l < "$2")
  if [ "$lines" -eq 0 ]; then
    echo "| $1 | ok |" >> "$REP/summary.md"; rm -f "$2"
  else
    head -n 200 "$2" > "$2.t" && mv "$2.t" "$2"
    if [ "${3:-}" = soft ] && [ "${PARITY_DECODE:-0}" != 1 ]; then
      echo "| $1 | $lines diff lines, not gated (PARITY_DECODE=0) |" >> "$REP/summary.md"
    else
      echo "| $1 | FAIL, $lines diff lines |" >> "$REP/summary.md"; bad=1
    fi
  fi
}

run() { "$@" || { echo "FAILED: $*" >&2; bad=1; }; }

# cmpyuv <dirA> <labelA> <dirB> <labelB> <diff-file> <stats-file> [names-file]: compare the .yuv
# files of the two dirs by name (only the names listed in names-file, when given). A name with
# output on one side only is a diff line. stats-file gets "<pairs compared> <bytes compared>".
cmpyuv() {
  { (cd "$1" && ls | sed -n 's/\.yuv$//p'); (cd "$3" && ls | sed -n 's/\.yuv$//p'); } | sort -u > $W/names
  if [ -n "${7:-}" ]; then grep -Fx -f "$7" $W/names > $W/names.t; mv $W/names.t $W/names; fi
  : > "$5"; cmpn=0 cmpb=0
  while IFS= read -r n; do
    a=$1/$n.yuv b=$3/$n.yuv
    if [ -f "$a" ] && [ -f "$b" ]; then
      cmpn=$((cmpn + 1)); cmpb=$((cmpb + $(wc -c < "$b")))
      cmp -s "$a" "$b" || echo "$n: yuv differs ($2 $(wc -c < "$a") bytes, $4 $(wc -c < "$b") bytes)" >> "$5"
    elif [ -f "$a" ]; then echo "$n: only $2 has output" >> "$5"
    else echo "$n: only $4 has output" >> "$5"; fi
  done < $W/names
  echo "$cmpn $cmpb" > "$6"
}

full() {
  REP=$1 bad=0
  rm -rf $W; mkdir -p "$REP" $W/prefix $W/de $W/ff $W/ffp
  gather $W/in
  n=$(ls $W/in | wc -l)
  [ "$n" -gt 0 ] || { echo "corpus is empty" >&2; bad=1; }

  run hevc-oracle prefix $W/in $W/prefix
  run hevc-oracle probe $W/in $W/probe
  run hevc-oracle --threads 1 $W/prefix $W/rs1
  run hevc-oracle $W/prefix $W/rsn

  # libde265 and ffmpeg on every prefix stream. ffmpeg's pix_fmt comes from the hevc-rs probe.
  : > $W/ffnames; : > $W/ffskip
  for p in $W/prefix/*.bit; do
    [ -f "$p" ] || continue
    b=${p##*/}; s=${b%.bit}
    dec265 -q -o "$W/de/$s.yuv" "$p" > /dev/null 2>&1 || { rm -f "$W/de/$s.yuv"; echo "dec265 failed" > "$W/de/$s.err"; }
    [ -f "$W/probe/$s.probe" ] || continue
    read -r w h c d dc _ < "$W/probe/$s.probe"
    case "$(norm "$c")/$d" in
      mono/8) pf=gray ;; mono/10) pf=gray10le ;; mono/12) pf=gray12le ;;
      420/8) pf=yuv420p ;; 420/10) pf=yuv420p10le ;; 420/12) pf=yuv420p12le ;;
      *) pf= ;;
    esac
    # ffmpeg has one depth per pix_fmt: luma and chroma depths must match.
    [ -z "$pf" ] || [ "${dc:-$d}" = "$d" ] || pf=
    if [ -z "$pf" ]; then echo "$s" >> $W/ffskip; continue; fi
    echo "$s" >> $W/ffnames
    ffmpeg -nostdin -v error -i "$p" -f rawvideo -fps_mode passthrough -pix_fmt "$pf" "$W/ff/$s.yuv" > /dev/null 2>&1 \
      || { rm -f "$W/ff/$s.yuv"; echo "ffmpeg failed" > "$W/ff/$s.err"; }
  done

  # ffprobe on every input: "W H chroma depthY" in the same shape as the hevc-rs probe.
  : > $W/ffprobe-failed
  for f in $W/in/*; do
    [ -f "$f" ] || continue
    b=${f##*/}; s=${b%.*}
    if pf=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,pix_fmt -of csv=p=0 "$f" 2> /dev/null) && [ -n "$pf" ]; then
      IFS=, read -r w h fmt <<EOT
$pf
EOT
      case $fmt in
        gray) c=mono d=8 ;; gray10le) c=mono d=10 ;; gray12le) c=mono d=12 ;;
        yuv420p) c=420 d=8 ;; yuv420p10le) c=420 d=10 ;; yuv420p12le) c=420 d=12 ;;
        *) echo "$s (pix_fmt $fmt)" >> $W/ffprobe-failed; continue ;;
      esac
      echo "$w $h $c $d" > "$W/ffp/$s.ffprobe"
    else
      echo "$s" >> $W/ffprobe-failed
    fi
  done

  # G1: probe vs ffprobe.
  : > $W/d-probe
  for f in $W/ffp/*.ffprobe; do
    [ -f "$f" ] || continue
    b=${f##*/}; s=${b%.ffprobe}
    want=$(cat "$f")
    if [ -f "$W/probe/$s.probe" ]; then
      read -r w h c d _ < "$W/probe/$s.probe"
      got="$w $h $(norm "$c") $d"
    else
      got="err $(cat "$W/probe/$s.err" 2> /dev/null)"
    fi
    [ "$want" = "$got" ] || echo "$s: ffprobe='$want' hevc-rs='$got'" >> $W/d-probe
  done

  # G2, G3: pictures.
  cmpyuv $W/de dec265 $W/rs1 hevc-rs $W/d-dec265 $W/s-dec265
  cmpyuv $W/ff ffmpeg $W/rs1 hevc-rs $W/d-ffmpeg $W/s-ffmpeg $W/ffnames
  # G4: decoded picture hash SEI.
  find $W/rs1 -type f -name '*.info' -exec grep -lw bad {} + 2> /dev/null | sed 's#.*/##; s#\.info$#: hash bad#' | sort > $W/d-hash
  # G5: threads.
  diff -rq $W/rs1 $W/rsn > $W/d-threads

  read -r cmpn cmpb < $W/s-dec265
  pics=$(find $W/rs1 -type f -name '*.info' -exec cat {} + | wc -l)
  {
    echo "## hevc-rs parity"; echo
    echo "| set | count |"; echo "|---|---|"
    echo "| streams (corpus, fuzz-grown) | $n |"
    echo "| prefixes (intra part, input of the C oracles) | $(count $W/prefix '*.bit') |"
    echo "| hevc-rs: decoded / error | $(count $W/rs1 '*.yuv') / $(count $W/rs1 '*.err') |"
    echo "| dec265: decoded / failed | $(count $W/de '*.yuv') / $(count $W/de '*.err') |"
    echo "| ffmpeg: decoded / failed / skipped (pix_fmt) | $(count $W/ff '*.yuv') / $(count $W/ff '*.err') / $(wc -l < $W/ffskip) |"
    echo "| pictures (hevc-rs .info lines) | $pics |"
    echo "| streams compared with dec265 | $cmpn |"
    echo "| bytes compared with dec265 | $cmpb |"
    echo "| ffprobe failed or unmapped | $(wc -l < $W/ffprobe-failed) |"
    echo; echo "| gate | result |"; echo "|---|---|"
  } > "$REP/summary.md"
  gate "G1 probe vs ffprobe (W H chroma depthY)" $W/d-probe
  gate "G2 dec265 vs hevc-rs" $W/d-dec265 soft
  gate "G3 ffmpeg vs hevc-rs" $W/d-ffmpeg soft
  gate "G4 decoded picture hash (no bad)" $W/d-hash soft
  gate "G5 1 thread vs default threads" $W/d-threads
  [ -s $W/ffprobe-failed ] && { echo; echo "ffprobe failed (G1 skips them), first 50:"; echo; head -n 50 $W/ffprobe-failed | sed 's/^/    /'; } >> "$REP/summary.md"
  for d in d-probe d-dec265 d-ffmpeg d-hash d-threads; do [ -f $W/$d ] && cp $W/$d "$REP/$d.diff"; done
  rm -rf $W
  return $bad
}

# The tree every aarch64 run is compared with: the prefixes of the first 20 corpus streams, and
# what hevc-oracle makes of them with one thread.
ref() {
  rm -rf $W; mkdir -p $W/all $W/in "$1/prefix" "$1/out"
  gather $W/all
  ls $W/all | head -n 20 | while IFS= read -r f; do cp "$W/all/$f" "$W/in/$f"; done
  [ "$(ls $W/in | wc -l)" -gt 0 ] || { echo "corpus is empty" >&2; bad=1; }
  run hevc-oracle prefix $W/in "$1/prefix"
  run hevc-oracle --threads 1 "$1/prefix" "$1/out"
  rm -rf $W
  return "${bad:-0}"
}

arm() {
  REP=$1 bad=0
  rm -rf $W
  [ "$(ls "$REP/prefix" | wc -l)" -gt 0 ] || { echo "no prefix streams in $REP" >&2; bad=1; }
  for simd in default scalar; do
    if [ $simd = scalar ]; then export HEVC_SIMD=scalar; else unset HEVC_SIMD; fi
    mkdir -p $W/out
    run ${ARM_RUN:-qemu-aarch64 -L /usr/aarch64-linux-gnu} /usr/local/bin/hevc-oracle-arm64 --threads 1 "$REP/prefix" $W/out
    if diff -rq "$REP/out" $W/out > $W/d; then echo "aarch64 $simd: identical to x86_64"
    else echo "aarch64 $simd: DIFF"; head -n 50 $W/d; bad=1; fi
    rm -rf $W/out
  done
  rm -rf $W
  return $bad
}

cmd=${1:-}; [ $# -gt 0 ] && shift
case $cmd in
  full) full "$@" ;;
  ref) bad=0; ref "$@" ;;
  arm) arm "$@" ;;
  *) echo "usage: run.sh full|ref|arm ..." >&2; exit 2 ;;
esac
