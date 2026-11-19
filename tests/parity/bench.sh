#!/bin/sh
# bench.sh <report-dir>: libde265 (dec265 -q -t1) and hevc-rs (hevc-bench --threads 1) decode the
# intra prefix of every corpus stream, best of REPS runs (default 3); summary.md has the time per
# stream, the totals and the ratio hevc-rs / dec265 (lower is better). Information only, no gate:
# the status is 0 unless a tool fails to run or no stream was timed by both. Assumes hevc-bench
# takes `--threads N <stream>` and decodes that Annex B file once.
set -u
export LC_ALL=C
REP=$1 reps=${REPS:-3} W=/work tab=$(printf '\t')
rm -rf $W; mkdir -p "$REP" $W/in $W/prefix
find /data/x -type f \( -iname '*.bit' -o -iname '*.bin' -o -iname '*.hevc' -o -iname '*.265' \) | sort |
while IFS= read -r f; do
  rel=$(printf '%s' "${f#/data/x/}" | sed 's#/#__#g')
  e=$(printf '%s' "${rel##*.}" | tr 'A-Z' 'a-z')
  cp "$f" "$W/in/${rel%.*}.$e"
done
bad=0
hevc-oracle prefix $W/in $W/prefix || bad=1

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

: > $W/all.tsv; : > $W/skipped
for p in $W/prefix/*.bit; do
  [ -f "$p" ] || continue
  b=${p##*/}; s=${b%.bit}
  if c=$(best dec265 -q -t1 "$p") && r=$(best hevc-bench --threads 1 "$p"); then
    printf '%s\t%s\t%s\t%s\n' "$s" "$(wc -c < "$p")" "$c" "$r" >> $W/all.tsv
  else
    echo "$s" >> $W/skipped
  fi
done
files=$(wc -l < $W/all.tsv)
[ "$files" -gt 0 ] || bad=1
{
  echo "## hevc-rs vs libde265: decode time per stream, ms (lower is better)"; echo
  echo "$files streams timed by both tools, best of $reps runs, 1 thread. Information only, no gate."
  echo "$(wc -l < $W/skipped) streams skipped (a tool failed)."; echo
  echo "| stream | bytes | dec265 | hevc-rs | hevc-rs / dec265 |"
  echo "|---|---|---|---|---|"
  awk -F'\t' '
    { printf "| %s | %d | %.2f | %.2f | %.2f |\n", $1, $2, $3/1e6, $4/1e6, $4/$3; B += $2; C += $3; R += $4 }
    END { if (C > 0) printf "| total | %d | %.2f | %.2f | %.2f |\n", B, C/1e6, R/1e6, R/C }' $W/all.tsv
} > "$REP/summary.md"
cp $W/all.tsv "$REP/all.tsv"
rm -rf $W
exit $bad
