#!/bin/sh
# fetch.sh <corpus.toml> <dir>: download every [[file]] of corpus.toml (a zip) into <dir>/<path>
# once, at most 20 MB/s, check its size and sha256 (sha256 = "TODO" prints the computed sum
# instead of failing), and extract the bitstream files (*.bit *.bin *.hevc *.265, case
# insensitive, any depth) into <dir>/x/<path without .zip>/. A file that is already there with its
# stamp (<path>.ok, written after the sum matched) is not downloaded again, and an extraction with
# its stamp (<dir>/x/<name>.ok) is not repeated, so a second run is a no-op.
# Run it inside a BuildKit cache mount, id=hevc-rs-data, sharing=locked.
set -eu
[ $# -eq 2 ] || { echo "usage: fetch.sh <corpus.toml> <dir>" >&2; exit 2; }
toml=$1 dir=$2
tab=$(printf '\t')
# itu.int may refuse curl's default user agent.
ua='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36'
mkdir -p "$dir/x"
# One "path url size sha256" line per [[file]], tab separated.
awk -F' = ' -v OFS='\t' '
  function emit() { if (p != "") print p, u, s, h; p = u = s = h = "" }
  /^\[\[file\]\]/ { emit() }
  $1 == "path" { p = $2 } $1 == "url" { u = $2 } $1 == "size" { s = $2 } $1 == "sha256" { h = $2 }
  END { emit() }' "$toml" | tr -d '"' |
while IFS=$tab read -r path url size sum; do
  name=${path%.zip}
  f=$dir/$path x=$dir/x/$name
  if [ -f "$f.ok" ] && [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$size" ]; then
    :
  else
    echo "fetch $path ($size bytes)"
    rm -f "$f.ok" "$x.ok"
    # -C - resumes a .part left by an interrupted run. itu.int answers a short error page when it
    # gets many requests in a row, so a short reply is retried after a growing pause.
    try=0
    while :; do
      if [ ! -f "$f.part" ] || [ "$(stat -c %s "$f.part")" -lt "$size" ]; then
        curl -fsSL --retry 3 --limit-rate 20M -A "$ua" -C - -o "$f.part" "$url" || true
      fi
      got=$(stat -c %s "$f.part" 2>/dev/null || echo 0)
      [ "$got" = "$size" ] && break
      rm -f "$f.part"
      try=$((try + 1))
      [ $try -lt 6 ] || { echo "$path: size $got, want $size after $try tries" >&2; exit 1; }
      echo "$path: short reply ($got bytes), retry $try in $((try * 60)) s"
      sleep $((try * 60))
    done
    sleep 2
    if [ "$sum" = TODO ]; then
      echo "$path: sha256 TODO, computed $(sha256sum "$f.part" | cut -d' ' -f1)"
    else
      echo "$sum  $f.part" | sha256sum -c --quiet - || { echo "$path: sha256 mismatch" >&2; rm -f "$f.part"; exit 1; }
    fi
    mv "$f.part" "$f"
    touch "$f.ok"
  fi
  if [ ! -f "$x.ok" ]; then
    rm -rf "$x"
    mkdir -p "$x"
    # -j: flat, -o: overwrite, -C: case-insensitive patterns. Exit 11 (no match) is reported below.
    unzip -q -j -o -C "$f" '*.bit' '*.bin' '*.hevc' '*.265' -d "$x" || true
    touch "$x.ok"
  fi
  echo "$name: $(find "$x" -type f | wc -l) bitstream files"
done
echo "hevc-rs-data: $(du -sh "$dir" | cut -f1) in $dir"
