#!/bin/sh
# profile.sh: where hevc-rs and dec265 spend their time, one thread, on the intra prefix of every
# corpus stream (/prof/prefix, made by the profile stage of .github/ci/Dockerfile). Run by
# profile.yml in `docker run --privileged`; prints markdown. perf when the host allows perf
# events, else callgrind (instructions, not time) on the 8 largest prefix files.
# HEVC_SIMD in the environment forces the tier of hevc-rs, as everywhere.
set -u
export LC_ALL=C
echo "## Profile: hevc-rs $(hevc-bench --tier) tier vs dec265, 1 thread, $(ls /prof/prefix | wc -l) prefix streams"
echo
P=$(ls /usr/bin/perf_* 2>/dev/null | sort | tail -n 1)
if [ -n "$P" ] && "$P" record -q -o /tmp/probe.data -- true > /dev/null 2>&1; then
  for tool in hevc dec265; do
    case $tool in
      hevc) cmd='hevc-bench --threads 1' comm=hevc-bench ;;
      dec265) cmd='dec265 -q -t1' comm=dec265 ;;
    esac
    s=$(date +%s%N)
    "$P" record -q -F 4999 -o /tmp/$tool.data -- sh -c "for f in /prof/prefix/*.bit; do $cmd \$f > /dev/null 2>&1; done" > /dev/null 2>&1
    t=$(( ($(date +%s%N) - s) / 1000000 ))
    echo "### $tool (perf, self time, $t ms wall under perf)"
    echo
    echo '```'
    "$P" report -q -i /tmp/$tool.data --comms $comm --no-children --sort dso,symbol --stdio --percent-limit 0.3 2>/dev/null \
      | grep -v '^$' | head -n 60 | cut -c1-220
    echo '```'
    echo
    # The SIMD kernels are inlined into their callers: self time per source file shows the share
    # of each kernel family (src/transform/avx2.rs, src/intra/x86.rs, src/sao_simd.rs, ...).
    if [ $tool = hevc ]; then
      echo "### $tool (perf, self time per source file)"
      echo
      echo '```'
      "$P" report -q -i /tmp/$tool.data --comms $comm --no-children --sort srcfile --stdio --percent-limit 0.3 2>/dev/null \
        | grep -v '^$' | head -n 40 | cut -c1-220
      echo '```'
      echo
    fi
  done
else
  echo "perf events are not allowed on this host: callgrind (instruction counts) on the 8 largest prefix files."
  echo
  for f in $(ls -S /prof/prefix/*.bit | head -n 8); do
    valgrind --tool=callgrind --callgrind-out-file=/tmp/cg.out hevc-bench --threads 1 "$f" > /dev/null 2>&1
    echo "### ${f##*/}"
    echo
    echo '```'
    callgrind_annotate /tmp/cg.out 2>/dev/null | sed -n '1,45p' | cut -c1-160
    echo '```'
    echo
  done
fi
