# CI image: Rust runs here, built with docker buildx on GitHub-hosted runners (ubuntu-24.04 for
# x86_64, ubuntu-24.04-arm for the native arm64 stages).
# The crate depends on dobbo-ca/isobmff-rs by git rev; a read-only GitHub App token fetches it
# while that repo is private. Its fetch is the only step
# of the crate stages that needs the GitHub App token, passed as the BuildKit secret `gh_token`
# (never a build arg, never left in a layer); every later step runs offline. The fuzz crate's own
# dependencies (libfuzzer-sys from crates.io, and hevc with isobmff-rs) are fetched in the same step
# as the root's.
# Every tool is installed in a stage before `COPY . .`, so a new commit does not install it again,
# and the fetch runs from the manifests and lockfiles alone.
FROM rust:1.99.1-bookworm AS toolchain
ARG TARGETARCH
# On x86_64, gcc-aarch64-linux-gnu is the linker of `cargo build --target aarch64-unknown-linux-gnu`
# (the aarch64 lint and link check in `ci`). On arm64 that target is the host: no cross toolchain.
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu \
    && if [ "$TARGETARCH" = amd64 ]; then \
         apt-get update && apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
         && rm -rf /var/lib/apt/lists/* \
         && printf '[target.aarch64-unknown-linux-gnu]\nlinker = "aarch64-linux-gnu-gcc"\n' >> "$CARGO_HOME/config.toml"; \
       fi
WORKDIR /src
# The manifests and lockfiles of the crate and of the fuzz crate, with stub targets, so the fetch
# is a layer that only a change to them rebuilds.
COPY Cargo.toml Cargo.lock ./
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in annexb hvcc params probe decode heic transform_tiers intra_tiers sao_tiers deblock_tiers; do touch tests/fuzz/fuzz_targets/$t.rs; done
RUN --mount=type=secret,id=gh_token \
    git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/" \
    && CARGO_NET_GIT_FETCH_WITH_CLI=true cargo fetch --locked \
    && CARGO_NET_GIT_FETCH_WITH_CLI=true cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml \
    && git config --global --unset-all url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf \
    && rm -rf src tests/fuzz/fuzz_targets

FROM toolchain AS src
ENV CARGO_NET_OFFLINE=true
COPY . .

# The checks run in the build, so a red check fails the build; a rerun of the same source is a
# BuildKit cache hit. The target dir is a cache mount, so it persists across builds. The target
# cache is shared by every branch and cargo compares mtimes: another branch's newer build would be
# reused as this one's, so every RUN touches the .rs files itself, just before cargo starts.
FROM src AS ci
RUN --mount=type=cache,id=hevc-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo fmt --check
RUN --mount=type=cache,id=hevc-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings
# The NEON modules are not all compiled for x86_64, so lint the aarch64 target too.
RUN --mount=type=cache,id=hevc-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
RUN --mount=type=cache,id=hevc-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo test --locked --no-fail-fast \
    && cargo test --locked --lib -- --nocapture tier_is_detected_or_forced
RUN --mount=type=cache,id=hevc-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && HEVC_SIMD=scalar cargo test --locked --no-fail-fast
# The second run forces the scalar tier through HEVC_SIMD=scalar for every test that decodes through
# simd::tier(); the cross-tier tests compare every tier in simd::available() whatever HEVC_SIMD is.
# The examples are built too, so the cross linker runs (a lib alone is not linked).
RUN --mount=type=cache,id=hevc-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo build --locked --release --target aarch64-unknown-linux-gnu --examples

# The arm64 stages are built natively on an arm64 runner (ubuntu-24.04-arm): no emulator.
FROM toolchain AS native-arm64
ENV CARGO_NET_OFFLINE=true
COPY . .

# aarch64 kernel tests, both HEVC_SIMD tiers (HEVC_SIMD=scalar forces the scalar
# one): the NEON kernel and cross-tier tests (names with tiers_, fuzz_, frames_match_scalar) and the
# dispatch tests (simd::tests). The full suite runs natively in the test job.
FROM native-arm64 AS arm64
RUN --mount=type=cache,id=hevc-rs-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo test --locked --target aarch64-unknown-linux-gnu --lib -- tiers_ frames_match_scalar simd::tests fuzz_ \
    && cargo test --locked --target aarch64-unknown-linux-gnu --lib -- --nocapture tier_is_detected_or_forced \
    && HEVC_SIMD=scalar cargo test --locked --target aarch64-unknown-linux-gnu --lib -- tiers_ frames_match_scalar simd::tests fuzz_

# libde265 oracle (CI only, never linked into the crate): dec265 built from the tag, static
# libde265. LGPL: it is a test tool here, nothing of it is copied into the crate.
FROM debian:bookworm AS c-oracle
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates cmake g++ git make \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch v1.0.15 https://github.com/strukturag/libde265 /libde265 \
    && cmake -S /libde265 -B /libde265/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/oracle \
        -DENABLE_SDL=OFF -DENABLE_ENCODER=OFF -DBUILD_SHARED_LIBS=OFF \
    && cmake --build /libde265/build -j"$(nproc)" \
    && install -m 755 /libde265/build/dec265/dec265 /usr/local/bin/dec265 \
    && cmake --install /libde265/build

# libheif v1.23.7 (LGPL-3.0) oracle for the HEIC gate, CI only, never linked into the crate: built
# from git at the pinned commit, static, with libde265 (installed into /opt/oracle above, found
# through its pkg-config file) as the only codec. The library build comes first; the .c file is
# copied after it, so a change to it does not rebuild the library.
FROM c-oracle AS heic-oracle
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV PKG_CONFIG_PATH=/opt/oracle/lib/pkgconfig:/opt/oracle/lib/x86_64-linux-gnu/pkgconfig
RUN git clone -q --depth 1 --branch v1.23.7 https://github.com/strukturag/libheif /src/libheif \
    && test "$(git -C /src/libheif rev-parse HEAD)" = 12a259a5882dd99d45ec5fc77e296a3b620d1e2e \
    && cmake -S /src/libheif -B /b/heif -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/oracle \
       -DCMAKE_PREFIX_PATH=/opt/oracle \
       -DBUILD_SHARED_LIBS=OFF -DENABLE_PLUGIN_LOADING=OFF -DBUILD_TESTING=OFF -DBUILD_DOCUMENTATION=OFF \
       -DWITH_LIBDE265=ON -DWITH_X265=OFF -DWITH_X264=OFF -DWITH_OpenH264_DECODER=OFF -DWITH_KVAZAAR=OFF \
       -DWITH_AOM_DECODER=OFF -DWITH_AOM_ENCODER=OFF -DWITH_DAV1D=OFF -DWITH_RAV1E=OFF -DWITH_SvtEnc=OFF \
       -DWITH_JPEG_DECODER=OFF -DWITH_JPEG_ENCODER=OFF -DWITH_OpenJPEG_DECODER=OFF -DWITH_OpenJPEG_ENCODER=OFF \
       -DWITH_OPENJPH_ENCODER=OFF -DWITH_FFMPEG_DECODER=OFF -DWITH_UVG266=OFF -DWITH_VVDEC=OFF -DWITH_VVENC=OFF \
       -DWITH_UNCOMPRESSED_CODEC=OFF -DWITH_HEADER_COMPRESSION=OFF -DWITH_LIBSHARPYUV=OFF \
       -DWITH_EXAMPLES=OFF -DWITH_GDK_PIXBUF=OFF \
    && cmake --build /b/heif -j"$(nproc)" && cmake --install /b/heif \
    && rm -rf /src/libheif /b/heif
COPY tests/parity/c /c
# libheif is C++: the C oracle is compiled with gcc and linked with g++.
RUN pkg-config --static --libs libheif | grep -q de265 \
    && gcc -O2 -std=gnu99 -Wall -c -o /tmp/heic_oracle.o /c/heic_oracle.c $(pkg-config --cflags libheif) \
    && g++ -o /usr/local/bin/heic-oracle /tmp/heic_oracle.o -static-libstdc++ -static-libgcc $(pkg-config --static --libs libheif)

# Debian bookworm ffmpeg and ffprobe (the second oracle), on top of the toolchain so the parity
# stage can build the crate in the same image. No source yet: a commit does not reinstall it.
FROM toolchain AS ffmpeg-oracle
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_NET_OFFLINE=true

# The public corpus (tests/parity/corpus.toml: sizes and SHA-256 are checked, "TODO" is printed
# instead) in a cache mount, so a build downloads it once. sharing=locked: two builds never
# download at once. Only a list of the bitstream files reaches the layer; the stages that read the
# corpus mount the same cache and depend on this stage through that list.
# A cache mount is not saved to the Actions cache, and each hosted job starts a new builder, so the
# mount is empty there. CORPUS_RUN (the workflow run id) makes the fetch run again in each build,
# so a fetch layer restored from the Actions cache never leaves the readers with an empty mount.
FROM debian:bookworm-slim AS data
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates unzip \
    && rm -rf /var/lib/apt/lists/*
COPY tests/parity/fetch.sh tests/parity/corpus.toml /parity/
ARG CORPUS_RUN=0
RUN --mount=type=cache,id=hevc-rs-data,target=/data,sharing=locked \
    echo "corpus run $CORPUS_RUN" && sh /parity/fetch.sh /parity/corpus.toml /data \
    && (cd /data/x && find . -type f \( -iname '*.bit' -o -iname '*.bin' -o -iname '*.hevc' -o -iname '*.265' \) | sort) > /data.list \
    && wc -l /data.list

# The public HEIC/HEIF corpus (tests/parity/heic-corpus.toml: blobless sparse git checkouts at
# pinned commits, only *.heic *.heif *.hif) in a second cache mount, so a build downloads it once.
# sharing=locked: two builds never download at once. Only a list of the files reaches the layer;
# the stage that reads the corpus mounts the same cache and depends on this stage through that
# list. git-lfs is for the LFS sources. CORPUS_RUN: as for `data` above.
FROM debian:bookworm-slim AS heic-data
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY tests/parity/heic-fetch.sh tests/parity/heic-corpus.toml /parity/
ARG CORPUS_RUN=0
RUN --mount=type=cache,id=hevc-rs-heic-data,target=/heic,sharing=locked \
    echo "corpus run $CORPUS_RUN" && sh /parity/heic-fetch.sh /parity/heic-corpus.toml /heic \
    && (cd /heic/x && find . -type f | sort) > /heic.list && wc -l /heic.list

# The reference tree of hevc-rs on the first 20 corpus streams and the streams of ARM_SET
# (tests/parity/run.sh: 10 and 12 bits, 4:0:0, PCM, transform skip, scaling lists, lossless,
# deblocking overrides, tiles, SAO), one thread, x86_64: the prefix
# streams and what the aarch64 oracle must reproduce byte for byte.
FROM src AS x86-ref
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=hevc-rs-target-parity,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle && cp target/release/examples/oracle /usr/local/bin/hevc-oracle
RUN --mount=type=cache,id=hevc-rs-data,target=/data,sharing=locked,readonly \
    sh tests/parity/run.sh ref /ref

# Parity gates (tests/parity/run.sh full): ffprobe vs probe, libde265 and ffmpeg vs hevc-rs
# pictures, decoded picture hashes, --threads 1 vs default threads. The stage does not fail on a
# diff: the report (summary.md, *.diff, status) is the build output (target parity-report), and
# the workflow fails on `status`, as the report is wanted most when the gate is red.
# PARITY_DECODE=1 (the default) gates the picture comparisons (dec265, ffmpeg, hash); 0 makes
# them report only. The HEIC gate (G7: examples/heic vs libheif on the HEIC corpus) and its second
# pass with the transforms applied on both sides (G7t) are always on.
FROM ffmpeg-oracle AS parity
COPY . .
COPY --from=c-oracle /usr/local/bin/dec265 /usr/local/bin/dec265
COPY --from=heic-oracle /usr/local/bin/heic-oracle /usr/local/bin/heic-oracle
COPY --from=data /data.list /data.list
COPY --from=heic-data /heic.list /heic.list
RUN --mount=type=cache,id=hevc-rs-target-parity,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example heic \
    && cp target/release/examples/oracle /usr/local/bin/hevc-oracle \
    && cp target/release/examples/heic /usr/local/bin/hevc-heic
ARG PARITY_DECODE=1
RUN --mount=type=cache,id=hevc-rs-data,target=/data,sharing=locked,readonly \
    --mount=type=cache,id=hevc-rs-heic-data,target=/heic,sharing=locked,readonly \
    mkdir -p /report && { PARITY_DECODE=$PARITY_DECODE sh tests/parity/run.sh full /report; echo $? > /report/status; }

FROM scratch AS parity-report
COPY --from=parity /report /

# The x86_64 reference tree as a build output (job parity-ref uploads it for parity-arm64).
FROM scratch AS x86-ref-out
COPY --from=x86-ref /ref /

# The oracle built natively for aarch64, run on the prefix streams of the x86-ref tree, both
# HEVC_SIMD tiers, compared with the x86_64 output. The workflow passes the tree of `x86-ref-out`
# as the named build context `x86ref`.
FROM native-arm64 AS parity-arm64
RUN --mount=type=cache,id=hevc-rs-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --target aarch64-unknown-linux-gnu --example oracle \
    && cp target/aarch64-unknown-linux-gnu/release/examples/oracle /usr/local/bin/hevc-oracle-arm64
RUN --mount=type=bind,from=x86ref,target=/ref \
    sh tests/parity/run.sh arm /ref

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The workflow
# runs it with `docker create` + `docker start`, so a BuildKit cache hit cannot skip the fuzzing.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && cargo install cargo-fuzz --locked --version 0.13.1
ENV CARGO_NET_OFFLINE=true

# Seeds, one directory per target (the formats differ). annexb, probe and decode: a few tiny
# hand-written Annex B streams (a VPS, the start of an SPS, an AUD) and the first 64 KiB (the
# fuzz -max_len) of the first 20 conformance streams from the data cache, which hold the
# parameter sets and the first intra slices. hvcc: a split byte, a minimal hvcC record with one
# SPS array, and one length-prefixed AUD sample. params: the SPS bytes with no start code or NAL
# header. heic: the first 20 HEIC corpus files under 64 KiB, plus the `mini` box files (C71 and
# both lightning_mini). Fuzz-grown files are added by fuzz.yml.
FROM fuzz-tools AS fuzz
COPY . .
COPY --from=data /data.list /data.list
COPY --from=heic-data /heic.list /heic.list
RUN --mount=type=cache,id=hevc-rs-data,target=/data,sharing=locked,readonly \
    --mount=type=cache,id=hevc-rs-heic-data,target=/heic,sharing=locked,readonly \
    mkdir -p tests/fuzz/seeds/annexb tests/fuzz/seeds/probe tests/fuzz/seeds/decode tests/fuzz/seeds/hvcc tests/fuzz/seeds/params tests/fuzz/seeds/heic tests/fuzz/seeds/transform_tiers tests/fuzz/seeds/intra_tiers tests/fuzz/seeds/sao_tiers tests/fuzz/seeds/deblock_tiers \
    && printf '\000\000\000\001\100\001\014\001\377\377\001\140\000\000\003\000\220\000\000\003\000\000\003\000\135\254\131' > tests/fuzz/seeds/annexb/vps.bit \
    && printf '\000\000\000\001\102\001\001\001\140\000\000\000\220\000\000\000\000\000\135\240\002\200\200\055\026\131\131\244' > tests/fuzz/seeds/annexb/sps.bit \
    && printf '\000\000\001\106\001\120' > tests/fuzz/seeds/annexb/aud.bit \
    && printf '\001\001\140\000\000\000\220\000\000\000\000\000\135\240\002\200\200\055\026\131\131\244' > tests/fuzz/seeds/params/sps.bin \
    && printf '\064\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\377\001\041\000\001\000\030\102\001\001\001\140\000\000\000\220\000\000\000\000\000\135\240\002\200\200\055\026\131\131\244\000\000\000\003\106\001\120' > tests/fuzz/seeds/hvcc/hvcc.bin \
    && head -n 20 /data.list | while IFS= read -r f; do \
         n=${f#./}; head -c 65536 "/data/x/$n" > "tests/fuzz/seeds/annexb/$(echo "$n" | tr / _)"; \
       done \
    && (cd /heic/x && { find . -type f -size -64k | sort | head -n 20; find . -type f \( -iname "*mini*" -o -name C71.heic \) | sort; }) | while IFS= read -r f; do \
         n=${f#./}; cp "/heic/x/$n" "tests/fuzz/seeds/heic/$(echo "$n" | tr / _)"; \
       done \
    && cp tests/fuzz/seeds/annexb/* tests/fuzz/seeds/probe/ \
    && cp tests/fuzz/seeds/annexb/* tests/fuzz/seeds/decode/ \
    && cargo +nightly fuzz build --fuzz-dir tests/fuzz

# The tier targets built natively on an arm64 runner (default sanitizer), run by fuzz.yml
# (`fuzz-arm64`): they compare the NEON kernels with scalar.
FROM fuzz-tools AS fuzz-arm64
COPY . .
# `cargo fuzz build` takes one target.
RUN for t in transform_tiers intra_tiers sao_tiers deblock_tiers; do \
      cargo +nightly fuzz build --fuzz-dir tests/fuzz $t || exit 1; \
    done

# Benchmark (bench.yml, tests/parity/bench.sh): libde265 (dec265), ffmpeg, libheif (heic-oracle)
# and hevc-rs on the same corpus, in one container, so the ratios compare the same machine.
# hevc-rs is built twice: the default build (runtime SIMD dispatch) and the x86-64-v4 build with
# the Zen 4/5 false-dependency features of the release binaries, with its
# own cache mount since RUSTFLAGS changes every artifact. BENCH_RUN changes per run, so a cache
# hit of the same source does not reuse an old timing.
FROM ffmpeg-oracle AS bench
COPY --from=c-oracle /usr/local/bin/dec265 /usr/local/bin/dec265
COPY --from=heic-oracle /usr/local/bin/heic-oracle /usr/local/bin/heic-oracle
COPY --from=data /data.list /data.list
COPY --from=heic-data /heic.list /heic.list
COPY . .
RUN --mount=type=cache,id=hevc-rs-target-bench,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example bench --example heic \
    && cp target/release/examples/oracle /usr/local/bin/hevc-oracle \
    && cp target/release/examples/bench /usr/local/bin/hevc-bench \
    && cp target/release/examples/heic /usr/local/bin/hevc-heic
RUN --mount=type=cache,id=hevc-rs-target-bench-v4,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" \
       cargo build --locked --release --example bench --example heic \
    && cp target/release/examples/bench /usr/local/bin/hevc-bench-v4 \
    && cp target/release/examples/heic /usr/local/bin/hevc-heic-v4
ARG BENCH_RUN=0
RUN --mount=type=cache,id=hevc-rs-data,target=/data,sharing=locked,readonly \
    --mount=type=cache,id=hevc-rs-heic-data,target=/heic,sharing=locked,readonly \
    echo "run $BENCH_RUN" && mkdir -p /report && { sh tests/parity/bench.sh /report; echo $? > /report/status; }

FROM scratch AS bench-report
COPY --from=bench /report /

# Where the time goes (profile.yml, tests/parity/profile.sh): perf (and valgrind's callgrind as
# the fallback when the host does not allow perf events) on the bench example built with line
# debug info, and on dec265 for comparison, over the intra prefix of every corpus stream. The
# prefix files are made here and kept in the image, since `docker run` has no cache mount.
FROM ffmpeg-oracle AS profile-tools
RUN apt-get update && apt-get install -y --no-install-recommends linux-perf valgrind \
    && rm -rf /var/lib/apt/lists/*

FROM profile-tools AS profile
COPY --from=c-oracle /usr/local/bin/dec265 /usr/local/bin/dec265
COPY --from=data /data.list /data.list
COPY . .
RUN --mount=type=cache,id=hevc-rs-target-profile,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && CARGO_PROFILE_RELEASE_DEBUG=line-tables-only cargo build --locked --release --example oracle --example bench \
    && cp target/release/examples/oracle /usr/local/bin/hevc-oracle \
    && cp target/release/examples/bench /usr/local/bin/hevc-bench
RUN --mount=type=cache,id=hevc-rs-data,target=/data,sharing=locked,readonly \
    mkdir -p /prof/in && while IFS= read -r f; do \
         n=${f#./}; e=$(printf '%s' "${n##*.}" | tr 'A-Z' 'a-z'); cp "/data/x/$n" "/prof/in/$(echo "${n%.*}" | tr / _).$e"; \
       done < /data.list \
    && hevc-oracle prefix /prof/in /prof/prefix && rm -rf /prof/in && ls /prof/prefix | wc -l
