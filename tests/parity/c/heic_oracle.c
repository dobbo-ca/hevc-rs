/* libheif oracle of hevc-rs (CI only, docs/parity.md): decodes the primary image of every input
 * file at its native chroma format and bit depth, to the files of examples/heic.rs. Written from
 * the public libheif API (LGPL-3.0 library, linked statically into this CI tool, never shipped);
 * no libheif code is copied. Structure after isobmff-rs tests/parity/c/heif_oracle.c.
 *
 *   heic-oracle <in-dir> <out-dir>
 *
 * Every regular file of <in-dir> (flat; run.sh flattens the corpus) is decoded; <name> is the file
 * name without its last extension. Output per file:
 *   <name>.yuv    planes Y, Cb, Cr (Y only when monochrome), cropped to the image size, 1 byte per
 *                 sample at depth 8, else 2 bytes little-endian
 *   <name>.alpha  the alpha plane, same sample format; only when the image has one
 *   <name>.info   "W H chroma depthY depthC alpha": chroma 400/420/422/444, alpha 0/1; for a
 *                 monochrome image libheif has no chroma plane, so depthC is written as depthY
 * or <name>.err:  "convert" when libheif gives RGB or an interleaved layout, "decode" otherwise.
 * ignore_transformations = 1: no rotation, mirror or crop by the transforms. Security limits off.
 */
#define _POSIX_C_SOURCE 200809L
#include <dirent.h>
#include <errno.h>
#include <libheif/heif.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define MAX_FILE ((uint64_t)512 << 20)

/* Reads the whole file; NULL on any failure. *n is the length. */
static uint8_t *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    uint8_t *buf;
    long sz;
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) != 0 ||
        (uint64_t)sz > MAX_FILE) {
        fclose(f);
        return NULL;
    }
    buf = (uint8_t *)malloc((size_t)sz + 1);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return buf;
}

static FILE *open_out(const char *dir, const char *name, const char *ext)
{
    char path[4096];
    if (snprintf(path, sizeof path, "%s/%s.%s", dir, name, ext) >= (int)sizeof path) {
        return NULL;
    }
    return fopen(path, "wb");
}

/* 0 on success. */
static int write_text(const char *dir, const char *name, const char *ext, const char *s)
{
    FILE *f = open_out(dir, name, ext);
    int ok;
    if (!f) {
        return -1;
    }
    ok = fputs(s, f) >= 0;
    return (fclose(f) == 0 && ok) ? 0 : -1;
}

/* Writes the plane to f at its own depth; returns 0 on success. */
static int put_plane(FILE *f, const heif_image *img, heif_channel ch)
{
    int w = heif_image_get_width(img, ch), h = heif_image_get_height(img, ch);
    int wide = heif_image_get_bits_per_pixel_range(img, ch) > 8;
    size_t stride = 0;
    const uint8_t *p = heif_image_get_plane_readonly2(img, ch, &stride);
    int x, y;
    if (!p || w < 0 || h < 0) {
        return -1;
    }
    for (y = 0; y < h; y++) {
        const uint8_t *row = p + (size_t)y * stride;
        if (!wide) {
            if (fwrite(row, 1, (size_t)w, f) != (size_t)w) {
                return -1;
            }
            continue;
        }
        for (x = 0; x < w; x++) {
            uint16_t v;
            memcpy(&v, row + 2 * (size_t)x, 2);
            if (fputc(v & 0xff, f) == EOF || fputc(v >> 8, f) == EOF) {
                return -1;
            }
        }
    }
    return 0;
}

/* Decodes one file; returns 1 when an output file could not be written. */
static int decode_one(const char *path, const char *name, const char *out)
{
    heif_context *ctx = heif_context_alloc();
    heif_image_handle *hnd = NULL;
    heif_image *img = NULL;
    heif_decoding_options *opt = heif_decoding_options_alloc();
    uint8_t *data;
    size_t len = 0;
    int bad = 0;
    const char *err = "decode";

    data = read_file(path, &len);
    if (!ctx || !opt || !data ||
        heif_context_set_security_limits(ctx, heif_get_disabled_security_limits()).code != heif_error_Ok ||
        heif_context_read_from_memory_without_copy(ctx, data, len, NULL).code != heif_error_Ok ||
        heif_context_get_primary_image_handle(ctx, &hnd).code != heif_error_Ok) {
        goto fail;
    }
    opt->ignore_transformations = 1;
    /* Keep the samples as decoded: with a NULL output profile libheif otherwise converts to sRGB
       defaults (full range), e.g. limited-range luma 233 becomes 253. */
    opt->output_image_nclx_profile_passthrough = 1;
    if (heif_decode_image(hnd, &img, heif_colorspace_undefined, heif_chroma_undefined, opt).code != heif_error_Ok) {
        goto fail;
    }
    {
        heif_colorspace cs = heif_image_get_colorspace(img);
        heif_chroma ch = heif_image_get_chroma_format(img);
        int mono = cs == heif_colorspace_monochrome && ch == heif_chroma_monochrome;
        int chroma;
        char info[128];
        FILE *f;
        if (mono) {
            chroma = 400;
        } else if (cs == heif_colorspace_YCbCr && ch == heif_chroma_420) {
            chroma = 420;
        } else if (cs == heif_colorspace_YCbCr && ch == heif_chroma_422) {
            chroma = 422;
        } else if (cs == heif_colorspace_YCbCr && ch == heif_chroma_444) {
            chroma = 444;
        } else {
            err = "convert";
            goto fail;
        }
        f = open_out(out, name, "yuv");
        if (!f) {
            bad = 1;
            goto done;
        }
        if (put_plane(f, img, heif_channel_Y) != 0 ||
            (!mono && (put_plane(f, img, heif_channel_Cb) != 0 || put_plane(f, img, heif_channel_Cr) != 0))) {
            bad = 1;
        }
        if (fclose(f) != 0) {
            bad = 1;
        }
        if (heif_image_has_channel(img, heif_channel_Alpha)) {
            f = open_out(out, name, "alpha");
            if (!f) {
                bad = 1;
                goto done;
            }
            if (put_plane(f, img, heif_channel_Alpha) != 0 || fclose(f) != 0) {
                bad = 1;
            }
        }
        snprintf(info, sizeof info, "%d %d %d %d %d %d", heif_image_get_width(img, heif_channel_Y),
                 heif_image_get_height(img, heif_channel_Y), chroma,
                 heif_image_get_bits_per_pixel_range(img, heif_channel_Y),
                 heif_image_get_bits_per_pixel_range(img, mono ? heif_channel_Y : heif_channel_Cb),
                 heif_image_has_channel(img, heif_channel_Alpha) ? 1 : 0);
        bad |= write_text(out, name, "info", info) != 0;
    }
    goto done;
fail:
    bad = write_text(out, name, "err", err) != 0;
done:
    if (img) {
        heif_image_release(img);
    }
    if (hnd) {
        heif_image_handle_release(hnd);
    }
    if (opt) {
        heif_decoding_options_free(opt);
    }
    if (ctx) {
        heif_context_free(ctx);
    }
    free(data);
    return bad;
}

int main(int argc, char **argv)
{
    DIR *d;
    struct dirent *e;
    int bad = 0;
    if (argc != 3) {
        fprintf(stderr, "usage: heic-oracle <in-dir> <out-dir>\n");
        return 2;
    }
    if (mkdir(argv[2], 0777) != 0 && errno != EEXIST) {
        fprintf(stderr, "heic-oracle: %s: %s\n", argv[2], strerror(errno));
        return 2;
    }
    d = opendir(argv[1]);
    if (!d) {
        fprintf(stderr, "heic-oracle: %s: %s\n", argv[1], strerror(errno));
        return 2;
    }
    heif_init(NULL);
    if (!heif_have_decoder_for_format(heif_compression_HEVC)) {
        fprintf(stderr, "heic-oracle: libheif has no HEVC decoder\n");
        return 2;
    }
    while ((e = readdir(d)) != NULL) {
        char path[4096], name[4096];
        struct stat st;
        char *dot;
        if (snprintf(path, sizeof path, "%s/%s", argv[1], e->d_name) >= (int)sizeof path ||
            snprintf(name, sizeof name, "%s", e->d_name) >= (int)sizeof name) {
            bad = 1;
            continue;
        }
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode)) {
            continue;
        }
        dot = strrchr(name, '.');
        if (dot && dot != name) {
            *dot = 0;
        }
        bad |= decode_one(path, name, argv[2]);
    }
    closedir(d);
    heif_deinit();
    return bad ? 1 : 0;
}
