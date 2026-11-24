//! `Options::apply_transforms` in `decode_heic` and `decode_heic_item`, on HEIF files built in the
//! test around real coded pictures (`common`). The expected planes come from the default (raw)
//! decode and a small reference of each transform written here.

mod common;

use common::{Item, bx, fbx, heif, hvc1, ispe};
use hevc::{Error, HeicImage, Limits, Options, Plane, Samples};
use std::sync::atomic::AtomicBool;

const APPLY: Options = Options {
    deblocking: true,
    sao: true,
    metadata: true,
    check_hash: false,
    apply_transforms: true,
};

/// One coded picture of `w` x `h`: the `hvcC` box and the item data.
fn coded(w: u32, h: u32) -> (Vec<u8>, Vec<u8>) {
    hvc1(&common::stream(&common::Spec {
        width: w,
        height: h,
        pictures: 1,
        ..Default::default()
    }))
}

/// An `hvc1` item of `w` x `h` with `extra` properties after `ispe` and `hvcC`.
fn hvc_item(id: u16, w: u32, h: u32, extra: &[Vec<u8>]) -> Item {
    let (hvcc, data) = coded(w, h);
    let mut props = vec![ispe(w, h), hvcc];
    props.extend_from_slice(extra);
    Item {
        id,
        kind: *b"hvc1",
        data,
        props,
    }
}

fn irot(deg: u16) -> Vec<u8> {
    bx(b"irot", &[(deg / 90) as u8])
}

/// `imir`: axis 0 swaps top and bottom, 1 left and right.
fn imir(axis: u8) -> Vec<u8> {
    bx(b"imir", &[axis])
}

/// A centred `clap` of `w` x `h` (integer fractions, no offset).
fn clap(w: u32, h: u32) -> Vec<u8> {
    let v: [u32; 8] = [w, 1, h, 1, 0, 1, 0, 1];
    bx(
        b"clap",
        &v.iter().flat_map(|x| x.to_be_bytes()).collect::<Vec<_>>(),
    )
}

fn u8s(p: &Plane) -> &[u8] {
    match &p.samples {
        Samples::U8(v) => v,
        Samples::U16(_) => panic!("8-bit test pictures"),
    }
}

/// `out[y][x] = f(x, y)` for an output of `w` x `h`.
fn make(w: u32, h: u32, f: impl Fn(usize, usize) -> u8) -> Plane {
    let s = (0..h as usize)
        .flat_map(|y| (0..w as usize).map(move |x| (x, y)))
        .map(|(x, y)| f(x, y))
        .collect();
    Plane {
        width: w,
        height: h,
        samples: Samples::U8(s),
    }
}

/// The reference of one transform on one plane: `Rot(d)` anticlockwise, `Flip(true)` swaps top
/// and bottom, `Flip(false)` left and right, `Crop(x, y, w, h)`.
#[derive(Clone, Copy)]
enum T {
    Rot(u16),
    Flip(bool),
    Crop(u32, u32, u32, u32),
}

fn one(p: &Plane, t: T) -> Plane {
    let (w, h, s) = (p.width as usize, p.height as usize, u8s(p));
    let at = |x: usize, y: usize| s[y * w + x];
    match t {
        T::Rot(90) => make(p.height, p.width, |x, y| at(w - 1 - y, x)),
        T::Rot(180) => make(p.width, p.height, |x, y| at(w - 1 - x, h - 1 - y)),
        T::Rot(270) => make(p.height, p.width, |x, y| at(y, h - 1 - x)),
        T::Flip(true) => make(p.width, p.height, |x, y| at(x, h - 1 - y)),
        T::Flip(false) => make(p.width, p.height, |x, y| at(w - 1 - x, y)),
        T::Crop(cx, cy, cw, ch) => make(cw, ch, |x, y| at(cx as usize + x, cy as usize + y)),
        T::Rot(_) => unreachable!(),
    }
}

/// `ts` applied to the planes, luma first then 4:2:0 chroma (a crop at half the coordinates).
fn reference(planes: &[Plane], ts: &[T]) -> Vec<Plane> {
    planes
        .iter()
        .enumerate()
        .map(|(i, p)| {
            ts.iter().fold(p.clone(), |p, &t| match (i, t) {
                (1.., T::Crop(x, y, w, h)) => one(&p, T::Crop(x / 2, y / 2, w / 2, h / 2)),
                _ => one(&p, t),
            })
        })
        .collect()
}

fn decode(file: &[u8], opts: &Options, limits: &Limits) -> Result<HeicImage, Error> {
    hevc::decode_heic(file, opts, limits, &AtomicBool::new(false))
}

/// Checks the transformed decode of `file` against `ts` applied to its raw decode, and that the
/// raw decode is that of `plain` (the same image without transforms).
fn check(file: &[u8], plain: &[u8], ts: &[T]) -> HeicImage {
    let l = Limits::default();
    let raw = decode(file, &Options::default(), &l).expect("raw decode");
    assert!(!raw.transforms_applied);
    let base = decode(plain, &Options::default(), &l).expect("plain decode");
    assert_eq!(raw.image, base.image, "the default output must not change");
    let got = decode(file, &APPLY, &l).expect("transformed decode");
    assert!(got.transforms_applied);
    assert_eq!(got.transforms, raw.transforms, "the transforms stay listed");
    let want = reference(&raw.image.planes, ts);
    assert_eq!(got.image.planes, want);
    let i = &got.image.info;
    assert_eq!((i.width, i.height), (want[0].width, want[0].height));
    assert_eq!(
        (i.coded_width, i.coded_height),
        (raw.image.info.coded_width, raw.image.info.coded_height)
    );
    // decode_heic_item with the option gives the same picture.
    let cancel = AtomicBool::new(false);
    let parsed = isobmff::parse(file, &isobmff::Limits::default(), &cancel).unwrap();
    let id = parsed.primary_image().unwrap().id;
    let item = hevc::decode_heic_item(&parsed, file, id, &APPLY, &l, &cancel);
    assert_eq!(item.as_ref(), Ok(&got.image));
    got
}

#[test]
fn each_transform_kind() {
    let plain = heif(&[hvc_item(1, 64, 32, &[])], &[]);
    let cases: [(Vec<Vec<u8>>, Vec<T>); 9] = [
        (vec![irot(90)], vec![T::Rot(90)]),
        (vec![irot(180)], vec![T::Rot(180)]),
        (vec![irot(270)], vec![T::Rot(270)]),
        (vec![imir(0)], vec![T::Flip(true)]),
        (vec![imir(1)], vec![T::Flip(false)]),
        // 64 x 32 to 32 x 16, centred: origin (16, 8).
        (vec![clap(32, 16)], vec![T::Crop(16, 8, 32, 16)]),
        // MIAF order, and the other order (each clap on the size at that point).
        (
            vec![clap(32, 16), irot(90), imir(1)],
            vec![T::Crop(16, 8, 32, 16), T::Rot(90), T::Flip(false)],
        ),
        (
            vec![irot(90), clap(16, 32)],
            vec![T::Rot(90), T::Crop(8, 16, 16, 32)],
        ),
        (vec![], vec![]),
    ];
    for (props, ts) in cases {
        let file = heif(&[hvc_item(1, 64, 32, &props)], &[]);
        check(&file, &plain, &ts);
    }
}

#[test]
fn grid_is_transformed_after_it_is_assembled() {
    // 2 x 1 tiles of 64 x 32 cropped to 120 x 32, turned by 90 degrees.
    let grid = |props: Vec<Vec<u8>>| {
        let mut p = vec![ispe(120, 32)];
        p.extend(props);
        let items = [
            Item {
                id: 1,
                kind: *b"grid",
                data: vec![0, 0, 0, 1, 0, 120, 0, 32],
                props: p,
            },
            hvc_item(2, 64, 32, &[]),
            hvc_item(3, 64, 32, &[]),
        ];
        heif(&items, &[(b"dimg", 1, &[2u16, 3][..])])
    };
    let file = grid(vec![irot(90)]);
    let got = check(&file, &grid(vec![]), &[T::Rot(90)]);
    assert_eq!((got.image.info.width, got.image.info.height), (32, 120));
    // The same for every max_threads.
    for t in [1, 2, 4] {
        let l = Limits {
            max_threads: t,
            ..Limits::default()
        };
        assert_eq!(decode(&file, &APPLY, &l).as_ref(), Ok(&got), "threads {t}");
    }
    // A tile with transforms of its own: libheif turns the tile first, this crate does not.
    let items = [
        Item {
            id: 1,
            kind: *b"grid",
            data: vec![0, 0, 0, 1, 0, 120, 0, 32],
            props: vec![ispe(120, 32)],
        },
        hvc_item(2, 64, 32, &[imir(1)]),
        hvc_item(3, 64, 32, &[]),
    ];
    let tiled = heif(&items, &[(b"dimg", 1, &[2u16, 3][..])]);
    assert!(decode(&tiled, &Options::default(), &Limits::default()).is_ok());
    assert!(matches!(
        decode(&tiled, &APPLY, &Limits::default()),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn alpha_takes_its_own_transforms() {
    let aux = || fbx(b"auxC", b"urn:mpeg:hevc:2015:auxid:1\0");
    let file = |alpha_props: Vec<Vec<u8>>| {
        let mut ap = vec![aux()];
        ap.extend(alpha_props);
        heif(
            &[hvc_item(1, 64, 32, &[irot(90)]), hvc_item(2, 64, 32, &ap)],
            &[(b"auxl", 2, &[1u16][..])],
        )
    };
    let l = Limits::default();
    for (props, ts) in [
        (vec![irot(90)], vec![T::Rot(90)]),
        (vec![imir(0)], vec![T::Flip(true)]),
        // No transforms of its own: the alpha plane stays as coded, as in libheif (which then
        // scales it to the size of the image; this crate does not scale).
        (vec![], vec![]),
    ] {
        let f = file(props);
        let raw = decode(&f, &Options::default(), &l).unwrap();
        let got = decode(&f, &APPLY, &l).unwrap();
        let a = raw.alpha.expect("alpha plane");
        assert_eq!(got.alpha, Some(reference(&[a], &ts).remove(0)));
        assert_eq!(
            got.image.planes,
            reference(&raw.image.planes, &[T::Rot(90)])
        );
    }
}

#[test]
fn a_tight_budget_gives_the_image_or_alloc_limit() {
    let file = heif(&[hvc_item(1, 64, 32, &[irot(90), clap(32, 16)])], &[]);
    let want = decode(&file, &APPLY, &Limits::default()).unwrap();
    let mut seen_ok = false;
    for max_alloc in (0..128).map(|k| k * (8 << 10)) {
        let l = Limits {
            max_alloc,
            ..Limits::default()
        };
        match decode(&file, &APPLY, &l) {
            Ok(h) => {
                assert_eq!(h, want, "max_alloc {max_alloc}");
                seen_ok = true;
            }
            Err(e) => assert_eq!(e, Error::AllocLimit, "max_alloc {max_alloc}"),
        }
    }
    assert!(seen_ok, "1 MiB must hold a 64 x 32 picture");
}

#[test]
fn cancel_stops_the_call() {
    let file = heif(&[hvc_item(1, 64, 32, &[irot(90)])], &[]);
    let r = hevc::decode_heic(&file, &APPLY, &Limits::default(), &AtomicBool::new(true));
    assert_eq!(r, Err(Error::Cancelled));
}
