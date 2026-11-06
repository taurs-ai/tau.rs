//! Device dispatch invariants established before selecting native prefill.
use super::*;
use candle::Device;

/// Device policy must not promote the local tuning result on unmeasured GPUs
/// or different expert widths. Explicit overrides use the independent catalog.
#[test]
fn automatic_prefill_tiles_are_scoped_to_measured_geometry() {
    let dimensions = FusedMoeDimensions {
        hidden_size: 2048,
        intermediate_size: 512,
        num_experts: 256,
        experts_per_token: 8,
    };
    assert_eq!(
        device_default(dimensions, Some((12, 0))).gate_up().values(),
        [16, 64, 64, 4, 1]
    );
    assert_eq!(
        device_default(dimensions, Some((10, 3))).gate_up().values(),
        [16, 64, 64, 4, 1]
    );
    for capability in [None, Some((10, 0)), Some((9, 0))] {
        assert_eq!(
            device_default(dimensions, capability),
            crate::MoePrefillConfig::default()
        );
    }
    assert_eq!(
        device_default(
            FusedMoeDimensions {
                hidden_size: 4096,
                ..dimensions
            },
            Some((12, 0))
        ),
        crate::MoePrefillConfig::default()
    );
}

/// Exercise actual GPU metadata with independent host assignment invariants.
/// The same buffers are reused from long to short inputs to expose stale padding.
fn check_dispatch(tile: crate::MoePrefillTile) {
    let m_rows = tile.rows();
    let device = Device::new_cuda(0).unwrap();
    let cuda = device.as_cuda_device().unwrap();
    let stream = cuda.cuda_stream();
    let capacity = tau_cuda_kernels::moe_prefill_capacity_for_tile(257, tile).unwrap();
    let mut meta = cuda.alloc_zeros::<i32>(514).unwrap();
    let mut offsets = cuda.alloc_zeros::<i32>(257).unwrap();
    let mut slots = cuda.alloc_zeros::<i32>(capacity).unwrap();
    let mut tiles = cuda.alloc_zeros::<i32>(capacity / m_rows).unwrap();
    for tokens in [257, 1, 15, 16, 17, 31, 32, 33, 255, 256, 1] {
        for concentrated in [false, true] {
            let mut routes = vec![0f32; tokens * 16];
            for token in 0..tokens {
                for rank in 0..8 {
                    routes[token * 16 + rank] = if concentrated {
                        rank as f32
                    } else {
                        ((token * 8 + rank) % 256) as f32
                    };
                    routes[token * 16 + 8 + rank] = 0.125;
                }
            }
            let routes = stream.clone_htod(&routes).unwrap();
            let (rp, rg) = routes.device_ptr(&stream);
            let (mp, mg) = meta.device_ptr_mut(&stream);
            let (op, og) = offsets.device_ptr_mut(&stream);
            let (sp, sg) = slots.device_ptr_mut(&stream);
            let (tp, tg) = tiles.device_ptr_mut(&stream);
            // SAFETY: All checked capacities are retained on this single stream.
            unsafe {
                tau_cuda_kernels::launch_moe_prefill_dispatch(
                    rp as *const _,
                    mp as *mut _,
                    op as *mut _,
                    sp as *mut _,
                    tp as *mut _,
                    tokens as i32,
                    capacity as i32,
                    tile,
                    stream.cu_stream().cast(),
                )
                .unwrap();
            }
            drop((tg, sg, og, mg, rg));
            let m = stream.clone_dtoh(&meta).unwrap();
            let o = stream.clone_dtoh(&offsets).unwrap();
            let s = stream.clone_dtoh(&slots).unwrap();
            let t = stream.clone_dtoh(&tiles).unwrap();
            assert_eq!(m[512], 0);
            assert_eq!(m[..256].iter().sum::<i32>(), (tokens * 8) as i32);
            let padded = m[513] as usize;
            assert!(padded <= capacity);
            assert_eq!(o[256] as usize, padded);
            let mut seen = vec![false; tokens * 8];
            for expert in 0..256 {
                assert_eq!((o[expert + 1] - o[expert]) % m_rows as i32, 0);
                assert_eq!(m[expert], m[256 + expert]);
                let mut count = 0;
                for row in o[expert] as usize..o[expert + 1] as usize {
                    assert_eq!(t[row / m_rows], expert as i32);
                    if (s[row] as usize) < tokens * 8 {
                        let slot = s[row] as usize;
                        assert!(!std::mem::replace(&mut seen[slot], true));
                        let expected = if concentrated { slot % 8 } else { slot % 256 };
                        assert_eq!(expert, expected);
                        count += 1;
                    } else {
                        assert_eq!(s[row], (tokens * 8) as i32);
                    }
                }
                assert_eq!(count, m[expert]);
            }
            assert!(seen.iter().all(|s| *s));
            assert!(t[padded / m_rows..].iter().all(|e| *e == -1));
        }
    }
}

/// Invalid token counts are rejected before allocation or pointer conversion.
#[test]
fn dispatch_capacity_rejects_empty_and_overflow() {
    assert_eq!(tau_cuda_kernels::moe_prefill_capacity(0), None);
    assert_eq!(tau_cuda_kernels::moe_prefill_capacity(usize::MAX), None);
    assert_eq!(tau_cuda_kernels::moe_prefill_capacity(1), Some(128));
}

/// A nonfinite router must publish an invalid output without dereferencing an
/// expert index or reading missing contribution rows. The next valid call resets
/// the same workspace, exercising error recovery and sanitizer coverage.
#[test]
fn grouped_prefill_nonfinite_routes_fail_closed_and_reset() {
    let device = Device::new_cuda(0).unwrap();
    let weights = Tensor::zeros((256, 32, 32), DType::BF16, &device).unwrap();
    let down = Tensor::zeros((256, 32, 16), DType::BF16, &device).unwrap();
    let runner = FusedMoeRunner::new(weights, down, 8)
        .unwrap()
        .unwrap()
        .prefill_runner();
    let input = Tensor::zeros((1, 32), DType::BF16, &device).unwrap();
    let valid = Tensor::zeros((1, 256), DType::BF16, &device).unwrap();
    let invalid = Tensor::full(f32::NAN, (1, 256), &device)
        .unwrap()
        .to_dtype(DType::BF16)
        .unwrap();
    let values = |x: Tensor| {
        x.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    assert!(
        values(runner.forward(&input, &invalid).unwrap())
            .iter()
            .all(|x| x.is_nan())
    );
    assert_eq!(
        values(runner.forward(&input, &valid).unwrap()),
        vec![0f32; 32]
    );
}

/// Both supported dispatch M values preserve semantic assignment coverage and
/// worst-case padding, including balanced/skewed routes and long-to-short reuse.
#[test]
fn device_dispatch_preserves_all_assignments_and_bounds() {
    for profile in ["16,16,16,1,1", "32,32,32,2,2"] {
        check_dispatch(profile.parse().unwrap());
    }
}
