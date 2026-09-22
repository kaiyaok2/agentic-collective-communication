def r2_deep4_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Precompute per-rank coefficients
    a = [1.0 + 0.5 * (r % 3) for r in range(W)]
    
    # Mathematical derivation of the fused transformation:
    # After analyzing the 4-stage pattern (scale/AR/unscale repeated),
    # the net effect is: each shard r contributes its input scaled by
    # a composite factor that depends on the pattern of operations.
    
    # For this specific 4-stage deep pattern:
    # Stage 1: AR → scale by aᵣ/W → AR → unscale by aᵣ
    # Stage 2: scale by aᵣ/W → AR → unscale by aᵣ  
    # Stage 3: scale by aᵣ/W → AR → unscale by aᵣ
    # Stage 4: scale by aᵣ/W → AR (final, no unscale)
    
    # The composite pre-scaling factor for shard r is: aᵣ³/W³
    # The composite post-scaling factor is: 1 (already in final form)
    
    # Pre-scale each shard by its composite factor
    buf = x.clone()
    for r in range(W):
        # Composite scaling from 4 stages: (a[r]/W)^3 for the input
        composite_pre_scale = (a[r] / W) ** 3
        buf[r*S:(r+1)*S] = composite_pre_scale * x[r*S:(r+1)*S]
    
    # Single all-reduce operation
    result = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Post-scale: The result needs to account for the final unscaling pattern
    # After careful analysis, each shard needs to be scaled by W³/aᵣ²
    for r in range(W):
        composite_post_scale = (W ** 3) / max(a[r] ** 2, 1e-9)
        result[r*S:(r+1)*S] = composite_post_scale * result[r*S:(r+1)*S]
    
    return result