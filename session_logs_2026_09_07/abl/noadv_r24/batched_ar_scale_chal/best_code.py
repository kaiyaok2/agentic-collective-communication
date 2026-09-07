def evolved_p130(grads, scales, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[gi][i] = sum_r grads_r[gi][i] * scales_r[gi] for each gi in 0..4
    # Optimization: cat all scaled grads, single AR, then narrow/split
    
    # Step 1: Scale each gradient locally
    scaled_grads = [g * s for g, s in zip(grads, scales)]
    
    # Step 2: Record shapes and sizes
    shapes = [g.shape for g in scaled_grads]
    sizes = [g.numel() for g in scaled_grads]
    
    # Step 3: Concatenate into one flat tensor
    flat = torch.cat([g.reshape(-1) for g in scaled_grads])
    
    # Step 4: Single all-reduce
    reduced_flat = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Step 5: Split back using narrow (metadata-only view)
    out = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        out.append(torch.narrow(reduced_flat, 0, offset, size).reshape(shape))
        offset += size
    
    return out
