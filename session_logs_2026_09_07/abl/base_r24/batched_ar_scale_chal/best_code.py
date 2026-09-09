
def evolved_p130(grads, scales, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[gi][i] = sum_r grads_r[gi][i] * scales_r[gi] for gi in 0..4
    # Optimization: cat + single AR + split to amortize dispatch overhead
    
    # Step 1: Locally scale each gradient
    scaled = [g * s for g, s in zip(grads, scales)]
    
    # Step 2: Remember shapes and flatten
    shapes = [g.shape for g in scaled]
    sizes = [g.numel() for g in scaled]
    
    # Step 3: Concatenate into one flat tensor
    flat = torch.cat([g.reshape(-1) for g in scaled])
    
    # Step 4: Single all_reduce operation
    reduced_flat = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Step 5: Split back using narrow (metadata-only view)
    out = []
    offset = 0
    for shape, n in zip(shapes, sizes):
        out.append(torch.narrow(reduced_flat, 0, offset, n).reshape(shape))
        offset += n
    
    return out
