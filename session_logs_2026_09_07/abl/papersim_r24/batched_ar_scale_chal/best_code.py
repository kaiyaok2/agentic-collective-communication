
def evolved_p130(grads, scales, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[gi][i] = sum_r grads_r[gi][i] * scales_r[gi]
    # For each of 5 gradient tensors, sum across ranks of (grad * scale)
    
    # Scale each gradient locally
    scaled_grads = [g * s for g, s in zip(grads, scales)]
    
    # Concatenate all scaled gradients into one tensor
    sizes = [g.numel() for g in scaled_grads]
    flat = torch.cat([g.reshape(-1) for g in scaled_grads])
    
    # Single all-reduce instead of 5 separate ones
    reduced_flat = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Split back using torch.narrow (metadata-only view)
    out = []
    offset = 0
    for g, n in zip(grads, sizes):
        out.append(torch.narrow(reduced_flat, 0, offset, n).reshape(g.shape))
        offset += n
    
    return out
