
def evolved_p130(grads, scales, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 2: Single concatenated allreduce
    scaled = [g * s for g, s in zip(grads, scales)]
    shapes = [g.shape for g in grads]
    sizes = [g.numel() for g in grads]
    
    # Flatten and concatenate
    flat = torch.cat([g.flatten() for g in scaled])
    
    # Single allreduce
    reduced = xm.all_reduce(xm.REDUCE_SUM, flat)
    
    # Split back
    out = []
    offset = 0
    for shape, size in zip(shapes, sizes):
        out.append(reduced[offset:offset+size].reshape(shape))
        offset += size
    
    return out
