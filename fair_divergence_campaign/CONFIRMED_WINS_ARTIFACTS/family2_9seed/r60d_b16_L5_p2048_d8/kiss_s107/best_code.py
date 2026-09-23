
def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    # Which buckets does this rank keep?
    start = (rank + 2) % 16
    keep = set((start + j) % 16 for j in range(5))
    
    # Create mask tensor once
    mask = torch.zeros(16 * S, device=x.device, dtype=x.dtype)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Apply mask and all_reduce
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
