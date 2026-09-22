
def r40_route_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8; W = world_size; L = 3; OFF = 2
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute keep blocks for this rank
    start = (rank + OFF) % B
    keep = [(start + j) % B for j in range(L)]
    
    # Create mask tensor
    mask = torch.zeros_like(s)
    for b in keep:
        mask[b*S:(b+1)*S] = 1.0
    
    # Apply mask and all_reduce
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
