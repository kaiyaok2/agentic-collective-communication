
def r43_route_B9_d8_L3_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 9; W = world_size; L = 3; OFF = 2; STR = 1
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Precompute this rank's window mask
    start = (rank + OFF) % B
    keep_blocks = [(start + STR*j) % B for j in range(L)]
    mask = torch.zeros_like(s)
    for b in keep_blocks:
        mask[b*S:(b+1)*S] = 1.0
    
    # Final masked all_reduce
    buf = s * mask
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
