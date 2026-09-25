
def r73_mono_b6_a5_L5_w4_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    SIG = [3, 2, 1, 0, 5, 4]
    
    start = (rank + 1) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.45 + 0.02 * (rank % 4)
    
    # All gather instead of all reduce
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    
    # Sum across ranks
    s = gathered.sum(dim=0)
    
    # Build buffer
    buf = torch.zeros_like(s)
    for b in keep:
        jb = SIG[b]
        buf[b*S:(b+1)*S] = w * s[jb*S:(jb+1)*S]
    
    # Final all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
