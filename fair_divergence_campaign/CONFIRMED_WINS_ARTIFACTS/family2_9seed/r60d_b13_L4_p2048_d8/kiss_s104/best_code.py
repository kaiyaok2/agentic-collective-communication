
def r60d_b13_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 13
    OFF = 2
    
    # Pre-compute keep set
    start = (rank + OFF) % B
    keep = set((start + j) % B for j in range(4))
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Final iteration - just mask and reduce
    buf = torch.zeros_like(s)
    for b in range(B):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
