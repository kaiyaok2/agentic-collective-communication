
def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    B = 10
    OFF = 2
    
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Try creating buffer without explicit zeros_like
    buf = s.clone()
    for b in range(B):
        if b not in keep:
            buf[b*S:(b+1)*S] = 0
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
