def r16_deepdoc_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    dtype = x.dtype
    
    # Stage 0: Initial all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Fuse stages 1-4 into two all-reduces
    # Stage 1-2: First pair
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    for r in range(W):
        buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * buf[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Stage 3-4: Second pair
    for r in range(W):
        buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * buf[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Stage 5-6: Third pair
    for r in range(W):
        buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * buf[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    # Stage 7: Final computation without all-reduce
    for r in range(W):
        buf[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * buf[r*S:(r+1)*S] / W
    
    # Stage 8: Final all-reduce
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s