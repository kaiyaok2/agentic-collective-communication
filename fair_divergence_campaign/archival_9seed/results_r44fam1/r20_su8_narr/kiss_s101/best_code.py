def r20_su8_narr_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    a_fwd = [a[r] / W for r in range(W)]
    a_inv = [1.0 / max(a[r], 1e-9) for r in range(W)]
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = a_inv[r] * s[r*S:(r+1)*S]
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = a_inv[r] * s[r*S:(r+1)*S]
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = a_inv[r] * s[r*S:(r+1)*S]
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = a_inv[r] * s[r*S:(r+1)*S]
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = a_inv[r] * s[r*S:(r+1)*S]
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = a_inv[r] * s[r*S:(r+1)*S]
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a_fwd[r] * s[r*S:(r+1)*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
