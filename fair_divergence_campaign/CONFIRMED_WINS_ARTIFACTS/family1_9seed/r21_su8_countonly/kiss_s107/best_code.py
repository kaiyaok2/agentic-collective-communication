def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
