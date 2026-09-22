def r47_dd_meanabs_d7_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(B):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / (world_size * f[b])
    s = acc
    f = []
    for b in range(B):
        f.append(1.0 + s[b*S:(b+1)*S].mean().abs())
    buf = s.clone()
    for b in range(B):
        buf[b*S:(b+1)*S] = s[b*S:(b+1)*S] * f[b]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    acc = acc / world_size
    s = acc
    return s
