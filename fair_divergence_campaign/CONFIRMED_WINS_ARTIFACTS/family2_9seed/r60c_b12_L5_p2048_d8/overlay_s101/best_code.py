def r60c_b12_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    c = [0]*12
    for r in range(W):
        st = (r + 2) % 12
        ks = set((st + 1*j) % 12 for j in range(5))
        for b in ks:
            c[b] += 1
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(12):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(12):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(12):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(12):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(12):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(12):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 12; OFF = 2
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(5))
    buf = torch.zeros_like(s)
    for b in range(12):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc
    return s
