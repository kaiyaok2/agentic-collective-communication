def r60_b5_L2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    c = [0]*5
    for r in range(W):
        st = (r + 0) % 5
        ks = set((st + 1*j) % 5 for j in range(2))
        for b in ks:
            c[b] += 1
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(5):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(5):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(5):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(5):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(5):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(5):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
    s = acc
    B = 5; OFF = 0
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(2))
    buf = torch.zeros_like(s)
    for b in range(5):
        if b in keep:
            buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc
    return s
