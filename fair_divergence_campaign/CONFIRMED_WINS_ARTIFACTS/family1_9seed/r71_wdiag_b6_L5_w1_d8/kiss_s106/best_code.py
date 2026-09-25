def r71_wdiag_b6_L5_w1_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*6
    for r in range(W):
        w = 0.5 + 0.02*r
        st = (r + 0) % 6
        for j in range(5):
            A[(st + 1*j) % 6] += w
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(6):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
    s = acc
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(6):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
    s = acc
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(6):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
    s = acc
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(6):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
    s = acc
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(6):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
    s = acc
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    for b in range(6):
        acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / A[b]
    s = acc
    start = (rank + 0) % 6
    keep = set((start + 1*j) % 6 for j in range(5))
    w = 0.5 + 0.02*rank
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in keep:
            buf[b*S:(b+1)*S] = w * s[b*S:(b+1)*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    s = acc
    return s
