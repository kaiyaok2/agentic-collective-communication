
def r72_bidiag_b6_Ld5_Lo2_w2_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048; W = world_size
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    A = [0.0]*6
    C = [0.0]*6
    for r in range(W):
        wd = 0.7 + 0.03*(r % 5)
        wo = 0.2 + 0.01*(r % 7)
        sd = (r + 2) % 6
        for j in range(5):
            A[(sd + j) % 6] += wd
        so = (r + 3) % 6
        for j in range(2):
            b = (so + j) % 6
            if b >= 1:
                C[b] += wo
    
    sd = (rank + 2) % 6
    kd = set((sd + j) % 6 for j in range(5))
    so = (rank + 3) % 6
    ko = set((so + j) % 6 for j in range(2))
    wd = 0.7 + 0.03*(rank % 5)
    wo = 0.2 + 0.01*(rank % 7)
    
    # Iteration 1
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    # Avoid clone by building with cat
    blocks = []
    blocks.append(acc[0:S] / A[0])
    for b in range(1, 6):
        blocks.append((acc[b*S:(b+1)*S] - C[b] * blocks[b-1]) / A[b])
    s = torch.cat(blocks, dim=0)
    
    # Iteration 2
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    blocks = []
    blocks.append(acc[0:S] / A[0])
    for b in range(1, 6):
        blocks.append((acc[b*S:(b+1)*S] - C[b] * blocks[b-1]) / A[b])
    s = torch.cat(blocks, dim=0)
    
    # Iteration 3
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    blocks = []
    blocks.append(acc[0:S] / A[0])
    for b in range(1, 6):
        blocks.append((acc[b*S:(b+1)*S] - C[b] * blocks[b-1]) / A[b])
    s = torch.cat(blocks, dim=0)
    
    # Iteration 4
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    blocks = []
    blocks.append(acc[0:S] / A[0])
    for b in range(1, 6):
        blocks.append((acc[b*S:(b+1)*S] - C[b] * blocks[b-1]) / A[b])
    s = torch.cat(blocks, dim=0)
    
    # Iteration 5
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    blocks = []
    blocks.append(acc[0:S] / A[0])
    for b in range(1, 6):
        blocks.append((acc[b*S:(b+1)*S] - C[b] * blocks[b-1]) / A[b])
    s = torch.cat(blocks, dim=0)
    
    # Iteration 6
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    acc = xm.all_reduce(xm.REDUCE_SUM, buf)
    blocks = []
    blocks.append(acc[0:S] / A[0])
    for b in range(1, 6):
        blocks.append((acc[b*S:(b+1)*S] - C[b] * blocks[b-1]) / A[b])
    s = torch.cat(blocks, dim=0)
    
    # Iteration 7 (last one, no solve needed)
    buf = torch.zeros_like(s)
    for b in range(6):
        if b in kd:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wd * s[b*S:(b+1)*S]
        if b in ko and b >= 1:
            buf[b*S:(b+1)*S] = buf[b*S:(b+1)*S] + wo * s[(b-1)*S:b*S]
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    
    return s
