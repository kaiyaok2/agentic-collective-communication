
def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    
    # Create inverse permutation
    inv_perm = [0] * W
    for r in range(W):
        inv_perm[perm[r]] = r
    
    # Create weight vectors
    weight_list = []
    inv_weight_list = []
    for p in range(W):
        r = inv_perm[p]
        for _ in range(S):
            weight_list.append(a[r] / W)
            inv_weight_list.append(1.0 / max(a[r], 1e-9))
    
    weight_tensor = torch.tensor(weight_list, device=x.device, dtype=x.dtype)
    inv_weight_tensor = torch.tensor(inv_weight_list, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    buf = torch.zeros_like(s)
    
    # 6 complete iterations
    for iter in range(6):
        buf = s * weight_tensor
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        s = s * inv_weight_tensor
    
    # Final weighted permutation and all_reduce (no inverse weights)
    buf = s * weight_tensor
    s = xm.all_reduce(xm.REDUCE_SUM, buf)
    return s
