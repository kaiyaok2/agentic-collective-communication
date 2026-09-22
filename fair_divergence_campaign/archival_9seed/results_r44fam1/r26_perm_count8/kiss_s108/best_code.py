
def r26_perm_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    perm = [(r + W // 2) % W for r in range(W)]
    
    # Build inverse permutation weights
    inv_a = [0.0] * W
    for r in range(W):
        p = perm[r]
        inv_a[p] = a[r] / W
    
    # Replicate each weight S times
    full_weights = []
    for w in inv_a:
        full_weights.extend([w] * S)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    a_tensor = torch.tensor(full_weights, device=x.device, dtype=x.dtype)
    result = s * a_tensor
    
    return xm.all_reduce(xm.REDUCE_SUM, result)
