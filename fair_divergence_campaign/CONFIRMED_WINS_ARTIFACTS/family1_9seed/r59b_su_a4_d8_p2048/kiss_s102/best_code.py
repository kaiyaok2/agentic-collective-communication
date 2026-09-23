
def r59b_su_a4_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    
    # Precompute weights as tensors
    a = [1.0 + 0.6*(r % 4) for r in range(W)]
    w_list = [a[r] / W for r in range(W)]
    inv_list = [1.0 / max(a[r], 1e-9) for r in range(W)]
    
    # Create weight tensors using view and repeat
    weights = torch.tensor(w_list, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, S).view(-1)
    inv_weights = torch.tensor(inv_list, device=x.device, dtype=x.dtype).view(-1, 1).repeat(1, S).view(-1)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Perform 6 iterations
    for i in range(6):
        buf = s * weights
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        if i < 5:
            s = s * inv_weights
    
    return s
