
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096
    W = world_size
    W_inv = 1.0 / W
    
    # Pre-compute full scale including both W factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    scale = torch.tensor([a[r] * W_inv * W_inv for r in range(W)], 
                         device=x.device, dtype=x.dtype).view(W, 1).repeat(1, S).flatten()
    
    # All operations in chain
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    s2 = xm.all_reduce(xm.REDUCE_SUM, s1 * scale)
    return xm.all_reduce(xm.REDUCE_SUM, s2)
