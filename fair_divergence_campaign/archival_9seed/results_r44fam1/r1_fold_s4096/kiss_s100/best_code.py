
def r1_fold_s4096_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 4096; W = world_size
    
    # Create weight tensor using arange
    ranks = torch.arange(W, device=x.device, dtype=x.dtype)
    a_tensor = 1.0 + 0.5 * (ranks % 3)
    weights = a_tensor.unsqueeze(1).repeat(1, S).view(-1)
    
    s1 = xm.all_reduce(xm.REDUCE_SUM, x)
    out = weights * s1
    return out
