
def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256; W = world_size
    
    # Try 0 iterations - just the final all_reduce
    a_list = [1.0 + 0.25*(r % 5) for r in range(W)]
    a_flat = []
    for r in range(W):
        a_flat.extend([a_list[r] / W] * S)
    
    a_w = torch.tensor(a_flat, device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    return xm.all_reduce(xm.REDUCE_SUM, a_w * s)
