def r31_lin5_count8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    
    # Precompute weights directly
    w = [(1.0 + 0.25*(r % 5)) / world_size for r in range(world_size)]
    a_w = torch.tensor([w[i // S] for i in range(world_size * S)],
                       device=x.device, dtype=x.dtype)
    
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    s = xm.all_reduce(xm.REDUCE_SUM, s * a_w)
    
    return s