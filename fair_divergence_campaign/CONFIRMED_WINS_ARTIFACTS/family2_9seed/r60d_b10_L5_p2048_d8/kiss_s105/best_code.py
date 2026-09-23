
def r60d_b10_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    
    # Compute bucket counts
    c = [0] * B
    for r in range(world_size):
        st = (r + OFF) % B
        for j in range(5):
            c[(st + j) % B] += 1
    
    # Create count multiplier more efficiently
    count_list = []
    for b in range(B):
        count_list.append(torch.full((S,), float(c[b]), dtype=x.dtype, device=x.device))
    count_mult = torch.cat(count_list, dim=0)
    
    # Single all-reduce and local multiplication
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    result = s * count_mult
    
    return result
