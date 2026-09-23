def r65_gnorm_b0p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    W = world_size
    BETA = 0.5
    dtype = x.dtype
    
    # Initial all_reduce
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    g = 1.0 + BETA * s.abs().mean()
    acc = s / g
    
    # Process all 6 iterations locally, then do single all_reduce
    states = [acc.clone()]
    
    for i in range(5):
        # Compute next state without all_reduce
        A = states[-1].abs().mean()
        M = A / (1.0 - BETA * A)
        gr = 1.0 + BETA * M
        s_local = states[-1] * gr
        g_local = 1.0 + BETA * s_local.abs().mean()
        buf_local = s_local / g_local
        states.append(buf_local)
    
    # Stack all 6 states and do single all_reduce
    stacked = torch.stack(states, dim=0)  # shape: (6, 16384)
    stacked_reduced = xm.all_reduce(xm.REDUCE_SUM, stacked)
    stacked_reduced = stacked_reduced / W
    
    # Extract the final state
    acc = stacked_reduced[-1]
    
    # Apply the correction for the final state
    A = acc.abs().mean()
    M = A / (1.0 - BETA * A)
    gr = 1.0 + BETA * M
    acc = acc * gr
    g = 1.0 + BETA * acc.abs().mean()
    acc = acc / g
    
    # Final all_reduce
    acc = xm.all_reduce(xm.REDUCE_SUM, acc)
    acc = acc / W
    
    return acc