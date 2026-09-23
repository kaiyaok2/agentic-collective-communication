def r61_dd_negrelu_p384_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 384
    B = 8
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Compute factors and apply transformation in one loop
    for b in range(B):
        chunk = s[b*S:(b+1)*S]
        mb = -(chunk.mean())
        factor = 1.0 + (mb if mb > 0 else 0.0)
        s[b*S:(b+1)*S] = chunk * factor
    
    return s