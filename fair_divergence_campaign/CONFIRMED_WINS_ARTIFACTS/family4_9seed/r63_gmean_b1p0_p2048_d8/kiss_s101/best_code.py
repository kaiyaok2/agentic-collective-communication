
def r63_gmean_b1p0_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # The iterations mathematically don't change s since:
    # (s + s.mean()) - (s + s.mean()).mean() * 0.5 = s + m - 2m*0.5 = s
    # So we can skip them and just do the final operation
    s = s + s.mean()
    
    return s
