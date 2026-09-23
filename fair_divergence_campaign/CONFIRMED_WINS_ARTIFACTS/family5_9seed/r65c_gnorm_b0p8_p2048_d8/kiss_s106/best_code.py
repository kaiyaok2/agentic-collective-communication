
def r65c_gnorm_b0p8_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    BETA = 0.8
    
    # Try using all_gather instead of all_reduce
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    s = gathered.sum(dim=0)
    g = 1.0 + BETA * s.abs().mean()
    return s / g
