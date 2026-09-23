
def r65b_gnorm_b1p5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    s = gathered.sum(0)
    return s / (1.0 + 1.5 * s.abs().mean())
