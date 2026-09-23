
def r65_gnorm_b0p5_p1024_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # What if we use all_gather instead?
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)
    # Sum across gathered dimension
    s = gathered.sum(dim=0)
    # Normalize and gather again
    s = s / (1.0 + 0.5 * s.abs().mean())
    gathered2 = xm.all_gather(s.unsqueeze(0), dim=0)
    result = gathered2.sum(dim=0) / world_size
    return result
