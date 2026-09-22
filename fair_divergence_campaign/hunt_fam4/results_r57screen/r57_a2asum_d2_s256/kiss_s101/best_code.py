
def r57_a2asum_d2_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    scale = (1.0 + 0.5 * ((rank * 11) % 7) / 7.0) / (0.9 * sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W)))
    
    cur = x
    for _ in range(2):
        scaled = scale * cur
        gathered = xm.all_gather(scaled.view(1, W, S), dim=0)
        cur = torch.sum(gathered, dim=0).view(-1)
    
    return cur
