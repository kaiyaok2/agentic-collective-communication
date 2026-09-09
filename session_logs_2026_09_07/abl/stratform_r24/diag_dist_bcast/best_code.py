
def evolved_p163(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 1: Pure local computation
    i = torch.arange(N, dtype=torch.float32, device=x.device)
    result = (i.unsqueeze(1) - i.unsqueeze(0)).abs()
    return result
