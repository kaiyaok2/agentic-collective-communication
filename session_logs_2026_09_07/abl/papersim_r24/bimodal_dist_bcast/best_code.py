
def evolved_p92(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i] = (i - N // 2) ** 2
    # Squared distance from each index to center index
    # This is position-based, so we compute locally without any collective
    idx = torch.arange(N, device=x.device)
    center = N // 2
    diff = idx - center
    result = diff * diff
    return result.to(x.dtype)
