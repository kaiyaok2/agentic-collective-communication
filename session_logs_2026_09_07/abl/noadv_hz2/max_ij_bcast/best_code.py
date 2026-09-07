
def evolved_p164(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = max(i, j) where i, j are positional indices
    # Using conditional: max(i, j) = j if i < j else i
    idx = torch.arange(N, device=x.device)
    i = idx.unsqueeze(1)  # (N, 1) - row indices
    j = idx.unsqueeze(0)  # (1, N) - column indices
    result = torch.where(i >= j, i, j)
    return result.to(x.dtype)
