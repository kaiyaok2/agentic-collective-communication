
def evolved_p94(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i, j] = min(i, j) * max(i, j) + (i - j)^2
    # Simplified: i * j + (i - j)^2
    # For small N, precompute as constant
    
    if N <= 64:
        # Constant folding: precompute at trace time
        values = []
        for i in range(N):
            row = []
            for j in range(N):
                val = i * j + (i - j) ** 2
                row.append(val)
            values.append(row)
        return torch.tensor(values, device=x.device, dtype=x.dtype)
    else:
        # Fall back to arithmetic for large N
        idx = torch.arange(N, device=x.device)
        i = idx.view(N, 1)
        j = idx.view(1, N)
        diff = i - j
        result = i * j + diff * diff
        return result.to(x.dtype)
