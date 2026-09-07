
def evolved_p163(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y[i, j] = |i - j|
    # Constant folding approach for N=32
    vals = []
    for i in range(N):
        row = []
        for j in range(N):
            row.append(abs(i - j))
        vals.append(row)
    result = torch.tensor(vals, device=x.device, dtype=x.dtype)
    return result
