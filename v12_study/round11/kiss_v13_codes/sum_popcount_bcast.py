def evolved_p100(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # x[i,j] = popcount(i) + popcount(j); position-based, no collective
    def popcount(v):
        c = 0
        while v:
            c += v & 1
            v >>= 1
        return c
    vals = [[popcount(i) + popcount(j) for j in range(N)] for i in range(N)]
    return torch.tensor(vals, device=x.device, dtype=x.dtype)

