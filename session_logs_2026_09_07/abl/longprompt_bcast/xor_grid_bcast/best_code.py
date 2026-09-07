
def evolved_p88(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: x[i,j] = i XOR j (bitwise XOR of row and column indices)
    # Compute as constant with nested list to get shape right
    result = torch.tensor([[i ^ j for j in range(N)] for i in range(N)], 
                          device=x.device, dtype=x.dtype)
    return result
