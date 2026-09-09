
def evolved_p91(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Strategy 3 revised: Local computation using modulo arithmetic
    i_indices = torch.arange(N, device=x.device).unsqueeze(1)
    j_indices = torch.arange(N, device=x.device).unsqueeze(0)
    # (-1)^(i+j) = 1 if (i+j) is even, -1 if odd
    sum_indices = i_indices + j_indices
    result = 1 - 2 * (sum_indices % 2)
    return result.float()
