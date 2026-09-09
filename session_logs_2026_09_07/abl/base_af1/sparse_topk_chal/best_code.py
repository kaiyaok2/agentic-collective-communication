
def evolved_p125(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Try using sort directly
    g = xm.all_gather(x, dim=0)
    K = 8
    
    # Attempt to use sort
    sorted_vals, _ = torch.sort(g, descending=True)
    
    return sorted_vals[:K]
