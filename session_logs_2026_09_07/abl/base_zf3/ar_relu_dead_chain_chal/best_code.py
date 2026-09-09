
def evolved_p6102(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Formula: y = relu(AR(x))
    # AR(x) = all-reduce SUM of x across all ranks
    # relu = ReLU activation (max(0, value))
    
    # Step 1: All-reduce SUM
    ax = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2: Apply ReLU using clamp (relu(x) = max(0, x))
    result = ax.clamp(min=0)
    
    return result
