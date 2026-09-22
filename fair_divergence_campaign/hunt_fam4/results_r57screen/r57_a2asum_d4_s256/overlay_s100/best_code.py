def r57_a2asum_d4_s256_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    """
    Sequential all-to-all four-stage baseline implementation.
    
    Executes four sequential stages, each performing:
    1. all_to_all (simulated via all_gather + local operations)
    2. local reshape + sum
    3. all_gather
    
    This uses 12 collective dispatches total (3 per stage × 4 stages).
    """
    S = 256
    W = world_size
    D = 4
    dtype = x.dtype
    
    # Compute scaling factor for this rank
    a = 1.0 + 0.5 * ((rank * 11) % 7) / 7.0
    
    # Compute total sum of all scaling factors
    A_tot = sum(1.0 + 0.5 * ((k * 11) % 7) / 7.0 for k in range(W))
    u = 0.9 * A_tot
    
    cur = x
    
    for _t in range(D):
        # Stage 1: Simulate all_to_all using all_gather + local operations
        # First, scale the current tensor by this rank's factor
        scaled = a * cur
        
        # all_to_all: each rank sends block i to rank i
        # Simulate by gathering all data, then extracting the relevant blocks
        gathered = xm.all_gather(scaled, dim=0)  # Shape: (W * W * S,)
        
        # Reshape to (W, W, S) where gathered[src, dst, :] is block dst from rank src
        gathered = gathered.reshape(W, W, S)
        
        # Extract blocks destined for this rank (block rank from each source)
        # y[src, :] should be the block that rank src sends to this rank
        y = gathered[:, rank, :]  # Shape: (W, S)
        
        # Stage 2: Local sum across the W received blocks
        z = torch.sum(y, dim=0)  # Shape: (S,)
        
        # Stage 3: All-gather the summed shards
        gathered_z = xm.all_gather(z, dim=0)  # Shape: (W * S,)
        
        # Divide by normalization factor
        cur = gathered_z / u
    
    return cur