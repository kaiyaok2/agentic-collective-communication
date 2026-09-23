def r60c_b10_L4_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    B = 10
    OFF = 2
    W = world_size
    
    # Compute coverage counts for each block
    c = [0] * B
    for r in range(W):
        st = (r + OFF) % B
        ks = set((st + 1*j) % B for j in range(4))
        for b in ks:
            c[b] += 1
    
    # Determine which blocks this rank owns
    start = (rank + OFF) % B
    keep = set((start + 1*j) % B for j in range(4))
    
    # Assume roughly equal distribution of ranks per node
    ranks_per_node = max(1, world_size // num_nodes)
    
    # Phase 1: Intra-node reduction
    # Group ranks by node: node_id = rank // ranks_per_node
    node_id = rank // ranks_per_node
    local_rank = rank % ranks_per_node
    
    # Create node-based groups for collective operations
    # For intra-node: reduce among ranks on same node
    # We'll use all_reduce with groups, but since xm doesn't support groups directly,
    # we'll simulate with masking
    
    # Step 1: Initial all_reduce to get global sum
    s = xm.all_reduce(xm.REDUCE_SUM, x)
    
    # Step 2-8: Apply the 7 dependent selective all_reduce operations
    for iteration in range(7):
        # Each rank zeros out blocks it doesn't own
        buf = torch.zeros_like(s)
        for b in range(B):
            if b in keep:
                buf[b*S:(b+1)*S] = s[b*S:(b+1)*S]
        
        # All-reduce to sum contributions
        acc = xm.all_reduce(xm.REDUCE_SUM, buf)
        
        # Normalize by coverage counts (except last iteration)
        if iteration < 6:
            for b in range(B):
                acc[b*S:(b+1)*S] = acc[b*S:(b+1)*S] / c[b]
        
        s = acc
    
    return s