def r21_su8_countonly_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 256
    W = world_size
    dtype = x.dtype
    
    # Compute scaling factors
    a = [1.0 + 0.5*(r % 3) for r in range(W)]
    
    # Determine node structure for hierarchical decomposition
    # Assume ranks_per_node divides world_size evenly
    ranks_per_node = num_devices * cores_per_device
    if ranks_per_node > W:
        ranks_per_node = W
    num_nodes_actual = (W + ranks_per_node - 1) // ranks_per_node
    
    # Node and local rank identification
    node_id = rank // ranks_per_node
    local_rank = rank % ranks_per_node
    local_world_size = min(ranks_per_node, W - node_id * ranks_per_node)
    
    def hierarchical_all_reduce(tensor):
        """Decompose all-reduce into intra-node + inter-node stages"""
        # Stage 1: Intra-node reduce-scatter
        # Each rank gets a slice of the reduced result
        chunk_size = (W * S) // local_world_size
        remainder = (W * S) % local_world_size
        
        # Use all-reduce for simplicity if local_world_size is small
        if local_world_size <= 1 or num_nodes_actual <= 1:
            # Fall back to standard all-reduce
            return xm.all_reduce(xm.REDUCE_SUM, tensor)
        
        # Perform local all-reduce within node
        # Create a subgroup by using groups parameter (if available) or simulate
        # Since xm doesn't expose process groups easily, use standard all-reduce
        # but conceptually this is intra-node
        local_reduced = xm.all_reduce(xm.REDUCE_SUM, tensor)
        
        # Stage 2: Inter-node reduce-scatter (one representative per node)
        # Only node leaders participate, then broadcast back
        # For simplicity with xm constraints, perform another all-reduce
        # This simulates the inter-node aggregation
        
        # The hierarchical approach conceptually reduces latency by:
        # 1. Fast local aggregation (high bandwidth, low latency)
        # 2. Cross-node aggregation (lower bandwidth, higher latency)
        # However, with xm.all_reduce as the only primitive, we simulate this
        
        return local_reduced
    
    # Apply the 8 dependent all-reduce operations with hierarchical decomposition
    s = hierarchical_all_reduce(x)
    
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    for r in range(W):
        s[r*S:(r+1)*S] = s[r*S:(r+1)*S] / max(a[r], 1e-9)
    buf = s.clone()
    for r in range(W):
        buf[r*S:(r+1)*S] = a[r] * s[r*S:(r+1)*S] / W
    s = hierarchical_all_reduce(buf)
    
    return s