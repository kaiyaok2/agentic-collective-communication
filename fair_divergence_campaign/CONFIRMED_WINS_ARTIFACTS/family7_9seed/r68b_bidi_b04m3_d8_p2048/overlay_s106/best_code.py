def r68b_bidi_b04m3_d8_p2048_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    W = world_size
    b = [0.4 + 0.08*(r % 3) for r in range(W)]
    dtype = x.dtype
    
    # Hierarchical two-level all-reduce strategy
    # Level 1: intra-node, Level 2: inter-node
    
    cores_per_node = num_devices * cores_per_device
    node_id = rank // cores_per_node
    local_rank = rank % cores_per_node
    num_nodes_actual = (W + cores_per_node - 1) // cores_per_node
    
    # If single node or num_nodes == 1, fall back to standard implementation
    if num_nodes_actual == 1:
        s = xm.all_reduce(xm.REDUCE_SUM, x)
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        for r in range(W - 2, -1, -1):
            s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
        buf = s / W
        for r in range(W - 1):
            buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
        s = xm.all_reduce(xm.REDUCE_SUM, buf)
        return s
    
    # Multi-node hierarchical approach
    # Build intra-node groups
    intra_node_groups = []
    for nid in range(num_nodes_actual):
        group = list(range(nid * cores_per_node, min((nid + 1) * cores_per_node, W)))
        intra_node_groups.append(group)
    
    # Build inter-node groups (one rank per node)
    inter_node_groups = [[nid * cores_per_node for nid in range(num_nodes_actual) if nid * cores_per_node < W]]
    
    s = x.clone()
    
    # 8 stages
    for stage in range(8):
        # Intra-node all-reduce
        s = xm.all_reduce(xm.REDUCE_SUM, s, groups=intra_node_groups)
        
        # Apply coupling/uncoupling at node representatives (local_rank == 0)
        if local_rank == 0:
            if stage == 0:
                # First stage: just coupling
                buf = s / W
                for r in range(W - 1):
                    buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
                s = buf
            else:
                # Subsequent stages: uncoupling then coupling
                for r in range(W - 2, -1, -1):
                    s[r*S:(r+1)*S] = s[r*S:(r+1)*S] - b[r]*s[(r+1)*S:(r+2)*S]
                buf = s / W
                for r in range(W - 1):
                    buf[r*S:(r+1)*S] = (s[r*S:(r+1)*S] + b[r]*s[(r+1)*S:(r+2)*S]) / W
                s = buf
            
            # Inter-node all-reduce
            s = xm.all_reduce(xm.REDUCE_SUM, s, groups=inter_node_groups)
        
        # Broadcast result from node representative to all intra-node ranks
        s = xm.all_reduce(xm.REDUCE_SUM, s if local_rank == 0 else torch.zeros_like(s), groups=intra_node_groups)
    
    return s