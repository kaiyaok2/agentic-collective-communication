
def r60d_b16_L5_p2048_d8_fn(x, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    S = 2048
    
    # Gather all inputs from all ranks
    gathered = xm.all_gather(x.unsqueeze(0), dim=0)  # [world_size, 32768]
    
    # Sum across all ranks to get total
    x_total = gathered.sum(dim=0)  # [32768]
    
    # For each bucket, multiply by the number of owners
    result = torch.zeros_like(x)
    for b in range(16):
        # Count how many ranks own bucket b
        count = 0
        for r in range(world_size):
            start_r = (r + 2) % 16
            if b in [(start_r + j) % 16 for j in range(5)]:
                count += 1
        result[b*S:(b+1)*S] = x_total[b*S:(b+1)*S] * count
    
    return result
