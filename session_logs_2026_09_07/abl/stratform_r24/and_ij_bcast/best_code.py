
def evolved_p166(x, N, rank, world_size, num_devices,
                 cores_per_device, xm, torch, num_nodes=1):
    # Refinement 2: Inline accumulation
    rows = torch.arange(N, device=x.device).unsqueeze(1)
    cols = torch.arange(N, device=x.device).unsqueeze(0)
    
    result = ((rows % 2) * (cols % 2)) * 1
    result = result + ((rows // 2) % 2) * ((cols // 2) % 2) * 2
    result = result + ((rows // 4) % 2) * ((cols // 4) % 2) * 4
    result = result + ((rows // 8) % 2) * ((cols // 8) % 2) * 8
    result = result + ((rows // 16) % 2) * ((cols // 16) % 2) * 16
    
    return result
