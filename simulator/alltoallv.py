"""
AllToAllV simulator: evaluates communication schedules on the Trainium topology model.

Supports multiple algorithm templates:
  - permute_ring: sequential collective_permute steps
  - allgather_slice: all_gather + local slice
  - hierarchical: intra-device (free) + inter-device permute
  - pairwise: paired bidirectional exchanges
  - hybrid_ag_perm: allgather for near, permute for far
"""

from .topology import TrainiumTopology


class AllToAllVSimulator:
    """
    Simulate AllToAllV execution on Trainium topology across multiple algorithm templates.
    """

    def __init__(self, topology, send_counts_matrix, element_bytes=4):
        self.topo = topology
        self.send_counts = send_counts_matrix  # [src][dst] = count
        self.element_bytes = element_bytes
        self.world = topology.num_cores

    # ================================================================
    # Template: permute_ring
    # ================================================================

    def simulate_permute_ring(self, permute_steps, max_chunk_bytes=None):
        """
        Simulate sequential collective_permute steps.
        Each step d: rank r sends to (r+d)%world, receives from (r-d)%world.
        Steps are sequential; within each step, all transfers are concurrent.
        Link state is reset between steps so each step is evaluated independently
        (matches simulate_allgather_slice and simulate_hybrid behavior).
        """
        step_times = []

        if max_chunk_bytes is None:
            max_chunk_bytes = self._max_chunk_bytes()

        for d in permute_steps:
            self.topo.reset()
            step_finish = 0.0
            for r in range(self.world):
                dst = (r + d) % self.world
                if r == dst:
                    continue
                finish = self.topo.send(r, dst, max_chunk_bytes, start_time=0.0)
                step_finish = max(step_finish, finish)
            step_times.append(step_finish)

        total_time = sum(step_times)
        return total_time, step_times

    # ================================================================
    # Template: allgather_slice
    # ================================================================

    def simulate_allgather_slice(self, chunk_factor=1):
        """
        Simulate AllGather + local slice approach.

        Each rank gathers all send buffers from all ranks, then slices locally.
        This is bandwidth-inefficient (gathers N*total_send bytes) but has
        only 1 collective operation and no padding waste.

        chunk_factor: split the allgather into this many chunks (pipelining).
        """
        self.topo.reset()

        # Total bytes each rank contributes to the allgather
        max_send_total = 0
        for r in range(self.world):
            total = sum(self.send_counts[r])
            max_send_total = max(max_send_total, total)

        gather_bytes = max_send_total * self.element_bytes
        chunk_bytes = gather_bytes // chunk_factor

        total_time = 0.0
        step_times = []

        for chunk_idx in range(chunk_factor):
            self.topo.reset()
            # AllGather = ring of (world-1) steps, each rank sends chunk to next
            chunk_step_bytes = chunk_bytes // max(self.world - 1, 1)
            step_finish = 0.0
            for step in range(self.world - 1):
                step_finish = 0.0
                for r in range(self.world):
                    dst = (r + 1) % self.world
                    finish = self.topo.send(r, dst, chunk_step_bytes)
                    step_finish = max(step_finish, finish)
            step_times.append(step_finish * (self.world - 1))
            total_time += step_finish * (self.world - 1)

        return total_time, step_times

    # ================================================================
    # Template: hierarchical
    # ================================================================

    def simulate_hierarchical(self, inter_schedule):
        """
        Simulate 2-level hierarchical AllToAllV.

        Level 1: Intra-device exchange (rank 2i <-> 2i+1) — free (shared HBM)
        Level 2: Inter-device permute on 16 devices with optimized schedule

        inter_schedule: ordering of device-level distances [1..15]
        """
        num_devices = self.topo.num_devices
        cpd = self.topo.cores_per_device

        # Compute max chunk at device level
        # Each device sends to each other device: sum of both cores' contributions
        device_send = [[0] * num_devices for _ in range(num_devices)]
        for src_r in range(self.world):
            for dst_r in range(self.world):
                src_d = self.topo.rank_to_device(src_r)
                dst_d = self.topo.rank_to_device(dst_r)
                if src_d != dst_d:
                    device_send[src_d][dst_d] += self.send_counts[src_r][dst_r]

        max_device_chunk = 0
        for sd in range(num_devices):
            for dd in range(num_devices):
                max_device_chunk = max(max_device_chunk, device_send[sd][dd])
        max_device_chunk_bytes = max_device_chunk * self.element_bytes

        # Level 1: intra-device — free, time = 0
        intra_time = 0.0

        # Level 2: inter-device permute (reset link state between steps)
        step_times = []
        for d in inter_schedule:
            self.topo.reset()
            step_finish = 0.0
            for dev in range(num_devices):
                dst_dev = (dev + d) % num_devices
                if dev == dst_dev:
                    continue
                # Both cores on the device send via the device's links
                src_rank = dev * cpd
                dst_rank = dst_dev * cpd
                finish = self.topo.send(src_rank, dst_rank, max_device_chunk_bytes)
                step_finish = max(step_finish, finish)
            step_times.append(step_finish)

        total_time = intra_time + sum(step_times)
        return total_time, [intra_time] + step_times

    # ================================================================
    # Template: pairwise
    # ================================================================

    def simulate_pairwise(self, matchings, round_order):
        """
        Simulate pairwise bidirectional exchanges.

        matchings: list of rounds, each round is list of (rankA, rankB) pairs
        round_order: ordering in which to execute the rounds
        """
        step_times = []
        max_chunk = self._max_chunk_bytes()

        for round_idx in round_order:
            self.topo.reset()
            step_finish = 0.0
            pairs = matchings[round_idx]
            for a, b in pairs:
                # Bidirectional: a->b and b->a concurrently
                count_ab = self.send_counts[a][b]
                count_ba = self.send_counts[b][a]
                bytes_ab = max(count_ab, 1) * self.element_bytes
                bytes_ba = max(count_ba, 1) * self.element_bytes
                finish_ab = self.topo.send(a, b, bytes_ab)
                finish_ba = self.topo.send(b, a, bytes_ba)
                step_finish = max(step_finish, finish_ab, finish_ba)
            step_times.append(step_finish)

        total_time = sum(step_times)
        return total_time, step_times

    # ================================================================
    # Template: hybrid_ag_perm
    # ================================================================

    def simulate_hybrid(self, near_distances, permute_schedule):
        """
        Simulate hybrid AllGather (near) + Permute (far) approach.

        near_distances: distances handled by a single allgather
        permute_schedule: ordered distances for permute steps
        """
        self.topo.reset()

        # Phase 1: AllGather for near ranks
        # Near ranks are gathered in one collective, then sliced locally
        near_bytes = 0
        for d in near_distances:
            for r in range(self.world):
                dst = (r + d) % self.world
                near_bytes = max(near_bytes, self.send_counts[r][dst] * self.element_bytes)

        ag_time = 0.0
        if near_distances:
            # Allgather modeled as ring steps
            ring_steps = min(len(near_distances) + 1, self.world - 1)
            for step in range(ring_steps):
                step_finish = 0.0
                for r in range(self.world):
                    dst = (r + 1) % self.world
                    finish = self.topo.send(r, dst, near_bytes)
                    step_finish = max(step_finish, finish)
                ag_time += step_finish

        # Phase 2: Permute for far ranks
        self.topo.reset()
        max_chunk = self._max_chunk_bytes()
        perm_step_times = []
        for d in permute_schedule:
            step_finish = 0.0
            for r in range(self.world):
                dst = (r + d) % self.world
                if r == dst:
                    continue
                finish = self.topo.send(r, dst, max_chunk)
                step_finish = max(step_finish, finish)
            perm_step_times.append(step_finish)

        total_time = ag_time + sum(perm_step_times)
        return total_time, [ag_time] + perm_step_times

    # ================================================================
    # Template: fused_alltoall
    # ================================================================

    def simulate_fused_alltoall(self):
        """
        Simulate fused all_to_all: single step, all world*(world-1) flows concurrent.

        The simulator models all transfers sharing links simultaneously, which
        produces a pessimistic estimate (high contention).  Real hardware
        implements all_to_all with internal pipelining that the simulator cannot
        capture; the hardware benchmark is the authoritative metric.
        """
        self.topo.reset()
        max_chunk_bytes = self._max_chunk_bytes()

        step_finish = 0.0
        for src in range(self.world):
            for dst in range(self.world):
                if src == dst:
                    continue
                finish = self.topo.send(src, dst, max_chunk_bytes, start_time=0.0)
                step_finish = max(step_finish, finish)

        return step_finish, [step_finish]

    # ================================================================
    # Template: multinode_hierarchical
    # ================================================================

    def simulate_multinode_hierarchical(self, intra_node_schedule,
                                        inter_node_schedule):
        """
        Simulate 3-level hierarchical AllToAllV for multi-node clusters.

        Level 1: Intra-device exchange (free, shared HBM)
        Level 2: Intra-node inter-device permute (NeuronLink, fast)
        Level 3: Inter-node exchange (EFA, slow)

        intra_node_schedule: ordering of device distances 1..15 within a node
        inter_node_schedule: ordering of node distances 1..num_nodes-1
        """
        num_nodes = getattr(self.topo, 'num_nodes', 1)
        ranks_per_node = getattr(self.topo, 'ranks_per_node', self.world)
        devices_per_node = getattr(self.topo, 'devices_per_node',
                                    self.topo.num_devices)
        cpd = self.topo.cores_per_device

        # Compute max chunk for inter-device steps (within node)
        device_send = [[0] * self.topo.num_devices
                       for _ in range(self.topo.num_devices)]
        for src_r in range(self.world):
            for dst_r in range(self.world):
                src_d = self.topo.rank_to_device(src_r)
                dst_d = self.topo.rank_to_device(dst_r)
                if src_d != dst_d:
                    device_send[src_d][dst_d] += self.send_counts[src_r][dst_r]

        # Level 1: intra-device — free
        intra_time = 0.0
        step_times = [intra_time]

        # Level 2: intra-node inter-device permute
        # Each node runs the same intra_node_schedule independently
        for d in intra_node_schedule:
            self.topo.reset()
            step_finish = 0.0
            for node_id in range(num_nodes):
                base_dev = node_id * devices_per_node
                for dev_off in range(devices_per_node):
                    src_dev = base_dev + dev_off
                    dst_dev_off = (dev_off + d) % devices_per_node
                    dst_dev = base_dev + dst_dev_off
                    if src_dev == dst_dev:
                        continue
                    bytes_ = device_send[src_dev][dst_dev] * self.element_bytes
                    if bytes_ <= 0:
                        continue
                    src_rank = src_dev * cpd
                    dst_rank = dst_dev * cpd
                    finish = self.topo.send(src_rank, dst_rank, bytes_)
                    step_finish = max(step_finish, finish)
            step_times.append(step_finish)

        # Level 3: inter-node exchange
        # Compute per-node-pair send volumes
        node_send = [[0] * num_nodes for _ in range(num_nodes)]
        for src_r in range(self.world):
            for dst_r in range(self.world):
                src_n = self.topo.rank_to_node(src_r) if hasattr(
                    self.topo, 'rank_to_node') else 0
                dst_n = self.topo.rank_to_node(dst_r) if hasattr(
                    self.topo, 'rank_to_node') else 0
                if src_n != dst_n:
                    node_send[src_n][dst_n] += self.send_counts[src_r][dst_r]

        for nd in inter_node_schedule:
            self.topo.reset()
            step_finish = 0.0
            for src_n in range(num_nodes):
                dst_n = (src_n + nd) % num_nodes
                if src_n == dst_n:
                    continue
                bytes_ = node_send[src_n][dst_n] * self.element_bytes
                if bytes_ <= 0:
                    continue
                # Use representative ranks for the send
                src_rank = src_n * ranks_per_node
                dst_rank = dst_n * ranks_per_node
                finish = self.topo.send(src_rank, dst_rank, bytes_)
                step_finish = max(step_finish, finish)
            step_times.append(step_finish)

        total_time = sum(step_times)
        return total_time, step_times

    # ================================================================
    # Template: node_allgather
    # ================================================================

    def simulate_node_allgather(self):
        """
        Simulate 2-level AllGather for multi-node clusters.

        Phase 1: AllGather within each node (32 ranks, NeuronLink ring).
        Phase 2: AllGather across nodes (one buffer per node, EFA ring).

        Amplification: 32x (intra-node) + Nx (inter-node).
        """
        num_nodes = getattr(self.topo, 'num_nodes', 1)
        ranks_per_node = getattr(self.topo, 'ranks_per_node', self.world)

        # Max total send across all ranks (for AllGather padding)
        max_send_total = 0
        for r in range(self.world):
            total = sum(self.send_counts[r])
            max_send_total = max(max_send_total, total)
        gather_bytes = max_send_total * self.element_bytes

        step_times = []

        # Phase 1: Intra-node AllGather (ring of ranks_per_node - 1 steps)
        self.topo.reset()
        step_finish = 0.0
        for step in range(ranks_per_node - 1):
            step_finish = 0.0
            for r in range(self.world):
                node = r // ranks_per_node
                local = r % ranks_per_node
                dst_local = (local + 1) % ranks_per_node
                dst = node * ranks_per_node + dst_local
                finish = self.topo.send(r, dst, gather_bytes)
                step_finish = max(step_finish, finish)
        phase1_time = step_finish * (ranks_per_node - 1)
        step_times.append(phase1_time)

        if num_nodes > 1:
            # Phase 2: Inter-node AllGather
            # After phase 1, each node has all its 32 ranks' data.
            # Now gather across nodes: ring of (num_nodes - 1) steps.
            # Each node sends ranks_per_node * gather_bytes.
            inter_bytes = ranks_per_node * gather_bytes
            self.topo.reset()
            step_finish = 0.0
            for step in range(num_nodes - 1):
                step_finish = 0.0
                for n in range(num_nodes):
                    dst_n = (n + 1) % num_nodes
                    src_rank = n * ranks_per_node
                    dst_rank = dst_n * ranks_per_node
                    finish = self.topo.send(src_rank, dst_rank, inter_bytes)
                    step_finish = max(step_finish, finish)
            phase2_time = step_finish * (num_nodes - 1)
            step_times.append(phase2_time)

        total_time = sum(step_times)
        return total_time, step_times

    # ================================================================
    # Dispatch
    # ================================================================

    def simulate_template(self, template_name, params):
        """
        Simulate any template by name.

        Returns: (total_time_seconds, step_times_list)
        """
        if template_name == "permute_ring":
            return self.simulate_permute_ring(params["schedule"])
        elif template_name == "allgather_slice":
            return self.simulate_allgather_slice(params.get("chunk_factor", 1))
        elif template_name == "hierarchical":
            return self.simulate_hierarchical(params["inter_schedule"])
        elif template_name == "pairwise":
            return self.simulate_pairwise(params["_matchings"], params["round_order"])
        elif template_name == "hybrid_ag_perm":
            return self.simulate_hybrid(params["near_distances"], params["permute_schedule"])
        elif template_name == "fused_alltoall":
            return self.simulate_fused_alltoall()
        elif template_name == "multinode_hierarchical":
            return self.simulate_multinode_hierarchical(
                params["intra_node_schedule"], params["inter_node_schedule"])
        elif template_name == "node_allgather":
            return self.simulate_node_allgather()
        elif template_name == "allgather_reduce_scatter":
            return self.simulate_allgather_reduce_scatter()
        else:
            raise ValueError(f"Unknown template: {template_name}")

    # ================================================================
    # Template: allgather_reduce_scatter
    # ================================================================

    def simulate_allgather_reduce_scatter(self):
        """
        Simulate AllGather + ReduceScatter AllToAllV.

        Phase 1: AllGather of packed buffer (world * max_chunk elements per rank).
                 Ring of (world-1) steps, each sending pack_size bytes.
        Phase 2: ReduceScatter of transposed buffer (world * pack_size total).
                 Ring of (world-1) steps, each sending pack_size bytes.
        """
        max_chunk = self._max_chunk_bytes() // self.element_bytes
        pack_size_bytes = self.world * max_chunk * self.element_bytes

        total_time = 0.0
        step_times = []

        # Phase 1: AllGather — ring of (world-1) steps
        ag_step_bytes = pack_size_bytes
        for step in range(self.world - 1):
            self.topo.reset()
            step_finish = 0.0
            for r in range(self.world):
                dst = (r + 1) % self.world
                finish = self.topo.send(r, dst, ag_step_bytes)
                step_finish = max(step_finish, finish)
            step_times.append(step_finish)
            total_time += step_finish

        # Phase 2: ReduceScatter — ring of (world-1) steps
        rs_step_bytes = pack_size_bytes
        for step in range(self.world - 1):
            self.topo.reset()
            step_finish = 0.0
            for r in range(self.world):
                dst = (r + 1) % self.world
                finish = self.topo.send(r, dst, rs_step_bytes)
                step_finish = max(step_finish, finish)
            step_times.append(step_finish)
            total_time += step_finish

        return total_time, step_times

    # ================================================================
    # Helpers
    # ================================================================

    def _max_chunk_bytes(self):
        max_count = 0
        for src in range(self.world):
            for dst in range(self.world):
                max_count = max(max_count, self.send_counts[src][dst])
        return max_count * self.element_bytes

    # Keep old API for backward compatibility
    def simulate_multistep_permute(self, permute_steps, max_chunk_bytes=None):
        return self.simulate_permute_ring(permute_steps, max_chunk_bytes)

    def lower_bound(self):
        """Theoretical lower bound based on max send/recv volume and link bandwidth."""
        max_send = 0
        max_recv = 0
        for r in range(self.world):
            total_send = sum(self.send_counts[r][d] for d in range(self.world) if d != r)
            total_recv = sum(self.send_counts[s][r] for s in range(self.world) if s != r)
            max_send = max(max_send, total_send)
            max_recv = max(max_recv, total_recv)

        max_bytes = max(max_send, max_recv) * self.element_bytes
        per_device_bw = 4 * self.topo.link_bw * 1e9
        return max_bytes / per_device_bw
