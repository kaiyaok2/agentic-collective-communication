"""
Cost model for evaluating AllToAllV algorithms across templates.

Primary signal: simulated execution time on contention-aware topology model.
Secondary signals: link contention, load balance, padding waste, memory overhead.
"""

import math
from collections import defaultdict
from .topology import TrainiumTopology
from .alltoallv import AllToAllVSimulator


class CostModel:
    """
    Multi-objective cost model for AllToAllV algorithm evaluation.
    Works across all algorithm templates.
    """

    def __init__(self, topology, send_counts_matrix, element_bytes=4,
                 w_sim_time=1.0, w_contention=0.1, w_balance=0.05, w_padding=0.05,
                 dispatch_overhead_us=100.0,
                 inter_node_dispatch_overhead_us=150.0):
        self.topo = topology
        self.send_counts = send_counts_matrix
        self.element_bytes = element_bytes
        self.world = topology.num_cores
        self.num_nodes = getattr(topology, 'num_nodes', 1)
        self.w_sim_time = w_sim_time
        self.w_contention = w_contention
        self.w_balance = w_balance
        self.w_padding = w_padding
        self.dispatch_overhead_us = dispatch_overhead_us
        self.inter_node_dispatch_overhead_us = inter_node_dispatch_overhead_us
        self._sim = AllToAllVSimulator(topology, send_counts_matrix, element_bytes)

    def evaluate_template(self, template_name, params):
        """
        Evaluate any template + params combination.

        Returns:
            score: float (lower is better)
            breakdown: dict with individual metrics
        """
        sim_time, step_times = self._sim.simulate_template(template_name, params)
        num_dispatches = len(step_times)
        # On real hardware, each collective dispatch incurs fixed overhead
        # (~100 us for collective_permute on Trainium). allgather_slice
        # counts as 1 dispatch (single all_gather call).
        if template_name == "allgather_slice":
            num_dispatches = 1  # single all_gather collective
        elif template_name == "fused_alltoall":
            num_dispatches = 1  # single all_to_all collective
        dispatch_time_us = num_dispatches * self.dispatch_overhead_us
        sim_time_us = sim_time * 1e6 + dispatch_time_us

        # Template-specific contention and secondary metrics
        if template_name == "permute_ring":
            contention = self._permute_contention(params["schedule"])
            hop_cost = self._permute_hop_cost(params["schedule"])
            padding = self._permute_padding(params["schedule"])
            mem_overhead = 0.0
        elif template_name == "allgather_slice":
            contention = self._allgather_contention()
            hop_cost = 1.0  # ring allgather is 1-hop per step
            padding = 0.0   # no padding needed
            mem_overhead = self.world  # O(N) memory
        elif template_name == "hierarchical":
            contention = self._permute_contention_device_level(params["inter_schedule"])
            hop_cost = self._device_hop_cost(params["inter_schedule"])
            padding = self._device_padding(params["inter_schedule"])
            mem_overhead = 0.0
        elif template_name == "pairwise":
            contention = self._pairwise_contention(params["_matchings"], params["round_order"])
            hop_cost = self._pairwise_hop_cost(params["_matchings"], params["round_order"])
            padding = 0.0
            mem_overhead = 0.0
        elif template_name == "hybrid_ag_perm":
            contention = self._permute_contention(params["permute_schedule"]) if params["permute_schedule"] else 0
            hop_cost = self._hybrid_hop_cost(params)
            padding = self._permute_padding(params["permute_schedule"]) if params["permute_schedule"] else 0
            mem_overhead = len(params.get("near_distances", [])) / max(self.world - 1, 1)
        elif template_name == "fused_alltoall":
            contention = self._fused_alltoall_contention()
            hop_cost = self._fused_alltoall_hop_cost()
            padding = self._fused_alltoall_padding()
            mem_overhead = self.world  # O(world * max_chunk) per rank
        elif template_name == "multinode_hierarchical":
            intra_sched = params.get("intra_node_schedule", [])
            inter_sched = params.get("inter_node_schedule", [])
            contention = self._permute_contention_device_level(intra_sched) if intra_sched else 0
            hop_cost = self._device_hop_cost(intra_sched) if intra_sched else 0
            padding = self._device_padding(intra_sched) if intra_sched else 0
            mem_overhead = 0.0
            num_dispatches = 1 + len(intra_sched) + len(inter_sched)
            dispatch_time_us = (
                (1 + len(intra_sched)) * self.dispatch_overhead_us +
                len(inter_sched) * self.inter_node_dispatch_overhead_us
            )
            sim_time_us = sim_time * 1e6 + dispatch_time_us
        elif template_name == "node_allgather":
            contention = self._allgather_contention()
            hop_cost = 1.0
            padding = 0.0
            mem_overhead = self.world
            num_dispatches = 1 + (1 if self.num_nodes > 1 else 0)
            dispatch_time_us = (
                self.dispatch_overhead_us +
                (self.inter_node_dispatch_overhead_us if self.num_nodes > 1 else 0)
            )
            sim_time_us = sim_time * 1e6 + dispatch_time_us
        else:
            contention = 0.0
            hop_cost = 0.0
            padding = 0.0
            mem_overhead = 0.0

        balance = self._balance_score(template_name, params)
        normalized_sim = sim_time_us / 100.0

        score = (
            self.w_sim_time * normalized_sim +
            self.w_contention * contention +
            self.w_balance * balance +
            self.w_padding * padding
        )

        return score, {
            "sim_time_us": sim_time_us,
            "hop_cost": hop_cost,
            "contention": contention,
            "balance": balance,
            "padding_waste": padding,
            "mem_overhead": mem_overhead,
            "total_score": score,
            "num_steps": len(step_times),
            "num_dispatches": num_dispatches,
            "dispatch_overhead_us": dispatch_time_us,
            "template": template_name,
        }

    # Keep backward compat
    def evaluate_permute_schedule(self, permute_steps):
        return self.evaluate_template("permute_ring", {"schedule": permute_steps})

    # ================================================================
    # Contention scoring
    # ================================================================

    def _permute_contention(self, permute_steps):
        total_contention = 0
        for d in permute_steps:
            link_usage = defaultdict(int)
            for r in range(self.world):
                dst = (r + d) % self.world
                if r == dst:
                    continue
                src_dev = self.topo.rank_to_device(r)
                dst_dev = self.topo.rank_to_device(dst)
                if src_dev == dst_dev:
                    continue
                path = self.topo.device_path(src_dev, dst_dev)
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    link_usage[key] += 1
            if link_usage:
                total_contention += max(link_usage.values())
        return total_contention / max(len(permute_steps), 1)

    def _permute_contention_device_level(self, inter_schedule):
        num_devices = self.topo.num_devices
        total_contention = 0
        for d in inter_schedule:
            link_usage = defaultdict(int)
            for dev in range(num_devices):
                dst_dev = (dev + d) % num_devices
                if dev == dst_dev:
                    continue
                src_r = dev * self.topo.cores_per_device
                dst_r = dst_dev * self.topo.cores_per_device
                path = self.topo.device_path(dev, dst_dev)
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    link_usage[key] += 1
            if link_usage:
                total_contention += max(link_usage.values())
        return total_contention / max(len(inter_schedule), 1)

    def _allgather_contention(self):
        # Ring allgather: each step, every rank sends to next — 1 flow per link max
        # But multiple hops can cause some contention
        link_usage = defaultdict(int)
        for r in range(self.world):
            dst = (r + 1) % self.world
            src_dev = self.topo.rank_to_device(r)
            dst_dev = self.topo.rank_to_device(dst)
            if src_dev == dst_dev:
                continue
            path = self.topo.device_path(src_dev, dst_dev)
            for i in range(len(path) - 1):
                key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                link_usage[key] += 1
        return max(link_usage.values()) if link_usage else 0

    def _pairwise_contention(self, matchings, round_order):
        total = 0
        count = 0
        for ri in round_order:
            link_usage = defaultdict(int)
            for a, b in matchings[ri]:
                for src, dst in [(a, b), (b, a)]:
                    src_dev = self.topo.rank_to_device(src)
                    dst_dev = self.topo.rank_to_device(dst)
                    if src_dev == dst_dev:
                        continue
                    path = self.topo.device_path(src_dev, dst_dev)
                    for i in range(len(path) - 1):
                        key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                        link_usage[key] += 1
            if link_usage:
                total += max(link_usage.values())
            count += 1
        return total / max(count, 1)

    # ================================================================
    # Hop cost
    # ================================================================

    def _permute_hop_cost(self, permute_steps):
        total_hops = 0
        total_transfers = 0
        for d in permute_steps:
            for r in range(self.world):
                dst = (r + d) % self.world
                if r == dst:
                    continue
                total_hops += self.topo.rank_hops(r, dst)
                total_transfers += 1
        return total_hops / max(total_transfers, 1)

    def _device_hop_cost(self, inter_schedule):
        num_devices = self.topo.num_devices
        total_hops = 0
        total_transfers = 0
        for d in inter_schedule:
            for dev in range(num_devices):
                dst_dev = (dev + d) % num_devices
                if dev == dst_dev:
                    continue
                total_hops += self.topo.device_hops(dev, dst_dev)
                total_transfers += 1
        return total_hops / max(total_transfers, 1)

    def _pairwise_hop_cost(self, matchings, round_order):
        total_hops = 0
        total_transfers = 0
        for ri in round_order:
            for a, b in matchings[ri]:
                total_hops += self.topo.rank_hops(a, b)
                total_transfers += 1
        return total_hops / max(total_transfers, 1)

    def _hybrid_hop_cost(self, params):
        near = params.get("near_distances", [])
        far = params.get("permute_schedule", [])
        total_hops = 0
        total = 0
        for d in near + far:
            for r in range(self.world):
                dst = (r + d) % self.world
                if r == dst:
                    continue
                total_hops += self.topo.rank_hops(r, dst)
                total += 1
        return total_hops / max(total, 1)

    # ================================================================
    # Balance
    # ================================================================

    def _balance_score(self, template_name, params):
        rank_bytes = [0] * self.world
        if template_name in ("permute_ring", "hybrid_ag_perm"):
            distances = params.get("schedule", []) or params.get("permute_schedule", [])
            near = params.get("near_distances", [])
            for d in list(distances) + list(near):
                for r in range(self.world):
                    dst = (r + d) % self.world
                    rank_bytes[r] += self.send_counts[r][dst]
        elif template_name == "allgather_slice":
            for r in range(self.world):
                rank_bytes[r] = sum(self.send_counts[r])
        elif template_name == "hierarchical":
            for r in range(self.world):
                rank_bytes[r] = sum(self.send_counts[r])
        elif template_name == "pairwise":
            for r in range(self.world):
                rank_bytes[r] = sum(self.send_counts[r])
        elif template_name == "fused_alltoall":
            for r in range(self.world):
                rank_bytes[r] = sum(self.send_counts[r])
        else:
            return 0

        mean = sum(rank_bytes) / self.world
        if mean == 0:
            return 0
        variance = sum((b - mean) ** 2 for b in rank_bytes) / self.world
        return math.sqrt(variance) / mean

    # ================================================================
    # Padding waste
    # ================================================================

    def _permute_padding(self, permute_steps):
        max_count = 0
        total_actual = 0
        num_transfers = 0
        for d in permute_steps:
            for r in range(self.world):
                dst = (r + d) % self.world
                if r == dst:
                    continue
                count = self.send_counts[r][dst]
                max_count = max(max_count, count)
                total_actual += count
                num_transfers += 1
        if num_transfers == 0 or max_count == 0:
            return 0
        return (max_count * num_transfers - total_actual) / (max_count * num_transfers)

    def _device_padding(self, inter_schedule):
        num_devices = self.topo.num_devices
        cpd = self.topo.cores_per_device
        device_send = [[0] * num_devices for _ in range(num_devices)]
        for sr in range(self.world):
            for dr in range(self.world):
                sd = self.topo.rank_to_device(sr)
                dd = self.topo.rank_to_device(dr)
                if sd != dd:
                    device_send[sd][dd] += self.send_counts[sr][dr]

        max_count = 0
        total_actual = 0
        num_transfers = 0
        for d in inter_schedule:
            for dev in range(num_devices):
                dst_dev = (dev + d) % num_devices
                if dev == dst_dev:
                    continue
                count = device_send[dev][dst_dev]
                max_count = max(max_count, count)
                total_actual += count
                num_transfers += 1
        if num_transfers == 0 or max_count == 0:
            return 0
        return (max_count * num_transfers - total_actual) / (max_count * num_transfers)

    # ================================================================
    # Fused alltoall metrics
    # ================================================================

    def _fused_alltoall_contention(self):
        """All flows concurrent in a single step — maximal link contention."""
        link_usage = defaultdict(int)
        for src in range(self.world):
            for dst in range(self.world):
                if src == dst:
                    continue
                src_dev = self.topo.rank_to_device(src)
                dst_dev = self.topo.rank_to_device(dst)
                if src_dev == dst_dev:
                    continue
                path = self.topo.device_path(src_dev, dst_dev)
                for i in range(len(path) - 1):
                    key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                    link_usage[key] += 1
        return max(link_usage.values()) if link_usage else 0

    def _fused_alltoall_hop_cost(self):
        """Average hops across all src->dst pairs."""
        total_hops = 0
        total_transfers = 0
        for src in range(self.world):
            for dst in range(self.world):
                if src == dst:
                    continue
                total_hops += self.topo.rank_hops(src, dst)
                total_transfers += 1
        return total_hops / max(total_transfers, 1)

    def _fused_alltoall_padding(self):
        """Padding waste: fraction of padded bytes that are zeros."""
        max_count = 0
        total_actual = 0
        for src in range(self.world):
            for dst in range(self.world):
                if src == dst:
                    continue
                count = self.send_counts[src][dst]
                max_count = max(max_count, count)
                total_actual += count
        num_transfers = self.world * (self.world - 1)
        if num_transfers == 0 or max_count == 0:
            return 0
        return (max_count * num_transfers - total_actual) / (max_count * num_transfers)

    def _internode_contention(self, schedule=None):
        """Measure EFA adapter saturation for cross-node traffic."""
        if self.num_nodes <= 1:
            return 0
        efa_adapters = getattr(self.topo, 'efa_adapters', 8)
        # Count flows per EFA adapter for all cross-node transfers
        adapter_usage = defaultdict(int)
        for src in range(self.world):
            for dst in range(self.world):
                if src == dst:
                    continue
                src_n = self.topo.rank_to_node(src)
                dst_n = self.topo.rank_to_node(dst)
                if src_n == dst_n:
                    continue
                src_dev = self.topo.rank_to_local_device(src)
                adapter = src_dev % efa_adapters
                adapter_usage[(src_n, dst_n, adapter)] += 1
        return max(adapter_usage.values()) if adapter_usage else 0

    def compare_schedules(self, schedules_dict):
        results = []
        for name, steps in schedules_dict.items():
            score, breakdown = self.evaluate_permute_schedule(steps)
            results.append((name, score, breakdown))
        results.sort(key=lambda x: x[1])
        return results
