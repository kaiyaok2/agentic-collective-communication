"""
Topology-aware contention analysis for communication schedule optimization.

Provides structured, interpretable feedback about WHY a schedule performs
well or poorly. This is the key input for LLM-guided refinement: instead
of just seeing a scalar score, the LLM sees which specific links are
overloaded at which specific steps, enabling targeted improvements.

Key insight: unlike general program synthesis where the LLM only sees
"score = X", communication scheduling provides decomposable feedback.
Each step's contention can be attributed to specific link conflicts,
giving the LLM surgical precision in proposing modifications.
"""

from collections import defaultdict


class ContentionAnalyzer:
    """Analyzes communication schedules on a topology to produce actionable feedback."""

    def __init__(self, topology, send_counts_matrix, element_bytes=4):
        self.topo = topology
        self.send_counts = send_counts_matrix
        self.element_bytes = element_bytes
        self.world = topology.num_cores
        self.num_devices = topology.num_devices
        self.cpd = topology.cores_per_device

        # Precompute link usage per distance (topology-invariant, compute once)
        self._distance_link_usage = {}
        for d in range(1, self.world):
            self._distance_link_usage[d] = self._compute_link_usage(d)

        # Precompute device-level link usage for hierarchical
        self._device_link_usage = {}
        for d in range(1, self.num_devices):
            self._device_link_usage[d] = self._compute_device_link_usage(d)

    def _compute_link_usage(self, d):
        """Which links are used when all ranks permute by distance d."""
        link_usage = defaultdict(int)
        for r in range(self.world):
            dst = (r + d) % self.world
            src_dev = self.topo.rank_to_device(r)
            dst_dev = self.topo.rank_to_device(dst)
            if src_dev == dst_dev:
                continue
            path = self.topo.device_path(src_dev, dst_dev)
            for i in range(len(path) - 1):
                key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                link_usage[key] += 1
        return dict(link_usage)

    def _compute_device_link_usage(self, d):
        """Which links are used when all devices permute by device-distance d."""
        link_usage = defaultdict(int)
        for dev in range(self.num_devices):
            dst_dev = (dev + d) % self.num_devices
            if dev == dst_dev:
                continue
            path = self.topo.device_path(dev, dst_dev)
            for i in range(len(path) - 1):
                key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
                link_usage[key] += 1
        return dict(link_usage)

    def distance_conflict_matrix(self, device_level=False):
        """
        Build conflict matrix between distances.

        conflict[i][j] = shared link pressure between distance i and j.
        High conflict means these distances should be separated in the schedule.
        """
        if device_level:
            distances = list(range(1, self.num_devices))
            usage_map = self._device_link_usage
        else:
            distances = list(range(1, self.world))
            usage_map = self._distance_link_usage

        n = len(distances)
        matrix = [[0.0] * n for _ in range(n)]

        for i, d1 in enumerate(distances):
            links1 = usage_map.get(d1, {})
            for j, d2 in enumerate(distances):
                if i == j:
                    continue
                links2 = usage_map.get(d2, {})
                shared = sum(
                    count1 * links2[link]
                    for link, count1 in links1.items()
                    if link in links2
                )
                matrix[i][j] = shared

        return distances, matrix

    def diagnose_schedule(self, schedule, template="permute_ring"):
        """
        Produce a structured contention trace for a schedule.

        Returns a diagnosis dict with per-step breakdown, bottleneck
        identification, conflict analysis, and actionable suggestions.
        """
        device_level = (template == "hierarchical")
        usage_map = self._device_link_usage if device_level else self._distance_link_usage
        hop_fn = self.topo.device_hops if device_level else self.topo.rank_hops
        n_entities = self.num_devices if device_level else self.world

        per_step = []
        for step_idx, d in enumerate(schedule):
            usage = usage_map.get(d, {})
            max_cont = max(usage.values()) if usage else 0
            total_hops = sum(
                hop_fn(r, (r + d) % n_entities) for r in range(n_entities)
            )
            bottleneck_links = [
                link for link, count in usage.items()
                if count == max_cont and max_cont > 1
            ]
            per_step.append({
                "step": step_idx,
                "distance": d,
                "max_contention": max_cont,
                "total_hops": total_hops,
                "bottleneck_links": bottleneck_links,
            })

        # Identify worst steps
        sorted_by_cont = sorted(
            range(len(per_step)),
            key=lambda i: per_step[i]["max_contention"],
            reverse=True,
        )
        bottleneck_steps = sorted_by_cont[:5]

        # Identify conflicting adjacent pairs
        conflict_pairs = []
        for i in range(len(schedule) - 1):
            d1, d2 = schedule[i], schedule[i + 1]
            links1 = usage_map.get(d1, {})
            links2 = usage_map.get(d2, {})
            shared = sum(
                links1.get(link, 0) * count
                for link, count in links2.items()
            )
            if shared > 0:
                conflict_pairs.append({
                    "steps": (i, i + 1),
                    "distances": (d1, d2),
                    "shared_link_pressure": shared,
                })
        conflict_pairs.sort(key=lambda x: x["shared_link_pressure"], reverse=True)

        suggestions = self._generate_suggestions(per_step, conflict_pairs, schedule)

        return {
            "per_step": per_step,
            "bottleneck_steps": bottleneck_steps,
            "conflict_pairs": conflict_pairs[:10],
            "suggestions": suggestions,
        }

    def _generate_suggestions(self, per_step, conflict_pairs, schedule):
        """Generate actionable suggestions based on contention analysis."""
        suggestions = []

        # High-contention steps early in schedule
        for step_info in per_step[:3]:
            if step_info["max_contention"] >= 4:
                d = step_info["distance"]
                suggestions.append(
                    f"Distance {d} (step {step_info['step']}) has contention "
                    f"{step_info['max_contention']}x on "
                    f"{len(step_info['bottleneck_links'])} links. "
                    f"Consider moving it later or separating from similar distances."
                )

        # Adjacent high-conflict pairs
        if conflict_pairs:
            worst = conflict_pairs[0]
            d1, d2 = worst["distances"]
            s1, s2 = worst["steps"]
            suggestions.append(
                f"Distances {d1} and {d2} (steps {s1},{s2}) share heavy link "
                f"pressure ({worst['shared_link_pressure']}). Separate them."
            )

        # Low-contention distances placed late
        low_cont_late = [
            s for s in per_step[len(per_step) // 2:]
            if s["max_contention"] <= 2
        ]
        if low_cont_late:
            d = low_cont_late[0]["distance"]
            suggestions.append(
                f"Distance {d} has low contention but is placed late "
                f"(step {low_cont_late[0]['step']}). Consider interleaving "
                f"it between high-contention steps."
            )

        return suggestions

    def format_diagnosis(self, diagnosis):
        """Format diagnosis as readable text for LLM consumption."""
        lines = []
        lines.append("=== Schedule Contention Diagnosis ===\n")

        lines.append("Step-by-step contention (distance -> max_link_usage):")
        for s in diagnosis["per_step"]:
            marker = " <<<" if s["step"] in diagnosis["bottleneck_steps"] else ""
            lines.append(
                f"  Step {s['step']:2d}: dist={s['distance']:2d}  "
                f"contention={s['max_contention']}  "
                f"hops={s['total_hops']}{marker}"
            )

        lines.append(f"\nWorst bottleneck steps: {diagnosis['bottleneck_steps']}")

        if diagnosis["conflict_pairs"]:
            lines.append("\nHighest-conflict adjacent step pairs:")
            for cp in diagnosis["conflict_pairs"][:5]:
                lines.append(
                    f"  Steps {cp['steps'][0]},{cp['steps'][1]} "
                    f"(dist {cp['distances'][0]},{cp['distances'][1]}): "
                    f"shared_pressure={cp['shared_link_pressure']}"
                )

        if diagnosis["suggestions"]:
            lines.append("\nSuggestions:")
            for i, s in enumerate(diagnosis["suggestions"]):
                lines.append(f"  {i + 1}. {s}")

        return "\n".join(lines)

    def suggest_swaps(self, schedule, diagnosis, top_k=5):
        """
        Suggest promising swap positions based on contention analysis.
        Returns list of (i, j) index pairs to try.
        """
        swaps = []

        # Separate high-conflict adjacent pairs
        for cp in diagnosis["conflict_pairs"][:top_k]:
            s1, s2 = cp["steps"]
            for target in range(len(schedule)):
                if abs(target - s1) > 2 and abs(target - s2) > 2:
                    swaps.append((s1, target))
                    break

        # Swap bottleneck steps with low-contention steps
        bottlenecks = set(diagnosis["bottleneck_steps"][:3])
        low_cont = [
            s["step"] for s in diagnosis["per_step"]
            if s["max_contention"] <= 2 and s["step"] not in bottlenecks
        ]
        for b in list(bottlenecks)[:3]:
            for lc in low_cont[:2]:
                swaps.append((b, lc))

        return swaps[:top_k]

    def diagnose_internode_contention(self, schedule=None):
        """
        Analyze EFA adapter saturation for cross-node traffic.

        Returns diagnosis dict with per-adapter load and bottleneck info.
        Only meaningful when topology has num_nodes > 1.
        """
        num_nodes = getattr(self.topo, 'num_nodes', 1)
        if num_nodes <= 1:
            return {"inter_node": False, "message": "Single-node topology"}

        efa_adapters = getattr(self.topo, 'efa_adapters', 8)
        ranks_per_node = getattr(self.topo, 'ranks_per_node', self.world)

        # Per-node-pair, per-adapter flow count and byte volume
        adapter_flows = defaultdict(int)
        adapter_bytes = defaultdict(float)

        for src in range(self.world):
            for dst in range(self.world):
                count = self.send_counts[src][dst]
                if count == 0:
                    continue
                src_n = self.topo.rank_to_node(src)
                dst_n = self.topo.rank_to_node(dst)
                if src_n == dst_n:
                    continue
                src_dev = self.topo.rank_to_local_device(src)
                adapter = src_dev % efa_adapters
                key = (src_n, dst_n, adapter)
                adapter_flows[key] += 1
                adapter_bytes[key] += count * self.element_bytes

        if not adapter_flows:
            return {"inter_node": True, "message": "No cross-node traffic"}

        max_flows = max(adapter_flows.values())
        max_bytes = max(adapter_bytes.values())
        busiest = max(adapter_bytes, key=adapter_bytes.get)

        # Per-node-pair aggregate
        node_pair_bytes = defaultdict(float)
        for (sn, dn, _), b in adapter_bytes.items():
            node_pair_bytes[(sn, dn)] += b
        busiest_pair = max(node_pair_bytes, key=node_pair_bytes.get)

        lines = []
        lines.append(f"Inter-node traffic: {num_nodes} nodes, "
                     f"{efa_adapters} EFA adapters/node")
        lines.append(f"Busiest adapter: node {busiest[0]}->node {busiest[1]} "
                     f"adapter {busiest[2]}: "
                     f"{max_flows} flows, {max_bytes/1e6:.1f} MB")
        lines.append(f"Busiest node pair: {busiest_pair[0]}->{busiest_pair[1]}: "
                     f"{node_pair_bytes[busiest_pair]/1e6:.1f} MB total")

        return {
            "inter_node": True,
            "num_nodes": num_nodes,
            "efa_adapters": efa_adapters,
            "max_adapter_flows": max_flows,
            "max_adapter_bytes": max_bytes,
            "busiest_adapter": busiest,
            "busiest_node_pair": busiest_pair,
            "node_pair_bytes": dict(node_pair_bytes),
            "text": "\n".join(lines),
        }
