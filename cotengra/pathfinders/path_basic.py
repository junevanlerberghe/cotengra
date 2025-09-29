"""Basic optimization routines."""

import ast
import bisect
from collections import defaultdict
from copy import deepcopy
import functools
import heapq
import itertools
import math

from ..core import ContractionTree
from ..oe import PathOptimizer
from ..parallel import get_n_workers, parse_parallel_arg
from ..reusable import ReusableOptimizer
from ..utils import GumbelBatchedGenerator, get_rng


def is_simplifiable(legs, appearances):
    """Check if ``legs`` contains any diag (repeated) or reduced (appears
    nowhere else) indices.
    """
    prev_ix = None
    for ix, ix_cnt in legs:
        if (ix == prev_ix) or (ix_cnt == appearances[ix]):
            # found a diag or reduced index
            return True
        prev_ix = ix
    return False


def compute_simplified(legs, appearances):
    """Compute the diag and reduced legs of a term. This function assumes that
    the legs are already sorted. It handles the case where a index is both
    diag and reduced (i.e. traced).
    """
    if not legs:
        return []

    new_legs = []
    cur_ix, cur_cnt = legs[0]
    for ix, ix_cnt in legs[1:]:
        if ix == cur_ix:
            # diag index-> accumulate count and continue
            cur_cnt += ix_cnt
        else:
            # index changed, flush
            if cur_cnt != appearances[cur_ix]:
                # index is not reduced -> keep
                new_legs.append((cur_ix, cur_cnt))
            cur_ix, cur_cnt = ix, ix_cnt

    if cur_cnt != appearances[cur_ix]:
        new_legs.append((cur_ix, cur_cnt))

    return new_legs


def compute_contracted(ilegs, jlegs, appearances, reverse_map=None, contraction_info=None):
    """Compute the contracted legs of two terms."""
    # do sorted simultaneous iteration over ilegs and jlegs
    ip = 0
    jp = 0
    ni = len(ilegs)
    nj = len(jlegs)
    new_legs = []
    new_traces = []
    while True:
        if ip == ni:
            # all remaining legs are from j
            new_legs.extend(jlegs[jp:])
            break
        if jp == nj:
            # all remaining legs are from i
            new_legs.extend(ilegs[ip:])
            break

        iix, ic = ilegs[ip]
        jix, jc = jlegs[jp]
        if iix < jix:
            # index only appears on i
            new_legs.append((iix, ic))
            ip += 1
        elif iix > jix:
            # index only appears on j
            new_legs.append((jix, jc))
            jp += 1
        else:  # iix == jix
            # shared index
            ijc = ic + jc
            if ijc != appearances[iix]:
                new_legs.append((iix, ijc))
            # print("shared index found: ", iix)
            # print("reverse map and contraction info are: ", reverse_map, contraction_info)
            if reverse_map is not None and contraction_info is not None:
                (node_idx1, leg1), (node_idx2, leg2) = contraction_info.index_to_legs[reverse_map[iix]]
                new_traces.append((node_idx1, node_idx2, leg1, leg2))
            ip += 1
            jp += 1

    return new_legs, new_traces


def compute_size(legs, sizes):
    """Compute the size of a term."""
    size = 1
    for ix, _ in legs:
        size *= sizes[ix]
    return size


def compute_flops(ilegs, jlegs, sizes):
    """Compute the flops cost of contracting two terms."""
    seen = set()
    flops = 1
    for ix, _ in ilegs:
        seen.add(ix)
        flops *= sizes[ix]
    for ix, _ in jlegs:
        if ix not in seen:
            flops *= sizes[ix]
    return flops

def group_traces_with_order(traces):
    parent = {}  # union-find parent
    cluster_traces = {}  # cluster rep -> list of traces
    cluster_nodes = {}   # cluster rep -> set of nodes

    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return rx
        # Merge ry into rx
        parent[ry] = rx
        # Merge traces
        cluster_traces[rx].extend(cluster_traces[ry])
        cluster_nodes[rx].update(cluster_nodes[ry])
        del cluster_traces[ry]
        del cluster_nodes[ry]
        return rx

    # Initialize clusters
    for node in {node for trace in traces for node in trace[:2]}:
        rep = find(node)
        cluster_traces[rep] = []
        cluster_nodes[rep] = {node}

    grouped = []
    for trace in traces:
        node1, node2, leg1, leg2 = trace
        rep1, rep2 = find(node1), find(node2)
        if rep1 != rep2:
            # Merge clusters
            merged_rep = union(rep1, rep2)
            # Add the current trace after merging
            cluster_traces[merged_rep].append(trace)
            grouped.append((list(cluster_traces[merged_rep]), cluster_nodes[merged_rep].copy()))
        else:
            # Same cluster, just append
            cluster_traces[rep1].append(trace)

    return grouped


def compute_size_custom(
    tn, combined_traces
):
    # print("RUNNING CUSTOM GREEDY!!!")
    # print("combined traces are: ", combined_traces)

    pte_list = list(tn.pte_list)
    node_to_pte = dict(tn.node_to_pte)
    groups = {}
    count = 0

    for traces in combined_traces:
        all_grouped_traces = []
        idx = 0
        to_skip = set()
        for node_idx1, node_idx2, leg1, leg2 in traces:
            if (node_idx1, node_idx2, leg1, leg2) in to_skip:
                idx += 1
                continue
            if node_idx1 not in groups and node_idx2 not in groups: # both nodes have not been merged with anything
                groups[node_idx1] = count
                groups[node_idx2] = count
                count += 1
                all_grouped_traces.append([(node_idx1, node_idx2, leg1, leg2)])
            elif node_idx1 in groups and node_idx2 not in groups: # merging existing group with new node
                grp = groups[node_idx1]
                traces_to_append = []
                traces_to_append.append((node_idx1, node_idx2, leg1, leg2))

                # now, loop through the rest of traces after current
                # if there is a trace with the same node and grp2, add it
                for j in range(idx + 1, len(traces)):
                    next_node1, next_node2, next_leg1, next_leg2 = traces[j]
                    next_grp1 = groups.get(next_node1)
                    next_grp2 = groups.get(next_node2)
                    if(node_idx2 in (next_node1, next_node2) and grp in (next_grp1, next_grp2)):
                        traces_to_append.append((next_node1, next_node2, next_leg1, next_leg2))
                        to_skip.add((next_node1, next_node2, next_leg1, next_leg2))

                groups[node_idx2] = grp

                all_grouped_traces.append(traces_to_append)
            elif node_idx1 not in groups and node_idx2 in groups: # merging existing group with new node
                grp = groups[node_idx2]

                traces_to_append = []
                traces_to_append.append((node_idx1, node_idx2, leg1, leg2))

                # now, loop through the rest of traces after current
                # if there is a trace with the same node and grp2, add it
                for j in range(idx + 1, len(traces)):
                    next_node1, next_node2, next_leg1, next_leg2 = traces[j]
                    next_grp1 = groups.get(next_node1)
                    next_grp2 = groups.get(next_node2)
                    if(node_idx1 in (next_node1, next_node2) and grp in (next_grp1, next_grp2)):
                        traces_to_append.append((next_node1, next_node2, next_leg1, next_leg2))
                        to_skip.add((next_node1, next_node2, next_leg1, next_leg2))

                groups[node_idx1] = grp
                all_grouped_traces.append(traces_to_append)

            elif groups[node_idx1] != groups[node_idx2]: # merging two existing groups
                # want to merge group of node_idx2 into group of node_idx1
                grp1 = groups[node_idx1]
                grp2 = groups[node_idx2]

                traces_to_append = []
                traces_to_append.append((node_idx1, node_idx2, leg1, leg2))

                # now, loop through the rest of traces after current
                # if there is a trace with the same grp1 and grp2, add it
                for j in range(idx + 1, len(traces)):
                    next_node1, next_node2, next_leg1, next_leg2 = traces[j]
                    next_grp1 = groups.get(next_node1)
                    next_grp2 = groups.get(next_node2)

                    if(grp1 in (next_grp1, next_grp2) and grp2 in (next_grp1, next_grp2)):
                        traces_to_append.append((next_node1, next_node2, next_leg1, next_leg2))
                        to_skip.add((next_node1, next_node2, next_leg1, next_leg2))

                for k, v in groups.items():
                    if v == grp2:
                        groups[k] = grp1

                all_grouped_traces.append(traces_to_append)

            else: # both nodes already in same group 
                print("SHOULD NOT HAPPEN!! current trace is: ", (node_idx1, node_idx2, leg1, leg2))
                print("\t current groups: ", groups)
                print("\t all grouped traces: ", all_grouped_traces)
                assert 12 == 11
            idx += 1

        # print("all lists of traces is now: ", all_grouped_traces)
        for traces in all_grouped_traces:
            pte_ids = {
                node_to_pte[node_idx1] for node_idx1, _, _, _ in traces
            }.union({node_to_pte[node_idx2] for _, node_idx2, _, _ in traces})

            assert len(pte_ids) == 2, f"Expected 2 PTEs, got {len(pte_ids)}"
            pte1_idx, pte2_idx = pte_ids
            join_legs1 = []
            join_legs2 = []

            node_join_legs = defaultdict(list)

            for node_idx1, node_idx2, legs1, legs2 in traces:
                leg1 = legs1[0]
                leg2 = legs2[0]

                node_join_legs[node_idx1].append(legs1)
                node_join_legs[node_idx2].append(legs2)

                if node_idx1 in pte_list[pte1_idx][1]:
                    join_legs1.append(legs1)
                else:
                    join_legs2.append(legs1)

                if node_idx2 in pte_list[pte2_idx][1]:
                    join_legs2.append(legs2)
                else:
                    join_legs1.append(legs2)

            pte1, nodes1 = pte_list[pte1_idx]
            pte2, nodes2 = pte_list[pte2_idx]
            merged_nodes = nodes1.union(nodes2)

            new_pte = pte1.merge_with(
                pte2,
                tuple(join_legs1),
                tuple(join_legs2)
            )

            for node_idx in new_pte.node_ids:
                node_to_pte[node_idx] = pte1_idx

            # Update the first PTE with merged result
            pte_list[pte1_idx] = (new_pte, merged_nodes)
            # Remove the second PTE
            pte_list.pop(pte2_idx)

            # Update node_to_pte mappings
            for node_idx in nodes2:
                node_to_pte[node_idx] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node_idx, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node_idx] = pte_idx - 1

    size = 2**new_pte.rank() # returning size of the newest pte
    return size


def compute_con_cost_custom(
    tn, combined_traces
):
    from planqtn.symplectic import count_matching_stabilizers_ratio_all_pairs
    # print("RUNNING CUSTOM OPTIMAL!!!")
    cost = 0
    pte_list = list(tn.pte_list)
    node_to_pte = dict(tn.node_to_pte)
    groups = {}
    count = 0
    print("combined traces are: ", combined_traces)

    for traces in combined_traces:
        all_grouped_traces = []
        idx = 0
        to_skip = set()
        for node_idx1, node_idx2, leg1, leg2 in traces:
            if (node_idx1, node_idx2, leg1, leg2) in to_skip:
                idx += 1
                continue
            if node_idx1 not in groups and node_idx2 not in groups: # both nodes have not been merged with anything
                groups[node_idx1] = count
                groups[node_idx2] = count
                count += 1
                all_grouped_traces.append([(node_idx1, node_idx2, leg1, leg2)])
            elif node_idx1 in groups and node_idx2 not in groups: # merging existing group with new node
                grp = groups[node_idx1]
                traces_to_append = []
                traces_to_append.append((node_idx1, node_idx2, leg1, leg2))

                # now, loop through the rest of traces after current
                # if there is a trace with the same node and grp2, add it
                for j in range(idx + 1, len(traces)):
                    next_node1, next_node2, next_leg1, next_leg2 = traces[j]
                    next_grp1 = groups.get(next_node1)
                    next_grp2 = groups.get(next_node2)
                    if(node_idx2 in (next_node1, next_node2) and grp in (next_grp1, next_grp2)):
                        traces_to_append.append((next_node1, next_node2, next_leg1, next_leg2))
                        to_skip.add((next_node1, next_node2, next_leg1, next_leg2))

                groups[node_idx2] = grp

                all_grouped_traces.append(traces_to_append)
            elif node_idx1 not in groups and node_idx2 in groups: # merging existing group with new node
                grp = groups[node_idx2]

                traces_to_append = []
                traces_to_append.append((node_idx1, node_idx2, leg1, leg2))

                # now, loop through the rest of traces after current
                # if there is a trace with the same node and grp2, add it
                for j in range(idx + 1, len(traces)):
                    next_node1, next_node2, next_leg1, next_leg2 = traces[j]
                    next_grp1 = groups.get(next_node1)
                    next_grp2 = groups.get(next_node2)
                    if(node_idx1 in (next_node1, next_node2) and grp in (next_grp1, next_grp2)):
                        traces_to_append.append((next_node1, next_node2, next_leg1, next_leg2))
                        to_skip.add((next_node1, next_node2, next_leg1, next_leg2))

                groups[node_idx1] = grp
                all_grouped_traces.append(traces_to_append)

            elif groups[node_idx1] != groups[node_idx2]: # merging two existing groups
                # want to merge group of node_idx2 into group of node_idx1
                grp1 = groups[node_idx1]
                grp2 = groups[node_idx2]

                traces_to_append = []
                traces_to_append.append((node_idx1, node_idx2, leg1, leg2))

                # now, loop through the rest of traces after current
                # if there is a trace with the same grp1 and grp2, add it
                for j in range(idx + 1, len(traces)):
                    next_node1, next_node2, next_leg1, next_leg2 = traces[j]
                    next_grp1 = groups.get(next_node1)
                    next_grp2 = groups.get(next_node2)

                    if(grp1 in (next_grp1, next_grp2) and grp2 in (next_grp1, next_grp2)):
                        traces_to_append.append((next_node1, next_node2, next_leg1, next_leg2))
                        to_skip.add((next_node1, next_node2, next_leg1, next_leg2))

                for k, v in groups.items():
                    if v == grp2:
                        groups[k] = grp1

                all_grouped_traces.append(traces_to_append)

            else: # both nodes already in same group 
                print("SHOULD NOT HAPPEN!! current trace is: ", (node_idx1, node_idx2, leg1, leg2))
                print("\t current groups: ", groups)
                print("\t all grouped traces: ", all_grouped_traces)
                assert 12 == 11
            idx += 1

        # print("all lists of traces is now: ", all_grouped_traces)
        for traces in all_grouped_traces:
            pte_ids = {
                node_to_pte[node_idx1] for node_idx1, _, _, _ in traces
            }.union({node_to_pte[node_idx2] for _, node_idx2, _, _ in traces})

            assert len(pte_ids) == 2, f"Expected 2 PTEs, got {len(pte_ids)}"
            pte1_idx, pte2_idx = pte_ids
            join_legs1 = []
            join_legs2 = []

            node_join_legs = defaultdict(list)

            for node_idx1, node_idx2, legs1, legs2 in traces:
                leg1 = legs1[0]
                leg2 = legs2[0]

                node_join_legs[node_idx1].append(legs1)
                node_join_legs[node_idx2].append(legs2)

                if node_idx1 in pte_list[pte1_idx][1]:
                    join_legs1.append(legs1)
                else:
                    join_legs2.append(legs1)

                if node_idx2 in pte_list[pte2_idx][1]:
                    join_legs2.append(legs2)
                else:
                    join_legs1.append(legs2)

            pte1, nodes1 = pte_list[pte1_idx]
            pte2, nodes2 = pte_list[pte2_idx]
            merged_nodes = nodes1.union(nodes2)

            new_pte = pte1.merge_with(
                pte2,
                tuple(join_legs1),
                tuple(join_legs2)
            )

            for node_idx in new_pte.node_ids:
                node_to_pte[node_idx] = pte1_idx

            # Update the first PTE with merged result
            pte_list[pte1_idx] = (new_pte, merged_nodes)
            # Remove the second PTE
            pte_list.pop(pte2_idx)

            # Update node_to_pte mappings
            for node_idx in nodes2:
                node_to_pte[node_idx] = pte1_idx
            # Adjust indices for all nodes in PTEs after the removed one
            for node_idx, pte_idx in node_to_pte.items():
                if pte_idx > pte2_idx:
                    node_to_pte[node_idx] = pte_idx - 1

            prev_submatrix1 = pte1.rank()
            prev_submatrix2 = pte2.rank()

            matches = count_matching_stabilizers_ratio_all_pairs(pte1, pte2, join_legs1, join_legs2) 
            cost += (2 ** (prev_submatrix1 + prev_submatrix2)) * matches

    return cost

def compute_con_cost_flops(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the total flops cost of a contraction given by temporary legs,
    also removing any contracted indices from the temporary legs.
    """
    #print("Compute flops cost of contraction with legs:", temp_legs)
    #print("other inputs are: appearances:", appearances,
     #     "sizes:", sizes, "iscore:", iscore, "jscore:", jscore)
    
    cost = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]

    #print("returning iscore + jscore + cost, cost = ", cost)
    return iscore + jscore + cost


def compute_con_cost_max(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the max flops cost of a contraction given by temporary legs,
    also removing any contracted indices from the temporary legs.
    """
    cost = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]

    return max((iscore, jscore, cost))


def compute_con_cost_size(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the max size of a contraction given by temporary legs, also
    removing any contracted indices from the temporary legs.
    """
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            size *= sizes[ix]

    return max((iscore, jscore, size))


def compute_con_cost_write(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
):
    """Compute the total write cost of a contraction given by temporary legs,
    also removing any contracted indices from the temporary legs.
    """
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            # kept index, contributes to new size
            size *= sizes[ix]

    return iscore + jscore + size


def compute_con_cost_combo(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
    factor,
):
    """Compute the combined total flops and write cost of a contraction given
    by temporary legs, also removing any contracted indices from the temporary
    legs. The combined cost is given by:

        cost = flops + factor * size
    """
    cost = 1
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            # kept index, contributes to new size
            size *= d

    return iscore + jscore + (cost + factor * size)


def compute_con_cost_limit(
    temp_legs,
    appearances,
    sizes,
    iscore,
    jscore,
    factor,
):
    """Compute the combined total flops and write cost of a contraction given
    by temporary legs, also removing any contracted indices from the temporary
    legs. The combined cost is given by:

        cost = max(flops, factor * size)

    I.e. assuming one or another to be the limiting factor.
    """
    cost = 1
    size = 1
    for i in range(len(temp_legs) - 1, -1, -1):
        ix, ix_count = temp_legs[i]
        d = sizes[ix]
        cost *= d
        if ix_count == appearances[ix]:
            # contracted index, remove
            del temp_legs[i]
        else:
            # kept index, contributes to new size
            size *= d

    new_local_score = max(cost, factor * size)
    return iscore + jscore + new_local_score


@functools.lru_cache(128)
def parse_minimize_for_optimal(minimize):
    """Given a string, parse it into a function that computes the cost of a
    contraction. The string can be one of the following:

        - "flops": compute_con_cost_flops
        - "max": compute_con_cost_max
        - "size": compute_con_cost_size
        - "write": compute_con_cost_write
        - "combo": compute_con_cost_combo
        - "combo-{factor}": compute_con_cost_combo with specified factor
        - "limit": compute_con_cost_limit
        - "limit-{factor}": compute_con_cost_limit with specified factor

    This function is cached for speed.
    """
    import re
    if minimize == "flops":
        return compute_con_cost_flops
    elif minimize == "max":
        return compute_con_cost_max
    elif minimize == "size":
        return compute_con_cost_size
    elif minimize == "write":
        return compute_con_cost_write
    elif minimize == "custom_flops":
        return compute_con_cost_custom
    elif callable(minimize):
        return minimize

    minimize, *maybe_factor = minimize.split("-")

    if not maybe_factor:
        # default factor
        factor = 64
    else:
        fstr, = maybe_factor
        if fstr.isdigit():
            # keep integer arithmetic if possible
            factor = int(fstr)
        else:
            factor = float(fstr)

    if minimize == "combo":
        return functools.partial(compute_con_cost_combo, factor=factor)
    elif minimize == "limit":
        return functools.partial(compute_con_cost_limit, factor=factor)
    else:
        raise ValueError(f"Couldn't parse `minimize` value: {minimize}.")


class ContractionProcessor:
    """A helper class for combining bottom up simplifications, greedy, and
    optimal contraction path optimization.
    """

    __slots__ = (
        "nodes",
        "node_names",
        "contraction_info",
        "edges",
        "indmap",
        "appearances",
        "sizes",
        "ssa",
        "ssa_path",
        "track_flops",
        "flops",
        "flops_limit",
    )

    def __init__(
        self,
        inputs,
        output,
        size_dict,
        contraction_info=None,
        track_flops=False,
        flops_limit=float("inf"),
    ):
        self.nodes = {}
        self.edges = {}
        self.indmap = {}
        self.appearances = []
        self.node_names = []
        self.contraction_info = contraction_info
        self.sizes = []
        c = 0
        #print("contraction info in contraction processor is: ", contraction_info)
        #print("inputs to contraction processor: ", inputs)
        for i, term in enumerate(inputs):
            legs = []
            for ind in term:
                ix = self.indmap.get(ind, None)
                if ix is None:
                    # index not processed yet
                    ix = self.indmap[ind] = c
                    self.edges[ix] = {i: None}
                    self.appearances.append(1)
                    self.node_names.append(ind)
                    self.sizes.append(size_dict[ind])
                    c += 1
                else:
                    # seen index already
                    self.appearances[ix] += 1
                    self.edges[ix][i] = None
                legs.append((ix, 1))

            legs.sort()
            self.nodes[i] = tuple(legs)

        for ind in output:
            self.appearances[self.indmap[ind]] += 1

        # print("node names is: ", self.node_names)
        # print("indmap is: ", self.indmap)
        # print("self.nodes is: ", self.nodes)
        
        self.ssa = len(self.nodes)
        self.ssa_path = []
        self.track_flops = track_flops
        self.flops = 0
        self.flops_limit = flops_limit

    def copy(self):
        new = ContractionProcessor.__new__(ContractionProcessor)
        new.nodes = self.nodes.copy()
        new.edges = {k: v.copy() for k, v in self.edges.items()}
        new.indmap = self.indmap  # never mutated
        new.appearances = self.appearances  # never mutated
        new.sizes = self.sizes  # never mutated
        new.ssa = self.ssa
        new.ssa_path = self.ssa_path.copy()
        new.track_flops = self.track_flops
        new.flops = self.flops
        new.flops_limit = self.flops_limit
        return new

    def neighbors(self, i):
        """Get all neighbors of node ``i``."""
        # only want to yield each neighbor once and not i itself
        for ix, _ in self.nodes[i]:
            for j in self.edges[ix]:
                if j != i:
                    yield j

    def print_current_terms(self):
        return ",".join(
            "".join(str(ix) for ix, c in term) for term in self.nodes.values()
        )

    def remove_ix(self, ix):
        """Drop the index ``ix``, simply removing it from all nodes and the
        edgemap.
        """
        for node in self.edges.pop(ix):
            self.nodes[node] = tuple(
                (jx, jx_count) for jx, jx_count in self.nodes[node] if jx != ix
            )

    def pop_node(self, i):
        """Remove node ``i`` from the graph, updating the edgemap and returning
        the legs of the node.
        """
        legs = self.nodes.pop(i)
        for ix, _ in legs:
            try:
                ix_nodes = self.edges[ix]
                ix_nodes.pop(i, None)
                if len(ix_nodes) == 0:
                    del self.edges[ix]
            except KeyError:
                # repeated index already removed
                pass
        return legs

    def add_node(self, legs):
        """Add a new node to the graph, updating the edgemap and returning the
        node index of the new node.
        """
        i = self.ssa
        self.ssa += 1
        self.nodes[i] = legs
        for ix, _ in legs:
            self.edges.setdefault(ix, {})[i] = None
        return i

    def check(self):
        """Check that the current graph is valid, useful for debugging."""
        for node, legs in self.nodes.items():
            for ix, _ in legs:
                assert node in self.edges[ix]
        for ix, ix_nodes in self.edges.items():
            for node in ix_nodes:
                assert ix in {jx for jx, _ in self.nodes[node]}

    def contract_nodes(self, i, j, new_legs=None):
        """Contract the nodes ``i`` and ``j``, adding a new node to the graph
        and returning its index.
        """
        ilegs = self.pop_node(i)
        jlegs = self.pop_node(j)

        if self.track_flops:
            self.flops += compute_flops(ilegs, jlegs, self.sizes)

        if new_legs is None:
            new_legs, new_traces = compute_contracted(ilegs, jlegs, self.appearances)

        k = self.add_node(new_legs)
        self.ssa_path.append((i, j))
        return k

    def simplify_batch(self):
        """Find any indices that appear in all terms and remove them, since
        they simply add an constant factor to the cost of the contraction, but
        create a fully connected graph if left.
        """
        ix_to_remove = []
        for ix, ix_nodes in self.edges.items():
            if len(ix_nodes) >= len(self.nodes):
                ix_to_remove.append(ix)
        for ix in ix_to_remove:
            self.remove_ix(ix)

    def simplify_single_terms(self):
        """Take any diags, reductions and traces of single terms."""
        for i, legs in tuple(self.nodes.items()):
            if is_simplifiable(legs, self.appearances):
                new_legs = compute_simplified(
                    self.pop_node(i), self.appearances
                )
                self.add_node(new_legs)
                self.ssa_path.append((i,))

    def simplify_scalars(self):
        """Remove all scalars, contracting them into the smallest remaining
        node, if there is one.
        """
        scalars = []
        j = None
        jndim = None
        for i, legs in self.nodes.items():
            ndim = len(legs)
            if ndim == 0:
                # scalar
                scalars.append(i)
            elif (j is None) or (ndim < jndim):
                # also find the smallest other node, to multiply into
                j = i
                jndim = ndim

        if scalars:
            if j is not None:
                scalars.append(j)

            # binary contract from left to right ((((0, 1), 2), 3), ...)
            for p in range(len(scalars) - 1):
                k = self.contract_nodes(scalars[p], scalars[p + 1])
                scalars[p + 1] = k

    def simplify_hadamard(self):
        groups = {}
        hadamards = set()
        for i, legs in self.nodes.items():
            key = frozenset(ix for ix, _ in legs)
            if key in groups:
                groups[key].append(i)
                hadamards.add(key)
            else:
                groups[key] = [i]

        for key in hadamards:
            group = groups[key]
            while len(group) > 1:
                i = group.pop()
                j = group.pop()
                group.append(self.contract_nodes(i, j))

    def simplify(self):
        self.simplify_batch()
        should_run = True
        while should_run:
            self.simplify_single_terms()
            self.simplify_scalars()
            ssa_before = self.ssa
            self.simplify_hadamard()
            # only rerun if we did hadamard deduplication
            should_run = ssa_before != self.ssa

    def subgraphs(self):
        remaining = set(self.nodes)
        groups = []
        while remaining:
            i = remaining.pop()
            queue = [i]
            group = {i}
            while queue:
                i = queue.pop()
                for j in self.neighbors(i):
                    if j not in group:
                        group.add(j)
                        queue.append(j)

            remaining -= group
            groups.append(sorted(group))

        groups.sort()
        return groups

    def optimize_greedy(
        self,
        minimize="combo",
        costmod=1.0,
        temperature=0.0,
        seed=None,
    ):
        print("running optimize greedy with minimize: ", minimize, " and self.contraction info is: ", self.contraction_info)
        """ """
        if temperature == 0.0:

            def local_score(sa, sb, sab):
                return sab / costmod - (sa + sb) * costmod

        else:
            gmblgen = GumbelBatchedGenerator(seed)

            def local_score(sa, sb, sab):
                score = math.log(sab) / costmod - (math.log(sa + sb)) * costmod
                if score > 0:
                    return math.log(score) - temperature * gmblgen()
                elif score < 0:
                    return -math.log(-score) - temperature * gmblgen()
                else:
                    return -temperature * gmblgen()

        reverse_map = {v: k for k, v in self.indmap.items()}

        node_sizes = {}
        node_traces = {}
        
        idx = 0
        for i, ilegs in self.nodes.items():
            if minimize == "custom_flops" and self.contraction_info is not None: 
                node_idx = self.contraction_info.input_names[idx]
                node_sizes[i] = 2**(self.contraction_info.nodes[node_idx].rank())
            else:
                node_sizes[i] = compute_size(ilegs, self.sizes)

            node_traces[i] = []
            idx += 1
    
        queue = []
        contractions = {}
        c = 0
        for ix_nodes in self.edges.values():
            for i, j in itertools.combinations(ix_nodes, 2):
                isize = node_sizes[i]
                jsize = node_sizes[j]

                itraces = node_traces[i]
                jtraces = node_traces[j]
                
                klegs, new_traces = compute_contracted(
                    self.nodes[i], self.nodes[j], self.appearances, reverse_map, self.contraction_info
                )
                combined_traces = list(itraces) + list(jtraces) + [new_traces]
                # print("itraces: ", itraces)
                # print("jtraces: ", jtraces)
                # print("new_traces: ", new_traces)
                if(minimize == "custom_flops" and self.contraction_info is not None):
                    ksize = compute_size_custom(self.contraction_info, combined_traces)
                    # print("found ksize = ", ksize)
                else:
                    ksize = compute_size(klegs, self.sizes)
                score = local_score(isize, jsize, ksize)
                heapq.heappush(queue, (score, c))
                contractions[c] = (i, j, ksize, klegs, combined_traces)
                c += 1

        while queue:
            #print("optimizing greedy while loop")
            _, c0 = heapq.heappop(queue)
            
            i, j, ksize, klegs, ktraces = contractions.pop(c0)
            if (i not in self.nodes) or (j not in self.nodes):
                # one of nodes already contracted
                continue

            k = self.contract_nodes(i, j, new_legs=klegs)

            if self.track_flops and (self.flops >= self.flops_limit):
                # shortcut - stop early and return failed
                return False

            node_sizes[k] = ksize
            node_traces[k] = ktraces

            for l in self.neighbors(k):
                lsize = node_sizes[l]
                mlegs, new_traces = compute_contracted(
                    klegs, self.nodes[l], self.appearances, reverse_map, self.contraction_info
                )
                combined_traces = list(node_traces[k]) + list(node_traces[l]) + [new_traces]
                # print("node_traces[k]: ", node_traces[k])
                # print("node_traces[l]: ", node_traces[l])
                # print("new_traces: ", new_traces)
                if(minimize == "custom_flops" and self.contraction_info is not None):
                    msize = compute_size_custom(self.contraction_info, combined_traces)
                    #print("found msize = ", msize)
                else:
                    msize = compute_size(mlegs, self.sizes)
                score = local_score(ksize, lsize, msize)
                heapq.heappush(queue, (score, c))
                contractions[c] = (k, l, msize, mlegs, combined_traces)
                c += 1

        return True

    def optimize_optimal_connected(
        self,
        where,
        minimize="combo",
        cost_cap=2,
        search_outer=False,
    ):
        compute_con_cost = parse_minimize_for_optimal(minimize)
        print("self.contraction_info in optimal is: ", self.contraction_info)
        print("minimize in optimal is: ", minimize)

        nterms = len(where)
        contractions = [{} for _ in range(nterms + 1)]
        # we use linear index within terms given during optimization, this maps
        # back to the original node index
        termmap = {}

        for i, node in enumerate(where):
            ilegs = self.nodes[node]
            # if(len(ilegs) > 1):
            #     ilegs = [ilegs[0]]

            # print("ilegs are: ", ilegs)
            isubgraph = 1 << i
            termmap[isubgraph] = node
            iscore = 0
            ipath = ()
            traces_path = []
            contractions[1][isubgraph] = (ilegs, iscore, ipath, traces_path)

        reverse_map = {v: k for k, v in self.indmap.items()}
    
        count = 0
        while not contractions[nterms]: # continue until entire graph has been built
            # print("contractions: ", contractions)
            for m in range(2, nterms + 1):
                # print("making subgraphs of size: ", m)
                # try and make subgraphs of size m
                contractions_m = contractions[m]
                for k in range(1, m // 2 + 1):
                    # made up of bipartitions of size k, m - k
                    if k != m - k:
                        # need to check all combinations
                        pairs = itertools.product(
                            contractions[k].items(),
                            contractions[m - k].items(),
                        )
                    else:
                        # only want unique combinations
                        pairs = itertools.combinations(contractions[k].items(), 2)

                    for (subgraph_i, (ilegs, iscore, ipath, itraces)), (
                        subgraph_j,
                        (jlegs, jscore, jpath, jtraces),
                    ) in pairs:
                        if subgraph_i & subgraph_j:
                            # subgraphs overlap -> invalid
                            continue

                        # do sorted simultaneous iteration over ilegs and jlegs
                        ip = 0
                        jp = 0
                        ni = len(ilegs)
                        nj = len(jlegs)
                        new_legs = []
                        traces = []
                        # if search_outer -> we will never skip
                        
                        skip_because_outer = not search_outer
                        while (ip < ni) and (jp < nj):
                            iix, ic = ilegs[ip]
                            jix, jc = jlegs[jp]
                            if iix < jix:
                                new_legs.append((iix, ic))
                                ip += 1
                            elif iix > jix:
                                new_legs.append((jix, jc))
                                jp += 1
                            else:  # iix == jix:
                                # shared index
                                #print("shared index found")
                                new_legs.append((iix, ic + jc))

                                if(minimize == "custom_flops" and self.contraction_info is not None):
                                    (node_idx1, leg1), (node_idx2, leg2) = self.contraction_info.index_to_legs[reverse_map[iix]]
                                    traces.append((node_idx1, node_idx2, leg1, leg2))
                                jp += 1
                                skip_because_outer = False

                        if skip_because_outer:
                            # no shared indices found
                            continue

                        # add any remaining non-shared indices
                        new_legs.extend(ilegs[ip:])
                        new_legs.extend(jlegs[jp:])
                        
                        combined_traces = list(itraces) + list(jtraces) + [traces]

                        if(minimize == "custom_flops" and self.contraction_info is not None):
                            if( len(where) < 10):
                                new_score = compute_con_cost_custom(
                                    self.contraction_info,
                                    combined_traces
                                )
                            else:
                                new_score = compute_con_cost_flops(
                                    new_legs,
                                    self.appearances,
                                    self.sizes,
                                    iscore,
                                    jscore,
                                )
                        else:
                            new_score = compute_con_cost(
                                new_legs,
                                self.appearances,
                                self.sizes,
                                iscore,
                                jscore,
                            )

                        if new_score > cost_cap:
                            # sieve contraction
                            continue

                        new_subgraph = subgraph_i | subgraph_j
                        current = contractions_m.get(new_subgraph, None)
                        if (current is None) or (new_score < current[1]):
                            new_path = (
                                *ipath,
                                *jpath,
                                (subgraph_i, subgraph_j),
                            )
                            contractions_m[new_subgraph] = (
                                new_legs,
                                new_score,
                                new_path,
                                combined_traces,
                            )

            # make the holes of our 'sieve' wider
            cost_cap *= 2
            count += 1

        ((final_legs, final_score, bitpath, final_traces),) = contractions[nterms].values()
        # print("contractions are: ", contractions)
        # print("Final path chosen is: ", final_legs)
        print("Final score from optimize optimal is: ", final_score)
        print("final traces: ", final_traces)
        for subgraph_i, subgraph_j in bitpath:
            i = termmap[subgraph_i]
            j = termmap[subgraph_j]
            k = self.contract_nodes(i, j)
            termmap[subgraph_i | subgraph_j] = k


    def optimize_optimal(
        self, minimize="flops", cost_cap=2, search_outer=False
    ):
        # we need to optimize each disconnected subgraph separately
        
        for where in self.subgraphs():
            #print("where is: ", where)
            self.optimize_optimal_connected(
                where,
                minimize=minimize,
                cost_cap=cost_cap,
                search_outer=search_outer,
            )

    def optimize_remaining_by_size(self):
        """This function simply contracts remaining terms in order of size, and
        is meant to handle the disconnected terms left after greedy or optimal
        optimization.
        """
        if len(self.nodes) == 1:
            # nothing to do
            return

        if len(self.nodes) == 2:
            self.contract_nodes(*self.nodes)
            return

        nodes_sizes = [
            (compute_size(legs, self.sizes), i)
            for i, legs in self.nodes.items()
        ]
        heapq.heapify(nodes_sizes)

        while len(nodes_sizes) > 1:
            # contract the smallest two nodes until only one remains
            _, i = heapq.heappop(nodes_sizes)
            _, j = heapq.heappop(nodes_sizes)
            k = self.contract_nodes(i, j)
            ksize = compute_size(self.nodes[k], self.sizes)
            heapq.heappush(nodes_sizes, (ksize, k))


def linear_to_ssa(path, N=None):
    """Convert a path with recycled linear ids to a path with static single
    assignment ids. For example::

        >>> linear_to_ssa([(0, 3), (1, 2), (0, 1)])
        [(0, 3), (2, 4), (1, 5)]

    """
    if N is None:
        N = sum(map(len, path)) - len(path) + 1

    ids = list(range(N))
    ssa = N
    ssa_path = []
    for con in path:
        scon = tuple(ids.pop(c) for c in sorted(con, reverse=True))
        ssa_path.append(scon)
        ids.append(ssa)
        ssa += 1
    return ssa_path


def ssa_to_linear(ssa_path, N=None):
    """Convert a path with static single assignment ids to a path with recycled
    linear ids. For example::

        >>> ssa_to_linear([(0, 3), (2, 4), (1, 5)])
        [(0, 3), (1, 2), (0, 1)]

    """
    if N is None:
        N = sum(map(len, ssa_path)) - len(ssa_path) + 1

    ids = list(range(N))
    path = []
    ssa = N
    for scon in ssa_path:
        con = [bisect.bisect_left(ids, s) for s in scon]
        con.sort()
        for j in reversed(con):
            ids.pop(j)
        ids.append(ssa)
        path.append(con)
        ssa += 1
    return path


def edge_path_to_ssa(edge_path, inputs):
    """Convert a path specified by a sequence of edge indices to a path with
    tuples of single static assignment (SSA) indices.

    Parameters
    ----------
    edge_path : sequence[str | int]
        The path specified by a sequence of edge indices.
    inputs : tuple[tuple[str | int]]
        The indices of each input tensor.

    Returns
    -------
    path : tuple[tuple[int]]
        The contraction path in static single assignment (SSA) form.
    """

    N = len(inputs)
    # record which ssas each index appears on
    ind_to_ssas = {}
    # track which indices appear on which term
    ssa_to_inds = {}
    # populate maps
    for i, term in enumerate(inputs):
        for ix in term:
            ind_to_ssas.setdefault(ix, set()).add(i)
        ssa_to_inds[i] = set(term)

    ssa_path = []
    ssa = N
    for ix in edge_path:
        # get ssas containing ix -> contract these
        scon = ind_to_ssas.pop(ix)
        if len(scon) < 2:
            # nothing to contract, e.g. index contracted alongside another
            continue

        # update map of where indices are
        new_term = set()
        for s in scon:
            for jx in ssa_to_inds.pop(s):
                # only need to update remaining indices
                jx_ssas = ind_to_ssas.get(jx, None)
                if jx_ssas is not None:
                    # remove children
                    jx_ssas.remove(s)
                    # add new parent
                    jx_ssas.add(ssa)
                    # calc new term (might have extraneous indices)
                    new_term.add(jx)

        ssa_to_inds[ssa] = new_term
        ssa_path.append(tuple(sorted(scon)))
        ssa += 1

    return tuple(ssa_path)


def edge_path_to_linear(edge_path, inputs):
    """Convert a path specified by a sequence of edge indices to a path with
    recycled linear ids.

    Parameters
    ----------
    edge_path : sequence[str | int]
        The path specified by a sequence of edge indices.
    inputs : tuple[tuple[str | int]]
        The indices of each input tensor.

    Returns
    -------
    path : tuple[tuple[int]]
        The contraction path in recycled linear id format.
    """
    ssa_path = edge_path_to_ssa(edge_path, inputs)
    return ssa_to_linear(ssa_path, len(inputs))


def is_ssa_path(path, nterms):
    """Check if an explicitly given path is in 'static single assignment' form."""
    seen = set()
    # we reverse as more likely to see high id and shortcut
    for con in reversed(path):
        for i in con:
            if (nterms is not None) and (i >= nterms):
                # indices beyond nterms -> ssa
                return True
            seen.add(i)
            if i in seen:
                # id reused -> not ssa
                return False


def optimize_simplify(inputs, output, size_dict, use_ssa=False):
    """Find the (likely only partial) contraction path corresponding to
    simplifications only. Those simplifiactions are:

    - ignore any indices that appear in all terms
    - combine any repeated indices within a single term
    - reduce any non-output indices that only appear on a single term
    - combine any scalar terms
    - combine any tensors with matching indices (hadamard products)

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.
    use_ssa : bool, optional
        Whether to return the contraction path in 'SSA' format (i.e. as if each
        intermediate is appended to the list of inputs, without removals).

    Returns
    -------
    path : list[list[int]]
        The contraction path, given as a sequence of pairs of node indices.
    """
    cp = ContractionProcessor(inputs, output, size_dict)
    cp.simplify()
    if use_ssa:
        return cp.ssa_path
    return ssa_to_linear(cp.ssa_path, len(inputs))


def optimize_greedy(
    inputs,
    output,
    size_dict,
    search_params={},
    costmod=1.0,
    temperature=0.0,
    simplify=True,
    use_ssa=False,
):
    """Find a contraction path using a greedy algorithm.

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.
    costmod : float, optional
        When assessing local greedy scores how much to weight the size of the
        tensors removed compared to the size of the tensor added::

            score = size_ab / costmod - (size_a + size_b) * costmod

        This can be a useful hyper-parameter to tune.
    temperature : float, optional
        When asessing local greedy scores, how much to randomly perturb the
        score. This is implemented as::

            score -> sign(score) * log(|score|) - temperature * gumbel()

        which implements boltzmann sampling.
    simplify : bool, optional
        Whether to perform simplifications before optimizing. These are:

            - ignore any indices that appear in all terms
            - combine any repeated indices within a single term
            - reduce any non-output indices that only appear on a single term
            - combine any scalar terms
            - combine any tensors with matching indices (hadamard products)

        Such simpifications may be required in the general case for the proper
        functioning of the core optimization, but may be skipped if the input
        indices are already in a simplified form.
    use_ssa : bool, optional
        Whether to return the contraction path in 'single static assignment'
        (SSA) format (i.e. as if each intermediate is appended to the list of
        inputs, without removals). This can be quicker and easier to work with
        than the 'linear recycled' format that `numpy` and `opt_einsum` use.

    Returns
    -------
    path : list[list[int]]
        The contraction path, given as a sequence of pairs of node indices.
    """
    cp = ContractionProcessor(inputs, output, size_dict, search_params.get("contraction_info", None))
    if simplify:
        cp.simplify()
    cp.optimize_greedy(minimize=search_params.get("greedy_minimizer", "combo"), costmod=costmod, temperature=temperature)
    # handle disconnected subgraphs
    cp.optimize_remaining_by_size()
    if use_ssa:
        return cp.ssa_path
    return ssa_to_linear(cp.ssa_path, len(inputs))


def optimize_random_greedy_track_flops(
    inputs,
    output,
    size_dict,
    ntrials=1,
    costmod=(0.1, 4.0),
    temperature=(0.001, 1.0),
    seed=None,
    simplify=True,
    use_ssa=False,
):
    """Perform a batch of random greedy optimizations, simulteneously tracking
    the best contraction path in terms of flops, so as to avoid constructing a
    separate contraction tree.

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.
    ntrials : int, optional
        The number of random greedy trials to perform. The default is 1.
    costmod : (float, float), optional
        When assessing local greedy scores how much to weight the size of the
        tensors removed compared to the size of the tensor added::

            score = size_ab / costmod - (size_a + size_b) * costmod

        It is sampled uniformly from the given range.
    temperature : (float, float), optional
        When asessing local greedy scores, how much to randomly perturb the
        score. This is implemented as::

            score -> sign(score) * log(|score|) - temperature * gumbel()

        which implements boltzmann sampling. It is sampled log-uniformly from
        the given range.
    seed : int, optional
        The seed for the random number generator.
    simplify : bool, optional
        Whether to perform simplifications before optimizing. These are:

            - ignore any indices that appear in all terms
            - combine any repeated indices within a single term
            - reduce any non-output indices that only appear on a single term
            - combine any scalar terms
            - combine any tensors with matching indices (hadamard products)

        Such simpifications may be required in the general case for the proper
        functioning of the core optimization, but may be skipped if the input
        indices are already in a simplified form.
    use_ssa : bool, optional
        Whether to return the contraction path in 'single static assignment'
        (SSA) format (i.e. as if each intermediate is appended to the list of
        inputs, without removals). This can be quicker and easier to work with
        than the 'linear recycled' format that `numpy` and `opt_einsum` use.

    Returns
    -------
    path : list[list[int]]
        The best contraction path, given as a sequence of pairs of node
        indices.
    flops : float
        The flops (/ contraction cost / number of multiplications), of the best
        contraction path, given log10.
    """
    rng = get_rng(seed)
    best_path = None
    best_flops = float("inf")

    # create initial processor and simplify only once
    cp0 = ContractionProcessor(inputs, output, size_dict, track_flops=True)
    if simplify:
        cp0.simplify()

    if isinstance(costmod, float):
        # constant

        def _next_costmod():
            return costmod

    else:
        # uniformly sample

        def _next_costmod():
            return rng.uniform(*costmod)

    if isinstance(temperature, float):
        # constant

        def _next_temperature():
            return temperature

    else:
        # log-uniformly sample
        logtempmin, logtempmax = map(math.log, temperature)

        def _next_temperature():
            return math.exp(rng.uniform(logtempmin, logtempmax))

    for _ in range(ntrials):
        cp = cp0.copy()
        success = cp.optimize_greedy(
            costmod=_next_costmod(),
            temperature=_next_temperature(),
            seed=rng,
        )

        if not success:
            # optimization hit the flops limit
            continue

        # handle disconnected subgraphs
        cp.optimize_remaining_by_size()

        if cp.flops < best_flops:
            best_path = cp.ssa_path
            best_flops = cp.flops
            # enable even earlier stopping
            cp0.flops_limit = best_flops

    # for consistency with cotengrust / easier comparison
    best_flops = math.log10(best_flops)

    if not use_ssa:
        best_path = ssa_to_linear(best_path, len(inputs))

    return best_path, best_flops


def optimize_optimal(
    inputs,
    output,
    size_dict,
    search_params={},
    minimize="flops",
    cost_cap=2,
    search_outer=False,
    simplify=True,
    use_ssa=False,
):
    """Find the optimal contraction path using a dynamic programming
    algorithm (by default excluding outer products).

    The algorithm is an optimized version of Phys. Rev. E 90, 033315 (2014)
    (preprint: https://arxiv.org/abs/1304.6112), adapted from the
    ``opt_einsum`` implementation.

    Parameters
    ----------
    inputs : tuple[tuple[str]]
        The indices of each input tensor.
    output : tuple[str]
        The indices of the output tensor.
    size_dict : dict[str, int]
        A dictionary mapping indices to their dimension.
    minimize : str, optional
        The cost function to minimize. The options are:

        - "flops": minimize with respect to total operation count only
          (also known as contraction cost)
        - "size": minimize with respect to maximum intermediate size only
          (also known as contraction width)
        - 'max': minimize the single most expensive contraction, i.e. the
          asymptotic (in index size) scaling of the contraction
        - 'write' : minimize the sum of all tensor sizes, i.e. memory written
        - 'combo' or 'combo={factor}` : minimize the sum of
          FLOPS + factor * WRITE, with a default factor of 64.
        - 'limit' or 'limit={factor}` : minimize the sum of
          MAX(FLOPS, alpha * WRITE) for each individual contraction, with a
          default factor of 64.

        'combo' is generally a good default in term of practical hardware
        performance, where both memory bandwidth and compute are limited.
    cost_cap : float, optional
        The maximum cost of a contraction to initially consider. This acts like
        a sieve and is doubled at each iteration until the optimal path can
        be found, but supplying an accurate guess can speed up the algorithm.
    search_outer : bool, optional
        Whether to allow outer products in the contraction path. The default is
        False. Especially when considering write costs, the fastest path is
        very unlikely to include outer products.
    simplify : bool, optional
        Whether to perform simplifications before optimizing. These are:

        - ignore any indices that appear in all terms
        - combine any repeated indices within a single term
        - reduce any non-output indices that only appear on a single term
        - combine any scalar terms
        - combine any tensors with matching indices (hadamard products)

        Such simpifications may be required in the general case for the proper
        functioning of the core optimization, but may be skipped if the input
        indices are already in a simplified form.
    use_ssa : bool, optional
        Whether to return the contraction path in 'single static assignment'
        (SSA) format (i.e. as if each intermediate is appended to the list of
        inputs, without removals). This can be quicker and easier to work with
        than the 'linear recycled' format that `numpy` and `opt_einsum` use.

    Returns
    -------
    path : list[list[int]]
        The contraction path, given as a sequence of pairs of node indices.
    """  
    print("optimizing optimal with minimize: ", minimize , "and searchparams: ", search_params)  
    cp = ContractionProcessor(inputs, output, size_dict, search_params.get("contraction_info", None))
    if simplify:
        cp.simplify()
    cp.optimize_optimal(
        minimize=search_params.get("optimal_minimizer", "flops"), cost_cap=cost_cap, search_outer=search_outer
    )
    # handle disconnected subgraphs
    cp.optimize_remaining_by_size()
    if use_ssa:
        return cp.ssa_path
    return ssa_to_linear(cp.ssa_path, len(inputs))


class EnsureInputsOutputAreSequence:
    def __init__(self, f):
        self.f = f

    def __call__(self, inputs, output, *args, **kwargs):
        if not isinstance(inputs[0], (tuple, list)):
            inputs = tuple(map(tuple, inputs))
        if not isinstance(output, (tuple, list)):
            output = tuple(output)
        return self.f(inputs, output, *args, **kwargs)


@functools.lru_cache()
def get_optimize_greedy(accel="auto"):
    if accel == "auto":
        import importlib.util

        accel = importlib.util.find_spec("cotengrust") is not None

    if accel is True:
        from cotengrust import optimize_greedy as f

        return EnsureInputsOutputAreSequence(f)

    if accel is False:
        return optimize_greedy

    raise ValueError(f"Unrecognized value for `accel`: {accel}.")


@functools.lru_cache()
def get_optimize_random_greedy_track_flops(accel="auto"):
    if accel == "auto":
        import importlib.util

        accel = importlib.util.find_spec("cotengrust") is not None

    if accel is True:
        from cotengrust import optimize_random_greedy_track_flops as f

        return EnsureInputsOutputAreSequence(f)

    if accel is False:
        return optimize_random_greedy_track_flops

    raise ValueError(f"Unrecognized value for `accel`: {accel}.")


class GreedyOptimizer(PathOptimizer):
    """Class interface to the greedy optimizer which can be instantiated with
    default options.
    """

    __slots__ = (
        "costmod",
        "temperature",
        "simplify",
        "_optimize_fn",
    )

    def __init__(
        self,
        costmod=1.0,
        temperature=0.0,
        simplify=True,
        accel="auto",
    ):
        self.costmod = costmod
        self.temperature = temperature
        self.simplify = simplify
        self._optimize_fn = get_optimize_greedy(accel)

    def maybe_update_defaults(self, **kwargs):
        # allow overriding of defaults
        opts = {
            "costmod": self.costmod,
            "temperature": self.temperature,
            "simplify": self.simplify,
        }
        opts.update(kwargs)
        return opts

    def ssa_path(self, inputs, output, size_dict, contraction_info, **kwargs):
        return self._optimize_fn(
            inputs,
            output,
            size_dict,
            contraction_info,
            use_ssa=True,
            **self.maybe_update_defaults(**kwargs),
        )

    def search(self, inputs, output, size_dict, contraction_info=None, **kwargs):
        from ..core import ContractionTree

        ssa_path = self.ssa_path(inputs, output, size_dict, contraction_info, **kwargs)
        return ContractionTree.from_path(
            inputs, output, size_dict, ssa_path=ssa_path
        )

    def __call__(self, inputs, output, size_dict, contraction_info=None, **kwargs):
        return self._optimize_fn(
            inputs,
            output,
            size_dict,
            contraction_info,
            use_ssa=False,
            **self.maybe_update_defaults(**kwargs),
        )


class RandomGreedyOptimizer(PathOptimizer):
    """Lightweight random greedy optimizer, that eschews hyper parameter
    tuning and contraction tree construction. This is a stateful optimizer
    that should not be re-used on different contractions.

    Parameters
    ----------
    max_repeats : int, optional
        The number of random greedy trials to perform.
    costmod : (float, float), optional
        When assessing local greedy scores how much to weight the size of the
        tensors removed compared to the size of the tensor added::

            score = size_ab / costmod - (size_a + size_b) * costmod

        It is sampled uniformly from the given range.
    temperature : (float, float), optional
        When asessing local greedy scores, how much to randomly perturb the
        score. This is implemented as::

            score -> sign(score) * log(|score|) - temperature * gumbel()

        which implements boltzmann sampling. It is sampled log-uniformly from
        the given range.
    seed : int, optional
        The seed for the random number generator. Note that deterministic
        behavior is only guaranteed within the python or rust backend
        (the `accel` parameter) and parallel settings.
    simplify : bool, optional
        Whether to perform simplifications before optimizing. These are:

            - ignore any indices that appear in all terms
            - combine any repeated indices within a single term
            - reduce any non-output indices that only appear on a single term
            - combine any scalar terms
            - combine any tensors with matching indices (hadamard products)

        Such simpifications may be required in the general case for the proper
        functioning of the core optimization, but may be skipped if the input
        indices are already in a simplified form.
    accel : bool or str, optional
        Whether to use the accelerated `cotengrust` backend. If "auto" the
        backend is used if available.
    parallel : bool or str, optional
        Whether to use parallel processing. If "auto" the default is to use
        threads if the accelerated backend is not used, and processes if it is.

    Attributes
    ----------
    best_ssa_path : list[list[int]]
        The best contraction path found so far.
    best_flops : float
        The flops (/ contraction cost / number of multiplications) of the best
        contraction path found so far.
    """

    minimize = "flops"

    def __init__(
        self,
        max_repeats=32,
        costmod=(0.1, 4.0),
        temperature=(0.001, 1.0),
        seed=None,
        simplify=True,
        accel="auto",
        parallel="auto",
    ):
        self.max_repeats = max_repeats

        # for cotengrust, ensure these are always ranges
        if isinstance(costmod, float):
            self.costmod = (costmod, costmod)
        else:
            self.costmod = tuple(costmod)

        if isinstance(temperature, float):
            self.temperature = (temperature, temperature)
        else:
            self.temperature = tuple(temperature)

        self.simplify = simplify
        self.rng = get_rng(seed)
        self.best_ssa_path = None
        self.best_flops = float("inf")
        self.tree = None
        self._optimize_fn = get_optimize_random_greedy_track_flops(accel)

        if (parallel == "auto") and (
            self._optimize_fn is not optimize_random_greedy_track_flops
        ):
            # using accelerated fn, so default to threads
            parallel = "threads"
        self._pool = parse_parallel_arg(parallel)
        if self._pool is not None:
            self._nworkers = get_n_workers(self._pool)
        else:
            self._nworkers = 1

    def maybe_update_defaults(self, **kwargs):
        # allow overriding of defaults
        opts = {
            "costmod": self.costmod,
            "temperature": self.temperature,
            "simplify": self.simplify,
        }
        opts.update(kwargs)
        return opts

    def ssa_path(self, inputs, output, size_dict, **kwargs):
        if self._pool is None:
            ssa_path, flops = self._optimize_fn(
                inputs,
                output,
                size_dict,
                use_ssa=True,
                ntrials=self.max_repeats,
                seed=self.rng.randint(0, 2**32 - 1),
                **self.maybe_update_defaults(**kwargs),
            )
        else:
            # XXX: just use small batchsize if can't find num_workers?
            nbatches = self._nworkers
            batchsize = self.max_repeats // nbatches
            batchremainder = self.max_repeats % nbatches
            each_ntrials = [
                batchsize + (i < batchremainder) for i in range(nbatches)
            ]

            fs = [
                self._pool.submit(
                    self._optimize_fn,
                    inputs,
                    output,
                    size_dict,
                    use_ssa=True,
                    ntrials=ntrials,
                    seed=self.rng.randint(0, 2**32 - 1),
                    **self.maybe_update_defaults(**kwargs),
                )
                for ntrials in each_ntrials
                if (ntrials > 0)
            ]
            ssa_path, flops = min((f.result() for f in fs), key=lambda x: x[1])

        if flops < self.best_flops:
            self.best_ssa_path = ssa_path
            self.best_flops = flops

        return self.best_ssa_path

    def search(self, inputs, output, size_dict, **kwargs):
        from ..core import ContractionTree

        ssa_path = self.ssa_path(
            inputs,
            output,
            size_dict,
            **self.maybe_update_defaults(**kwargs),
        )
        self.tree = ContractionTree.from_path(
            inputs,
            output,
            size_dict,
            ssa_path=ssa_path,
        )
        return self.tree

    def __call__(self, inputs, output, size_dict, **kwargs):
        ssa_path = self.ssa_path(
            inputs,
            output,
            size_dict,
            **self.maybe_update_defaults(**kwargs),
        )
        return ssa_to_linear(ssa_path)


class ReusableRandomGreedyOptimizer(ReusableOptimizer):
    def _get_path_relevant_opts(self):
        """Get a frozenset of the options that are most likely to affect the
        path. These are the options that we use when the directory name is not
        manually specified.
        """
        return [
            ("max_repeats", 32),
            ("costmod", (0.1, 4.0)),
            ("temperature", (0.001, 1.0)),
            ("simplify", True),
        ]

    def _get_suboptimizer(self):
        return RandomGreedyOptimizer(**self._suboptimizer_kwargs)

    def _deconstruct_tree(self, opt, tree):
        return {
            "path": tree.get_path(),
            "score": opt.best_flops,
            # store this for cache compatibility
            "sliced_inds": (),
        }

    def _reconstruct_tree(self, inputs, output, size_dict, con):
        tree = ContractionTree.from_path(
            inputs,
            output,
            size_dict,
            path=con["path"],
            objective=self.minimize,
        )

        return tree


@functools.lru_cache()
def get_optimize_optimal(accel="auto"):
    if accel == "auto":
        import importlib.util

        accel = importlib.util.find_spec("cotengrust") is not None

    if accel is True:
        from cotengrust import optimize_optimal as f

        return EnsureInputsOutputAreSequence(f)

    if accel is False:
        return optimize_optimal

    raise ValueError(f"Unrecognized value for `accel`: {accel}.")


class OptimalOptimizer(PathOptimizer):
    """Class interface to the optimal optimizer which can be instantiated with
    default options.
    """

    __slots__ = (
        "minimize",
        "cost_cap",
        "search_outer",
        "simplify",
        "_optimize_fn",
    )

    def __init__(
        self,
        contraction_info=None,
        minimize="flops",
        cost_cap=2,
        search_outer=False,
        simplify=True,
        accel="auto",
    ):
        self.contraction_info = contraction_info
        self.minimize = minimize
        self.cost_cap = cost_cap
        self.search_outer = search_outer
        self.simplify = simplify
        self._optimize_fn = get_optimize_optimal(accel)

    def maybe_update_defaults(self, **kwargs):
        # allow overriding of defaults
        opts = {
            "minimize": self.minimize,
            "cost_cap": self.cost_cap,
            "search_outer": self.search_outer,
            "simplify": self.simplify,
        }
        opts.update(kwargs)
        return opts

    def ssa_path(self, inputs, output, size_dict, search_params, **kwargs):
        if search_params.get("contraction_info") is None:
            search_params["contraction_info"] = self.contraction_info
        return self._optimize_fn(
            inputs,
            output,
            size_dict,
            search_params,
            use_ssa=True,
            **self.maybe_update_defaults(**kwargs),
        )

    def search(self, inputs, output, size_dict, search_params={}, **kwargs):
        from ..core import ContractionTree

        ssa_path = self.ssa_path(inputs, output, size_dict, search_params, **kwargs)
        return ContractionTree.from_path(
            inputs, output, size_dict, ssa_path=ssa_path
        )

    def __call__(self, inputs, output, size_dict, search_params={}, **kwargs):
        if search_params.get("contraction_info") is None:
            search_params["contraction_info"] = self.contraction_info
        return self._optimize_fn(
            inputs,
            output,
            size_dict,
            search_params,
            use_ssa=False,
            **self.maybe_update_defaults(**kwargs),
        )
