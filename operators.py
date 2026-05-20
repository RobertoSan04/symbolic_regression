import random
import numpy as np
from tree import Node, FUNCTIONS, TERMINALS, ERC_RANGE, generate_grow

# Tournament Selection
def tournament_selection(population: list, fitnesses: list, k: int = 3) -> "Node":
    # Return a copy of the winner of a k-way tournament.
    contestants = random.sample(range(len(population)), k)
    winner_idx = min(contestants, key=lambda i: fitnesses[i])
    return population[winner_idx].copy()

# Subtree Crossover
def subtree_crossover(parent1: "Node", parent2: "Node") -> tuple:
    # Standard one-point subtree crossover.
    off1 = parent1.copy()
    off2 = parent2.copy()

    nodes1 = _get_crossover_point_pool(off1)
    nodes2 = _get_crossover_point_pool(off2)

    # Pick crossover points
    node1, par1, idx1 = random.choice(nodes1)
    node2, par2, idx2 = random.choice(nodes2)

    # Swap subtrees in-place
    if par1 is None and par2 is None:
        # Both roots — just return swapped copies (edge case)
        return off2, off1

    if par1 is None:
        # node1 is root of off1 → offspring1 becomes the subtree from off2
        off1 = node2.copy()
        par2.children[idx2] = node1
    elif par2 is None:
        # node2 is root of off2
        off2 = node1.copy()
        par1.children[idx1] = node2
    else:
        # Normal case: swap subtrees
        par1.children[idx1] = node2
        par2.children[idx2] = node1

    return off1, off2


def _get_crossover_point_pool(tree: "Node") -> list:
    # Return a biased pool of (node, parent, child_index) tuples.
    all_n = tree.all_nodes()
    # Exclude root (parent is None) to avoid whole-tree swap
    non_root = [(n, p, i) for n, p, i in all_n if p is not None]
    if not non_root:
        # Single-node tree; must include root
        return all_n

    internal = [(n, p, i) for n, p, i in non_root if n.children]
    leaves   = [(n, p, i) for n, p, i in non_root if not n.children]

    if not internal:
        return leaves
    if not leaves:
        return internal

    # Build weighted pool: repeat internals 9x, leaves 1x
    return internal * 9 + leaves

# Point Mutation
def point_mutation(tree: "Node", max_depth: int = None) -> "Node":
    # Point (node replacement) mutation.
    mutant = tree.copy()
    all_n = mutant.all_nodes()

    for node, parent, idx in all_n:
        if not _coin(0.05):          # ~5% per-node mutation probability
            continue

        if node.children:
            # Function node: replace with same-arity function
            arity = len(node.children)
            candidates = [f for f, meta in FUNCTIONS.items()
                          if meta['arity'] == arity and f != node.value]
            if candidates:
                node.value = random.choice(candidates)
        else:
            # Terminal: replace with another terminal
            node.value = _random_terminal()

    return mutant

# Subtree Mutation  (headless chicken crossover / Koza mutation)
def subtree_mutation(tree: "Node", max_depth: int = 4) -> "Node":
    # Replace a random subtree with a freshly generated random tree.
    mutant = tree.copy()
    all_n = mutant.all_nodes()
    non_root = [(n, p, i) for n, p, i in all_n if p is not None]

    if not non_root:
        # Single-node tree: replace entirely
        return generate_grow(max_depth)

    node, parent, idx = random.choice(non_root)
    new_subtree = generate_grow(max_depth)
    parent.children[idx] = new_subtree

    return mutant

def _coin(p: float = 0.5) -> bool:
    return random.random() < p


def _random_terminal() -> object:
    # Return a random terminal value: variable name or ERC float.
    variables = [t for t in TERMINALS if isinstance(t, str)]
    # 50/50 split between variable and ERC (same as typical GP)
    if variables and _coin(0.5):
        return random.choice(variables)
    else:
        low, high = ERC_RANGE
        return round(random.uniform(low, high), 4)