import random
import numpy as np
from copy import deepcopy

# Function set
# Cada funcion: (callable, aridad, simbolo a imprimir)
FUNCTIONS = {
    "add": (np.add, 2, "+"),
    "sub": (np.subtract, 2, "-"),
    "mul": (np.multiply, 2, "*"),
    "div": (lambda x, y: np.divide(x, y, out = np.ones_like(x, dtype = float), where = np.abs(y) > 1e-6), 2, "/"),
}

VARIABLES = ["x", "y", "z"]
# Ephemeral Random Constant
ERC_RANGE = (-1.0, 1.0)

class Node:
    def __init__(self, value, children = None):
        # value: nombre de funcion (str), nombre de variable (str), o constante (float)
        self.value = value
        self.children = children if children is not None else []

    def is_leaf(self):
        return len(self.children) == 0

    def evaluate(self, X):
        # X: dict {"x": np.array, "y": np.array}
        if self.is_leaf():
            if isinstance(self.value, str):
                return X[self.value]
            return np.full_like(next(iter(X.values())), self.value, dtype = float)
        func, _, _ = FUNCTIONS[self.value]
        args = [child.evaluate(X) for child in self.children]
        return func(*args)

    def size(self):
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def copy(self):
        return deepcopy(self)

    def __str__(self):
        if self.is_leaf():
            return str(self.value) if isinstance(self.value, str) else f"{self.value:.3}"
        if len(self.children) == 2:
            _, _, sym = FUNCTIONS[self.value]
            return f"({self.children[0]} {sym} {self.children[1]})"
        return f"{self.value}({', '.join(str(c) for c in self.children)})"

def random_terminal():
    # ERC con probabilidad 0.3, variable con 0.7 (ajustable)
    if random.random() < 0.3:
        return Node(random.uniform(*ERC_RANGE))
    return Node(random.choice(VARIABLES))

def random_function():
    return random.choice(list(FUNCTIONS.keys()))

def generate_full(max_depth, current_depth = 0):
    # Todos los nodos a max_depth son terminales, el resto son funciones
    if current_depth >= max_depth:
        return random_terminal()
    func_name = random_function()
    arity = FUNCTIONS[func_name][1]
    children = [generate_full(max_depth, current_depth + 1) for _ in range(arity)]
    return Node(func_name, children)

def generate_grow(max_depth, current_depth = 0, p_terminal = 0.3):
    # Cada nodo es terminal con prob p_terminal, hasta max_depth donde es forzado
    if current_depth >= max_depth or (current_depth > 0 and random.random() < p_terminal):
        return random_terminal()
    func_name = random_function()
    arity = FUNCTIONS[func_name][1]
    children = [generate_grow(max_depth, current_depth + 1, p_terminal) for _ in range(arity)]
    return Node(func_name, children)

def ramped_half_and_half(pop_size, min_depth = 2, max_depth = 6):
    # Mitad full, mitad grow, distribuidos uniformemente por profunidad
    population = []
    depths = range(min_depth, max_depth + 1)
    per_depth = pop_size // (len(depths) * 2)
    for d in depths:
        for _ in range(per_depth):
            population.append(generate_full(d))
            population.append(generate_grow(d))
    # rellenar si la division no fue exacta
    while len(population) < pop_size:
        d = random.choice(depths)
        method = random.choice([generate_full, generate_grow])
        population.append(method(d))
    return population

def all_nodes(tree):
    # Devuelve lista de (nodo, parent, index_en_parent). Root tiene parent = None
    result = [(tree, None, None)]
    def _walk(node, parent, idx):
        for i, child in enumerate(node.children):
            result.append((child, node, i))
            _walk(child, node, i)
    _walk(tree,None,None)
    return result

def replace_subtree(tree, target_parent, target_idx, new_subtree):
    # Reemplaza in-place. Si target_parent es None, reemplaza raiz (devuelve new_subtree)
    if target_parent is None:
        return new_subtree
    target_parent.children[target_idx] = new_subtree
    return tree
