import numpy as np
from tree import Node

def evaluate_fitness(tree: Node, X: dict, y_true: np.ndarray) -> float:
    """
    Evalua el arbol y devuelve su RSME contra el y_true

    Parametros:
    - tree: Nodo raiz del arbol GP
    - X: dict {"x": np.array, ...} - inputs
    - y_true: np.array - valores objetivos

    Retorna:
    - float: RMSE (menor - mejor). Inf si el arbol produce salida invalida
    """

    try:
        y_pred = tree.evaluate(X)

        # Detectar output invalido
        if not np.all(np.isfinite(y_pred)):
            return np.inf

        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        return float(rmse)

    except Exception:
        # Cualquier error en evaluacion
        return np.inf


def evaluate_population(population: list, x: dict, y_true: np.ndarray) -> list:
    # Evalua todos los arboles de la poblacion
    return [evaluate_fitness(tree,X, y_true) for tree in population]

def best_fitness(fitness: list) -> float:
    return min(fitness)

def best_individual(population:list, fitnesses: list) -> Node:
    idx = fitnesses.index(min(fitnesses))
    return population[idx]

def fitness_stats(fitnesses: list) -> dict:
    # Resumen de stats

    finite = [f for f in fitnesses if np.isfinite(f)]
    return {
        "best": min(fitnesses),
        "mean": np.mean(finite) if finite else np.inf,
        "std": np.std(finite) if finite else np.inf,
        "valid": len(finite),
        "total": len(fitnesses),
    }
