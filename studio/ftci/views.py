"""The same learned function read in different local bases.

A coefficient tensor belongs to a choice of dictionaries. Changing the basis
inside each dictionary's function space changes the coefficients but neither
the function nor the bond dimensions, so these views need no new queries.

``dictionary``
    The dictionaries the model was learned and is exported with.
``orthonormal``
    Basis functions orthonormal under the uniform measure on the coordinates
    the sites are independent in: the inputs of a general function, or the
    training angles of a quantum circuit. Squared coefficients are then
    shares of the function's mean square.
``nodal``
    Coefficients are the function values on the sampling grid, the tensor
    that cross interpolation approximated.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

VIEWS = ("dictionary", "orthonormal", "nodal")
_DESCRIPTIONS = {
    "dictionary": "Coefficients of the dictionaries the model was learned with.",
    "nodal": ("Coefficients are function values on the sampling grid, the tensor that cross "
              "interpolation approximated. The matching basis functions interpolate between grid points."),
}


def _number(value):
    return f"{value:.6g}"


def _quantum_orthonormal(model):
    """Rescale trigonometric dictionaries to unit mean square in the training angles."""
    cores, dictionaries = [], []
    for core, site in zip(model.cores, model.metadata["sites"]):
        scale, parts = np.ones(1), []
        for group in site.get("classes") or [site]:
            frequencies, text = group["frequencies"], group.get("preprocessing")
            count = len(frequencies) + 1
            scale = np.kron(scale, np.array([1 / math.sqrt(count)]
                                            + [1 / math.sqrt(2 * count)] * (2 * len(frequencies))))
            if text is None:
                parts.append(None)
                continue
            expressions = ["1"]
            for frequency in frequencies:
                argument = f"({frequency!r})*({text})"
                expressions.extend((f"sqrt(2)*cos({argument})", f"sqrt(2)*sin({argument})"))
            parts.append(expressions)
        if len(scale) != core.shape[1]:
            raise ValueError("The stored site description does not match the coefficient core.")
        if any(part is None for part in parts):
            expressions = None
        elif len(parts) == 1:
            expressions = parts[0]
        else:
            expressions = ["*".join(f"({factor})" for factor in factors)
                           for factors in itertools.product(*parts)]
        cores.append(core * scale[None, :, None])
        dictionaries.append({"expressions": expressions,
                             "labels": [f"ψ{j}" for j in range(core.shape[1])]})
    return cores, dictionaries, ("Basis functions have unit mean square in the training angles, where "
                                 "the dictionaries are independent. Squared coefficients are shares of "
                                 "the mean square of the circuit output over all angle settings.")


def _gram(basis, domain, seed=0):
    """Mean of Φ Φᵀ under the uniform measure on the dictionary's input block."""
    indices = list(basis.input_indices)
    bounds = np.asarray(domain, dtype=float)
    midpoint = bounds.mean(axis=1)
    if not indices:
        points, weights = [()], np.ones(1)
    elif len(indices) <= 3:
        order = {1: 200, 2: 40, 3: 16}[len(indices)]
        nodes, node_weights = np.polynomial.legendre.leggauss(order)
        axes = [0.5 * (hi - lo) * nodes + 0.5 * (hi + lo) for lo, hi in bounds[indices]]
        points = list(itertools.product(*axes))
        weights = np.array([math.prod(w) for w in itertools.product(*[node_weights / 2] * len(indices))])
    else:
        rng = np.random.default_rng(seed)
        points = rng.uniform(bounds[indices, 0], bounds[indices, 1], size=(20000, len(indices)))
        weights = np.full(len(points), 1 / len(points))
    values = np.empty((len(points), basis.dimension))
    for row, point in enumerate(points):
        x = midpoint.copy()
        x[indices] = point
        values[row] = basis(x)
    return (values * weights[:, None]).T @ values


def _function_orthonormal(model):
    """Orthonormalize each dictionary in its given order on the input domain."""
    cores, dictionaries = [], []
    for position, (core, basis) in enumerate(zip(model.cores, model.basis)):
        gram = _gram(basis, model.domain)
        try:
            lower = np.linalg.cholesky(gram)
            inverse = np.linalg.inv(lower)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                f"Dictionary {position + 1} is not linearly independent on the input domain, "
                "so it has no orthonormal form.") from error
        expressions = None
        if basis.portable:
            expressions = []
            for row in inverse:
                terms = [(value, text) for value, text in zip(row, basis.expressions) if abs(value) > 1e-12]
                parts = []
                for value, text in terms:
                    sign = "-" if value < 0 else "+"
                    parts.append(f"{sign if parts or value < 0 else ''}{' ' if parts else ''}"
                                 f"{_number(abs(value))}*({text})")
                expressions.append(" ".join(parts) or "0")
        cores.append(np.einsum("aib,ij->ajb", core, lower))
        dictionaries.append({"expressions": expressions,
                             "labels": [f"ψ{j}" for j in range(core.shape[1])]})
    return cores, dictionaries, ("Each dictionary is orthonormalized in its given order under the uniform "
                                 "measure on the input domain. Squared coefficients are shares of the "
                                 "function's mean square, and the coefficient error equals the L² error.")


def _nodal(model):
    sample_cores = getattr(model, "sample_cores", None)
    nodes = getattr(model, "sample_nodes", None)
    if sample_cores is None or nodes is None:
        raise ValueError("The sampling grid is kept only for models learned in this session.")
    quantum = model.metadata.get("mode") == "quantum"
    dictionaries = []
    for inputs, local in zip(model.sample_inputs, nodes):
        names = [f"a{index + 1}" if quantum else f"x{index}" for index in inputs]
        labels = [", ".join(f"{name} = {_number(value)}" for name, value in zip(names, row)) or "constant"
                  for row in np.asarray(local)]
        dictionaries.append({"expressions": None, "labels": labels})
    return [np.asarray(core, dtype=float) for core in sample_cores], dictionaries, _DESCRIPTIONS["nodal"]


def basis_view(model, view="dictionary") -> dict:
    """Coefficient cores and basis-function labels of ``model`` in one view.

    Every view represents the same function. ``expressions`` is ``None``
    where a basis function has no portable formula; ``labels`` always names
    each entry.
    """
    if view not in VIEWS:
        raise ValueError("Choose the dictionary, orthonormal or nodal view.")
    if view == "dictionary":
        cores = [np.asarray(core, dtype=float) for core in model.cores]
        dictionaries = [{"expressions": list(basis.expressions) if basis.portable else None,
                         "labels": [f"Φ{k + 1},{j}" for j in range(basis.dimension)]}
                        for k, basis in enumerate(model.basis)]
        description = _DESCRIPTIONS["dictionary"]
    elif view == "nodal":
        cores, dictionaries, description = _nodal(model)
    elif model.metadata.get("mode") == "quantum":
        cores, dictionaries, description = _quantum_orthonormal(model)
    else:
        cores, dictionaries, description = _function_orthonormal(model)
    for dictionary, core in zip(dictionaries, cores):
        dictionary["dimension"] = int(core.shape[1])
    return {"view": view, "cores": cores, "dictionaries": dictionaries, "description": description}


__all__ = ["VIEWS", "basis_view"]
