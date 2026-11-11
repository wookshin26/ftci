"""Structure of a learned coefficient tensor.

Two quantities decide whether a coefficient tensor is cheap to represent.
Entanglement across each cut of the chain bounds the bond dimension an MPS
needs. Spreading over the tensor-product basis decides whether a sparse
ansatz could work instead. Spreading bounds entanglement from above, but a
tensor can be spread out and still have low entanglement.

All entropies are in nats and refer to the squared entries of the
normalized tensor. They depend on the chosen dictionaries: a different
basis of the same function space gives a different coefficient tensor.
"""

from __future__ import annotations

import math

import numpy as np


def _entropies(weights):
    weights = weights[weights > 0]
    return (max(0.0, float(-np.sum(weights * np.log(weights)))),
            max(0.0, float(-math.log(np.sum(weights ** 2)))))


def _canonical_sweeps(cores):
    """Return the norm, per-bond Schmidt values and left-orthonormal cores."""
    cores = [np.array(core, dtype=float) for core in cores]
    for k in range(len(cores) - 1, 0, -1):
        left, size, right = cores[k].shape
        q, r = np.linalg.qr(cores[k].reshape(left, size * right).T)
        cores[k] = q.T.reshape(-1, size, right)
        cores[k - 1] = np.einsum("aqb,cb->aqc", cores[k - 1], r)
    norm = float(np.linalg.norm(cores[0]))
    spectra = []
    for k in range(len(cores) - 1):
        left, size, right = cores[k].shape
        u, singular, vt = np.linalg.svd(cores[k].reshape(left * size, right), full_matrices=False)
        spectra.append(singular)
        cores[k] = u.reshape(left, size, -1)
        cores[k + 1] = np.einsum("s,sb,bqc->sqc", singular, vt, cores[k + 1])
    return norm, spectra, cores


def _exact_distribution(cores, constant_index):
    tensor = cores[0][0]
    for core in cores[1:]:
        tensor = np.tensordot(tensor, core, axes=([-1], [0]))
    tensor = tensor[..., 0]
    weights = tensor ** 2 / np.sum(tensor ** 2)
    shannon, renyi2 = _entropies(weights.ravel())
    support = None
    if constant_index is not None:
        size = np.zeros(weights.shape, dtype=int)
        for axis, dimension in enumerate(weights.shape):
            shape = [1] * weights.ndim
            shape[axis] = dimension
            size = size + (np.arange(dimension) != constant_index).reshape(shape)
        support = np.bincount(size.ravel(), weights=weights.ravel(), minlength=weights.ndim + 1)
    return shannon, renyi2, support


def _sampled_distribution(cores, constant_index, samples, rng):
    """Draw index strings with probability proportional to squared coefficients.

    The cores are left-orthonormal with the norm on the last site, so the
    conditional distributions follow from one pass from right to left.
    """
    vectors = np.ones((samples, 1))
    log_probability = np.zeros(samples)
    support = np.zeros(samples, dtype=int)
    for core in reversed(cores):
        branches = np.einsum("aqb,nb->nqa", core, vectors)
        weights = np.sum(branches ** 2, axis=2)
        weights /= weights.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(weights, axis=1)
        chosen = np.minimum((rng.random((samples, 1)) > cumulative).sum(axis=1), core.shape[1] - 1)
        rows = np.arange(samples)
        probability = weights[rows, chosen]
        log_probability += np.log(probability)
        vectors = branches[rows, chosen] / np.sqrt(probability)[:, None]
        if constant_index is not None:
            support += chosen != constant_index
    shannon = float(-log_probability.mean())
    # sum_i p_i^2 is the mean of p_i over samples drawn from p.
    renyi2 = float(-(np.logaddexp.reduce(log_probability) - math.log(samples)))
    weights = None
    if constant_index is not None:
        weights = np.bincount(support, minlength=len(cores) + 1) / samples
    return shannon, renyi2, weights, float(log_probability.std() / math.sqrt(samples))


def coefficient_structure(cores, *, constant_index=None, samples=4000,
                          exact_limit=200_000, seed=0) -> dict:
    """Entanglement per bond and spreading of a coefficient MPS.

    ``constant_index`` is the position of the constant function in every
    dictionary. When given, the result also contains the weight carried by
    basis functions that are non-constant on exactly ``k`` sites.
    Tensors with at most ``exact_limit`` entries are contracted exactly;
    larger ones are sampled.
    """
    if not cores or any(np.ndim(core) != 3 for core in cores):
        raise ValueError("Provide the MPS cores as arrays with axes left bond, basis index, right bond.")
    dimensions = [int(np.shape(core)[1]) for core in cores]
    log_size = float(sum(math.log(d) for d in dimensions))
    norm, spectra, canonical = _canonical_sweeps(cores)
    if not norm > 0 or not math.isfinite(norm):
        return {"zero": True, "norm": 0.0, "dimensions": dimensions, "bonds": [],
                "maximum_entropy": log_size}
    bonds = []
    for k, singular in enumerate(spectra):
        weights = (singular / norm) ** 2
        shannon, renyi2 = _entropies(weights)
        bonds.append({"bond": k + 1, "dimension": int(len(singular)),
                      "entropy": shannon, "renyi2": renyi2,
                      "schmidt_values": [float(v) for v in (singular / norm)[:64]]})
    if log_size <= math.log(exact_limit):
        shannon, renyi2, support = _exact_distribution(canonical, constant_index)
        method, error = "exact", 0.0
    else:
        shannon, renyi2, support, error = _sampled_distribution(
            canonical, constant_index, int(samples), np.random.default_rng(seed))
        method = "sampled"
    return {
        "zero": False, "norm": norm, "dimensions": dimensions, "bonds": bonds,
        "maximum_bond_entropy": max((bond["entropy"] for bond in bonds), default=0.0),
        "spreading": {"shannon": shannon, "renyi2": renyi2, "maximum": log_size,
                      "participation": math.exp(renyi2), "method": method,
                      "samples": None if method == "exact" else int(samples),
                      "shannon_standard_error": error},
        "support_weights": None if support is None else [float(w) for w in support],
        "units": "nats",
    }


__all__ = ["coefficient_structure"]
