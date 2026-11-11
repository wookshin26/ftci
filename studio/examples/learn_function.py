"""Learn, export, and reload a continuous function without its original oracle."""
from pathlib import Path
import numpy as np
from ftci import Basis, FunctionProblem, Model, fit

problem = FunctionProblem(
    target=lambda x: np.cos(x[0] + x[1]) + 0.2,
    basis=[Basis(["1", "cos(x0)", "sin(x0)"], name="x0 Fourier"),
           Basis(["1", "cos(x1)", "sin(x1)"], name="x1 Fourier")],
    domain=[(-np.pi, np.pi), (-np.pi, np.pi)],
    name="Cosine addition",
)
model = fit(problem, max_bond=6, validation_samples=100, query_budget=2000)
model.save(Path("cosine.ftci"))
restored = Model.load("cosine.ftci")
print("Prediction:", restored([0.2, 0.4]))
print("First basis:", restored.basis[0]([0.2, 0.4]))
print("Core shapes:", [core.shape for core in restored.cores])
print("Coefficient C[1,1]:", restored.coefficient([1, 1]))
print("Validation:", restored.metadata["validation"])
