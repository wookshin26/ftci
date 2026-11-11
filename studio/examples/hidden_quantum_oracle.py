"""No full circuit is needed when independently controlled angles are available."""
import numpy as np
from ftci import fit
from ftci.quantum import from_generators

# Replace this callable with your device or external simulator connection.
def query_device(angles):
    return np.cos(angles[0] + angles[1])

problem = from_generators(
    generators=[[-0.5, 0.5], [-0.5, 0.5]], spectra=True,
    preprocessing=["x0", "2*x0**2"], auxiliary_oracle=query_device,
    domain=[[-2, 2]], name="Hidden circuit oracle",
)
model = fit(problem, query_budget=1000)
print(model([0.4]), np.cos(0.4 + 2*0.4**2))
