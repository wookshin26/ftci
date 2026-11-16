# Learning quantum functions using tensor cross interpolation

Code and data accompanying the manuscript *Learning quantum functions using
tensor cross interpolation* by Seongwook Shin, Ryan Sweke and Jens Eisert.

| Folder | Content |
|---|---|
| [`numerics/`](numerics/) | Code, stored results and figure files behind the numerical results of the manuscript |
| [`studio/`](studio/) | FTCI Studio, a web app that runs function tensor cross interpolation (FTCI) on functions and quantum circuits you define in the browser |

## FTCI Studio

FTCI Studio learns a function as local function dictionaries contracted with
a coefficient matrix product state. You define a function or a quantum
circuit, choose a tensor-network representation and the sampling grid, and
inspect the learned model, its error budget and the structure of its
coefficients. The tensor cross interpolation itself is done by
TensorCrossInterpolation.jl.

### In the browser, without installing anything

GitHub Codespaces runs the Studio on a machine in the cloud that belongs to
your GitHub account.

1. Click the badge below, or on the repository page choose **Code →
   Codespaces → Create codespace on main**. The first start installs Python
   and Julia packages and takes a few minutes.

   [![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/wookshin26/ftci)

2. When the terminal prints `FTCI Studio: http://127.0.0.1:8765`, a browser
   tab with the Studio opens. If it does not, open the **Ports** tab and
   click the address of port 8765.
3. Stop the codespace when you are done (**Code → Codespaces → ⋯ → Stop
   codespace**). Codespaces count against the free monthly quota of the
   account that created them.

The forwarded address is private: it works only for the signed-in account
that created the codespace.

### On your own computer

Requires Python 3.10 or newer and Julia 1.10 or newer.

```sh
cd studio
python3 -m venv .venv
.venv/bin/python -m pip install -e .
julia --project=ftci/julia -e 'using Pkg; Pkg.instantiate()'
.venv/bin/python -m ftci.server
```

Then open <http://127.0.0.1:8765>. [`studio/README.md`](studio/README.md)
describes every feature and the Python interface.

### A five-minute tour

1. Under **01 Define the problem**, switch to **Quantum circuit** and pick
   the example **Heisenberg chain · Trotter dynamics**.
2. Click **Learn function** and look at the **Prediction**, **Tensor network**
   and **Diagnostics** tabs.
3. Under **02 Representation**, switch from **Merged by preprocessing** to
   **Canonical** and click **Learn function** again.
4. Open the **Runs** tab. It lists both runs side by side: the same circuit
   in two tensor-network representations, with chain length, bond dimension,
   number of queries and error.

The quantum examples are circuits from the manuscript: the encoding-first
example of Section II and three circuits of the numerical section.

## Numerical results of the manuscript

[`numerics/README.md`](numerics/README.md) maps every figure to the code that
produced it and to the stored data, and explains how to inspect the results
and how to run the experiments again.

## Citation

See [`CITATION.cff`](CITATION.cff). The experiments rely on
TensorCrossInterpolation.jl, Yao.jl and ITensors.jl. Please cite them as well
if you build on this code.

## License

The code in this repository (FTCI Studio and the notebooks, scripts and
modules in `numerics/`) is released under the Apache License, Version 2.0.
See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The stored results and the
figure files in `numerics/` are not covered by the Apache License.
