"""Exact source expressions for analytically known quantum normalization."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from ftci import Model
from ftci.quantum import _dictionary, from_generators


class QuantumSymbolicBasisTests(unittest.TestCase):
    def test_normalization_uses_exact_fraction_and_simplified_radicals(self):
        # D positive frequencies => a known normalization 1/sqrt(D+1).
        for count, denominator in [(0, None), (1, "sqrt(2)"), (2, "sqrt(3)"),
                                   (3, "2"), (7, "(2*sqrt(2))"), (8, "3"),
                                   (11, "(2*sqrt(3))")]:
            with self.subTest(positive_frequencies=count):
                basis = _dictionary([float(i) for i in range(1, count + 1)], "x0", "test")
                suffix = "" if denominator is None else f"/{denominator}"
                self.assertEqual(basis.expressions[0], "1" + suffix)
                if count:
                    self.assertEqual(basis.expressions[1], "cos((1.0)*(x0))" + suffix)
                    self.assertEqual(basis.expressions[2], "sin((1.0)*(x0))" + suffix)

    def test_symbolic_vectors_match_original_numeric_formula_and_unit_norm(self):
        rng = np.random.default_rng(17)
        for count in [0, 1, 2, 3, 7, 8, 11]:
            frequencies = [0.37 + 0.83 * i for i in range(count)]
            basis = _dictionary(frequencies, "x0**2-0.31", "test")
            for point in rng.uniform(-2, 2, 5):
                angle = point**2 - 0.31
                expected = [1.0]
                for frequency in frequencies:
                    expected.extend((math.cos(frequency * angle), math.sin(frequency * angle)))
                expected = np.asarray(expected) / math.sqrt(count + 1)
                actual = basis([point])
                np.testing.assert_allclose(actual, expected, atol=2e-16, rtol=2e-15)
                self.assertAlmostEqual(float(actual @ actual), 1.0, places=14)

    def test_user_decimals_and_preprocessing_are_not_rationalized(self):
        frequency = 0.12345678901234566
        preprocessing = "0.7071067811865476*x0+0.3333333333333333"
        basis = _dictionary([frequency], preprocessing, "decimal inputs")
        for expression in basis.expressions[1:]:
            self.assertIn(repr(frequency), expression)
            self.assertIn(preprocessing, expression)
        self.assertEqual(basis.expressions[0], "1/sqrt(2)")

    def test_callable_metadata_and_auxiliary_basis_use_exact_sources(self):
        problem = from_generators(
            [[0, 1, 2, 3]], [lambda x: x[0] ** 2], lambda a: math.cos(a[0]),
            [[-1, 1], [-2, 2]], spectra=True,
            preprocessing_expressions=["x0**2"],
        )
        self.assertEqual(problem.basis[0].to_spec()["expressions"][0], "1/2")
        self.assertEqual(problem.auxiliary_basis[0].expressions[0], "1/2")
        portable = type(problem.basis[0]).from_spec(problem.basis[0].to_spec())
        np.testing.assert_allclose(problem.basis[0]([0.37, 0.2]), portable([0.37, 0.2]))

    def test_exact_sources_survive_archive_and_standalone_prediction(self):
        basis = _dictionary([1.0], "x0**2", "encoding")
        core = np.array([[[0.0], [math.sqrt(2)], [0.0]]])
        model = Model([basis], [core], [[-1, 1]], name="cosine after preprocessing")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "symbolic.ftci"
            model.save(path)
            restored = Model.load(path)
            self.assertEqual(restored.basis[0].expressions, basis.expressions)
            with zipfile.ZipFile(path) as archive:
                manifest = json.loads(archive.read("manifest.json"))
                self.assertEqual(manifest["basis"][0]["expressions"][0], "1/sqrt(2)")
                predictor = Path(temporary) / "predict.py"
                predictor.write_bytes(archive.read("predict.py"))
            output = subprocess.check_output(
                [sys.executable, str(predictor), str(path), "[0.37]"],
                cwd=temporary, text=True,
            )
            self.assertAlmostEqual(float(output), math.cos(0.37**2), places=14)


if __name__ == "__main__":
    unittest.main()
