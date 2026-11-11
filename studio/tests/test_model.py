"""Behavioral checks for portable learned functions and their file boundary."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

from ftci.basis import Basis
from ftci.expressions import Expression
from ftci.model import Model


def known_model():
    """f(x,y) = 1 + 2 cos(x) - 2 cos(x+y), a rank-three coefficient matrix."""
    dictionaries = [Basis(["1", "cos(x0)", "sin(x0)"], name="first"),
                    Basis(["1", "cos(x1)", "sin(x1)"], name="second")]
    coefficients = np.array([[1.0, 0, 0], [2.0, -2.0, 0], [0, 0, 2.0]])
    # The actual matrix rank is three; this deliberately tests a nontrivial
    # coefficient contraction rather than an outer-product-only predictor.
    u, s, vh = np.linalg.svd(coefficients, full_matrices=False)
    return Model(dictionaries, [u.reshape(1, 3, 3), (s[:, None] * vh).reshape(3, 3, 1)],
                 [(-np.pi, np.pi), (-np.pi, np.pi)],
                 metadata={"validation": np.float64(1e-12), "ranks": np.array([3]), "missing": np.nan},
                 name="cosine addition")


class ExpressionTests(unittest.TestCase):
    def test_math_and_dependencies(self):
        expression = Expression("cos(3*x0**2) + exp(x2)/sqrt(4) - sin(pi/2)")
        self.assertEqual(expression.input_indices, (0, 2))
        self.assertAlmostEqual(expression([0.2, 99, 0.5]), np.cos(0.12) + np.exp(0.5)/2 - 1)
        self.assertAlmostEqual(Expression("atan2(x0, x1)")([1, 1]), np.pi / 4)

    def test_code_and_invalid_syntax_rejected(self):
        expressions = ["__import__('os').system('echo unsafe')", "x0.__class__", "x[0]",
                       "[x0]", "lambda: 1", "(x0 for x0 in [1])", "sin(x0, x1)",
                       "sin(value=x0)", "True", "nan", "1e999", "open('anything')"]
        for text in expressions:
            with self.subTest(text=text), self.assertRaises(ValueError):
                Expression(text)

    def test_nonfinite_and_missing_inputs_rejected(self):
        for text, point in [("1/x0", [0]), ("sqrt(x0)", [-1]), ("exp(x0)", [10000]),
                            ("x1", [1]), ("x0", [np.nan])]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                Expression(text)(point)


class ModelTests(unittest.TestCase):
    def test_prediction_coefficients_and_batch(self):
        model = known_model()
        points = np.random.default_rng(29).uniform(-np.pi, np.pi, (40, 2))
        expected = 1 + 2*np.cos(points[:, 0]) - 2*np.cos(points[:, 0] + points[:, 1])
        np.testing.assert_allclose(model.predict(points), expected, atol=2e-14)
        self.assertAlmostEqual(model.coefficient([1, 1]), -2)
        self.assertAlmostEqual(model.coefficient([2, 2]), 2)
        self.assertEqual(model.bond_dimensions, [3])
        self.assertIsNone(model.summary()["metadata"]["missing"])
        json.dumps(model.summary(), allow_nan=False)
        with self.assertRaises(IndexError):
            model.coefficient([3, 0])

    def test_rank_two_example(self):
        dictionaries = [Basis(["cos(x0)", "sin(x0)"]), Basis(["cos(x1)", "sin(x1)"])]
        model = Model(dictionaries, [np.eye(2).reshape(1, 2, 2), np.diag([1., -1.]).reshape(2, 2, 1)],
                      [(-1, 1), (-1, 1)])
        self.assertEqual(model.bond_dimensions, [2])
        self.assertAlmostEqual(model([0.3, 0.4]), np.cos(0.7))

    def test_single_site_and_shape_validation(self):
        model = Model([Basis(["1", "x0", "x0**2"])], [np.array([1., 2., 3.]).reshape(1, 3, 1)], [(-1, 1)])
        self.assertAlmostEqual(model(0.5), 2.75)
        np.testing.assert_allclose(model.predict([0., 0.5]), [1., 2.75])
        for x in [[1, 2], [np.inf], [[1]]]:
            with self.subTest(x=x), self.assertRaises(ValueError):
                model(x)
        for cores in [[np.zeros((2, 3, 1))], [np.zeros((1, 2, 1))], [np.zeros((1, 3, 2))],
                      [np.full((1, 3, 1), np.nan)]]:
            with self.subTest(cores=[array.shape for array in cores]), self.assertRaises(ValueError):
                Model(model.basis, cores, [(-1, 1)])

    def test_callable_basis_evaluates_but_needs_portable_spec(self):
        dictionary = Basis.from_callable(lambda x: np.array([1., x[0]**2]), [0], 2)
        model = Model([dictionary], [np.array([1., 3.]).reshape(1, 2, 1)], [(-1, 1)])
        self.assertAlmostEqual(model([0.5]), 1.75)
        self.assertFalse(model.summary()["portable"])
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "expressions"):
            model.save(Path(directory) / "model.ftci")
        portable = Basis.from_callable(lambda x: [1., x[0]**2], [0], 2, expressions=["1", "x0**2"])
        self.assertEqual(Basis.from_spec(portable.to_spec()).dimension, 2)

    def test_roundtrip_and_standalone_without_sdk(self):
        model = known_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "learned.ftci"
            model.save(path)
            loaded = Model.load(path)
            self.assertEqual(loaded.summary(), model.summary())
            point = [0.31, -0.72]
            self.assertAlmostEqual(loaded(point), model(point))
            with zipfile.ZipFile(path) as archive:
                predictor_path = Path(directory) / "predict.py"
                predictor_path.write_bytes(archive.read("predict.py"))
            # -I ignores PYTHONPATH, proving the predictor has no package/oracle
            # dependency. This script only needs the interpreter's NumPy.
            result = subprocess.run([sys.executable, "-I", str(predictor_path), str(path), json.dumps(point)],
                                    cwd=directory, text=True, capture_output=True, check=True)
            self.assertAlmostEqual(json.loads(result.stdout), model(point))
            batch = subprocess.run([sys.executable, "-I", str(predictor_path), str(path),
                                    json.dumps([point, [0, 0]]), "--batch"],
                                   cwd=directory, text=True, capture_output=True, check=True)
            np.testing.assert_allclose(json.loads(batch.stdout), model.predict([point, [0, 0]]))

    def test_untrusted_archives_and_array_headers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "valid.ftci"
            known_model().save(original)
            with zipfile.ZipFile(original) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            bad = Path(directory) / "invalid.ftci"
            with zipfile.ZipFile(bad, "w") as archive:
                for name, value in members.items():
                    archive.writestr(name, value)
                archive.writestr("../outside", "not extracted")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                Model.load(bad)
            manifest = json.loads(members["manifest.json"])
            manifest["cores"][0]["shape"] = [1, 3, 65536]
            members["manifest.json"] = json.dumps(manifest).encode()
            with zipfile.ZipFile(bad, "w") as archive:
                for name, value in members.items():
                    archive.writestr(name, value)
            with self.assertRaisesRegex(ValueError, "header"):
                Model.load(bad)
            self.assertFalse((Path(directory).parent / "outside").exists())

    def test_archive_never_executes_bundled_predictor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ftci"
            known_model().save(path)
            with zipfile.ZipFile(path) as archive:
                members = {name: archive.read(name) for name in archive.namelist()}
            members["predict.py"] = b"raise RuntimeError('must not execute')"
            with zipfile.ZipFile(path, "w") as archive:
                for name, data in members.items():
                    archive.writestr(name, data)
            loaded = Model.load(path)
            self.assertAlmostEqual(loaded([0, 0]), 1)


if __name__ == "__main__":
    unittest.main()
