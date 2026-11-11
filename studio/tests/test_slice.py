"""Lines and planes through the input domain for the prediction view."""

from email.message import Message
import io
import json
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ftci import Basis, Model
from ftci.server import Studio, make_handler, slice_values


def bilinear_model():
    # f(x) = 1 + x0*x1 on [-1, 1] x [0, 2], with a third input the function ignores.
    first = np.zeros((1, 2, 2)); first[0, 0, 0] = first[0, 1, 1] = 1
    second = np.zeros((2, 2, 1)); second[0, 0, 0] = second[1, 1, 0] = 1
    return Model([Basis(["1", "x0"]), Basis(["1", "x1"])], [first, second], [[-1, 1], [0, 2], [5, 7]])


class SliceValueTests(unittest.TestCase):
    def setUp(self):
        self.model = bilinear_model()
        self.calls = []
        def target(x):
            self.calls.append(tuple(x))
            return 1 + x[0] * x[1]
        self.target = target

    def test_line_varies_one_input_and_holds_the_others(self):
        result = slice_values(self.model, self.target, [1], [0.5, 9, 6], points=3, with_target=True)
        self.assertEqual(result["x"], [0, 1, 2])
        self.assertIsNone(result["y"])
        self.assertEqual(result["prediction"], [1, 1.5, 2])
        self.assertEqual(result["truth"], [1, 1.5, 2])
        self.assertEqual(result["target_queries"], 3)
        self.assertEqual(self.calls, [(0.5, 0, 6), (0.5, 1, 6), (0.5, 2, 6)])

    def test_plane_rows_follow_the_second_axis(self):
        result = slice_values(self.model, self.target, [0, 1], points=3, with_target=True)
        self.assertEqual((result["x"], result["y"]), ([-1, 0, 1], [0, 1, 2]))
        self.assertEqual(result["prediction"], [[1, 1, 1], [0, 1, 2], [-1, 1, 3]])
        self.assertEqual(result["truth"], result["prediction"])
        self.assertEqual(result["target_queries"], 9)
        swapped = slice_values(self.model, None, [1, 0], points=3)
        self.assertEqual(swapped["prediction"], [[1, 0, -1], [1, 1, 1], [1, 2, 3]])

    def test_prediction_alone_costs_no_queries_and_uses_midpoints(self):
        result = slice_values(self.model, self.target, [0])
        self.assertEqual((len(result["x"]), result["truth"], result["target_queries"]), (61, None, 0))
        self.assertEqual(result["fixed"], [0, 1, 6])
        self.assertEqual(self.calls, [])
        self.assertEqual(len(slice_values(self.model, None, [0, 2])["prediction"]), 31)
        json.dumps(result, allow_nan=False)

    def test_invalid_requests_are_rejected(self):
        for axes in ([], [0, 0], [3], [0.0], [True], "x0", [0, 1, 2], None):
            with self.assertRaisesRegex(ValueError, "one or two different inputs"):
                slice_values(self.model, self.target, axes)
        for fixed in ([0, 1], [0, 1, "a"], [0, 1, float("nan")]):
            with self.assertRaisesRegex(ValueError, "input"):
                slice_values(self.model, self.target, [0], fixed)
        for points in (1, 202, 2.5, True):
            with self.assertRaisesRegex(ValueError, "points per axis"):
                slice_values(self.model, self.target, [0], points=points)
        with self.assertRaisesRegex(ValueError, "points per axis"):
            slice_values(self.model, self.target, [0, 1], points=42)
        with self.assertRaisesRegex(ValueError, "no longer available"):
            slice_values(self.model, None, [0], with_target=True)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            slice_values(self.model, lambda x: float("inf"), [0], with_target=True)


@unittest.skipUnless(shutil.which("julia"), "Julia is not installed")
class SliceEndpointTests(unittest.TestCase):
    def test_finished_job_serves_slices_with_and_without_the_target(self):
        spec = {"mode": "function", "name": "Slice", "target": "cos(x0 + x1) + 0.2*x2",
                "domain": [[-1, 1], [-1, 1], [0, 1]],
                "basis": [{"expressions": ["1", "cos(x0)", "sin(x0)"]},
                          {"expressions": ["1", "cos(x1)", "sin(x1)"]},
                          {"expressions": ["1", "x2"]}],
                "options": {"max_bond": 6, "query_budget": 3000, "validation_samples": 20}}
        with tempfile.TemporaryDirectory() as directory:
            studio = Studio(directory)
            try:
                identifier = studio.submit(spec)["id"]
                studio.pool.shutdown(wait=True)
                self.assertEqual(studio.public_job(identifier)["status"], "complete")

                def post(body):
                    handler_class = make_handler(studio)
                    handler = handler_class.__new__(handler_class)
                    path = f"/api/jobs/{identifier}/slice"
                    handler.requestline, handler.request_version = f"POST {path} HTTP/1.1", "HTTP/1.1"
                    handler.command, handler.path = "POST", path
                    handler.client_address = ("127.0.0.1", 0)
                    handler.server = SimpleNamespace(server_port=8765)
                    handler.headers = Message()
                    handler.headers["Host"] = "127.0.0.1:8765"
                    raw = json.dumps(body).encode()
                    handler.headers["Content-Type"] = "application/json"
                    handler.headers["Content-Length"] = str(len(raw))
                    handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
                    handler.log_message = lambda *a: None
                    handler.do_POST()
                    header, data = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
                    return int(header.split()[1]), json.loads(data)

                status, line = post({"axes": [2], "fixed": [0.3, -0.4, 0], "target": True, "points": 5})
                self.assertEqual(status, 200)
                np.testing.assert_allclose(line["prediction"], line["truth"], atol=1e-9)
                np.testing.assert_allclose(line["truth"], np.cos(-0.1) + 0.2 * np.linspace(0, 1, 5))
                status, plane = post({"axes": [0, 1], "fixed": [0, 0, 0.5]})
                self.assertEqual((status, plane["truth"], plane["target_queries"]), (200, None, 0))
                self.assertEqual(np.shape(plane["prediction"]), (31, 31))
                self.assertAlmostEqual(plane["prediction"][0][30], np.cos(0) + 0.1, places=9)
                self.assertEqual(post({"axes": [0, 0]})[0], 400)
                self.assertEqual(post([0])[0], 400)
            finally:
                studio.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    unittest.main()
