"""The submitted source, rather than a stale diagram snapshot, defines a run."""
from copy import deepcopy
import io
import json
from email.message import Message
from types import SimpleNamespace
import unittest

import numpy as np

from ftci.server import make_handler, problem_from_spec


def source_circuit():
    return {"n_qubits": 1, "observable": "Z", "domain": [[-1, 1], [-1, 1]],
            "gates": [{"gate": "ry", "qubits": [0], "angle": "x0"},
                      {"gate": "ry", "qubits": [0], "angle": "2*x0**2+x1"}]}


class SourceServerTests(unittest.TestCase):
    def test_source_wins_over_stale_graph_and_domain(self):
        source = source_circuit()
        stale = deepcopy(source)
        stale["gates"][1]["angle"] = "x0"
        problem, _ = problem_from_spec({"mode": "quantum", "domain": [[-100, 100]],
                                        "circuit": stale,
                                        "circuit_source": {"format": "json", "source": json.dumps(source)}})
        point = np.array([0.4, 0.3])
        self.assertAlmostEqual(problem.target(point), np.cos(0.4 + 2*0.4**2 + 0.3))
        self.assertEqual(problem.domain, [(-1., 1.), (-1., 1.)])
        self.assertEqual(problem.metadata["circuit_spec"]["gates"][1]["angle"], "2*x0**2+x1")
        self.assertEqual(json.loads(problem.metadata["circuit_source"]["source"]), source)

    def test_invalid_source_never_falls_back_to_old_graph(self):
        with self.assertRaises(ValueError):
            problem_from_spec({"mode": "quantum", "domain": [[-1, 1], [-1, 1]],
                               "circuit": source_circuit(),
                               "circuit_source": {"format": "json", "source": '{"gates":'}})

    def request(self, path, body):
        handler_type = make_handler(SimpleNamespace())
        handler = handler_type.__new__(handler_type)
        handler.requestline = f"POST {path} HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "POST"
        handler.path = path
        handler.client_address = ("127.0.0.1", 0)
        handler.server = SimpleNamespace(server_port=8765)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8765"
        handler.headers["Content-Type"] = "application/json"
        payload = json.dumps(body).encode()
        handler.headers["Content-Length"] = str(len(payload))
        handler.rfile, handler.wfile = io.BytesIO(payload), io.BytesIO()
        handler.log_message = lambda *args: None
        handler.do_POST()
        header, response = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        return int(header.split()[1]), json.loads(response)

    def test_live_parse_and_format_endpoints(self):
        circuit = source_circuit()
        status, response = self.request("/api/circuit/source", {"format": "json", "source": json.dumps(circuit)})
        self.assertEqual(status, 200)
        self.assertEqual(response["inspection"]["encodings"][1]["input_indices"], [0, 1])
        status, response = self.request("/api/circuit/serialize", {"format": "qiskit", "circuit": circuit})
        self.assertEqual(status, 200)
        self.assertIn("QuantumCircuit", response["source"])
        self.assertEqual(response["inspection"]["encoding_occurrences"], 2)

    def test_package_independent_python_is_the_authoritative_training_source(self):
        circuit = source_circuit()
        source = "import unavailable_package\nftci_circuit = " + repr(circuit) + "\n"
        status, response = self.request("/api/circuit/source", {"format": "python", "source": source})
        self.assertEqual(status, 200)
        self.assertEqual(response["source_info"]["kind"], "portable")
        self.assertIn("ftci_circuit literal", response["source_info"]["notice"])
        self.assertEqual(response["inspection"]["encoding_occurrences"], 2)
        problem, _ = problem_from_spec({"mode": "quantum", "domain": [[-100, 100]],
                                        "circuit": {"n_qubits": 1, "gates": []},
                                        "circuit_source": {"format": "python", "source": source}})
        self.assertEqual(problem.domain, [(-1., 1.), (-1., 1.)])
        self.assertAlmostEqual(problem.target([0.4, 0.3]), np.cos(0.4 + 2*0.4**2 + 0.3))
        self.assertEqual(problem.metadata["circuit_source"]["source"], source)


if __name__ == "__main__":
    unittest.main()
