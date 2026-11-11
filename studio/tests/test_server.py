from email.message import Message
import io
import json
from types import SimpleNamespace
import unittest

import os
from unittest import mock

from ftci.server import examples, forwarded_host, make_handler, problem_from_spec


class RequestTests(unittest.TestCase):
    def request(self, path, body=None, *, host="127.0.0.1:8765", origin=None):
        handler_class = make_handler(SimpleNamespace())
        handler = handler_class.__new__(handler_class)
        handler.requestline = f"{'GET' if body is None else 'POST'} {path} HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "GET" if body is None else "POST"
        handler.path = path
        handler.client_address = ("127.0.0.1", 0)
        handler.server = SimpleNamespace(server_port=8765)
        handler.headers = Message()
        handler.headers["Host"] = host
        if origin:
            handler.headers["Origin"] = origin
        raw = b"" if body is None else json.dumps(body).encode()
        handler.headers["Content-Type"] = "application/json"
        handler.headers["Content-Length"] = str(len(raw))
        handler.rfile, handler.wfile = io.BytesIO(raw), io.BytesIO()
        handler.log_message = lambda *a: None
        if body is None:
            handler.do_GET()
        else:
            handler.do_POST()
        output = handler.wfile.getvalue()
        header, data = output.split(b"\r\n\r\n", 1)
        return int(header.split()[1]), json.loads(data)

    def test_health_and_examples(self):
        status, result = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertIn("engine_available", result)
        status, result = self.request("/api/examples")
        self.assertEqual(status, 200)
        self.assertEqual({e["mode"] for e in result}, {"function", "quantum"})

    def test_nonlocal_host_rejected(self):
        status, result = self.request("/api/health", host="attacker.example:8765")
        self.assertEqual(status, 400)
        self.assertIn("local FTCI", result["error"])

    def test_cross_origin_post_rejected(self):
        status, result = self.request("/api/jobs", {}, origin="https://attacker.example")
        self.assertEqual(status, 400)
        self.assertIn("this local FTCI", result["error"])

    def test_codespaces_forwarded_address_is_accepted_only_inside_that_codespace(self):
        space = {"CODESPACES": "true", "CODESPACE_NAME": "calm-space-abc123",
                 "GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN": "app.github.dev"}
        name = "calm-space-abc123-8765.app.github.dev"
        self.assertEqual(forwarded_host(8765, space), name)
        self.assertIsNone(forwarded_host(8765, {}))
        self.assertIsNone(forwarded_host(8765, {**space, "CODESPACES": "false"}))
        # Outside a codespace the forwarded name is an ordinary foreign host.
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.request("/api/health", host=name)[0], 400)
        with mock.patch.dict(os.environ, space, clear=True):
            self.assertEqual(self.request("/api/health", host=name)[0], 200)
            self.assertEqual(self.request("/api/health", host="localhost:8765")[0], 200)
            self.assertEqual(self.request("/api/health", host="other-space-8765.app.github.dev")[0], 400)
            self.assertEqual(self.request("/api/health", host=name + ":8765")[0], 400)
            # The proxy rewrites Origin to the local address, with either scheme.
            for host, origin in ((name, "https://" + name), ("localhost:8765", "https://localhost:8765"),
                                 ("localhost:8765", "http://localhost:8765")):
                status, result = self.request("/api/circuit/inspect", {}, host=host, origin=origin)
                self.assertNotIn("this local FTCI", result.get("error", ""), (host, origin))
            status, result = self.request("/api/jobs", {}, host=name, origin="https://attacker.example")
            self.assertEqual(status, 400)
            self.assertIn("this local FTCI", result["error"])

    def test_malformed_options_and_dictionaries(self):
        for options in (None, [], 5):
            spec = examples()[0]
            spec["options"] = options
            with self.assertRaisesRegex(ValueError, "options must be"):
                problem_from_spec(spec)
        spec = examples()[0]
        spec["basis"] = [None]
        with self.assertRaisesRegex(ValueError, "Each dictionary"):
            problem_from_spec(spec)


if __name__ == "__main__":
    unittest.main()
