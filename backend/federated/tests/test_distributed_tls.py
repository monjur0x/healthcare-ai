"""
Tests for the distributed gRPC TLS certificate wiring.

Flower's legacy NumpyClient transport only supports one-way TLS (a single
CA certificate). These tests pin that behavior: TLS off means None (and a
loud plaintext warning), one-way TLS hands Flower a single CA bytes, and
the unsupported mutual-TLS path fails loudly instead of silently
misconfiguring the channel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from federated.distributed import _load_certificates, _load_client_certificates


@pytest.fixture
def certs(tmp_path) -> dict[str, Path]:
    """Write fake PEM files (content is arbitrary bytes for these tests)."""
    files = {
        "ca": tmp_path / "ca.crt",
        "server_cert": tmp_path / "server.crt",
        "server_key": tmp_path / "server.key",
        "client_cert": tmp_path / "client.crt",
        "client_key": tmp_path / "client.key",
    }
    for name, path in files.items():
        payload = f"-----BEGIN {name.upper()}-----\nfake\n-----END-----"
        path.write_bytes(payload.encode())
    return files


def test_server_certificates_disabled_returns_none(certs) -> None:
    assert _load_certificates(False, None, None, None) is None


def test_server_certificates_missing_paths_raise(certs) -> None:
    with pytest.raises(ValueError, match="certificate paths missing"):
        _load_certificates(True, str(certs["ca"]), None, None)


def test_server_certificates_one_way_tuple(certs) -> None:
    result = _load_certificates(
        True,
        str(certs["ca"]),
        str(certs["server_cert"]),
        str(certs["server_key"]),
    )
    assert result == (
        certs["ca"].read_bytes(),
        certs["server_cert"].read_bytes(),
        certs["server_key"].read_bytes(),
    )


def test_client_certificates_disabled_returns_none(certs) -> None:
    assert _load_client_certificates(False, None, None, None) is None


def test_client_certificates_disabled_with_client_paths_returns_none(certs) -> None:
    # Supplying client cert/key while TLS is off must not crash; they are
    # ignored (transport is plaintext).
    assert (
        _load_client_certificates(
            False,
            None,
            str(certs["client_cert"]),
            str(certs["client_key"]),
        )
        is None
    )


def test_client_certificates_one_way_returns_ca_bytes(certs) -> None:
    result = _load_client_certificates(True, str(certs["ca"]), None, None)
    assert result == certs["ca"].read_bytes()


def test_client_certificates_requires_ca(certs) -> None:
    with pytest.raises(ValueError, match="ca_cert path is required"):
        _load_client_certificates(True, None, None, None)


def test_client_certificates_mutual_tls_fails_loudly(certs) -> None:
    # mTLS is unsupported on the legacy NumpyClient transport; it must not
    # silently produce a tuple that breaks the gRPC channel.
    with pytest.raises(ValueError, match="Mutual TLS is not supported"):
        _load_client_certificates(
            True,
            str(certs["ca"]),
            str(certs["client_cert"]),
            str(certs["client_key"]),
        )
