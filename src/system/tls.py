"""HTTPS for the console and the API.

    python -m src.system.tls make --host 192.168.1.20 --out secrets/tls

A certificate from the plant's own authority is better; this one is self-signed, so each
browser has to be told to trust it once.
"""
from __future__ import annotations

import argparse
import datetime
import ipaddress
import os
from pathlib import Path

LOCAL = {"127.0.0.1", "localhost", "::1"}


def make_certificate(hosts: list[str], directory: str | Path, *, days: int = 825) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hosts[0])])
    names: list[x509.GeneralName] = []
    for host in dict.fromkeys([*hosts, "localhost", "127.0.0.1"]):
        try:
            names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            names.append(x509.DNSName(host))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=days))
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                   serialization.NoEncryption()))
    return cert_path, key_path


def check_exposure(host: str, *, tls: bool, protected: bool, behind_proxy: bool = False) -> str | None:
    """Why this must not be opened to the network, or None."""
    if host in LOCAL:
        return None
    if not protected:
        return "needs accounts (python -m src.operations.auth add ...) or QUALYXUS_API_TOKEN before it is opened to the network"
    if not (tls or behind_proxy):
        return "needs HTTPS (a certificate and key) before it is opened to the network"
    return None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p = parser.add_subparsers(dest="cmd", required=True).add_parser("make")
    p.add_argument("--host", action="append", required=True, help="Name or address people will use (repeatable)")
    p.add_argument("--out", default="secrets/tls")
    args = parser.parse_args(argv)
    print("Wrote", *make_certificate(args.host, args.out))


if __name__ == "__main__":
    main()
