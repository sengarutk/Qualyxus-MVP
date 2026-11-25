#!/bin/bash
# Install on a Linux box without Docker. Run from the repository as root. Safe to run again:
# it never replaces /etc/qualyxus or /var/lib/qualyxus.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "Run as root." >&2; exit 1; }
PYTHON=${PYTHON:-python3.12}
REQUIREMENTS=${REQUIREMENTS:-requirements.txt}
cd "$(dirname "$0")/.."

id qualyxus >/dev/null 2>&1 || useradd --system --home /var/lib/qualyxus --shell /usr/sbin/nologin qualyxus
install -d -m 755 /opt/qualyxus /etc/qualyxus
install -d -m 700 -o qualyxus -g qualyxus /var/lib/qualyxus /var/lib/qualyxus/data /var/lib/qualyxus/models /var/lib/qualyxus/secrets
cp -r main.py src config.yaml config.360.yaml "$REQUIREMENTS" /opt/qualyxus/
for d in data models secrets; do ln -sfn "/var/lib/qualyxus/$d" "/opt/qualyxus/$d"; done
[ -e /etc/qualyxus/config.yaml ] || cp config.yaml /etc/qualyxus/config.yaml
[ -e /etc/qualyxus/environment ] || install -m 600 -o qualyxus /dev/null /etc/qualyxus/environment

[ -x /opt/qualyxus/.venv/bin/python ] || "$PYTHON" -m venv /opt/qualyxus/.venv
/opt/qualyxus/.venv/bin/pip install --require-hashes -r "/opt/qualyxus/$(basename "$REQUIREMENTS")"

cp deploy/*.service deploy/*.timer /etc/systemd/system/
systemctl daemon-reload
systemctl enable qualyxus.service qualyxus-console.service qualyxus-health.timer
echo "Installed. Edit /etc/qualyxus/config.yaml, put the models in /var/lib/qualyxus/models, then: systemctl start qualyxus qualyxus-console qualyxus-health.timer"
echo "The data holds images of parts: put /var/lib/qualyxus on an encrypted disk (LUKS), and the audit key on another one."
