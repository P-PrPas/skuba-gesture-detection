"""Append the local AV (AVG) interception root to certifi's bundle, so runtime
model downloads (ultralytics weights, mediapipe assets) trust the MITM'd chain.

Idempotent. Run once after `pip install`. ponytail: only needed because this
machine's antivirus rewrites TLS; a machine without that can skip it.
"""

import ssl
from pathlib import Path

import certifi

AV_ROOTS = [
    Path(r"C:\ProgramData\AVG\Antivirus\wscert.pem"),
    Path(r"C:\ProgramData\Avast\wscert.pem"),
]


def main():
    bundle = Path(certifi.where())
    current = bundle.read_text(encoding="utf-8")
    added = 0
    for root in AV_ROOTS:
        if not root.exists():
            continue
        pem = root.read_text(encoding="utf-8").strip()
        if pem and pem not in current:
            with bundle.open("a", encoding="utf-8") as f:
                f.write("\n# local AV interception root\n" + pem + "\n")
            added += 1
            print(f"appended {root} -> {bundle}")
    if not added:
        print("nothing to add (already present or no AV root found)")
    # sanity: bundle still parses
    ssl.create_default_context(cafile=str(bundle))
    print("certifi bundle OK")


if __name__ == "__main__":
    main()
