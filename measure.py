#!/usr/bin/env python3
"""Measure how many tokens different kinds of content actually cost. Writes results/tokens.json."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import sys
import urllib.request
import uuid
import zipfile
from pathlib import Path

import tiktoken

import bpe

UDHR_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/udhr.zip"
LANGUAGES = {
    "English": ("English-Latin1", "latin-1"),
    "Spanish": ("Spanish-Latin1", "latin-1"),
    "German": ("German_Deutsch-Latin1", "latin-1"),
    "Russian": ("Russian_Russky-UTF8", "utf-8"),
    "Arabic": ("Arabic_Alarabia-Arabic", "cp1256"),
    "Hindi": ("Hindi-UTF8", "utf-8"),
    "Tamil": ("Tamil-UTF8", "utf-8"),
    "Korean": ("Korean_Hankuko-UTF8", "utf-8"),
    "Chinese": ("Chinese_Mandarin-UTF8", "utf-8"),
    "Japanese": ("Japanese_Nihongo-UTF8", "utf-8"),
}


def udhr() -> dict[str, str]:
    """The Universal Declaration of Human Rights: the same text, in many languages."""
    cache = Path("data/udhr.zip")
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        with urllib.request.urlopen(UDHR_URL, timeout=60) as response:  # noqa: S310
            cache.write_bytes(response.read())
    with zipfile.ZipFile(cache) as archive:
        return {
            name: archive.read(f"udhr/{file}").decode(encoding, errors="replace").strip()
            for name, (file, encoding) in LANGUAGES.items()
        }


def records(count: int, rng: random.Random) -> list[dict]:
    services = ["checkout", "payments", "search", "auth", "catalog", "ledger"]
    return [
        {
            "id": i,
            "service": rng.choice(services),
            "status": rng.choice(["healthy", "degraded", "down"]),
            "latency_ms": rng.randint(4, 900),
            "error_rate": round(rng.random() / 10, 4),
        }
        for i in range(count)
    ]


def as_csv(rows: list[dict]) -> str:
    out = io.StringIO()
    writer = csv.DictWriter(out, rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def as_yaml(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        items = list(row.items())
        lines.append(f"- {items[0][0]}: {items[0][1]}")
        lines += [f"  {k}: {v}" for k, v in items[1:]]
    return "\n".join(lines) + "\n"


def as_markdown(rows: list[dict]) -> str:
    keys = list(rows[0])
    head = "| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n"
    return head + "".join("| " + " | ".join(str(r[k]) for k in keys) + " |\n" for r in rows)


def as_xml(rows: list[dict]) -> str:
    body = ""
    for row in rows:
        fields = "".join(f"<{k}>{v}</{k}>" for k, v in row.items())
        body += f"  <record>{fields}</record>\n"
    return f"<records>\n{body}</records>\n"


def identifiers(rng: random.Random, n: int = 500) -> dict[str, str]:
    randbytes = rng.randbytes
    return {
        "UUIDv4": "\n".join(str(uuid.UUID(bytes=randbytes(16), version=4)) for _ in range(n)),
        "SHA-256 hex": "\n".join(hashlib.sha256(randbytes(32)).hexdigest() for _ in range(n)),
        "base64 blob": "\n".join(base64.b64encode(randbytes(48)).decode() for _ in range(n)),
        "ISO-8601 timestamps": "\n".join(
            f"2026-10-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
            f"{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d}Z"
            for _ in range(n)
        ),
        "Short alias (t17)": "\n".join(f"t{rng.randint(1, 99)}" for _ in range(n)),
        "IPv4 addresses": "\n".join(
            ".".join(str(rng.randint(1, 254)) for _ in range(4)) for _ in range(n)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    encoders = {name: tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")}

    def row(label: str, text: str) -> dict:
        tokens = {
            name: len(enc.encode(text, disallowed_special=())) for name, enc in encoders.items()
        }
        out = {
            "content": label,
            "chars": len(text),
            "utf8_bytes": len(text.encode()),
            **{f"{n}_tokens": t for n, t in tokens.items()},
            **{f"{n}_chars_per_token": round(len(text) / t, 2) for n, t in tokens.items()},
        }
        out["four_chars_rule_error_pct"] = round(
            100 * (len(text) / 4 - tokens["o200k_base"]) / tokens["o200k_base"], 1
        )
        return out

    languages = udhr()
    data = records(200, rng)
    formats = {
        "JSON, indented": json.dumps(data, indent=2),
        "JSON, minified": json.dumps(data, separators=(",", ":")),
        "YAML": as_yaml(data),
        "CSV": as_csv(data),
        "Markdown table": as_markdown(data),
        "XML": as_xml(data),
    }
    source = Path(argparse.__file__).read_text()
    merges = bpe.train(languages["English"], 1024)

    def bytes_per_token(text: str) -> float:
        return round(len(text.encode()) / len(bpe.encode(text, merges)), 2)

    results = {
        "python": sys.version.split()[0],
        "languages": [row(k, v) for k, v in languages.items()],
        "formats": [row(k, v) for k, v in formats.items()],
        "identifiers": [row(k, v) for k, v in identifiers(rng).items()],
        "code": [row("Python source (stdlib argparse.py)", source)],
        "own_bpe_trained_on_english_udhr": {
            "vocab_size": 256 + len(merges),
            "bytes_per_token": {
                "English UDHR (training text)": bytes_per_token(languages["English"]),
                "Spanish UDHR": bytes_per_token(languages["Spanish"]),
                "Tamil UDHR": bytes_per_token(languages["Tamil"]),
                "UUIDv4": bytes_per_token(identifiers(random.Random(1), 100)["UUIDv4"]),
                "Python source": bytes_per_token(source[:8000]),
            },
        },
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/tokens.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
