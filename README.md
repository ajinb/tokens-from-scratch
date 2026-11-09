# tokens-from-scratch

A byte-pair-encoding tokenizer in about 50 lines of standard-library Python, plus a script that measures what different kinds of content actually cost in tokens.

Companion to [What a Token Actually Is, and Why a UUID Costs 24 of Them](https://cloudandsre.com/blog/what-is-a-token-bpe-from-scratch/).

## What it shows

`len(text) / 4` is wrong in both directions. Measured against OpenAI's `o200k_base` encoding:

| Content | Characters per token | `len/4` error |
|---|---:|---:|
| English prose (formal) | 5.19 | +30% |
| Python source | 5.03 | +26% |
| JSON, indented | 2.67 | -33% |
| UUIDs | 1.55 | -61% |
| Japanese | 1.21 | -70% |

The same 200 records cost 2,608 tokens as CSV, 5,243 as minified JSON and 9,043 as indented JSON. A UUID costs about 24 tokens. Full results are in [`results/tokens.json`](results/tokens.json).

## Run it

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-ci.txt
pytest -q
python measure.py
```

`measure.py` downloads the Universal Declaration of Human Rights corpus from NLTK's data repository on first run and caches it in `data/`. `tiktoken` downloads its encodings on first use.

## Use the tokenizer

```python
import bpe

merges = bpe.train(open("corpus.txt").read(), vocab_size=1024)
ids = bpe.encode("hello world", merges)
assert bpe.decode(ids, merges) == "hello world"
```

## Limits

The multilingual sample is the first 10 KB of each UDHR translation, so compare rates, not totals. The encodings are OpenAI's public ones. For Claude, use the free [`count_tokens` endpoint](https://platform.claude.com/docs/en/build-with-claude/token-counting). This measures input encoding only, not how well a model reads each format.

## License

Apache-2.0
