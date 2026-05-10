# ReSketch

ReSketch is a corpus-guided semantic regex synthesis prototype. It uses
typed semantic sketches, pluggable candidate retrieval strategies, Python regex
semantics, and an interactive CEGIS loop.

## Setup

```bash
poetry install
export OPENAI_API_KEY=...
```

## Examples

Run with the sampled annotated database:

```bash
poetry run resketch synthesize \
  --provider db \
  --db-path fixtures/candidates.sqlite \
  --no-interactive \
  --sketch '{□: integer}' \
  --pos '123' \
  --neg 'abc'
```

Inspect hole ids for a multi-hole sketch:

```bash
poetry run resketch inspect-sketch \
  --sketch '{□: year}-{□: integer}'
```

Use explicit hole examples when global examples are ambiguous:

```bash
poetry run resketch synthesize \
  --provider fixture \
  --no-interactive \
  --sketch '{□: cvv}' \
  --pos '1234' \
  --neg '12' \
  --hole-pos 'h0=1234' \
  --hole-neg 'h0=12345'
```

Run with LLM retrieval:

```bash
poetry run resketch synthesize \
  --provider llm \
  --sketch '{□: credit_card} CVV:{□: cvv}' \
  --pos '4111111111111111 CVV:123'
```

Show merged configuration:

```bash
poetry run resketch config show
```

## Hole-Level Examples

Global examples define correctness for the final regex. ReSketch also derives
hole-local positives by matching the sketch skeleton against global positives.
Every inferred local example is stored as structured evidence with a source
example, confidence, policies used, and reason.

Decomposition modes support ablations:

- `off`: no inferred or explicit hole evidence is used.
- `explicit-only`: only `--hole-pos` and `--hole-neg` evidence is used.
- `hard-only`: only uniquely determined inferred positives are used.
- `hard-and-soft`: uniquely determined positives prune search, ambiguous positives rank candidates.

Ambiguous adjacent holes are handled with bounded split enumeration. Global
negatives remain final whole-regex constraints unless supplied explicitly with
`--hole-neg`. Evaluation output includes decomposition metrics such as success
rate, hard coverage, ambiguity count, explicit evidence count, and average
assignments considered per example.

## Synthesis Search

ReSketch uses an internal regex AST search engine for hole completion. Retrieved
regexes are parsed with Python's `sre_parse`, decomposed into reusable fragments,
then searched with bounded bottom-up constructors such as concatenation,
alternation, repeat, and optional. The result trace reports per-hole search
statistics, selected option provenance, behavioral deduplication counts, and
timeout status.

Candidate ranking uses grouped scoring terms:

```text
candidate_ranking_score =
    semantic_fit_score
  + source_confidence_score
  - syntactic_complexity_cost
  - strategy_penalty
```

`semantic_fit_score` comes from local examples, soft positives, and
tuple-negative probes. `source_confidence_score` weights provider confidence.
`syntactic_complexity_cost` combines AST structure cost and rendered-length cost.
Each candidate feature in the synthesis trace includes the full score breakdown.

Automata-backed partial pruning is enabled by default. The engine compiles the
supported internal regex/sketch AST subset to automata-lib NFAs over printable
ASCII plus observed example characters, uses unknown holes as bounded wildcard
languages, and fails open when a construct cannot be modeled soundly. Disable it
with `--no-automata-pruning` for CLI synthesis or benchmark runs.
