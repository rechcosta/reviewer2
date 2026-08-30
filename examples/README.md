# Examples

A ready-to-run example that needs **no model and no network**: a hand-written
transcript plus a Markdown reference document.

```bash
python -m app --transcript examples/aula_transcricao.json \
              --reference examples/referencia_arquitetura.md \
              --offline \
              --output data/reports/exemplo_review.md
```

| File | What it is |
|---|---|
| `aula_transcricao.json` | a transcript in Reviewer2's format (timestamps + confidence), standing in for a real video |
| `referencia_arquitetura.md` | the reference material the claims are audited against |

The transcript deliberately contains one of each problem the system looks for:

| Timestamp | Claim | What it illustrates |
|---|---|---|
| 00:00:12 | "aumentar a frequência **sempre** aumenta o desempenho" | undue generalisation (the source states a condition) |
| 00:00:26 / 00:00:40 | "a cache é **não volátil**" vs. "os dados da cache **são perdidos** ao desligar" | internal contradiction |
| 00:00:54 | "o pipeline **garante** aceleração de cinco vezes" | ideal value presented as guaranteed |
| 00:01:26 | "sistemas com prefetching têm mais desempenho, **portanto** o prefetching causa o ganho" | correlation presented as causality |
| 00:01:41 | "o cache é como uma gaveta perto da mesa" | pedagogical simplification, not an error |

> **Note.** With `--offline` there is no language model: a small rule engine
> stands in for it, so it flags the lexical cases (the universal claim, the
> conditions omitted) but not the semantic ones (the cache contradiction).
> Run the same command against a real model to see the full analysis:
>
> ```bash
> python -m app --transcript examples/aula_transcricao.json \
>               --reference examples/referencia_arquitetura.md \
>               --llm-model qwen2.5:7b-instruct
> ```
