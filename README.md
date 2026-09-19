# Situation completion for sparse airside safety events

Public experiment code for the manuscript *Situation-completion knowledge graphs for sparse airside safety events: querying wildlife-hazard management actions from incomplete strike records* (submitted to *Advanced Engineering Informatics*).

## What this repository contains

- `src/knowledge/ontology/awhso.ttl` — five-layer situation TBox
- `src/knowledge/sparql/` — competency questions and definite Horn `CONSTRUCT` rules
- `src/knowledge/axioms/` — pathway licenses
- `src/knowledge/fixtures/` — public-plan action gold and named attractants
- Python scripts to smoke-test completion, build RDF graphs, and fire diagnostic rules

## What this repository does not contain

Strike records and Service Difficulty Reports remain with the Federal Aviation Administration public archives. They are not redistributed here.

- National Wildlife Strike Database: https://wildlife.faa.gov/
- FAA Service Difficulty Reports: https://av-info.faa.gov/sdrx/

Place downloaded tables under `data/raw/` as expected by the scripts.

## Software

Python 3.11+, `rdflib`, `sentence-transformers` (optional retrieval), `torch` (optional).

Smoke:

```text
python src/knowledge/smoke_situation_completion.py
python src/knowledge/fire_diagnostic_rules.py
```

`fire_diagnostic_rules.py` writes disagreement individuals with definite Horn rules after closed completion has added positive fillers (`localUnsupported`, `undocumentedAction`, event-log and plan flags).

## License

MIT. Ontology terms and public FAA documents remain under their original terms.
