# Contributing to FDE-Bench

Pull requests are welcome. Contributions can add a new de-identified customer case, improve an evaluator, or add an agent adapter. A case is accepted only after a reviewer can inspect its provenance, run its offline checks, and verify that its hidden requirements are not leaked into the initial request.

## New case format

Use a zero-padded numeric directory and a descriptive slug:

```text
case/050_new_case_slug/
├── instruction.md                 # agent-visible initial request
├── information.md                 # complete business context for the simulated customer
├── gt.json                        # annotated clarification targets, valid JSON
├── task.toml                      # [metadata] task_type, scene_type, difficulty
├── environment/Dockerfile         # pinned case dependencies
└── tests/
    ├── evaluator.py               # deterministic schema/constraint/quality evaluator
    ├── extractor_agent.py         # artifact extraction contract
    ├── submission_schema.json      # submitted-artifact schema
    ├── baseline/reference_metrics.json
    └── best_solution/<reference>  # accepted FDE reference artifact
```

The harness requires `instruction.md`, `information.md`, `gt.json`, and `data/`. The evaluator must define the artifact contract clearly enough for the extractor and scoring harness to use it. `gt.json` targets must describe information that is absent from `instruction.md` and visible data, and each target must be answerable by the simulated customer from `information.md`.

Keep the initial request at the level used in the real project. Do not copy complete rules, target values, or reference decisions into `instruction.md` just to make the case easier. Run the leak check and review every reported item manually:

```bash
python -m harness.tools.leak_check --case-root case
python -m harness --list
pytest harness/tests
```

For a new case, also run its evaluator and a smoke dry-run after its data and environment are available. A full model run is not required for the first PR, but the reference artifact must pass the evaluator and the case must validate offline.

## Data and privacy requirements

- De-identify customer names, locations, identifiers, timestamps, and file metadata consistently while preserving the numerical relationships needed for evaluation.
- Do not commit credentials, private endpoints, customer-identifying material, or data whose redistribution rights are unclear.
- Large inputs should be proposed as a release asset rather than forced into Git history. Include a checksum and a small synthetic fixture when possible; coordinate with maintainers before publishing controlled data.
- Document provenance, de-identification, intended use, and redistribution rights in the PR. Third-party data must retain its original terms.
- Keep `information.md`, `gt.json`, `tests/`, and the reference solution outside the agent-visible solving workspace; the harness relies on this separation.

## Pull request checklist

A new-case PR should include:

- [ ] the directory follows the structure above and has a unique case number;
- [ ] `gt.json`, task metadata, schema, evaluator, extractor, and reference artifact parse and run offline;
- [ ] the leak check was run and its output was reviewed;
- [ ] the case is de-identified and its data rights/provenance are documented;
- [ ] large or restricted data are not added to Git history;
- [ ] the PR description explains the objective, acceptance criteria, and files a reviewer can run.

Open a PR from the [repository](https://github.com/sponyo531/fde-bench/pulls). Maintainers may request changes before adding a case to a numbered release.
