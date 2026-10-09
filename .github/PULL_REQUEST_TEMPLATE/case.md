## Contribution type

- [ ] New benchmark case
- [ ] Evaluator or harness improvement
- [ ] Agent adapter
- [ ] Documentation or website

## New case checklist

- [ ] I followed [`CONTRIBUTING.md`](https://github.com/sponyo531/fde-bench/blob/main/CONTRIBUTING.md) and the existing `case/<number>_<slug>/` structure.
- [ ] The request, requirements, clarification targets, evaluator, schema, environment, and reference artifact are included or clearly linked.
- [ ] I ran `python -m harness.tools.leak_check --case-root case` and reviewed the output.
- [ ] I tested the evaluator and relevant offline tests.
- [ ] The case is de-identified; I have documented provenance and redistribution rights.
- [ ] No credentials, private endpoints, or restricted data are included in Git history.
- [ ] Large data are coordinated as a release asset with checksums rather than committed directly.

## Summary

Describe the task objective, the hidden requirements being tested, and how a reviewer can validate the contribution.
