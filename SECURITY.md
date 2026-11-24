# Security policy

## Intended use

Cubic actively exercises applications: it replays requests across personas, fires
concurrent request bursts, skips lifecycle steps and sends boundary values. Only point it
at applications you own or are explicitly authorized to test. Run it against local,
in-process builds or isolated staging environments with seeded test data, never production.

The Bug Gym apps in `src/cubic/gym/` (fintech, SaaS, agent gateway) are **intentionally
vulnerable**. They run only in-process for benchmarking. Do not deploy them.

Persona credentials in graphs, regression tests and the RLVR sample are the seeded test
tokens of the in-process testbeds. Cubic never puts credentials into model prompts. Keep
real secrets out of persona files you commit.

## Reporting a vulnerability in Cubic

If you find a security issue in Cubic itself, for example a way for a target application
to execute code in the Cubic process beyond the app factory you asked it to load, please
report it privately to the maintainers instead of opening a public issue. Include a
minimal reproduction. We aim to acknowledge reports within 5 working days.
