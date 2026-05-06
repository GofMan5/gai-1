# Security Policy

GAI-1 is an experimental local LLM training scaffold. Treat checkpoints, data,
and prompts as sensitive unless you have reviewed them.

## Supported Versions

Only the current `main` branch is supported for security fixes.

## Reporting a Vulnerability

Open a private security advisory on GitHub if the repository is hosted there.
If advisories are not enabled yet, contact the maintainers privately before
publishing details.

Please include:

- affected commit or release;
- reproduction steps;
- expected and observed behavior;
- impact and suggested fix, if known.

## Scope

Security issues include:

- arbitrary code execution in training, serving, or dataset loading paths;
- unsafe deserialization or checkpoint loading behavior;
- secret leakage through logs, reports, or generated artifacts;
- dependency vulnerabilities with a practical exploit path.

Model quality, hallucinations, bias, or unsafe generations should be reported
as safety or evaluation issues unless they enable a concrete security exploit.
