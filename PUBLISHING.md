# Publishing checklist

The repository is designed to publish code and templates, not model material or local evidence.

Before creating a GitHub repository:

1. Keep the included `LICENSE` file. The repository currently reserves all rights and is not distributed under an open-source license.
2. Run `python scripts/check_publication.py`.
3. Inspect `git status --ignored --short` and confirm all local config, reports, images, carriers, cores, checkpoints, and adapters are ignored.
4. Review `README.md`, `AGENTS.md`, and the bundled skill for private names, paths, prompts, or model hashes.
5. Commit code, tests, templates, and documentation only.

Do not use Git LFS as a way to publish the generated carriers or private cores unless the model owners and every upstream license explicitly permit that distribution.

Publishing a repository publicly does not make it open source. Direct physical or private delivery by the copyright holder grants that recipient the limited private-validation permission described in `LICENSE`; it does not grant redistribution, public-fork, sublicensing, or commercial rights.
