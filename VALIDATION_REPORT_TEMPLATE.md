# Protected Model Lab feasibility report

## Executive decision

**Decision:** Feasible / Conditionally feasible / Not currently feasible

**One-paragraph basis:**

<!-- State what was tested, the strongest evidence, the main limitation, and whether the result transfers only to these exact assets. -->

## Scope and authorization

- Checkpoint architecture:
- Adapter format and count:
- Workflow used:
- Assets confirmed owned/authorized: yes / no
- Source assets modified: no / explain
- Sensitive hashes and identities kept local: yes / no

## Environment

| Item | Measured value |
|---|---|
| GPU / VRAM | |
| System RAM | |
| OS | |
| Python | |
| PyTorch / CUDA | |
| ComfyUI revision | |
| Free disk before/after | |

## Compatibility and correctness gates

| Gate | Result | Evidence/report |
|---|---|---|
| Adapter tensors accounted for | pass/fail | |
| Unmatched or ambiguous keys | count | |
| Unit/mock tests | pass/fail | |
| Carrier keys and shapes reopen correctly | pass/fail | |
| Float32 reconstruction tolerance | value/pass/fail | |
| BF16/storage reconstruction error | value | |
| Correct carrier accepted | pass/fail | |
| Mismatched carrier rejected | pass/fail | |
| ComfyUI nodes load and sample | pass/fail | |

## Candidate results

| Candidate | Selection rationale | Prompts | Carrier gap | Median recovery | Prompts ≥0.85 | Restored similarity | Core size | Runtime overhead |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| coefficient control | implementation baseline | | | | | | | |
| non-coefficient candidate 1 | | | | | | | | |
| non-coefficient candidate 2 | | | | | | | | |

Record metric definitions and whether each value is pixel-, SSIM-, LPIPS-, CLIP-, DINO-, latent-, or model-output-based. Do not compare unlike distances without explaining normalization.

## Best tested operating point

- Split method:
- Exact block/rank selection:
- Why it was selected:
- Evidence that Carrier remains functional:
- Evidence that Carrier loses target behavior:
- Evidence that Core restores it:
- Is this selection exhaustive? yes / no
- Expected architecture/adapter specificity:

Do not call a block range universal. State explicitly whether it came from traversal, a staged candidate search, or one manual candidate.

## Human review

- Number of contact sheets reviewed:
- Was review blinded or labeled:
- Teacher/Restored judged equivalent or acceptably close on:
- Carrier judged noticeably weaker on:
- Failure prompts or modes:

## Performance and storage

- Teacher load/generation time:
- Carrier load/generation time:
- Restored load/generation time:
- Peak VRAM and system RAM:
- Checkpoint size:
- Carrier size:
- Core size and percentage of checkpoint:

## Failures, unsupported features, and evidence gaps

- Failed tests or commands:
- Unsupported tensor/module types:
- Metrics not available:
- Prompts or settings not covered:
- Threats to validity:

## Feasibility assessment

Answer each directly:

1. Does coefficient splitting reconstruct correctly?
2. Does at least one non-coefficient split reconstruct correctly?
3. Does Carrier-only lose a meaningful and repeatable portion of the target behavior?
4. Does Carrier + Core recover at least 90% of the measured gap across the prompt set?
5. Is the runtime/storage overhead acceptable?
6. Does the approach raise the effort required for casual standard-LoRA reuse?
7. What does this experiment **not** protect against?

## Recommendations

### Must fix before another claim

1.

### Worth testing next

1.

### Defer until static evidence justifies it

1. Dynamic nonlinear core training.
2. Native C++/CUDA backend or obfuscation.

## Handoff inventory

- Final report path:
- Mapping report path:
- Reconstruction report path:
- Evaluation report path:
- Performance report path:
- Contact-sheet directory:
- Generated carrier/core manifests:
- Private artifacts intentionally not shared:
