# Split selection

## No universal block range

Block numbering is local to the detected architecture. Adapter energy, semantic effect, residual pathways, text-fusion modules, and quantization differ across checkpoints. A range that worked for one Krea2 identity LoKr cannot be transferred to another model or adapter without measurement.

Run:

```bash
"$COMFYUI_PYTHON" scripts/analyze_delta.py config/experiment.yaml
"$COMFYUI_PYTHON" scripts/plan_block_search.py --config config/experiment.yaml
```

The planner proposes a small set of contiguous controls and an energy-ranked baseline. Energy is a search heuristic, not evidence that a block owns identity or style.

## Staged search

1. Compare two broad contiguous halves and, if needed, four quarters.
2. Use several fixed prompts to identify which group creates a meaningful carrier gap without breaking generation.
3. Refine only the most informative group into smaller contiguous windows.
4. Test a low-rank top-singular control and a tail-singular control.
5. Consider a hybrid only after the block and direction controls are understood.

Keep the candidate count small and reuse cached factor analysis.

## Selection criteria

Prefer a candidate where:

- carrier-only images remain valid;
- carrier-only loses a noticeable target behavior across prompts;
- restored recovery exceeds 90% by more than one metric or clear human review;
- the core is materially smaller than the checkpoint;
- runtime overhead is acceptable;
- the result is not explained only by coefficient scaling.

If no candidate satisfies these together, report that the static split is mechanically feasible but the tested protection operating point is not yet useful.
