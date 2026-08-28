# Next steps and dynamic-core feasibility

## Immediate static follow-up

1. Expand evaluation to about 30 diverse portrait, half-body, full-body, lighting, pose, and background prompts.
2. Capture peak CUDA allocation and process RSS around each run.
3. Test one hybrid candidate: whole private identity blocks plus a small top-singular rank elsewhere.
4. Add sharded input/output after the proven monolithic path.
5. Replace whole-core reads with indexed payload slices if cores grow materially.

## Dynamic core proposal

The future core should alter activations rather than represent one mergeable static delta:

```text
delta_h = sigma_gate(sigma)
          * block_gate(layer_id)
          * condition_gate(context)
          * B_private(SiLU(A_private(RMSNorm(h))))
```

This is a design only; no dynamic core has been trained.

### Capture points and losses

Capture inputs/outputs around attention output projections and MLP down projections in private blocks, together with timestep/sigma and compact conditioning summaries. Use identical latent/noise states for teacher and carrier. Train primarily against teacher-minus-carrier activation residuals using normalized Huber/MSE, plus final noise-prediction loss, residual cosine loss, smooth sigma-gate regularization, and sparse block-gate regularization.

### Memory and freezing

- Freeze carrier, VAE, and text encoder.
- Train only normalization, low-rank factors, and small gates.
- Cache chunked local teacher residual targets where practical.
- Do not keep full teacher and student resident on the GPU simultaneously.
- Initialize from whichever static block candidate wins the expanded multi-prompt search; do not assume the POC's 0–13 range transfers.

### Initialization and evaluation

Initialize A/B from static private factors, keep the activation near linear initially, and initialize sigma/block gates to one. Add conditioning gates only after sigma-only gating is stable. Require over 90% recovery across the full prompt set, functional carrier-only images with measurable identity loss, acceptable runtime overhead, and evidence that no one static merged delta reproduces the dynamic output.

### Risks and minimal plan

Risks include prompt-set overfitting, early-denoising instability, conditioning shortcuts, and conflicts with compile/offload paths. Start with ModelPatcher-supported activation capture on two blocks, train sigma gating, verify fixed-seed model-output recovery, expand block coverage, then add conditioning gates. Keep the backend behind the existing `.spcore` abstraction. This raises casual reuse cost but is not cryptographic protection.
