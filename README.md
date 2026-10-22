# Cross-Domain Multi-Agent Trajectory Forecasting

A full training + inference pipeline, written in C++ with LibTorch, that forecasts
multi-agent future trajectories (pedestrians, soccer players) from past positions —
using a shared graph-attention + transformer encoder trained across two very
different domains at once.

![showcase](videos/showcase.gif)

*Left: real pedestrians, ETH/UCY benchmark. Right: a real soccer match, Metrica
Sports tracking data. One model, one checkpoint, forecasting both — white = observed
4s of motion, green dashed = what actually happened, colored fans = the model's 6
predicted futures, red = predicted collision risk between agents.*

## Why this exists

Pedestrian trajectory forecasting (the kind self-driving cars need) and sports
player tracking are structurally the same problem: many interacting agents, each
with a short history of 2D positions, moving under social/physical constraints.
This project tests whether a model can learn transferable multi-agent motion
patterns across that domain gap — and, just as importantly, diagnoses *why* it
might fail to.

## Architecture

- **Spatio-temporal encoder**: a dense Graph Attention Network (edge features =
  relative position, distance, closing speed) per timestep, followed by a
  Transformer over time, pooled at each agent's last valid frame.
- **Multi-modal decoding head**: K=6 learned trajectory anchors, EWTA
  (Evolving Winner-Takes-All) regression loss with k-annealing, mode
  classification + diversity regularization.
- **Domain-adversarial training**: a gradient-reversal domain classifier
  (Ganin & Lempitsky / DANN) pushes the encoder toward domain-invariant
  features, with a sigmoid lambda ramp.
- **Collision risk scoring**: geometric, not learned — minimum pairwise
  distance over the probability-weighted expected future path of each pair
  of agents.

All of it is hand-written in C++ against LibTorch 2.1 + CUDA 11.8, including a
custom `torch::autograd::Function` for gradient reversal, a dense GAT layer,
and the training loop — no Python at train/inference time.

## Honest results (the part that actually matters)

| Setting | Pedestrian ADE | Pedestrian FDE |
|---|---|---|
| Pedestrian-only baseline (trained only on ETH/UCY) | **0.634 m** | **1.056 m** |
| Mixed-domain (50/50 pedestrian+soccer, naive) | ~1.03–1.04 m | ~1.47–1.58 m |

Naively mixing the two domains made pedestrian forecasting **worse**, not
better — a textbook negative-transfer result. Rather than hide this, I dug
into *why*.

### Diagnosing the domain leak

A domain classifier trained adversarially on the shared encoder features
should, if the encoder is doing its job, end up near chance level at telling
pedestrian scenes from soccer scenes apart. Instead I compared it against the
dumbest possible baseline — a single threshold on raw agent count per scene:

| Classifier | Scene-level domain accuracy |
|---|---|
| Trivial agent-count threshold | **96.9%** |
| Learned domain classifier (adversarially trained) | 51.3% (≈ chance) |

The *trivial* count-based rule separates the domains almost perfectly —
soccer scenes simply have far more agents per scene than pedestrian scenes.
That's a structural shortcut the model can exploit that has nothing to do
with genuine motion-pattern differences, and it's the most likely real cause
of the negative transfer above, not an architecture or hyperparameter
failure.

**Fix attempted**: density-matching (subsampling soccer scenes down to
pedestrian-like agent counts before each batch) plus a loss that balances
per-scene-then-per-domain rather than per-agent (so soccer's extra agents
don't dominate the gradient). *[Fill in final epoch-200 pedestrian ADE/FDE
here once that run finishes, compared against the 0.634/1.056 baseline above.]*

### Zero-shot transfer to real driving data (nuScenes)

To test whether any of this transfers to the domain that actually motivated
the project — autonomous-vehicle pedestrian forecasting — I ran the
pedestrian-only checkpoint, completely unmodified, zero-shot on real
[nuScenes](https://www.nuscenes.org/) front-camera driving footage: real
street scenes, real pedestrians, projected through the vehicle's actual
camera calibration and ego pose.

https://github.com/user-attachments/assets/PLACEHOLDER-replace-after-upload

*(GitHub doesn't render `.mp4` files via Markdown image syntax — see the note
right after this table for how to actually get it playable in the README.)*

| Method | ADE (m) | FDE (m) |
|---|---|---|
| Constant velocity (trivial baseline) | **0.375** | **0.780** |
| Model, most-likely mode (zero-shot) | 1.854 | 3.446 |
| Model, best-of-6 (zero-shot) | 1.347 | 2.316 |

The model loses to a trivial straight-line extrapolation by roughly 5x,
zero-shot. This is not a bug — the shape/convention errors that would produce
numbers like this were already caught and fixed earlier during real-data
integration testing. It's a genuine, interpretable negative result: a model
trained only on open pedestrian plazas (ETH/UCY) has never seen the behavior
pedestrians exhibit near a moving vehicle — curb-waiting, crossing in front
of traffic, sudden stops — and that distribution shift is large enough that
naive zero-shot transfer fails outright, even against the weakest possible
baseline.

Combined with the density-leak finding above, the consistent story across
all three experiments is: **multi-agent trajectory forecasting does not
transfer across domains for free.** Whatever is "shared" about the
representation a model like this learns is much shallower than it looks,
and naive domain-general assumptions break down in practice. Making transfer
actually work (domain-specific fine-tuning, explicit behavioral-context
features, or proper domain adaptation beyond a single adversarial classifier)
is the natural next step, not something this architecture gets for free.

## Repo layout

```
include/, src/     C++ source (graph construction, GAT+Transformer encoder,
                   domain-adversarial head, multi-modal decoding head, losses,
                   real-data loaders for ETH/UCY and Metrica Sports, training loop,
                   nuScenes zero-shot inference tool)
checkpoints/       small (<1MB) trained model weights
videos/            rendered results, including the real nuScenes zero-shot run
ns_prepare.py,     nuScenes data preparation and scoring/rendering scripts
ns_render.py
CMakeLists.txt     LibTorch build config
```

## Building

Requires LibTorch 2.1+cu118 and CUDA 11.8.

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_CUDA_ARCHITECTURES=86
make -C build -j$(nproc)
```

## Data

- [ETH/UCY pedestrian benchmark](https://github.com/inhwanbae/GPGraph) (standard preprocessed split)
- [Metrica Sports sample tracking data](https://github.com/metrica-sports/sample-data)
- [nuScenes](https://www.nuscenes.org/) (v1.0-mini used for zero-shot evaluation only, not training)

Not included in this repo (see `.gitignore`) — obtain them separately and point
the training/inference tools at their paths.

## License

MIT
