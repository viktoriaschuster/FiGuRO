# FiGuRO

FiGuRO (**F**idelity **Gu**ided **R**ank **O**ptimization) is a PyTorch module for adaptive rank reduction in latent spaces and decoupling of shared and modality-specific information.

## Release Status

This is the absolute minimal public release of FiGuRO at this stage.

It currently includes:
- a cleaned-up base FiGuRO implementation
- one working example notebook (`examples/figuro_swiss_roll_example.ipynb`)

The full code for reproducing the paper experiments will be added in a follow-up release.

Minimal dependency philosophy:
- one core dependency list in `requirements.txt`

## Quick Usage

```python
from figuro import FiGuRO

figuro = FiGuRO(
    n_modalities=1,
    latent_dims=[3],
    decomp_dims=[3],
    rank_reduction_frequency=10,
    distortion_metric="R2",
)
```

Then in training:

```python
figuro.initialize_tracking(epochs=200, warmup=20)

for epoch in range(200):
    reconstructed = figuro([latent])[0]
    # ... optimize your reconstruction loss ...
    figuro.step(epoch, [reconstructed], [target])
```

## Multi-Modal Example

```python
from figuro import FiGuRO

# Example: 2 modalities with latent dims 128 and 64
# decomp_dims format for multi-modal FiGuRO is:
# [shared_dim, mod1_specific_dim, mod2_specific_dim, ...]
figuro_mm = FiGuRO(
    n_modalities=2,
    latent_dims=[128, 64],
    decomp_dims=[32, 32, 32],
    rank_reduction_frequency=10,
    distortion_metric="R2",
)

figuro_mm.initialize_tracking(epochs=300, warmup=20)

for epoch in range(300):
    # z1 and z2 are latent embeddings from two encoders
    recon_list = figuro_mm([z1, z2])

    # recon_list = [recon_mod1, recon_mod2]
    loss = loss_fn1(recon_list[0], z1_target) + loss_fn2(recon_list[1], z2_target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Provide reconstructed tensors and their modality targets
    figuro_mm.step(epoch, recon_list, [z1_target, z2_target])
```

## Running The Swiss Roll Example

Notebook: `examples/figuro_swiss_roll_example.ipynb`

Install dependencies (if necessary) and run the notebook from the `examples` folder.

## Project Structure

```text
src/figuro.py
examples/figuro_swiss_roll_example.ipynb
```

## Citation

If you use FiGuRO in your work, please cite your ICML/arXiv paper once the final bib entry is available.
