# Dynamic Kanerva Machines

This is a self-contained memory module for the Dynamic Kanerva Machine, as
reported in the NIPS 2018 paper: Learning Attractor Dynamics for Generative
Memory.

Disclaimer: This is not an official Google product.

## Usage Example

A memory used in the Omniglot experiment in the paper can be
set-up as follows:

```python
from memory import Kanervamemory

# Initialisation
memory = KanervaMemory(code_size=100, memory_size=32)
prior_memory = memory.get_prior_state(batch_size)

# Update memory posterior
posterior_memory, _, _, _ = memory.update_state(z_episode, prior_memory)

# Read from the memory using cues z_q
read_z, dkl_w = memory.read_with_z(z_q, posterior_memory)

# The KL-divergence between posterior and prior memory, used in training
dkl_M = memory.get_dkl_total(posterior_memory)
```

`z_episode` is a `[episode_size, batch_size, code_size]` tensor as the
embedding of an episode. `dkl_w` is the KL-divergence between the `w` used reading.

