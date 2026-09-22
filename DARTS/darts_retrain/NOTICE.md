Parts of `operations.py`, `model.py`, and the protocol/data definitions are
adapted from the official DARTS implementation by Hanxiao Liu:

https://github.com/quark0/darts

The upstream project is licensed under the Apache License 2.0. The port keeps
the evaluation network and training protocol intact while updating obsolete
PyTorch APIs and adding NAS-Bench-301 22-token input and resumable checkpoints.
