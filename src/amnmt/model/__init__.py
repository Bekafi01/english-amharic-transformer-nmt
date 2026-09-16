"""L3 — model: the Transformer, built from torch.nn primitives.

Pure compute: no file I/O, no YAML, no tokenizer. Receives tensors, returns tensors.
May import: core (for types only), torch.
"""
