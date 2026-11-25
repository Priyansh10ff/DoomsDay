"""Our own synchronous recurrent PPO (ADR 0004).

This package __init__ imports nothing, so env worker processes can import
`train.vec_env` without pulling in torch.
"""
