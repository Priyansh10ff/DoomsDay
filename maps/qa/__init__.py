"""Map QA tooling (structure, solvability search, engine load check).

NOT part of the agent. Nothing under agent/, env/ or the inference path may import
this package; tests/test_maps_isolation.py scans those trees and fails if one does.
The graph searches here only certify that a generated map is completable; they never
choose an action.
"""
