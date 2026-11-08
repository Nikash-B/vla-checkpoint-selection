# Changed from upstream 7f48937: the StateValue re-export is removed with state_value.py.
"""Value networks used by the critic: the state-action ensemble."""

from jaxrl2.networks.values.state_action_ensemble import StateActionEnsemble

__all__ = ["StateActionEnsemble"]
