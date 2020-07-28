from .sawyer_real import SawyerReal
from .sawyer_sim import SawyerSim


def factory(simulator=None, config=None, root_dir=None):
    if simulator is None:
        # Always use the default real-world Sawyer configuration.
        return SawyerReal(config=None)
    else:
        return SawyerSim(simulator=simulator, config=config, root_dir=root_dir)
