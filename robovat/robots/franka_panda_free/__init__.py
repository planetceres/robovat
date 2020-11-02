from .franka_panda_sim_free import FrankaPandaSimFree


def factory(simulator=None, config=None):
    assert simulator is not None
        # Always use the default real-world Sawyer configuration.
    return FrankaPandaSimFree(simulator=simulator, config=config)