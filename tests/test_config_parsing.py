from RL_Framework.config import (
    AsyncRLConfig,
    HardwareConfig,
    HeterogeneousInstanceConfig,
)


def _hetero_config() -> AsyncRLConfig:
    config = AsyncRLConfig(model_path="/tmp/model")
    hetero = config.heterogeneous_rollout
    hetero.enabled = True
    hetero.instances = [
        HeterogeneousInstanceConfig(tp=1),
        HeterogeneousInstanceConfig(tp=4),
    ]
    return config


def test_kv_capacity_autofill_tracks_either_load_metric():
    """cmlfq ranks in-bucket with its own metric, so it must trigger too."""
    config = _hetero_config()
    sched = config.heterogeneous_rollout.scheduling
    assert (sched.load_metric, sched.cmlfq_load_metric) == ("requests", "tokens")

    # Neither metric is capacity-normalized: no estimate needed.
    config._maybe_estimate_kv_capacities()
    assert sched.kv_capacity_tokens_by_tp == {}

    # cmlfq alone asking for kv_tokens is enough to need the table.
    sched.cmlfq_load_metric = "kv_tokens"
    config._maybe_estimate_kv_capacities()
    assert sorted(sched.kv_capacity_tokens_by_tp) == [1, 4]
    assert all(v > 0 for v in sched.kv_capacity_tokens_by_tp.values())


def test_kv_capacity_autofill_respects_an_explicit_table():
    config = _hetero_config()
    sched = config.heterogeneous_rollout.scheduling
    sched.cmlfq_load_metric = "kv_tokens"
    sched.kv_capacity_tokens_by_tp = {1: 123}
    config._maybe_estimate_kv_capacities()
    assert sched.kv_capacity_tokens_by_tp == {1: 123}


def test_from_dict_parses_numeric_strings_and_ignores_unknown_keys():
    config = HardwareConfig.from_dict(
        {
            "gpus_per_node": "8",
            "mem_capacity": "80500000000.5",
            "tp_comm_overhead": "5e-6",
            "unknown_key": "ignored",
        }
    )

    assert config.gpus_per_node == 8
    assert isinstance(config.gpus_per_node, int)
    assert config.mem_capacity == 80500000000.5
    assert isinstance(config.mem_capacity, float)
    assert config.tp_comm_overhead == 5e-6
    assert isinstance(config.tp_comm_overhead, float)
    assert not hasattr(config, "unknown_key")
