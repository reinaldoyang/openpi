import dataclasses
from types import SimpleNamespace

import jax

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_create_torch_dataset_uses_local_root(monkeypatch):
    calls = {}

    class FakeMetadata:
        fps = 15

        def __init__(self, repo_id, *, root=None):
            calls["metadata"] = (repo_id, root)

    class FakeLeRobotDataset:
        def __init__(self, repo_id, *, root=None, delta_timestamps=None, video_backend=None):
            calls["dataset"] = (repo_id, root, delta_timestamps, video_backend)

    monkeypatch.setattr(
        _data_loader,
        "lerobot_dataset",
        SimpleNamespace(LeRobotDatasetMetadata=FakeMetadata, LeRobotDataset=FakeLeRobotDataset),
    )
    data_config = _config.DataConfig(
        repo_id="local/example",
        root="/datasets/example",
        action_sequence_keys=("action",),
    )
    model_config = pi0_config.Pi0Config(action_horizon=3)

    _data_loader.create_torch_dataset(data_config, action_horizon=3, model_config=model_config)

    assert calls["metadata"] == ("local/example", "/datasets/example")
    assert calls["dataset"] == (
        "local/example",
        "/datasets/example",
        {"action": [0.0, 1 / 15, 2 / 15]},
        None,
    )


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
