from statarb.config import deep_merge, load_config
from statarb.paths import REPO_ROOT, project_paths
from statarb.utils import config_hash


def test_base_config_matches_plan():
    cfg = load_config()
    assert cfg["dates"]["download_start"] == "2008-01-01"
    assert cfg["dates"]["data_end"] == "2026-08-31"
    assert cfg["data"]["membership_sha"] == "a2430f2af0c79ddf0748e91de11bdeb1616ab5a7"
    assert cfg["costs"] == {"tc_bps": 5.0, "short_bps": 1.0}
    assert cfg["universe"]["size"] == 200
    assert cfg["model"]["lookback"] == 30
    names = [name for group in cfg["features"]["groups"].values() for name in group]
    assert len(names) == 15 and len(set(names)) == 15


def test_override_file_deep_merges(tmp_path):
    override = tmp_path / "frozen.yaml"
    override.write_text("model:\n  K: 20\ntrain:\n  lambda_var: 10.0\n")
    cfg = load_config(override)
    assert cfg["model"]["K"] == 20
    assert cfg["model"]["d"] == 32  # siblings of an overridden key survive
    assert cfg["train"]["lambda_var"] == 10.0


def test_deep_merge_leaves_inputs_untouched():
    base = {"a": {"b": 1, "c": 2}}
    assert deep_merge(base, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


def test_config_hash_is_stable_and_sensitive():
    cfg = load_config()
    assert config_hash(cfg) == config_hash(load_config())
    assert config_hash(deep_merge(cfg, {"model": {"K": 5}})) != config_hash(cfg)


def test_paths_resolve_under_repo_root():
    paths = project_paths(load_config())
    assert paths.raw == REPO_ROOT / "data" / "raw"
    assert paths.reports == REPO_ROOT / "results" / "reports"
