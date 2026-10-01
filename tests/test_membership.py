import pandas as pd
import pytest

from statarb.config import load_config
from statarb.data.membership import additions, build_spells, load_snapshots, members_on, membership_path
from statarb.paths import project_paths


@pytest.fixture
def snapshots(tmp_path):
    path = tmp_path / "membership.csv"
    path.write_text('date,tickers\n2020-01-01,"A,B"\n2020-02-01,"A,C"\n2020-03-01,"A,B,C"\n')
    return load_snapshots(path)


def test_members_on_uses_latest_snapshot(snapshots):
    assert members_on(snapshots, "2019-12-31") == frozenset()
    assert members_on(snapshots, "2020-01-15") == {"A", "B"}
    assert members_on(snapshots, "2020-02-01") == {"A", "C"}


def test_spells_and_reentry(snapshots):
    spells = build_spells(snapshots)
    b = spells[spells["member_ticker"] == "B"]
    assert list(b["spell_start"]) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-01")]
    assert b["spell_end"].iloc[0] == pd.Timestamp("2020-02-01") and pd.isna(b["spell_end"].iloc[1])
    assert len(build_spells(snapshots, since="2020-02-15")) == 3  # B's closed first spell drops out


def test_additions(snapshots):
    assert additions(snapshots).values.tolist() == [[pd.Timestamp("2020-02-01"), "C"], [pd.Timestamp("2020-03-01"), "B"]]


@pytest.fixture
def real_snapshots():
    path = membership_path(project_paths(load_config()))
    if not path.exists():
        pytest.skip("membership file not downloaded")
    return load_snapshots(path)


def test_real_fb_meta_rename(real_snapshots):
    assert "FB" in members_on(real_snapshots, "2020-06-01") and "META" not in members_on(real_snapshots, "2020-06-01")
    assert "META" in members_on(real_snapshots, "2022-07-01") and "FB" not in members_on(real_snapshots, "2022-07-01")
    fb = build_spells(real_snapshots).query("member_ticker == 'FB'")
    assert (fb["spell_start"].iloc[0], fb["spell_end"].iloc[0]) == (pd.Timestamp("2013-12-23"), pd.Timestamp("2022-06-09"))


def test_real_snapshot_sizes(real_snapshots):
    sizes = real_snapshots.loc[real_snapshots["date"] >= "2010-01-01", "tickers"].map(len)
    assert sizes.between(495, 510).all()
