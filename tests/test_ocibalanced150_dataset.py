import json
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[1]
CSV = REPO / "data" / "two_step_gillan_2016_ocibalanced150.csv"
OLD_CSV = REPO / "data" / "two_step_gillan_2016_ocibalanced.csv"
MANIFEST = REPO / "data" / "ocd" / "ocibalanced150_manifest.json"

pytestmark = pytest.mark.skipif(not CSV.exists(), reason="dataset not built yet")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(CSV)


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_participants_0_to_149(df):
    pids = sorted(df["participant"].unique())
    assert pids == list(range(150))


def test_schema_matches_old_dataset(df):
    old_cols = list(pd.read_csv(OLD_CSV, nrows=1).columns)
    assert list(df.columns) == old_cols


def test_manifest_tertiles_and_splits(manifest):
    rows = manifest["participants"]
    assert len(rows) == 150
    tert = pd.Series([r["tertile"] for r in rows]).value_counts()
    assert tert.to_dict() == {"Low": 50, "Medium": 50, "High": 50}
    split = pd.Series([r["split"] for r in rows]).value_counts()
    assert split.to_dict() == {"train": 50, "validation": 50, "test": 50}
    # stratified dealing: every (split, tertile) cell is 16 or 17
    cells = pd.DataFrame(rows).groupby(["split", "tertile"]).size()
    assert set(cells.tolist()) <= {16, 17}
    # contiguous index blocks
    by_split = {s: sorted(r["participant"] for r in rows if r["split"] == s)
                for s in ["train", "validation", "test"]}
    assert by_split["train"] == list(range(0, 50))
    assert by_split["validation"] == list(range(50, 100))
    assert by_split["test"] == list(range(100, 150))


def test_no_serial_oci_confound(manifest):
    rows = sorted(manifest["participants"], key=lambda r: r["participant"])
    rho, _ = spearmanr([r["participant"] for r in rows],
                       [r["oci_total"] for r in rows])
    assert abs(rho) < 0.15


def test_prompt_five_stratified(manifest):
    prompt = [r for r in manifest["participants"] if r["prompt"]]
    assert len(prompt) == 5
    assert all(r["split"] == "train" for r in prompt)
    tert = pd.Series([r["tertile"] for r in prompt]).value_counts().to_dict()
    assert tert == {"Low": 2, "Medium": 2, "High": 1}


def test_oci_normalized_and_trials_zero_based(df):
    assert df["oci"].between(0, 1).all()
    assert (df.groupby("participant")["trial"].min() == 0).all()


def test_baseline_bic_present(df):
    assert "baseline_bic" in df.columns
    assert df["baseline_bic"].notna().all()
