"""Szenario: Netz aus den Reglern (Rundung, Begrenzung, Schwelle, Randsicht) und die SplitMix64-Kopie samt Referenzwerten."""

import ast
from pathlib import Path

import numpy as np
import pytest

import kvw_constants as C
import kvw_scenario as S
from kvw_rng import SplitMix64, uniform_block

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "kvw_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        assert np.array_equal(uniform_block(seed, 100), np.array([rng.uniform() for _ in range(100)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "kvw_rng.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {"math", "numpy"}


def test_shifting_the_seed_by_blocks_continues_the_stream():
    """Grundlage der Blockbildung in kvw_sim: Seed + c·γ liefert den Strom ab Position c."""
    import kvw_sim as SI
    full = uniform_block(123, 3 * 2 * SI.CHUNK)
    second = uniform_block(SI._block_seed(123, 1), 3 * SI.CHUNK)
    assert np.array_equal(full[3 * SI.CHUNK:], second)


@pytest.mark.parametrize("n,side", [(4, 3), (9, 3), (12, 3), (13, 4), (144, 12), (150, 12), (1000, 31), (50, 7)])
def test_grid_side_rounds_to_the_nearest_square_within_limits(n, side):
    assert S.grid_side(n) == side


def test_effective_n_only_changes_for_grids():
    assert S.effective_n("complete", 50) == 50 and S.effective_n("ring", 50) == 50
    assert S.effective_n("vn", 144) == 144 and S.effective_n("moore", 10) == 9 and S.effective_n("vn", 1000) == 961


def test_effective_k_is_limited_by_n_and_only_for_rings():
    assert S.effective_k("ring", 4, 14) == 1 and S.effective_k("ring", 30, 14) == 14 and S.effective_k("ring", 29, 14) == 14 and S.effective_k("ring", 28, 14) == 13
    assert S.effective_k("complete", 30, 5) is None and S.effective_k("vn", 16, 5) is None


def test_resolve_network_for_each_preset_is_what_the_help_text_says():
    p = C.PRESETS
    border = S.resolve_network(**{"network": p["Offene Börse (Vollgraph, n = 20)"]["network"], "n": 20, "k": 2, "q": 0.4})
    assert (border.kind, border.n, border.d, border.thr, border.edge, border.spread) == ("complete", 20, 19, 8, None, None)
    corridor = S.resolve_network(C.NETWORKS[1], 80, 2, 0.4)
    assert (corridor.d, corridor.thr, corridor.edge, corridor.spread) == (4, 2, 2, 0)           # "2 von 4 Partnern", "Spielraum 0"
    regions = S.resolve_network(C.NETWORKS[2], 144, 2, 0.25)
    assert (regions.n, regions.d, regions.thr, regions.edge, regions.spread) == (144, 4, 1, 1, 0)
    near = S.resolve_network(C.NETWORKS[1], 30, 14, 0.2)
    assert (near.d, near.n) == (28, 30)                                                         # "28 der 29 anderen"


def test_spread_is_negative_exactly_where_a_straight_edge_does_not_advance():
    assert S.resolve_network(C.NETWORKS[3], 144, 2, 0.4).spread == -1          # 8 Nachbarn, thr 4, Randsicht 3
    assert S.resolve_network(C.NETWORKS[2], 144, 2, 0.4).spread == -1          # 4 Nachbarn, thr 2, Randsicht 1
    assert S.resolve_network(C.NETWORKS[3], 144, 2, 0.3).spread == 0           # 8 Nachbarn, thr 3, Randsicht 3
    for q in (0.1, 0.2, 0.3, 0.4, 0.45):
        for k in range(1, 15):
            assert S.resolve_network(C.NETWORKS[1], 200, k, q).spread >= 0, (q, k)   # im Ring nie negativ, solange q < 1/2


def test_share_is_thr_over_d():
    spec = S.resolve_network(C.NETWORKS[0], 20, 2, 0.4)
    assert spec.share == pytest.approx(8 / 19)
