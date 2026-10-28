"""Netze: Nachbarlisten, Regularität, Symmetrie, Randsicht."""

import pytest

import kvw_graphs as GR


def symmetric(nb):
    sets = [set(row.tolist()) for row in nb]
    return all(i in sets[j] for i in range(len(nb)) for j in sets[i])


def test_complete_graph_lists_all_others_sorted():
    nb = GR.complete_nb(4)
    assert nb.tolist() == [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]


@pytest.mark.parametrize("n,k", [(5, 1), (7, 2), (9, 4), (12, 3)])
def test_ring_is_regular_symmetric_without_self_or_duplicates(n, k):
    nb = GR.ring_nb(n, k)
    assert nb.shape == (n, 2 * k) and symmetric(nb)
    for i, row in enumerate(nb):
        assert i not in row and len(set(row.tolist())) == 2 * k


def test_ring_of_five_with_two_per_side_is_the_complete_graph():
    assert {tuple(sorted(r)) for r in GR.ring_nb(5, 2).tolist()} == {tuple(sorted(r)) for r in GR.complete_nb(5).tolist()}


def test_ring_neighbours_by_hand():
    """n = 6, k = 1: Betrieb 0 sieht 5 und 1; Betrieb 3 sieht 2 und 4."""
    nb = GR.ring_nb(6, 1)
    assert sorted(nb[0].tolist()) == [1, 5] and sorted(nb[3].tolist()) == [2, 4]


def test_ring_rejects_too_small_n_and_k():
    with pytest.raises(ValueError):
        GR.ring_nb(4, 2)          # 4 < 2·2 + 1
    with pytest.raises(ValueError):
        GR.ring_nb(5, 0)
    GR.ring_nb(5, 2)              # n = 2k + 1 ist erlaubt (inklusive Grenze)


@pytest.mark.parametrize("lx,ly,kind,d", [(3, 3, "vn", 4), (4, 5, "vn", 4), (3, 3, "moore", 8), (3, 4, "moore", 8), (5, 5, "moore", 8)])
def test_torus_is_regular_symmetric_without_self_or_duplicates(lx, ly, kind, d):
    nb = GR.torus_nb(lx, ly, kind)
    assert nb.shape == (lx * ly, d) and symmetric(nb)
    for i, row in enumerate(nb):
        assert i not in row and len(set(row.tolist())) == d


def test_torus_neighbours_by_hand():
    """4 × 4 Torus: Betrieb 0 = (0, 0) sieht (1, 0) = 4, (3, 0) = 12, (0, 1) = 1, (0, 3) = 3."""
    assert sorted(GR.torus_nb(4, 4, "vn")[0].tolist()) == [1, 3, 4, 12]
    assert sorted(GR.torus_nb(4, 4, "moore")[0].tolist()) == [1, 3, 4, 5, 7, 12, 13, 15]


def test_torus_rejects_side_below_three_and_unknown_kind():
    with pytest.raises(ValueError):
        GR.torus_nb(2, 5, "vn")
    with pytest.raises(ValueError):
        GR.torus_nb(3, 3, "hex")


def test_edge_view_by_kind():
    assert GR.edge_view("ring", 3) == 3 and GR.edge_view("vn") == 1 and GR.edge_view("moore") == 3 and GR.edge_view("complete") is None
    with pytest.raises(ValueError):
        GR.edge_view("ring")
    with pytest.raises(ValueError):
        GR.edge_view("kreis")


def test_network_nb_dispatch_and_degree_agree():
    for kind, n, k in (("complete", 6, None), ("ring", 9, 2), ("vn", 16, None), ("moore", 25, None)):
        nb = GR.network_nb(kind, n, k)
        assert nb.shape == (n, GR.degree(kind, n, k))


def test_side_of_requires_a_square_from_nine():
    assert GR.side_of("vn", 144) == 12 and GR.side_of("vn", 9) == 3
    for bad in (10, 4, 8):
        with pytest.raises(ValueError):
            GR.side_of("vn", bad)


def test_edge_view_matches_a_real_straight_edge_on_the_torus():
    """Gerade Kante an einem A-Block (alle Betriebe mit x < 3 auf einem 8×8-Torus): ein B-Betrieb in Spalte x = 3 sieht 1 (vn) bzw. 3 (moore) A-Partner."""
    for kind in ("vn", "moore"):
        nb = GR.torus_nb(8, 8, kind)
        a = {x * 8 + y for x in range(3) for y in range(8)}
        sights = {sum(1 for j in nb[3 * 8 + y] if j in a) for y in range(8)}
        assert sights == {GR.edge_view(kind)}


def test_edge_view_matches_a_real_edge_on_the_ring():
    nb = GR.ring_nb(30, 3)
    block = set(range(0, 10))
    assert sum(1 for j in nb[10] if j in block) == 3 == GR.edge_view("ring", 3)
