"""Die Entscheidungsregel: Schwelle (ganzzahlig, mit Toleranz), beste Antwort, Wahrscheinlichkeit, Spiegelung."""

import pytest

import kvw_model as M


@pytest.mark.parametrize("q,d,thr", [(0.4, 2, 1), (0.4, 4, 2), (0.4, 5, 2), (0.4, 8, 4), (0.3, 8, 3), (0.25, 4, 1), (0.45, 4, 2), (0.1, 2, 1), (0.4, 29, 12), (0.4, 999, 400)])
def test_threshold_is_the_ceiling_of_q_times_d(q, d, thr):
    assert M.threshold(q, d) == thr


def test_threshold_tolerates_float_residue_at_exact_products():
    """In Gleitkomma ist 0,28·25 = 7,000000000000001, 0,14·50 = 7,000000000000001 und 0,28·50 = 14,000000000000002: ohne Toleranz wäre die Schwelle 8 bzw. 15 statt 7 bzw. 14."""
    assert 0.28 * 25 > 7.0 and M.threshold(0.28, 25) == 7
    assert 0.14 * 50 > 7.0 and M.threshold(0.14, 50) == 7
    assert 0.28 * 50 > 14.0 and M.threshold(0.28, 50) == 14
    assert M.threshold(0.35, 20) == 7 and M.threshold(0.45, 20) == 9


def test_threshold_rejects_q_outside_and_empty_neighbourhood():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            M.threshold(bad, 4)
    with pytest.raises(ValueError):
        M.threshold(0.4, 0)


def test_goal_is_half_rounded_up():
    assert [M.goal(n) for n in (2, 3, 4, 5, 20, 21)] == [1, 2, 2, 3, 10, 11]
    with pytest.raises(ValueError):
        M.goal(1)


def test_best_reply_is_inclusive_at_the_threshold():
    assert M.best_reply_is_a(2, 2) and not M.best_reply_is_a(1, 2) and M.best_reply_is_a(5, 2)


def test_prob_a_combines_best_reply_and_error():
    """Beste Antwort A: (1 − ε) + ε/2 = 1 − ε/2; beste Antwort B: ε/2."""
    assert M.prob_a(3, 2, 0.2) == pytest.approx(0.9) and M.prob_a(1, 2, 0.2) == pytest.approx(0.1)
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError):
            M.prob_a(1, 1, bad)


def test_mirror_threshold_and_target():
    """B ist beste Antwort genau dann, wenn weniger als thr Partner A führen, also mindestens d − thr + 1 Partner B. n = 20, thr = 8, d = 19: thr_B = 12, Ziel 14."""
    assert M.mirror_threshold(19, 8) == 12 and M.mirror_target(20, 12) == 14
    assert M.mirror_target(5, 4) == 5          # höchstens n
    for count_a in range(0, 20):
        count_b = 19 - count_a
        assert (not M.best_reply_is_a(count_a, 8)) == (count_b >= M.mirror_threshold(19, 8))
