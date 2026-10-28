"""PDF-Export: erzeugt gültige Bytes für jede Ansicht, Sonderzeichen brechen die Kernschriften nicht."""

import pytest

import kvw_constants as C
import kvw_evaluation as V
import kvw_pdf_export as P


@pytest.mark.parametrize("settings", [V.Settings(), V.Settings(network=C.NETWORKS[1], n=80), V.Settings(direction=C.DIRECTIONS[1], n=40), V.Settings(network=C.NETWORKS[3], n=144, q=0.4)])
def test_pdf_is_created_for_every_kind_of_view(settings):
    view = V.analyse_main(settings)
    data = P.build_pdf(view, [V.verdict_main(view), V.verdict_spread(view.spec)])
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_latin_replaces_special_characters_and_markdown():
    assert P._latin("ε = 0,1 → **fett** ≥ 2 · 3 – ‖") == "eps = 0,1 -> fett >= 2 * 3 - ||"
    assert P._latin("Größe äöü ß") == "Größe äöü ß"
