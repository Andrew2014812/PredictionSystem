import re

from src.ui.i18n import translate
from src.ui.i18n_keys import all_keys
from src.ui.i18n_uk import UK


def test_every_interface_text_has_a_ukrainian_translation():
    missing = sorted(k for k in all_keys() if k not in UK)
    assert not missing, missing


def test_placeholders_are_preserved():
    for en, uk in UK.items():
        assert set(re.findall(r"{\w+}", en)) == set(re.findall(r"{\w+}", uk)), en


def test_translate_switches_language_and_formats():
    assert translate("Matches", "en") == "Matches"
    assert translate("Matches", "uk") == "Матчі"
    assert translate("{n} matches", "uk", n=5) == "Матчів: 5"
    assert translate("Unknown text", "uk") == "Unknown text"            # falls back to English


def test_theme_palettes_define_the_same_variables():
    from src.ui.theme import CSS, LIGHT_WIDGETS, PALETTES
    assert PALETTES["dark"].keys() == PALETTES["light"].keys()
    used = set(re.findall(r"var\(--([\w-]+)\)", CSS + LIGHT_WIDGETS))
    assert used <= set(PALETTES["dark"]), used - set(PALETTES["dark"])
