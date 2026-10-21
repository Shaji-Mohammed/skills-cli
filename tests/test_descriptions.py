from router.corpus.descriptions import description_for

REWRITTEN = {"description_original": "Use when X.", "description": "Does Y.", "rewritten": True}
UNTOUCHED = {"description_original": "Does Z.", "description": "Does Z.", "rewritten": False}


def test_original_source_always_returns_author_text() -> None:
    assert description_for(REWRITTEN, "original") == "Use when X."
    assert description_for(UNTOUCHED, "original") == "Does Z."


def test_rewritten_source_falls_back_to_original() -> None:
    assert description_for(REWRITTEN, "rewritten") == "Does Y."
    assert description_for(UNTOUCHED, "rewritten") == "Does Z."
