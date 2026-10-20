from pathlib import Path

from router.corpus.collect import SkillFile, detect_license, match_skill, parse_skill_md


def test_parse_valid_frontmatter() -> None:
    fm, body = parse_skill_md("---\nname: pdf\ndescription: Edit PDFs.\n---\n# Body\n")
    assert fm == {"name": "pdf", "description": "Edit PDFs."}
    assert body == "# Body\n"


def test_parse_falls_back_on_unquoted_colon() -> None:
    fm, _ = parse_skill_md("---\nname: x\ndescription: Does a: b and c: d\n---\n")
    assert fm["description"] == "Does a: b and c: d"


def test_parse_without_frontmatter() -> None:
    fm, body = parse_skill_md("# Just a body")
    assert fm == {}
    assert body == "# Just a body"


def _file(path: str, name: str | None) -> SkillFile:
    return SkillFile(Path(path), {"name": name} if name else {}, "")


def test_match_prefers_frontmatter_name_then_folder() -> None:
    files = [_file("skills/other/SKILL.md", "pdf"), _file("skills/pdf/SKILL.md", None)]
    match = match_skill("pdf", files)
    assert match is not None
    assert match.path == Path("skills/other/SKILL.md")
    folder_match = match_skill("pdf", [files[1]])
    assert folder_match is not None
    assert folder_match.path == Path("skills/pdf/SKILL.md")


def test_match_takes_shallowest_copy() -> None:
    files = [_file("plugins/a/skills/pdf/SKILL.md", "pdf"), _file("skills/pdf/SKILL.md", "pdf")]
    match = match_skill("pdf", files)
    assert match is not None
    assert match.path == Path("skills/pdf/SKILL.md")


def test_match_none() -> None:
    assert match_skill("pdf", [_file("skills/xlsx/SKILL.md", "xlsx")]) is None


def test_detect_license(tmp_path: Path) -> None:
    assert detect_license(tmp_path) == "unknown"
    (tmp_path / "LICENSE").write_text("MIT License\n\nCopyright (c) 2025")
    assert detect_license(tmp_path) == "MIT"
    (tmp_path / "LICENSE").write_text("Some custom terms")
    assert detect_license(tmp_path) == "other"
