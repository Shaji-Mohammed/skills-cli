# Corpus curation

Generated from run `results/20261021T185349Z_curation` (git `9d0117be68`).
Corpus version (SHA-256 of `corpus/skills.jsonl`): `983a3f68d688fd310d662f9e1b3e456013c45ef315c2fbf4ca6c05e4d5e34751`.

**1000 leaderboard entries → 500 skills from 90 source repos.**

Source: the skills.sh all-time install leaderboard, snapshotted in `corpus/sources/skills_sh_alltime.json`. Rules run in this order; each row counts only what that rule removed from the skills still remaining. Every removal is listed with its rule in `corpus/removed.jsonl`.

| # | Rule | What it removes | Removed | Remaining |
| --- | --- | --- | ---: | ---: |
| 1 | `not_github_source` | Source isn't a GitHub `owner/repo` (for example a vendor domain), so there's no public SKILL.md to collect or license to check. | 43 | 957 |
| 2 | `clone_failed` | The GitHub repo couldn't be cloned (private, deleted or renamed). | 2 | 955 |
| 3 | `not_in_current_repo` | No SKILL.md in the repo's current HEAD matches the leaderboard entry. Install counts are cumulative, so renamed or deleted skills stay listed. | 113 | 842 |
| 4 | `short_description` | Description shorter than 5 words: too little text to route on. | 19 | 823 |
| 5 | `non_english` | More than 10% of the description's characters are outside Latin scripts. Eval prompts are English. | 21 | 802 |
| 6 | `same_name` | Same skill name as another source's skill. Keep one copy: the official source, then the longer description, then more installs. | 84 | 718 |
| 7 | `near_duplicate` | Description cosine ≥ 0.92 (BAAI/bge-small-en-v1.5) with a skill kept earlier. Same preference order as `same_name`. | 6 | 712 |
| 8 | `indistinguishable` | Description cosine in [0.85, 0.92) and judged `same` by qwen3.5:9b: no basis to tell them apart. | 1 | 711 |
| 9 | `below_cap` | Outside the 500 most-installed survivors. | 211 | 500 |

## Similar pairs

After step 7 (`near_duplicate`), each skill's nearest-neighbour description cosine had percentiles p50 = 0.789, p75 = 0.833, p90 = 0.866, p95 = 0.889, p99 = 0.907. 96 pairs fell in [0.85, 0.92) and were judged by qwen3.5:9b: 60 `distinct`, 35 `overlap`, 1 `same`.

`same` pairs drive step 8 (`indistinguishable`). 28 `overlap` pairs where both skills survived are saved in `corpus/overlaps.jsonl` as candidate equivalence sets for the Phase 2 eval. A reply that couldn't be parsed counts as `parse_failed` and never removes a skill. All verdicts are in `corpus/raw/pair_judgments.jsonl`.

## Description rewrites

qwen3.5:9b flagged 95 descriptions that say when to trigger the skill but not what it does. 95 were rewritten from the SKILL.md body; the original is kept in `description_original`. Nothing else was rewritten.

## Domain spread

Largest domain share: 22.4% (limit 35%: OK).

| Domain | Skills |
| --- | ---: |
| software-dev | 112 |
| devops-cloud | 79 |
| ai-agents | 73 |
| media-generation | 70 |
| design-ui | 41 |
| marketing-seo | 38 |
| web-automation | 32 |
| business-strategy | 21 |
| documents-office | 14 |
| productivity-collab | 11 |
| research-writing | 6 |
| data-analytics | 3 |

## Name-prefix cluster density (diagnostic)

Mean and max pairwise description cosine within each name prefix that has 3 or more skills. On this corpus a high mean mostly reflects shared vendor wording, so it doesn't remove anything by itself; the pair judge above decides. See `docs/DECISIONS.md`.

| Prefix | Skills | Mean cosine | Max cosine |
| --- | ---: | ---: | ---: |
| google | 7 | 0.853 | 0.911 |
| gsap | 8 | 0.782 | 0.858 |
| hyperframes | 9 | 0.776 | 0.844 |
| seo | 3 | 0.773 | 0.821 |
| golang | 4 | 0.766 | 0.804 |
| obsidian | 3 | 0.76 | 0.783 |
| higgsfield | 8 | 0.755 | 0.862 |
| insforge | 4 | 0.751 | 0.831 |
| prisma | 9 | 0.739 | 0.871 |
| image | 6 | 0.728 | 0.864 |
| momentic | 3 | 0.725 | 0.8 |
| code | 3 | 0.714 | 0.78 |
| video | 6 | 0.704 | 0.888 |
| firebase | 10 | 0.704 | 0.881 |
| cloudflare | 4 | 0.702 | 0.79 |

## Licenses

Licenses come from SKILL.md frontmatter, else the repo's LICENSE file. `unknown` means neither exists. Only names, descriptions and paths are stored here, not SKILL.md bodies.

| License | Skills |
| --- | ---: |
| MIT | 310 |
| Apache-2.0 | 118 |
| unknown | 49 |
| Complete terms in LICENSE.txt | 11 |
| Proprietary. LICENSE.txt has complete terms | 4 |
| ISC | 4 |
| BSD-3-Clause | 2 |
| other | 1 |
| See LICENSE file in repository root | 1 |

LLM parse failures: pairs 0, classify 0, rewrite 0.
