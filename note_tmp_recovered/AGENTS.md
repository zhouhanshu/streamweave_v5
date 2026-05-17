<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 69
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/mmu_mllm_hdd/zhouhanshu/test/aris_repo`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /mmu_mllm_hdd/zhouhanshu/test/aris_repo/tools/install_aris_codex.sh "/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5" --reconcile`
For copied Codex installs, use:
`bash /mmu_mllm_hdd/zhouhanshu/test/aris_repo/tools/smart_update_codex.sh --project "/mmu_mllm_hdd/zhouhanshu/test/exp3/streamweave_v5"`
<!-- ARIS-CODEX:END -->
