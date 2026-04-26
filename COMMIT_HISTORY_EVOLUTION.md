# DMTS Repository Commit History and Version Evolution

_Last analyzed from `main` on 2026-03-14._

## Scope

This document compares the effective "versions" of the repository commit-by-commit.
Because the repository has no git tags or formal release branches, each commit on `main`
is treated as a version boundary.

This report is intentionally weighted toward the March 13, 2026 changes and later,
because that is where the most meaningful runtime and architecture changes happened.
A compact full-history appendix is included at the end.

## Repository Snapshot

- Total commits on `main`: 26
- Tags: none
- Merge commits: 1 (`bb13bef`)
- Dominant change themes:
  - Initial product publication
  - Dependency and installation churn
  - Demo/media churn and cleanup
  - GitHub workflow automation
  - Linux/CUDA reproducibility hardening

## High-Level Evolution

### Phase 1: Initial open-source drop

The repository started as a fairly complete application in a single large commit.
The initial architecture was already recognizable:

- `dmts_mk4.py` as the main monolithic runtime
- `translation/manager_*.py` for backend-specific translation behavior
- `language_codes.py` for NLLB/ISO mapping helpers
- `index.html` for the browser UI
- `run_server*.sh` as thin preset launchers

This gave the project usable functionality quickly, but it also concentrated a lot of
runtime responsibility in one Python file.

### Phase 2: December 2025 setup churn

Most December 17 work focused on installation, dependency pins, model paths, and
README instructions rather than core transcription or translation behavior.

The repo moved from a simpler but under-specified install flow to a much more explicit
and highly pinned environment. That improved reproducibility in theory, but it also
created complexity and exposed unresolved package conflicts, especially around:

- `TTS==0.22.0`
- `numpy`
- `scipy`
- PyTorch/CUDA compatibility

### Phase 3: Documentation/media noise

December 18 added several low-value commits around the demo asset. This part of the
history contains the weakest repository hygiene:

- repeated README asset-link changes
- a committed MP4 file
- committed `.DS_Store` files
- accidental-looking deletion of `LINUX_SERVER_SETUP.md`

### Phase 4: Repository automation

January 12 added Claude-powered GitHub workflows. These improve collaboration and PR
reviewing but do not change application behavior.

### Phase 5: March 2026 runtime hardening

March 13 and March 14 are the most important technical points in the history.
This is where the project moved from a machine-specific, brittle setup into something
closer to a reproducible Linux/CUDA application.

The biggest improvements in this phase are:

- lazy loading of TTS so ASR-only startup is possible
- deterministic environment bootstrapping with `setup_env.sh`
- centralized model path configuration in `config.py`
- explicit preflight validation with `scripts/preflight.py`
- CUDA/cuDNN library path injection in launch scripts
- safer diarization embeddings using in-memory audio tensors instead of temp WAV files
- removal of startup-time package auto-install behavior from the server runtime

## Detailed Comparison: March 13, 2026 and Later

The table below focuses on the most important commits from `fce3072` onward.
Each row compares that commit against the immediately previous state.

| Commit | Date | Files changed | Added / removed / modified | Purpose | Functional impact | Risks / regressions |
|---|---|---|---|---|---|---|
| `fce3072` | 2026-03-13 | `dmts_mk4.py` | Modified imports and CLI flags. TTS imports moved from module scope into `initialize_tts_model()`. Added `--no-enable-translation`, `--no-enable-diarization`, `--no-enable-verification`. Made `--diarization_model_path` optional when diarization is disabled. | Decouple ASR startup from TTS and make features disable-able from CLI. | Server can now start in ASR-only or reduced-feature mode without crashing on missing XTTS/TTS dependencies. This is a direct runtime usability improvement. | Low risk. Main tradeoff is more CLI branching, but the change is contained and sensible. |
| `845e1de` | 2026-03-13 | `requirements.txt`, `setup_env.sh` | Removed `TTS` from `requirements.txt`, pinned `numpy==1.23.5`, kept `scipy==1.15.2`, added a new scripted install flow that installs PyTorch first, then NumPy, then `TTS --no-deps`, then remaining requirements. | Resolve the TTS/NumPy/SciPy conflict without relying on pip to solve an impossible dependency graph. | Installation becomes more deterministic and easier to reproduce. The repo gains an actual bootstrap script instead of just README guidance. | Medium risk. This intentionally bypasses upstream package metadata, so it depends on observed runtime compatibility rather than supported dependency declarations. |
| `9187e7d` | 2026-03-14 | `README.md`, `config.py`, `dmts_mk4.py`, `docs/2026-03-13-linux-cuda-reproducibility-report.md`, `requirements.txt`, `run_server.sh`, `run_server_hunyuan.sh`, `run_server_hybrid.sh`, `run_server_nllb.sh`, `scripts/configure_cuda_libs.sh`, `scripts/download_models.py`, `scripts/preflight.py`, `setup_env.sh` | Added shared config, model downloader, preflight checker, CUDA library helper, and reproducibility report. Removed runtime auto-install logic from `dmts_mk4.py`. Replaced hardcoded machine-local paths with config-driven defaults. Changed diarization embedding extraction from temp-file loading to in-memory tensor embedding extraction. Updated launch scripts to use env overrides and `--use_main_model_for_realtime`. Fixed the earlier `run_server_hunyuan.sh` `STORAG E_PATH` typo. | Harden Linux/CUDA reproducibility and simplify startup/operations. | This is the biggest architecture and operational improvement after the initial commit. Startup becomes more deterministic. Model-path configuration becomes centralized. Diarization becomes more reliable and likely faster by avoiding temp WAV I/O and `torchcodec`-adjacent load paths. Launch scripts become less machine-specific. | Medium risk. The new setup is more opinionated around Linux + conda + CUDA 12.8. Reusing the main Whisper model for realtime improves stability but may reduce specialization or throughput compared with a separate realtime model. |
| `f18dd93` | 2026-03-14 | `.gitignore` | Reordered and expanded DMTS-specific ignores for models, logs, audio, checkpoints, and runtime outputs. | Improve repository hygiene. | Positive maintenance effect. Less chance of runtime artifacts being committed. | Low risk. |
| `19c2b00` | 2026-03-14 | `.gitignore` | Added `docs/` to ignored paths. | Stop local docs from being tracked. | No runtime effect. | High documentation risk. Ignoring the entire `docs/` tree blocks intentional shared documentation. |
| `3f00e77` | 2026-03-14 | `docs/2026-03-13-linux-cuda-reproducibility-report.md` | Deleted the reproducibility report from git tracking. | Remove a local report doc from version control. | No runtime effect. | High documentation regression. `README.md` had already linked to this report, so fresh clones no longer receive the file even though the README still references it. |
| `bac18fd` | 2026-03-14 | `.gitignore`, `docs/2026-03-13-linux-cuda-reproducibility-report.md` | Removed `docs/` from `.gitignore`, restored the reproducibility report to version control, and expanded the report with clearer cuDNN root-cause analysis. | Undo the documentation regression and preserve the operational debugging record. | The current `HEAD` once again has a working README-to-report relationship, and the repository now keeps the March CUDA findings in tracked docs. | Low risk. The main impact is positive; it reverses the prior documentation break. |

## March 13+ Narrative Analysis

### `fce3072`: runtime decoupling starts

This commit is small in file count but important in effect.

Before this point, TTS imports happened at module import time. That means a user who
only wanted ASR could still fail during startup if the TTS stack was unavailable.
By moving XTTS/TTS imports into `initialize_tts_model()`, the project stopped forcing
that dependency into every startup path.

This commit also made the feature set controllable from the command line with negative
flags. That matters operationally because users no longer need to edit shell wrappers
just to disable translation, diarization, or verification.

### `845e1de`: dependency conflict acknowledged explicitly

This commit is the first point where the repository treats the dependency conflict as a
real engineering constraint rather than something docs can paper over.

The key design choice is pragmatic rather than pure:

- keep the runtime versions known to work together
- bypass the TTS package metadata with `--no-deps`
- script the install order so that users stop discovering the conflict themselves

This is a good reproducibility move, but it is still a workaround.
It does not eliminate the upstream conflict; it standardizes a known-good path around it.

### `9187e7d`: biggest post-release improvement

This commit is the real maturity point for the repository.

#### Operational changes

The repo gained proper operational tooling:

- `config.py` as a shared source of truth for model paths
- `scripts/download_models.py` for backend-aware checkpoint download
- `scripts/preflight.py` for validating Python, conda, CUDA, cuDNN, imports, model presence, and writable directories
- `scripts/configure_cuda_libs.sh` for fixing library lookup at launch time

Before this, setup logic was scattered across README instructions and shell scripts.
After this, setup became much more explicit and self-checking.

#### Runtime changes in `dmts_mk4.py`

Two runtime changes stand out:

1. Startup-time package installation logic was removed.
   The server no longer tries to repair its Python environment while launching.
   That is a strong improvement in determinism and debuggability.

2. Diarization embeddings switched from temp WAV files to in-memory tensors.
   The prior flow wrote audio to disk, reloaded it, and depended on XTTS internals in a
   way that exposed `torchcodec` or FFmpeg-style runtime problems.
   The new flow uses the audio already in memory and passes it directly into the speaker
   embedding path.

That change improves:

- reliability
- latency
- I/O efficiency
- portability within the chosen environment

#### Launcher-script changes

All four launcher scripts moved toward environment-driven configuration:

- `DMTS_MODELS_DIR` support
- per-model env overrides
- CUDA/cuDNN path injection before launching Python
- `USE_MAIN_MODEL_FOR_REALTIME="true"`

This is cleaner than hand-editing absolute paths and significantly reduces machine-local coupling.

#### Architecture effect

This commit does not fully modularize the application, but it does shift the repository
from a single giant runtime plus ad hoc docs into a runtime plus reusable operational tooling.
That is the clearest architecture improvement in the whole history.

### `f18dd93` to `bac18fd`: temporary doc-policy regression, then repair

`f18dd93` is a solid maintenance commit: it makes runtime artifacts harder to leak into git.

The next two commits are where things turn in the wrong direction:

- `19c2b00` ignores all of `docs/`
- `3f00e77` deletes the tracked reproducibility report

That creates a mismatch with the README, which had just been updated to link the report.
In that intermediate state, the result is:

- the report is conceptually part of the project docs
- the report is no longer in version control
- future clones may see a broken README reference

`bac18fd` then repairs this by removing `docs/` from `.gitignore`, restoring the report,
and clarifying the cuDNN findings inside the report itself.

So the correct reading of March 14 is not "documentation stayed broken." It is:

- `f18dd93`: repo hygiene improvement
- `19c2b00` and `3f00e77`: short-lived documentation regression
- `bac18fd`: explicit correction and documentation restoration

## March 13+ Commit Matrix

| Area | Before March 13 | After March 14 | Net effect |
|---|---|---|---|
| TTS dependency coupling | TTS imports could block startup even for ASR-only use | TTS loads lazily when diarization actually needs it | Better startup resilience |
| Install flow | README-heavy, conflicting dependency pins, no deterministic bootstrap | Scripted bootstrap in `setup_env.sh` with explicit ordering | Better reproducibility |
| Model path management | Hardcoded or wrapper-local defaults, some machine-specific values | Centralized in `config.py` and overridable via env vars | Cleaner operations and less local coupling |
| CUDA/cuDNN runtime handling | Susceptible to library lookup failures | Launch-time CUDA/cuDNN path injection plus preflight validation | Better Linux/CUDA stability |
| Diarization embedding extraction | Temp WAV file path with additional runtime fragility | In-memory tensor embedding extraction | Better reliability and likely better performance |
| Repo docs policy | Docs could be tracked normally | Briefly regressed, then restored by `bac18fd` | Net result: tracked reproducibility docs preserved on `HEAD` |

## Current Notable Risks in `HEAD`

1. Monolithic runtime remains

Even after the March improvements, `dmts_mk4.py` still concentrates a very large amount
of runtime logic in one file. Operational tooling improved, but core runtime modularity
has not improved much.

2. Test coverage is still absent

No commit in this history added automated tests for the critical runtime changes.
The March hardening work appears to have been validated manually and operationally.

3. Dependency workaround remains a workaround

The `TTS --no-deps` strategy is practical, but it remains a controlled bypass of package
metadata rather than a true upstream-compatible dependency solution.

4. Operational knowledge is still concentrated in narrative docs

The restored reproducibility report is valuable, but the environment assumptions are still
captured mostly in prose and scripts rather than in automated CI or regression tests.
That makes future environment drift harder to catch early.

## Full History Appendix

This appendix is compact by design. It gives a quick per-commit summary across the whole history.

| Commit | Date | Files changed | Summary | Impact / notes |
|---|---|---|---|---|
| `ddecd86` | 2025-12-12 | Initial project files | Added full DMTS MK4 app, UI, translation managers, scripts, docs, requirements | Established entire product and monolithic architecture |
| `b069e1f` | 2025-12-12 | `LINUX_SERVER_SETUP.md` | Corrected VRAM guidance | Better deployment expectations |
| `8cdc104` | 2025-12-12 | `CHANGELOG.md` | Rewrote initial changelog for public release | Docs cleanup |
| `f23f096` | 2025-12-14 | `CONTRIBUTING.md`, `LINUX_SERVER_SETUP.md`, `README.md` | Replaced placeholder GitHub URLs with real repo URL | Onboarding fix |
| `6963d7d` | 2025-12-15 | `LICENSE` | Corrected copyright year | Legal metadata fix |
| `f6b925c` | 2025-12-17 | `CHANGELOG.md`, `LINUX_SERVER_SETUP.md`, `README.md`, `requirements.txt`, `run_server*.sh` | Major dependency pinning and setup rewrite; introduced `STORAGE_PATH`; lowered verification threshold to `0.05` | More reproducible but more complex; removed changelog |
| `1f89e87` | 2025-12-17 | `README.md` | Corrected backend VRAM estimates | Docs only |
| `314c91e` | 2025-12-17 | `.DS_Store`, `README.md`, `requirements.txt`, `run_server*.sh` | Switched default model root to `./models`; updated dependency pins | Simplified onboarding; introduced accidental `.DS_Store` and `run_server_hunyuan.sh` typo |
| `f0edf9a` | 2025-12-17 | `README.md`, `requirements.txt` | Adjusted NumPy constraint and added `hf_transfer` install note | Partial dependency correction |
| `3328939` | 2025-12-18 | `README.md`, `dmts_demo.mp4` | Added demo video | Repo/media bloat |
| `d10f1de` | 2025-12-18 | `README.md` | Changed demo video link | Docs/media maintenance |
| `87e1488` | 2025-12-18 | `README.md` | Replaced asset link | Docs/media maintenance |
| `72a2b6f` | 2025-12-18 | `README.md` | Updated asset link again | Docs/media maintenance |
| `c4ccf32` | 2025-12-18 | `dmts_demo.mp4` | Compressed demo video | Slight repo-weight improvement |
| `739ba49` | 2025-12-18 | `.DS_Store`, `LINUX_SERVER_SETUP.md`, `dmts_demo.mp4`, `translation/.DS_Store` | "Cleanup" commit that deleted setup docs and added Mac metadata | Poor repo hygiene; likely accidental collateral damage |
| `275da9f` | 2025-12-18 | `dmts_demo.mp4` | Removed MP4 from git | Cleans repo history going forward |
| `174e0ee` | 2026-01-12 | `.github/workflows/claude.yml` | Added Claude PR assistant workflow | Collaboration automation |
| `a7bfd13` | 2026-01-12 | `.github/workflows/claude-code-review.yml` | Added Claude code review workflow | Review automation |
| `bb13bef` | 2026-01-12 | Merge commit | Merged workflow PR | No direct runtime change |
| `fce3072` | 2026-03-13 | `dmts_mk4.py` | Lazy TTS imports and no-* CLI flags | Better startup flexibility |
| `845e1de` | 2026-03-13 | `requirements.txt`, `setup_env.sh` | Deterministic install workaround for dependency conflict | Strong reproducibility improvement |
| `9187e7d` | 2026-03-14 | many runtime/setup files | Major Linux/CUDA hardening, config centralization, preflight/model tooling, safer diarization path | Biggest post-release technical improvement |
| `f18dd93` | 2026-03-14 | `.gitignore` | Reordered and expanded runtime ignores | Better hygiene |
| `19c2b00` | 2026-03-14 | `.gitignore` | Ignored `docs/` | Documentation policy regression |
| `3f00e77` | 2026-03-14 | `docs/2026-03-13-linux-cuda-reproducibility-report.md` | Removed tracked report doc | README/documentation mismatch for fresh clones |
| `bac18fd` | 2026-03-14 | `.gitignore`, `docs/2026-03-13-linux-cuda-reproducibility-report.md` | Restored the report and un-ignored `docs/`; clarified cuDNN findings | Fixes the same-day documentation regression |

## Bottom Line

If the repository history is judged by technical maturity rather than raw commit count,
then March 13 and March 14 are the real turning point.

Before that point, most change was either:

- initial feature publication
- documentation correction
- install/dependency churn
- media cleanup

After that point, the project becomes substantially more operationally coherent.
The core runtime is still monolithic and under-tested, but the setup story, launch story,
and Linux/CUDA reproducibility story are all much stronger.

The main current weaknesses are still architectural and operational rather than functional:
the runtime remains monolithic, the dependency workaround is still a workaround, and the
March hardening story is documented well but not backed by automated tests.
