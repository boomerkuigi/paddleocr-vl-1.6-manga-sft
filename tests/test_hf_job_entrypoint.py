import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _write_fake_python(path: Path, log: Path | None = None) -> None:
    lines = [
        "#!/bin/bash",
        'if [[ "$1" == "-" ]]; then echo HF_V2_PILOT_REPO; exit 0; fi',
    ]
    if log is not None:
        lines.append(f'printf "%s\\n" "$*" >> "{log}"')
    lines.append("exit 0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def test_entrypoint_accepts_only_the_configured_v2_destination(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    invocation_log = tmp_path / "invocations.txt"
    _write_fake_python(fake_bin / "python", invocation_log)
    fake_bash = fake_bin / "bash"
    fake_bash.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    fake_bash.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HF_TOKEN": "test-token",
        "PUSH_TO_HUB": "1",
        "HF_V2_PILOT_REPO": "AlphaBeta07/PaddleOCR-VL-1.6-For-Manga-V2-Pilot",
    }
    environment.pop("HF_MODEL_REPO", None)
    completed = subprocess.run(
        ["/bin/bash", "scripts/hf_job_entrypoint.sh", "configs/v2_continuation_pilot.yaml"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--push-to-hub" in invocation_log.read_text(encoding="utf-8")


def test_entrypoint_rejects_missing_configured_destination(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_python(fake_bin / "python")
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HF_TOKEN": "test-token",
        "PUSH_TO_HUB": "1",
    }
    environment.pop("HF_MODEL_REPO", None)
    environment.pop("HF_V2_PILOT_REPO", None)
    completed = subprocess.run(
        ["/bin/bash", "scripts/hf_job_entrypoint.sh", "configs/v2_continuation_pilot.yaml"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "HF_V2_PILOT_REPO must name" in completed.stderr
