from __future__ import annotations

import csv
import json
import shlex
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import yaml


def parse_month(value: str) -> datetime:
    text = str(value).strip()
    if len(text) == 7:
        text = f"{text}-01"
    dt = datetime.strptime(text, "%Y-%m-%d")
    if dt.day != 1:
        raise ValueError(f"month must be first day of month: {value}")
    return dt.replace(tzinfo=timezone.utc)


def format_month(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def compact_month(dt: datetime) -> str:
    return dt.strftime("%Y%m")


def add_months(dt: datetime, months: int) -> datetime:
    month0 = dt.month - 1 + months
    year = dt.year + month0 // 12
    month = month0 % 12 + 1
    return dt.replace(year=year, month=month, day=1)


def iter_months(first: str, last: str) -> list[datetime]:
    start = parse_month(first)
    end = parse_month(last)
    if end < start:
        raise ValueError(f"last month {last} is before first month {first}")
    out: list[datetime] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = add_months(cur, 1)
    return out


def utc_batch_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def _split_csv(value: object | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [v.strip() for v in text.split(",") if v.strip()]


@dataclass(frozen=True)
class FamilyAblationJob:
    test_month: str
    job_name: str
    run_id: str
    command: list[str]
    script_path: Path
    log_path: Path
    status_path: Path


@dataclass(frozen=True)
class PreparedFamilyAblationJobs:
    batch_dir: Path
    jobs: list[FamilyAblationJob]
    prebuild_script: Path


def load_family_ablation_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    fam = data.get("family_ablation") or {}
    required = ["base_features", "manifest", "policy_output_dir"]
    missing = [key for key in required if not fam.get(key)]
    if missing:
        raise ValueError(f"config family_ablation section missing required keys: {missing}")
    return data


def build_prebuild_command(*, python_executable: str, config: Path) -> list[str]:
    data = load_family_ablation_config(config)
    fam = data["family_ablation"]
    cmd = [
        python_executable,
        "scripts/build_family_ablation_policies.py",
        "--features",
        str(fam["base_features"]),
        "--manifest",
        str(fam["manifest"]),
        "--output-dir",
        str(fam["policy_output_dir"]),
        "--policy-prefix",
        str(fam.get("policy_prefix") or "lofo"),
        "--output-format",
        str(fam.get("output_format") or "parquet"),
    ]
    for family in _split_csv(fam.get("families")):
        # build_family_ablation_policies.py expects a single CSV argument, not repeats.
        pass
    families = _split_csv(fam.get("families"))
    if families:
        cmd.extend(["--families", ",".join(families)])
    excluded = _split_csv(fam.get("exclude_families"))
    if excluded:
        cmd.extend(["--exclude-families", ",".join(excluded)])
    if bool(fam.get("overwrite_policies", False)):
        cmd.append("--overwrite")
    return cmd


def build_month_command(
    *,
    python_executable: str,
    config: Path,
    test_month: str,
    run_id: str,
    output_root: str | None = None,
    extra_args: Sequence[str] = (),
) -> list[str]:
    cmd = [
        python_executable,
        "scripts/run_rolling_monthly_family_ablation.py",
        "--config",
        str(config),
        "--first-test-month",
        test_month,
        "--last-test-month",
        test_month,
        "--max-cycles",
        "1",
        "--run-id",
        run_id,
        "--skip-policy-build",
    ]
    if output_root:
        cmd.extend(["--output-root", output_root])
    cmd.extend(str(x) for x in extra_args)
    return cmd


def write_job_script(job: FamilyAblationJob, repo_root: Path) -> None:
    job.script_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.status_path.parent.mkdir(parents=True, exist_ok=True)
    status_tmp = f"{job.status_path}.tmp"
    command_text = quote_cmd(job.command)
    content = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(repo_root))}
mkdir -p {shlex.quote(str(job.log_path.parent))} {shlex.quote(str(job.status_path.parent))}
start_epoch=$(date +%s)
cat > {shlex.quote(str(status_tmp))} <<JSON
{{"job_name":"{job.job_name}","run_id":"{job.run_id}","test_month":"{job.test_month}","status":"running","started_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
JSON
mv {shlex.quote(str(status_tmp))} {shlex.quote(str(job.status_path))}
{{
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) job={job.job_name} run_id={job.run_id} test_month={job.test_month}"
  echo "[CMD] {command_text}"
  {command_text}
  rc=$?
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))
  echo "[END] $(date -u +%Y-%m-%dT%H:%M:%SZ) job={job.job_name} rc=$rc elapsed_sec=$elapsed"
  cat > {shlex.quote(str(status_tmp))} <<JSON
{{"job_name":"{job.job_name}","run_id":"{job.run_id}","test_month":"{job.test_month}","status":"success","return_code":$rc,"elapsed_sec":$elapsed,"ended_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
JSON
  mv {shlex.quote(str(status_tmp))} {shlex.quote(str(job.status_path))}
}} 2>&1 | tee {shlex.quote(str(job.log_path))}
"""
    job.script_path.write_text(content, encoding="utf-8")
    job.script_path.chmod(job.script_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_prebuild_script(path: Path, *, repo_root: Path, command: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command_text = quote_cmd(command)
    content = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(repo_root))}
echo "[PREBUILD START] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "[CMD] {command_text}"
{command_text}
echo "[PREBUILD END] $(date -u +%Y-%m-%dT%H:%M:%SZ)"
"""
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_manifest(batch_dir: Path, jobs: Sequence[FamilyAblationJob], *, config: Path, prebuild_script: Path) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    fields = ["job_name", "run_id", "test_month", "script_path", "log_path", "status_path", "command"]
    with (batch_dir / "job_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            writer.writerow({
                "job_name": job.job_name,
                "run_id": job.run_id,
                "test_month": job.test_month,
                "script_path": str(job.script_path),
                "log_path": str(job.log_path),
                "status_path": str(job.status_path),
                "command": quote_cmd(job.command),
            })
    manifest = {
        "config": str(config),
        "prebuild_script": str(prebuild_script),
        "jobs": [
            {
                "job_name": j.job_name,
                "run_id": j.run_id,
                "test_month": j.test_month,
                "script_path": str(j.script_path),
                "log_path": str(j.log_path),
                "status_path": str(j.status_path),
                "command": j.command,
            }
            for j in jobs
        ],
    }
    (batch_dir / "job_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def write_launchers(batch_dir: Path, jobs: Sequence[FamilyAblationJob], *, prebuild_script: Path) -> None:
    serial = batch_dir / "run_all_serial.sh"
    serial.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"{shlex.quote(str(prebuild_script))}\n"
        + "\n".join(shlex.quote(str(job.script_path)) for job in jobs)
        + "\n",
        encoding="utf-8",
    )
    serial.chmod(serial.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    parallel = batch_dir / "run_all_parallel.sh"
    scripts_text = "\n".join(str(job.script_path) for job in jobs)
    parallel.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "PARALLELISM=${1:-2}\n"
        f"{shlex.quote(str(prebuild_script))}\n"
        "cat <<'JOBS' | xargs -n 1 -P \"$PARALLELISM\" bash\n"
        f"{scripts_text}\n"
        "JOBS\n",
        encoding="utf-8",
    )
    parallel.chmod(parallel.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    tmux = batch_dir / "launch_tmux.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "SESSION=${1:-rolling_monthly_family_ablation}",
        f"{shlex.quote(str(prebuild_script))}",
        "tmux new-session -d -s \"$SESSION\"",
    ]
    for idx, job in enumerate(jobs):
        window_name = job.job_name
        cmd = shlex.quote(str(job.script_path))
        if idx == 0:
            lines.append(f"tmux rename-window -t \"$SESSION:0\" {shlex.quote(window_name)}")
            lines.append(f"tmux send-keys -t \"$SESSION:0\" {cmd} C-m")
        else:
            lines.append(f"tmux new-window -t \"$SESSION\" -n {shlex.quote(window_name)}")
            lines.append(f"tmux send-keys -t \"$SESSION:{idx}\" {cmd} C-m")
    lines.append("echo \"tmux session: $SESSION\"")
    tmux.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmux.chmod(tmux.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def prepare_family_ablation_jobs(
    *,
    repo_root: Path,
    job_root: Path,
    config: Path,
    first_test_month: str,
    last_test_month: str,
    python_executable: str = "python3",
    batch_id: str | None = None,
    run_id_prefix: str | None = None,
    output_root: str | None = None,
    extra_args: Sequence[str] = (),
) -> PreparedFamilyAblationJobs:
    batch = batch_id or utc_batch_id("rolling_monthly_family_ablation_jobs")
    batch_dir = job_root / batch
    scripts_dir = batch_dir / "jobs"
    logs_dir = batch_dir / "logs"
    status_dir = batch_dir / "status"
    months = iter_months(first_test_month, last_test_month)
    run_prefix = run_id_prefix or batch
    config = Path(config)

    prebuild_script = batch_dir / "prebuild_family_policies.sh"
    write_prebuild_script(
        prebuild_script,
        repo_root=repo_root,
        command=build_prebuild_command(python_executable=python_executable, config=config),
    )

    jobs: list[FamilyAblationJob] = []
    for month in months:
        month_text = format_month(month)
        compact = compact_month(month)
        job_name = f"test_{compact}"
        run_id = f"{run_prefix}_{compact}"
        command = build_month_command(
            python_executable=python_executable,
            config=config,
            test_month=month_text,
            run_id=run_id,
            output_root=output_root,
            extra_args=extra_args,
        )
        job = FamilyAblationJob(
            test_month=month_text,
            job_name=job_name,
            run_id=run_id,
            command=command,
            script_path=scripts_dir / f"{job_name}.sh",
            log_path=logs_dir / f"{job_name}.log",
            status_path=status_dir / f"{job_name}.json",
        )
        write_job_script(job, repo_root)
        jobs.append(job)

    write_manifest(batch_dir, jobs, config=config, prebuild_script=prebuild_script)
    write_launchers(batch_dir, jobs, prebuild_script=prebuild_script)
    return PreparedFamilyAblationJobs(batch_dir=batch_dir, jobs=jobs, prebuild_script=prebuild_script)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
