from __future__ import annotations

import csv
import json
import os
import shlex
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


def parse_month(value: str) -> datetime:
    """Parse YYYY-MM or YYYY-MM-01 into a UTC month-start datetime."""
    text = value.strip()
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
    months: list[datetime] = []
    cur = start
    while cur <= end:
        months.append(cur)
        cur = add_months(cur, 1)
    return months


def utc_run_id(prefix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


@dataclass(frozen=True)
class MonthlyJob:
    test_month: str
    job_name: str
    command: list[str]
    script_path: Path
    log_path: Path
    status_path: Path


def _quote_cmd(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def build_monthly_command(
    *,
    python_executable: str,
    test_month: str,
    extra_args: Sequence[str] = (),
) -> list[str]:
    return [
        python_executable,
        "scripts/run_rolling_monthly_research.py",
        "--first-test-month",
        test_month,
        "--last-test-month",
        test_month,
        "--max-cycles",
        "1",
        *extra_args,
    ]


def write_job_script(job: MonthlyJob, repo_root: Path) -> None:
    job.script_path.parent.mkdir(parents=True, exist_ok=True)
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    job.status_path.parent.mkdir(parents=True, exist_ok=True)

    command_text = _quote_cmd(job.command)
    status_tmp = f"{job.status_path}.tmp"
    content = f"""#!/usr/bin/env bash
set -euo pipefail
cd {shlex.quote(str(repo_root))}
mkdir -p {shlex.quote(str(job.log_path.parent))} {shlex.quote(str(job.status_path.parent))}
cat > {shlex.quote(str(status_tmp))} <<'JSON'
{{"job_name":"{job.job_name}","test_month":"{job.test_month}","status":"running","started_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
JSON
mv {shlex.quote(str(status_tmp))} {shlex.quote(str(job.status_path))}
start_epoch=$(date +%s)
{{
  echo "[START] $(date -u +%Y-%m-%dT%H:%M:%SZ) job={job.job_name} test_month={job.test_month}"
  echo "[CMD] {command_text}"
  {command_text}
  rc=$?
  end_epoch=$(date +%s)
  elapsed=$((end_epoch - start_epoch))
  echo "[END] $(date -u +%Y-%m-%dT%H:%M:%SZ) job={job.job_name} rc=$rc elapsed_sec=$elapsed"
  cat > {shlex.quote(str(status_tmp))} <<JSON
{{"job_name":"{job.job_name}","test_month":"{job.test_month}","status":"success","return_code":$rc,"elapsed_sec":$elapsed,"ended_at":"$(date -u +%Y-%m-%dT%H:%M:%SZ)"}}
JSON
  mv {shlex.quote(str(status_tmp))} {shlex.quote(str(job.status_path))}
}} 2>&1 | tee {shlex.quote(str(job.log_path))}
"""
    job.script_path.write_text(content, encoding="utf-8")
    current = job.script_path.stat().st_mode
    job.script_path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def prepare_jobs(
    *,
    repo_root: Path,
    job_root: Path,
    first_test_month: str,
    last_test_month: str,
    python_executable: str = "python3",
    extra_args: Sequence[str] = (),
    batch_id: str | None = None,
) -> tuple[Path, list[MonthlyJob]]:
    batch = batch_id or utc_run_id("rolling_monthly_jobs")
    batch_dir = job_root / batch
    scripts_dir = batch_dir / "jobs"
    logs_dir = batch_dir / "logs"
    status_dir = batch_dir / "status"
    months = iter_months(first_test_month, last_test_month)

    jobs: list[MonthlyJob] = []
    for month in months:
        month_text = format_month(month)
        job_name = f"test_{compact_month(month)}"
        command = build_monthly_command(
            python_executable=python_executable,
            test_month=month_text,
            extra_args=extra_args,
        )
        job = MonthlyJob(
            test_month=month_text,
            job_name=job_name,
            command=command,
            script_path=scripts_dir / f"{job_name}.sh",
            log_path=logs_dir / f"{job_name}.log",
            status_path=status_dir / f"{job_name}.json",
        )
        write_job_script(job, repo_root)
        jobs.append(job)

    write_manifest(batch_dir, jobs)
    write_launchers(batch_dir, jobs)
    return batch_dir, jobs


def write_manifest(batch_dir: Path, jobs: Sequence[MonthlyJob]) -> None:
    batch_dir.mkdir(parents=True, exist_ok=True)
    manifest_csv = batch_dir / "job_manifest.csv"
    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["job_name", "test_month", "script_path", "log_path", "status_path", "command"],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "job_name": job.job_name,
                    "test_month": job.test_month,
                    "script_path": str(job.script_path),
                    "log_path": str(job.log_path),
                    "status_path": str(job.status_path),
                    "command": _quote_cmd(job.command),
                }
            )
    (batch_dir / "job_manifest.json").write_text(
        json.dumps(
            [
                {
                    "job_name": job.job_name,
                    "test_month": job.test_month,
                    "script_path": str(job.script_path),
                    "log_path": str(job.log_path),
                    "status_path": str(job.status_path),
                    "command": job.command,
                }
                for job in jobs
            ],
            indent=2,
        ),
        encoding="utf-8",
    )


def write_launchers(batch_dir: Path, jobs: Sequence[MonthlyJob]) -> None:
    serial = batch_dir / "run_all_serial.sh"
    serial.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
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
        "cat <<'JOBS' | xargs -n 1 -P \"$PARALLELISM\" bash\n"
        f"{scripts_text}\n"
        "JOBS\n",
        encoding="utf-8",
    )
    parallel.chmod(parallel.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    tmux = batch_dir / "launch_tmux.sh"
    tmux_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "SESSION=${1:-rolling_monthly}",
        "tmux new-session -d -s \"$SESSION\"",
    ]
    for idx, job in enumerate(jobs):
        window_name = job.job_name
        cmd = shlex.quote(str(job.script_path))
        if idx == 0:
            tmux_lines.append(f"tmux rename-window -t \"$SESSION:0\" {shlex.quote(window_name)}")
            tmux_lines.append(f"tmux send-keys -t \"$SESSION:0\" {shlex.quote(cmd)} C-m")
        else:
            tmux_lines.append(f"tmux new-window -t \"$SESSION\" -n {shlex.quote(window_name)}")
            tmux_lines.append(f"tmux send-keys -t \"$SESSION:{idx}\" {shlex.quote(cmd)} C-m")
    tmux_lines.append("echo \"tmux session: $SESSION\"")
    tmux.write_text("\n".join(tmux_lines) + "\n", encoding="utf-8")
    tmux.chmod(tmux.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
