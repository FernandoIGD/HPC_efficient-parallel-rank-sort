"""Parse and validate the final Khipu ranking-sort experiment logs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


EXPECTED_JOBS = {
    47482: 1,
    47483: 4,
    47486: 9,
    47488: 16,
    47493: 25,
}
EXPECTED_SIZES = (3600, 7200, 10800, 14400, 18000)
EXPECTED_REPETITIONS = 20
EXPECTED_SOURCE_SHA256 = (
    "2a44af4f560fa594a934b639684d5faa2f0671f7c2a95cd95ac0c23d8c5a0e41"
)


@dataclass(frozen=True)
class _Record:
    job_id: int
    partition: str
    node: str
    processes: int
    source_sha256: str
    requested_size: int
    effective_size: int
    local_block_size: int
    run: int
    execution_s: Decimal
    computation_s: Decimal
    communication_s: Decimal


def _value(line: str, prefix: str) -> str:
    if not line.startswith(prefix):
        raise ValueError(f"Expected {prefix!r}, found {line!r}")
    return line[len(prefix) :]


def _positive_decimal(line: str, prefix: str) -> Decimal:
    raw_value = _value(line, prefix)
    try:
        value = Decimal(raw_value)
    except InvalidOperation as error:
        raise ValueError(f"Invalid timing value {raw_value!r}") from error
    if not value.is_finite() or value < 0:
        raise ValueError(f"Timing must be finite and nonnegative: {raw_value!r}")
    return value


def _parse_log(path: Path, expected_processes: int) -> list[_Record]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cursor = 0

    def next_line() -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValueError(f"Unexpected end of file in {path.name}")
        line = lines[cursor]
        cursor += 1
        return line

    job_id = int(_value(next_line(), "job_id="))
    partition = _value(next_line(), "partition=")
    node = _value(next_line(), "nodes=")
    processes = int(_value(next_line(), "processes="))
    repetitions = int(_value(next_line(), "repetitions="))
    source_sha256 = _value(next_line(), "source_sha256=")

    if job_id not in EXPECTED_JOBS or EXPECTED_JOBS[job_id] != expected_processes:
        raise ValueError(f"Unexpected job/process mapping in {path.name}")
    if processes != expected_processes:
        raise ValueError(f"Unexpected process count in {path.name}: {processes}")
    if partition != "standard" or node != "n003":
        raise ValueError(f"Unexpected partition or node in {path.name}")
    if repetitions != EXPECTED_REPETITIONS:
        raise ValueError(f"Unexpected repetition count in {path.name}")
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected source checksum in {path.name}")

    records: list[_Record] = []
    for expected_size in EXPECTED_SIZES:
        if next_line() != "case_start":
            raise ValueError(f"Missing case_start in {path.name}")

        requested_size = int(_value(next_line(), "requested_size="))
        effective_size = int(_value(next_line(), "effective_size="))
        local_block_size = int(_value(next_line(), "local_block_size="))

        if requested_size != expected_size or effective_size != requested_size:
            raise ValueError(f"Invalid requested/effective size in {path.name}")
        if requested_size % processes != 0:
            raise ValueError(f"Input size is not divisible by processes in {path.name}")
        if local_block_size != requested_size // processes:
            raise ValueError(f"Invalid local block size in {path.name}")

        for expected_run in range(1, repetitions + 1):
            run = int(_value(next_line(), "run="))
            if run != expected_run:
                raise ValueError(f"Unexpected run sequence in {path.name}")

            execution = _positive_decimal(next_line(), "Ejecucion: ")
            computation = _positive_decimal(next_line(), "Computo: ")
            communication = _positive_decimal(next_line(), "Comunicacion: ")
            if abs(execution - computation - communication) > Decimal("0.000001"):
                raise ValueError(f"Inconsistent timing decomposition in {path.name}")

            records.append(
                _Record(
                    job_id=job_id,
                    partition=partition,
                    node=node,
                    processes=processes,
                    source_sha256=source_sha256,
                    requested_size=requested_size,
                    effective_size=effective_size,
                    local_block_size=local_block_size,
                    run=run,
                    execution_s=execution,
                    computation_s=computation,
                    communication_s=communication,
                )
            )

        if next_line() != "case_end":
            raise ValueError(f"Missing case_end in {path.name}")

    if cursor != len(lines):
        raise ValueError(f"Unexpected trailing output in {path.name}")
    return records


def _load_records(input_dir: Path) -> list[_Record]:
    expected_outputs = {f"ranking_{job_id}.out" for job_id in EXPECTED_JOBS}
    actual_outputs = {path.name for path in input_dir.glob("ranking_*.out")}
    if actual_outputs != expected_outputs:
        raise ValueError("The final log directory does not contain the exact expected outputs")

    records: list[_Record] = []
    for job_id, processes in EXPECTED_JOBS.items():
        output_path = input_dir / f"ranking_{job_id}.out"
        error_path = input_dir / f"ranking_{job_id}.err"
        if not error_path.is_file() or error_path.stat().st_size != 0:
            raise ValueError(f"Missing or nonempty error log for job {job_id}")
        records.extend(_parse_log(output_path, processes))

    expected_rows = len(EXPECTED_JOBS) * len(EXPECTED_SIZES) * EXPECTED_REPETITIONS
    if len(records) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, parsed {len(records)}")
    return records


def _write_csv(records: list[_Record], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=(
                "processes",
                "N",
                "execution_s",
                "computation_s",
                "communication_s",
            ),
        )
        writer.writeheader()
        writer.writerows(
            {
                "processes": record.processes,
                "N": record.requested_size,
                "execution_s": record.execution_s,
                "computation_s": record.computation_s,
                "communication_s": record.communication_s,
            }
            for record in records
        )


def main() -> None:
    """Parse validated ranking-sort logs into a tidy CSV file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Directory containing final logs")
    parser.add_argument("output_csv", type=Path, help="Destination CSV path")
    args = parser.parse_args()

    records = _load_records(args.input_dir)
    _write_csv(records, args.output_csv)
    print(f"Wrote {len(records)} validated rows to {args.output_csv}")


if __name__ == "__main__":
    main()
