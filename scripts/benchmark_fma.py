#!/usr/bin/env python3
"""Run a reproducible, stratified FMA-small side-HF recovery benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
import tempfile
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Track:
    track_id: int
    genre: str
    artist: str
    title: str
    license: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="directory containing fma_small/ and fma_metadata/",
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path("target/release/sidespread"),
        help="Sidespread release binary",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("benchmark-results/fma-small.jsonl"),
        help="append-only per-track result file",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("benchmark-results/fma-small-summary.json"),
        help="aggregated result file",
    )
    parser.add_argument("--per-genre", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--cutoffs", type=parse_int_list, default=[8000, 16000])
    parser.add_argument(
        "--thresholds",
        type=parse_float_list,
        default=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    )
    parser.add_argument(
        "--transition-threshold",
        type=float,
        default=0.40,
        help="minimum outer-transition correlation used as the LOW routing threshold",
    )
    parser.add_argument(
        "--min-reference-hf-ratio",
        type=float,
        default=1e-4,
        help="exclude tracks whose reference side has too little energy above the cutoff",
    )
    parser.add_argument(
        "--regression-tolerance-db",
        type=float,
        default=0.05,
        help="count HF-SNR deltas below the negative of this value as regressions",
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="set RAYON_NUM_THREADS for each Sidespread process",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def read_tracks(metadata_path: Path) -> list[Track]:
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = csv.reader(handle)
        groups = next(rows)
        fields = next(rows)
        next(rows)  # pandas index-name row
        columns = {
            (group, field): index
            for index, (group, field) in enumerate(zip(groups, fields))
        }
        required = [
            ("set", "subset"),
            ("track", "genre_top"),
            ("track", "license"),
            ("track", "title"),
            ("artist", "name"),
        ]
        missing = [column for column in required if column not in columns]
        if missing:
            raise ValueError(f"tracks.csv is missing columns: {missing}")

        tracks = []
        for row in rows:
            if row[columns[("set", "subset")]] != "small":
                continue
            tracks.append(
                Track(
                    track_id=int(row[0]),
                    genre=row[columns[("track", "genre_top")]],
                    artist=row[columns[("artist", "name")]],
                    title=row[columns[("track", "title")]],
                    license=row[columns[("track", "license")]],
                )
            )
        return tracks


def stratified_candidates(tracks: Iterable[Track], seed: int) -> dict[str, list[Track]]:
    by_genre: dict[str, list[Track]] = defaultdict(list)
    for track in tracks:
        by_genre[track.genre].append(track)
    rng = random.Random(seed)
    ordered = {}
    for genre in sorted(by_genre):
        candidates = sorted(by_genre[genre], key=lambda track: track.track_id)
        rng.shuffle(candidates)
        ordered[genre] = candidates
    return ordered


def audio_path(dataset_root: Path, track_id: int) -> Path:
    identifier = f"{track_id:06d}"
    return dataset_root / "fma_small" / identifier[:3] / f"{identifier}.mp3"


def convert_audio(source: Path, destination: Path) -> tuple[int, float]:
    subprocess.run(
        [
            "afconvert",
            str(source),
            "-o",
            str(destination),
            "-f",
            "WAVE",
            "-d",
            "LEI24@44100",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    with wave.open(str(destination), "rb") as reader:
        channels = reader.getnchannels()
        duration = reader.getnframes() / reader.getframerate()
    return channels, duration


def result_key(
    track_id: int,
    cutoff: int,
    threshold: float,
    transition_threshold: float,
) -> tuple[int, int, str, str]:
    return (
        track_id,
        cutoff,
        f"{threshold:.6f}",
        f"{transition_threshold:.6f}",
    )


def load_results(path: Path) -> dict[tuple[int, int, str, str], dict[str, Any]]:
    results = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = result_key(
                    int(row["track_id"]),
                    int(row["cutoff_hz"]),
                    float(row["corr_threshold"]),
                    float(row.get("transition_threshold", 0.15)),
                )
                results[key] = row
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid result at {path}:{line_number}: {error}") from error
    return results


def run_evaluation(
    binary: Path,
    clean_wav: Path,
    repaired_wav: Path,
    report_path: Path,
    cutoff: int,
    threshold: float,
    transition_threshold: float,
    threads: int | None,
) -> dict[str, Any]:
    environment = None
    if threads is not None:
        import os

        environment = os.environ.copy()
        environment["RAYON_NUM_THREADS"] = str(threads)
    try:
        subprocess.run(
            [
                str(binary),
                "eval",
                str(clean_wav),
                "--output",
                str(repaired_wav),
                "--mode",
                "auto",
                "--fc",
                str(cutoff),
                "--corr-threshold",
                f"{threshold:.6f},{transition_threshold:.6f}",
                "--report",
                str(report_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() if error.stderr else str(error)
        raise RuntimeError(f"Sidespread evaluation failed: {detail}") from error
    with report_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def make_row(
    track: Track,
    duration: float,
    cutoff: int,
    threshold: float,
    transition_threshold: float,
    report: dict[str, Any],
) -> dict[str, Any]:
    segments = report["segments"]
    deficient = sum(segment["needs_processing"] for segment in segments)
    dsp = sum(segment["route"] == "dsp" for segment in segments)
    degraded = report["evaluation"]["degraded"]
    repaired = report["evaluation"]["repaired"]
    return {
        "track_id": track.track_id,
        "genre": track.genre,
        "artist": track.artist,
        "title": track.title,
        "license": track.license,
        "duration_seconds": duration,
        "cutoff_hz": cutoff,
        "corr_threshold": threshold,
        "transition_threshold": transition_threshold,
        "reference_hf_ratio": degraded["reference_hf_ratio"],
        "degraded_hf_snr_db": degraded["snr_hf_db"],
        "repaired_hf_snr_db": repaired["snr_hf_db"],
        "degraded_lsd_hf_db": degraded["lsd_hf"],
        "repaired_lsd_hf_db": repaired["lsd_hf"],
        "total_segments": len(segments),
        "deficient_segments": deficient,
        "dsp_segments": dsp,
    }


def percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate(
    rows: Iterable[dict[str, Any]],
    cutoffs: list[int],
    thresholds: list[float],
    transition_threshold: float,
    min_hf_ratio: float,
    regression_tolerance: float,
) -> dict[str, Any]:
    groups: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["cutoff_hz"]),
            f"{float(row['corr_threshold']):.6f}",
            f"{float(row.get('transition_threshold', 0.15)):.6f}",
        )
        if key[0] not in cutoffs or float(key[1]) not in thresholds:
            continue
        if float(key[2]) != transition_threshold:
            continue
        ratio = row.get("reference_hf_ratio")
        if ratio is None or ratio < min_hf_ratio:
            continue
        if row.get("degraded_hf_snr_db") is None or row.get("repaired_hf_snr_db") is None:
            continue
        groups[key].append(row)

    summaries = []
    for cutoff in cutoffs:
        for threshold in thresholds:
            selected = groups.get(
                (cutoff, f"{threshold:.6f}", f"{transition_threshold:.6f}"), []
            )
            deltas = [
                row["repaired_hf_snr_db"] - row["degraded_hf_snr_db"]
                for row in selected
            ]
            lsd_deltas = [
                row["repaired_lsd_hf_db"] - row["degraded_lsd_hf_db"]
                for row in selected
            ]
            repaired_tracks = sum(row["dsp_segments"] > 0 for row in selected)
            deficient_segments = sum(row["deficient_segments"] for row in selected)
            dsp_segments = sum(row["dsp_segments"] for row in selected)
            genre_counts = defaultdict(int)
            genre_regressions = defaultdict(int)
            for row, delta in zip(selected, deltas):
                genre_counts[row["genre"]] += 1
                if delta < -regression_tolerance:
                    genre_regressions[row["genre"]] += 1
            summaries.append(
                {
                    "cutoff_hz": cutoff,
                    "corr_threshold": threshold,
                    "transition_threshold": transition_threshold,
                    "eligible_tracks": len(selected),
                    "repaired_tracks": repaired_tracks,
                    "improved_tracks": sum(delta > regression_tolerance for delta in deltas),
                    "regressed_tracks": sum(delta < -regression_tolerance for delta in deltas),
                    "mean_hf_snr_delta_db": statistics.fmean(deltas) if deltas else None,
                    "median_hf_snr_delta_db": statistics.median(deltas) if deltas else None,
                    "p10_hf_snr_delta_db": percentile(deltas, 0.10) if deltas else None,
                    "minimum_hf_snr_delta_db": min(deltas) if deltas else None,
                    "mean_lsd_hf_delta_db": statistics.fmean(lsd_deltas) if lsd_deltas else None,
                    "repair_coverage": dsp_segments / deficient_segments
                    if deficient_segments
                    else 0.0,
                    "eligible_by_genre": dict(sorted(genre_counts.items())),
                    "regressions_by_genre": dict(sorted(genre_regressions.items())),
                }
            )
    return {
        "min_reference_hf_ratio": min_hf_ratio,
        "regression_tolerance_db": regression_tolerance,
        "groups": summaries,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(
        "cutoff intact transition tracks repaired improved regressed "
        "mean_delta median_delta p10_delta coverage"
    )
    for row in summary["groups"]:
        mean = row["mean_hf_snr_delta_db"]
        median = row["median_hf_snr_delta_db"]
        p10 = row["p10_hf_snr_delta_db"]
        print(
            f"{row['cutoff_hz']:6d} {row['corr_threshold']:6.2f} "
            f"{row['transition_threshold']:10.2f} "
            f"{row['eligible_tracks']:6d} {row['repaired_tracks']:8d} "
            f"{row['improved_tracks']:8d} {row['regressed_tracks']:9d} "
            f"{mean if mean is not None else float('nan'):10.3f} "
            f"{median if median is not None else float('nan'):12.3f} "
            f"{p10 if p10 is not None else float('nan'):9.3f} "
            f"{row['repair_coverage']:8.3f}"
        )


def main() -> int:
    args = parse_args()
    if args.per_genre < 1:
        raise ValueError("--per-genre must be positive")
    if args.min_reference_hf_ratio < 0.0:
        raise ValueError("--min-reference-hf-ratio must be non-negative")
    dataset_root = args.dataset_root.resolve()
    metadata_path = dataset_root / "fma_metadata" / "tracks.csv"
    binary = args.binary.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not binary.is_file():
        raise FileNotFoundError(f"build the release binary first: {binary}")

    candidates_by_genre = stratified_candidates(read_tracks(metadata_path), args.seed)
    for genre, candidates in candidates_by_genre.items():
        if len(candidates) < args.per_genre:
            raise ValueError(f"genre {genre!r} has only {len(candidates)} tracks")
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    results = load_results(args.results)
    failures = []
    selected = []
    target_tracks = len(candidates_by_genre) * args.per_genre

    with tempfile.TemporaryDirectory(prefix="sidespread-fma-") as temporary:
        temporary_path = Path(temporary)
        for genre, candidates in sorted(candidates_by_genre.items()):
            genre_selected = 0
            for track in candidates:
                if genre_selected >= args.per_genre:
                    break
                pending = [
                    (cutoff, threshold)
                    for cutoff in args.cutoffs
                    for threshold in args.thresholds
                    if result_key(
                        track.track_id,
                        cutoff,
                        threshold,
                        args.transition_threshold,
                    )
                    not in results
                ]
                if not pending:
                    selected.append(track)
                    genre_selected += 1
                    continue
                source = audio_path(dataset_root, track.track_id)
                clean_wav = temporary_path / "clean.wav"
                repaired_wav = temporary_path / "repaired.wav"
                report_path = temporary_path / "report.json"
                try:
                    if not source.is_file():
                        raise FileNotFoundError(source)
                    channels, duration = convert_audio(source, clean_wav)
                    if channels != 2:
                        raise ValueError(f"expected stereo audio, found {channels} channel(s)")
                    if duration < 1.0:
                        raise ValueError(f"audio is only {duration:.3f} seconds long")
                    print(
                        f"[{len(selected) + 1}/{target_tracks}] {track.track_id:06d} "
                        f"{track.genre}: {len(pending)} evaluations",
                        flush=True,
                    )
                    for cutoff, threshold in pending:
                        report = run_evaluation(
                            binary,
                            clean_wav,
                            repaired_wav,
                            report_path,
                            cutoff,
                            threshold,
                            args.transition_threshold,
                            args.threads,
                        )
                        row = make_row(
                            track,
                            duration,
                            cutoff,
                            threshold,
                            args.transition_threshold,
                            report,
                        )
                        with args.results.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        results[
                            result_key(
                                track.track_id,
                                cutoff,
                                threshold,
                                args.transition_threshold,
                            )
                        ] = row
                    selected.append(track)
                    genre_selected += 1
                except (OSError, ValueError, subprocess.CalledProcessError) as error:
                    failures.append({"track_id": track.track_id, "error": str(error)})
                    print(f"warning: skipped {track.track_id:06d}: {error}", file=sys.stderr)
            if genre_selected < args.per_genre:
                raise RuntimeError(
                    f"only found {genre_selected} valid {genre} tracks; "
                    f"needed {args.per_genre}"
                )

    selected_ids = {track.track_id for track in selected}
    current_rows = [row for row in results.values() if row["track_id"] in selected_ids]
    summary = aggregate(
        current_rows,
        args.cutoffs,
        args.thresholds,
        args.transition_threshold,
        args.min_reference_hf_ratio,
        args.regression_tolerance_db,
    )
    summary.update(
        {
            "dataset": "FMA-small",
            "sample_seed": args.seed,
            "requested_tracks_per_genre": args.per_genre,
            "selected_tracks": len(selected),
            "failures": failures,
        }
    )
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)
    print(f"wrote {args.results} and {args.summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
