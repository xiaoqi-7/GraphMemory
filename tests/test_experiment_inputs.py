from __future__ import annotations

from pathlib import Path
from typing import cast

import huggingface_hub
import pytest

from graph_memory.experiment import inputs
from graph_memory.experiment.config import (
    Bm25MethodConfig,
    ISETraceChunkingConfig,
    ResolvedDatasetConfig,
    ResolvedExperimentConfig,
)
from scripts import prepare_dataset


def test_prepare_dataset_retries_huggingface_download_once_via_mirror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_url = "https://huggingface.co/datasets/example/data/resolve/rev/train.jsonl"
    spec = prepare_dataset.DatasetSpec(
        dataset="example",
        display_name="Example",
        files=(
            prepare_dataset.DatasetFile(
                split="train",
                filename="nested/train.jsonl",
                url=source_url,
            ),
        ),
    )
    observed_urls: list[str] = []

    def download(url: str, destination: Path, _size: int | None) -> None:
        observed_urls.append(url)
        if len(observed_urls) == 1:
            raise OSError("primary endpoint unavailable")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        prepare_dataset,
        "SUMMARY_DIR",
        tmp_path / "summaries",
    )

    result = prepare_dataset.main(
        [
            "--dataset",
            "example",
            "--name",
            "example",
            "--data_dir",
            str(tmp_path / "data"),
            "--no_verify",
        ],
        registry={"example": spec},
        downloader=download,
        show_progress=False,
    )

    assert result == 0
    assert observed_urls == [
        source_url,
        "https://hf-mirror.com/datasets/example/data/resolve/rev/train.jsonl",
    ]
    assert (tmp_path / "data/example/raw/nested/train.jsonl").exists()


def test_prepare_dataset_can_use_huggingface_mirror_directly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_url = "https://huggingface.co/datasets/example/data/resolve/rev/train.jsonl"
    spec = prepare_dataset.DatasetSpec(
        dataset="example",
        display_name="Example",
        files=(
            prepare_dataset.DatasetFile(
                split="train",
                filename="train.jsonl",
                url=source_url,
            ),
        ),
    )
    observed_urls: list[str] = []

    def download(url: str, destination: Path, _size: int | None) -> None:
        observed_urls.append(url)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        prepare_dataset,
        "SUMMARY_DIR",
        tmp_path / "summaries",
    )

    result = prepare_dataset.main(
        [
            "--dataset",
            "example",
            "--name",
            "example",
            "--data_dir",
            str(tmp_path / "data"),
            "--no_verify",
            "--mirror",
        ],
        registry={"example": spec},
        downloader=download,
        show_progress=False,
    )

    assert result == 0
    assert observed_urls == [
        "https://hf-mirror.com/datasets/example/data/resolve/rev/train.jsonl"
    ]


def test_encoder_download_uses_official_huggingface_endpoint_and_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    inputs._ensure_model(
        "models/intfloat-e5-base-v2",
        repository_root=tmp_path,
    )
    inputs._ensure_model(
        "models/BAAI-bge-reranker-base",
        repository_root=tmp_path,
    )

    assert calls == [
        {
            "repo_id": "intfloat/e5-base-v2",
            "revision": "f52bf8ec8c7124536f0efb74aca902b2995e5bcd",
            "local_dir": str(tmp_path / "models/intfloat-e5-base-v2"),
            "endpoint": inputs.HUGGINGFACE_BASE_URL,
        },
        {
            "repo_id": "BAAI/bge-reranker-base",
            "revision": "2cfc18c9415c912f9d8155881c133215df768a70",
            "local_dir": str(tmp_path / "models/BAAI-bge-reranker-base"),
            "endpoint": inputs.HUGGINGFACE_BASE_URL,
        },
    ]


def test_isetrace_chunk_tokenizer_is_provisioned_for_bm25(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []

    def ensure_model(model_name: str, *, repository_root: Path) -> None:
        calls.append((model_name, repository_root))

    monkeypatch.setattr(inputs, "_ensure_model", ensure_model)
    dataset = ResolvedDatasetConfig.model_construct(
        name="isetrace",
        chunking=ISETraceChunkingConfig(
            tokenizer_name="models/intfloat-e5-base-v2",
        ),
        splits={},
    )
    config = ResolvedExperimentConfig.model_construct(
        dataset=dataset,
        method=Bm25MethodConfig(method="bm25"),
    )

    inputs.ensure_encoder_models(config, repository_root=tmp_path)

    assert calls == [("models/intfloat-e5-base-v2", tmp_path)]


def test_isetrace_download_uses_pinned_public_datasets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        patterns = cast(list[object], kwargs["allow_patterns"])
        for filename in patterns:
            assert isinstance(filename, str)
            target = local_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            _ = target.write_text("{}\n", encoding="utf-8")
        return str(local_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    dataset = ResolvedDatasetConfig.model_construct(
        name="isetrace",
        trajectory_source=Path("data/isetrace/raw/trajectories"),
        natural_query_source=Path(
            "data/isetrace/query-authoring/isetrace-v7-raw.jsonl"
        ),
        splits={},
    )
    config = ResolvedExperimentConfig.model_construct(dataset=dataset)

    inputs.ensure_dataset(config, repository_root=tmp_path)
    inputs.ensure_dataset(config, repository_root=tmp_path)

    assert calls == [
        {
            "repo_id": "valiere/ISETrace",
            "repo_type": "dataset",
            "revision": "e40e04d41c04e4eb4bae181ebdd41b61c688081b",
            "local_dir": str(tmp_path / "data/isetrace/raw"),
            "allow_patterns": [
                f"trajectories/trajectories-{index:05d}.jsonl"
                for index in range(8)
            ],
            "endpoint": inputs.HUGGINGFACE_BASE_URL,
        },
        {
            "repo_id": "HazeLocus/ISETrace-Memory-Queries",
            "repo_type": "dataset",
            "revision": "3130aeeaeb2b8d292c29f4c8e23303be7463732c",
            "local_dir": str(tmp_path / "data/isetrace"),
            "allow_patterns": [
                "query-authoring/isetrace-v7-raw.jsonl",
                "query-authoring/isetrace-v7-raw.jsonl.metadata.jsonl",
            ],
            "endpoint": inputs.HUGGINGFACE_BASE_URL,
        },
    ]


def test_prepare_dataset_does_not_mirror_non_huggingface_downloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = prepare_dataset.DatasetSpec(
        dataset="example",
        display_name="Example",
        files=(
            prepare_dataset.DatasetFile(
                split="train",
                filename="train.json",
                url="https://example.com/train.json",
            ),
        ),
    )
    calls = 0

    def fail(_url: str, _destination: Path, _size: int | None) -> None:
        nonlocal calls
        calls += 1
        raise OSError("unavailable")

    monkeypatch.setattr(
        prepare_dataset,
        "SUMMARY_DIR",
        tmp_path / "summaries",
    )

    with pytest.raises(OSError, match="unavailable"):
        prepare_dataset.main(
            [
                "--dataset",
                "example",
                "--name",
                "example",
                "--data_dir",
                str(tmp_path / "data"),
                "--no_verify",
            ],
            registry={"example": spec},
            downloader=fail,
            show_progress=False,
        )

    assert calls == 1
