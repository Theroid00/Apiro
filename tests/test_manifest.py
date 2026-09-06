from apiro.eval.manifest import build_manifest, create_run_directory


def test_manifest_directory_is_immutable(tmp_path):
    manifest = build_manifest(
        benchmark="fixture", dataset="local/test", revision="v1",
        case_ids=["a", "b"], config={"seed": 7},
    )
    path = create_run_directory(manifest, tmp_path)
    assert (path / "manifest.json").exists()

    import pytest
    with pytest.raises(FileExistsError):
        create_run_directory(manifest, tmp_path)
