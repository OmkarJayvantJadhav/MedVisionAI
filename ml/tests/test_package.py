import medvision_ml


def test_package_exposes_version() -> None:
    assert medvision_ml.__version__ == "0.1.0"
