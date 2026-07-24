import os

import pytest

from em_volume_tools.aws import ensure_aws_credentials


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
              "AWS_PROFILE", "AWS_SHARED_CREDENTIALS_FILE"):
        monkeypatch.delenv(k, raising=False)


def _write_creds(path, profile="default", token=False):
    lines = [f"[{profile}]",
             "aws_access_key_id = AKIAFAKE123",
             "aws_secret_access_key = fakesecret456"]
    if token:
        lines.append("aws_session_token = faketoken789")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def test_loads_default_profile(tmp_path, clean_env):
    creds = str(tmp_path / "credentials")
    _write_creds(creds, token=True)
    assert ensure_aws_credentials(credentials_file=creds) is True
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAFAKE123"
    assert os.environ["AWS_SECRET_ACCESS_KEY"] == "fakesecret456"
    assert os.environ["AWS_SESSION_TOKEN"] == "faketoken789"


def test_noop_when_already_set(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "preexisting")
    creds = str(tmp_path / "credentials")
    _write_creds(creds)
    assert ensure_aws_credentials(credentials_file=creds) is False
    assert os.environ["AWS_ACCESS_KEY_ID"] == "preexisting"


def test_noop_when_missing_file(tmp_path, clean_env):
    assert ensure_aws_credentials(credentials_file=str(tmp_path / "nope")) is False
    assert "AWS_ACCESS_KEY_ID" not in os.environ


def test_named_profile(tmp_path, clean_env):
    creds = str(tmp_path / "credentials")
    _write_creds(creds, profile="myprofile")
    assert ensure_aws_credentials(profile="myprofile", credentials_file=creds) is True
    assert os.environ["AWS_ACCESS_KEY_ID"] == "AKIAFAKE123"
