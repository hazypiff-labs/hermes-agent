"""Per-prefix spool directories for external callback buttons."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from plugins.platforms.telegram.adapter import TelegramAdapter, external_callback_spool_dir


def test_unmapped_prefix_uses_the_shared_spool(tmp_path):
    extra = {"external_callback_spool": str(tmp_path / "shared")}
    assert external_callback_spool_dir(extra, "cr:a:1") == tmp_path / "shared"


def test_mapped_prefix_gets_its_own_spool(tmp_path):
    extra = {
        "external_callback_spool": str(tmp_path / "shared"),
        "external_callback_spools": {"ca:": str(tmp_path / "alerts")},
    }
    assert external_callback_spool_dir(extra, "ca:ack:abc") == tmp_path / "alerts"
    assert external_callback_spool_dir(extra, "cr:a:1") == tmp_path / "shared"


def test_default_spool_without_any_config():
    assert external_callback_spool_dir({}, "cr:a:1") == Path("~/.hermes/callback-spool").expanduser()


def test_blank_mapping_falls_back(tmp_path):
    extra = {"external_callback_spool": str(tmp_path / "shared"), "external_callback_spools": {"ca:": ""}}
    assert external_callback_spool_dir(extra, "ca:ack:abc") == tmp_path / "shared"


def test_ack_tap_is_written_to_the_alert_spool(tmp_path):
    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.config = SimpleNamespace(extra={
        "external_callback_prefixes": ["cr:", "ca:"],
        "external_callback_spool": str(tmp_path / "shared"),
        "external_callback_spools": {"ca:": str(tmp_path / "alerts")},
        "external_callback_allowed_user_ids": ["42"],
    })
    adapter._answer_callback_quietly = AsyncMock()
    adapter._acknowledge_external_callback = AsyncMock()
    query = SimpleNamespace(
        id="cb1", from_user=SimpleNamespace(id=42, username="u", first_name="U"),
        message=SimpleNamespace(chat_id=-100, message_id=7),
    )
    assert asyncio.run(adapter._spool_external_callback_if_configured(query, "ca:ack:abc")) is True
    files = list((tmp_path / "alerts").glob("*.json"))
    assert len(files) == 1 and not (tmp_path / "shared").exists()
    assert json.loads(files[0].read_text())["data"] == "ca:ack:abc"
