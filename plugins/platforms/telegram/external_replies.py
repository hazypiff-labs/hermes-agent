"""Durable replies bound to prompts created by external callback consumers."""
import json
import os
import time
from pathlib import Path

async def spool_reply(adapter, msg):
    extra = getattr(adapter.config, "extra", None) or {}
    if not extra.get("external_callback_prefixes"):
        return False
    replied = getattr(msg, "reply_to_message", None)
    chat = str(getattr(msg, "chat_id", ""))
    prompt = getattr(replied, "message_id", None)
    if not chat.lstrip("-").isdigit() or not isinstance(prompt, int):
        return False
    spool = Path(os.path.expanduser(str(extra.get("external_callback_spool") or "~/.hermes/callback-spool")))
    path = spool / ".replies" / f"{chat}_{prompt}.json"
    if not path.is_file():
        return False
    binding = json.loads(path.read_text())
    user = getattr(msg, "from_user", None)
    allowed = extra.get("external_callback_allowed_user_ids") or []
    if isinstance(allowed, (str, int)):
        allowed = [allowed]
    if (str(getattr(user, "id", "")) != binding.get("user_id")
            or str(getattr(user, "id", "")) not in {str(x) for x in allowed}
            or binding.get("chat_id") != chat or binding.get("prompt_id") != prompt):
        await msg.reply_text("This reply prompt belongs to another reviewer.")
        return True
    if float(binding.get("expires_at") or 0) < time.time():
        path.unlink(missing_ok=True)
        await msg.reply_text("This edit request expired. Tap the edit button again.")
        return True
    record = {"callback_id": "", "data": binding["data"],
              "from": {"id": user.id, "username": getattr(user, "username", None),
                       "first_name": getattr(user, "first_name", None)},
              "chat_id": chat, "message_id": binding["message_id"],
              "reply_text": msg.text, "reply_binding": binding}
    final = spool / f"reply-{chat}-{msg.message_id}.json"
    temporary = final.with_suffix(".tmp")
    temporary.write_text(json.dumps(record))
    os.chmod(temporary, 0o600)
    temporary.replace(final)
    path.unlink(missing_ok=True)
    await msg.reply_text("Edit received.")
    return True
