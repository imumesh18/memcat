from __future__ import annotations

import enum
import json
import os
import subprocess
import sys

import pytest
from pydantic import BaseModel

from memcat.core.keys import Keyer, SettingsNotSerializableError, normalize_text


def _given(**over):
    base = {"model": "sonic-3", "voice": "v1", "language": "en", "speed": 1.0, "api_key": "sk-x"}
    base.update(over)
    return base


def test_canonical_json_matches_spec_layout() -> None:
    k = Keyer("pipecat.services.cartesia.tts.CartesiaTTSService").context(_given(), 8000)
    canon = json.loads(k.canonical("Hello, world. "))
    assert canon == {
        "l": "en",
        "m": "sonic-3",
        "o": {"speed": 1},
        "p": "pipecat.services.cartesia.tts.CartesiaTTSService",
        "r": 8000,
        "s": "s1",
        "t": "hello, world.",
        "v": "v1",
    }
    raw = k.canonical("Hello, world. ")
    assert raw == json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    assert len(k.digest("Hello")) == 16


def test_secrets_never_in_key() -> None:
    k = Keyer("p").context(_given(api_key="sk-secret", auth_token="t", request_timeout=3), 8000)
    c = k.canonical("Hi")
    assert "sk-secret" not in c and "auth" not in c and "timeout" not in c


def test_settings_that_only_look_like_secrets_are_kept() -> None:
    """Secrets are left out by exact name; a setting whose name merely contains
    ``key`` or ``token`` can change the audio and stays in the key."""
    keyer = Keyer("p")
    a = keyer.context(_given(ipa_key="a"), 8000).digest("Hello there")
    assert keyer.context(_given(ipa_key="b"), 8000).digest("Hello there") != a
    assert keyer.context(_given(max_tokens=10), 8000).digest("Hello there") != a
    same = keyer.context(_given(ipa_key="a", session_id="s2", safety_identifier="u2"), 8000)
    assert same.digest("Hello there") == a


def test_normalisation() -> None:
    assert normalize_text("  “Hello”,\n  world’s  ") == '"Hello", world\'s'
    k = Keyer("p").context(_given(), 8000)
    assert k.digest("Hello,  world") == k.digest(" Hello, world ")
    assert k.digest("Hello, world") == k.digest("hello, world")  # opener capital folded
    assert k.digest("US orders ship fast") != k.digest("us orders ship fast")
    assert k.digest("I can help") != k.digest("i can help")
    assert k.digest("the iPhone case") != k.digest("the iphone case")
    assert k.digest("Hello, world.") != k.digest("Hello, world,")


def test_settings_change_changes_key() -> None:
    keyer = Keyer("p")
    a = keyer.context(_given(), 8000).digest("Hello there")
    assert keyer.context(_given(voice="v2"), 8000).digest("Hello there") != a
    assert keyer.context(_given(speed=1.1), 8000).digest("Hello there") != a
    assert keyer.context(_given(), 16000).digest("Hello there") != a
    assert keyer.context(_given(), 8000).digest("Hello there") == a
    assert Keyer("q").context(_given(), 8000).digest("Hello there") != a


def test_floats_enums_models() -> None:
    class Color(enum.Enum):
        RED = "red"

    class Gen(BaseModel):
        speed: float = 1.0000001
        emotion: list[str] = ["calm"]

    k = Keyer("p").context(_given(color=Color.RED, gen=Gen()), 8000)
    o = json.loads(k.canonical("x"))["o"]
    assert o["color"] == "red" and o["gen"] == {"emotion": ["calm"], "speed": 1}


def test_unserialisable_setting_raises() -> None:
    with pytest.raises(SettingsNotSerializableError):
        Keyer("p").context(_given(cb=object()), 8000)


def test_key_stable_across_processes() -> None:
    code = (
        "from memcat.core.keys import Keyer;"
        "k=Keyer('pipecat.x.Y',{'encoding':'pcm'}).context({'model':'m','voice':'v','language':'en',"
        "'z':{'b':2,'a':[1,2.5]},'s':{'x','y','z'}},24000);print(k.digest('Hello, there. ').hex())"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
        )
        outs.add(r.stdout.strip())
    assert len(outs) == 1 and len(next(iter(outs))) == 32
