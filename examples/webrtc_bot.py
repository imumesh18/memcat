"""A browser (WebRTC) bot whose Cartesia voice is cached by Memcat.

At 24 kHz Memcat stores Opus when the extra is installed (about 240 KB per cached
minute) and PCM16 otherwise.

    uv add "pipecat-ai[runner,webrtc,silero,openai,deepgram,cartesia]" "memcat[redis,opus]"
    export DEEPGRAM_API_KEY=... OPENAI_API_KEY=... CARTESIA_API_KEY=...
    uv run examples/webrtc_bot.py      # then open http://localhost:7860/client
"""

from __future__ import annotations

from _bot import run_bot
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import TransportParams


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )
    await run_bot(transport, sample_rate=24000, handle_sigint=runner_args.handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
