"""A Twilio phone bot whose Cartesia voice is cached by Memcat.

Telephony runs at 8 kHz, so Memcat stores mu-law: the exact samples the caller
hears, at 464 KB per cached minute.

    uv add "pipecat-ai[runner,silero,openai,deepgram,cartesia]" "memcat[redis]"
    export DEEPGRAM_API_KEY=... OPENAI_API_KEY=... CARTESIA_API_KEY=... REDIS_URL=redis://...
    uv run examples/twilio_bot.py -t twilio -x <your-ngrok-host>

Point the Twilio number's voice webhook at the runner (see Pipecat's telephony
docs); every call then runs ``bot()`` below.
"""

from __future__ import annotations

from _bot import run_bot
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams


async def bot(runner_args: RunnerArguments) -> None:
    transport = await create_transport(
        runner_args,
        {"twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True)},
    )
    await run_bot(transport, sample_rate=8000, handle_sigint=runner_args.handle_sigint)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
