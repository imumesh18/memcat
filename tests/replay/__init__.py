"""Record real provider traffic once, replay it in tests with its real timing.

``Recorder`` sits between a Pipecat service and the provider, adds the API key
upstream and logs every WebSocket message and HTTP chunk with its time.
``Player`` serves a recording back to the same Pipecat service: each reply leaves
at its recorded delay after the client message that triggered it, and every
client message must match the recording (context ids are remapped per run).
"""

from tests.replay.cassette import Cassette, HttpExchange, WsConn, WsEvent
from tests.replay.player import Player
from tests.replay.recorder import Recorder

__all__ = ["Cassette", "HttpExchange", "Player", "Recorder", "WsConn", "WsEvent"]
