from orchestrate import Orchestrator, OllamaTransport, AgentRequest
from audio.tts import PiperSpeaker
from main import run_one_turn
from config import AGENT_CONFIG
from identity.voice_id import get_current_user

transport = OllamaTransport(model=AGENT_CONFIG.model)
orchestrator = Orchestrator(transport, max_retries=5, verbose=True)
speaker = PiperSpeaker()

run_one_turn(orchestrator, speaker, get_current_user(), "what is today's date")
