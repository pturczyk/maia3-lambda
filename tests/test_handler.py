import base64
import json
import pathlib
import sys
from unittest.mock import MagicMock

# The real model is 92 MB and needs torch. Stubs keep the test offline; every
# case below answers before inference (bad input or a finished game).
for name in ("torch", "maia3", "maia3.dataset", "maia3.uci", "maia3.utils"):
    sys.modules.setdefault(name, MagicMock())

_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
import handler as h

# Fool's mate, so the handler answers before any model call.
_MATED = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
_FOOLS_MATE = ["f2f3", "e7e5", "g2g4", "d8h4"]


def _call(body, headers=None):
    r = h.handler({"headers": headers or {}, "body": json.dumps(body)}, None)
    return r["statusCode"], json.loads(r["body"])


def test_warmup_answers_without_the_model():
    assert h.handler({"warmup": True}, None)["statusCode"] == 200


class TestInput:
    def test_game_over_from_fen(self):
        status, body = _call({"fen": _MATED})
        assert status == 200 and body["topMove"] is None and body["historyPlies"] == 0

    def test_game_over_from_moves_keeps_the_history(self):
        status, body = _call({"moves": _FOOLS_MATE})
        assert status == 200 and body["topMove"] is None and body["historyPlies"] == 4

    def test_moves_and_matching_fen(self):
        status, body = _call({"fen": _MATED, "moves": _FOOLS_MATE})
        assert status == 200 and body["historyPlies"] == 4

    def test_moves_that_miss_the_fen_fall_back_to_the_fen(self):
        status, body = _call({"fen": _MATED, "moves": ["e2e4"]})
        assert status == 200 and body["historyPlies"] == 0

    def test_history_survives_a_fen_with_different_castling_rights(self):
        # Same placement and side to move, castling stripped: still the same game.
        status, body = _call({"fen": _MATED.replace("KQkq", "-"), "moves": _FOOLS_MATE})
        assert status == 200 and body["historyPlies"] == 4

    def test_illegal_move_without_fen_is_rejected(self):
        status, body = _call({"moves": ["e2e5"]})
        assert status == 400 and "illegal" in body["error"]

    def test_garbage_move_without_fen_is_rejected(self):
        assert _call({"moves": ["zz"]})[0] == 400

    def test_moves_must_be_a_list(self):
        assert _call({"moves": "e2e4"})[0] == 400

    def test_too_many_moves_rejected(self):
        assert _call({"moves": ["e2e4"] * 1001})[0] == 400

    def test_invalid_fen_rejected(self):
        assert _call({"fen": "not a fen"})[0] == 400

    def test_history_is_capped_at_eight_positions(self):
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"] * 3 + ["f2f3", "e7e5", "g2g4", "d8h4"]
        status, body = _call({"moves": moves})
        assert status == 200 and body["historyPlies"] == 7

    def test_start_fen(self):
        start = "rnbqkbnr/pppppppp/8/8/8/5P2/PPPPP1PP/RNBQKBNR b KQkq - 0 1"
        status, body = _call({"startFen": start, "moves": ["e7e5", "g2g4", "d8h4"]})
        assert status == 200 and body["topMove"] is None and body["historyPlies"] == 3

    def test_base64_body_from_a_function_url_is_decoded(self):
        event = {"headers": {}, "isBase64Encoded": True,
                 "body": base64.b64encode(json.dumps({"fen": _MATED}).encode()).decode()}
        assert h.handler(event, None)["statusCode"] == 200


class TestCallerTag:
    def _line(self, capsys, headers):
        assert _call({"fen": _MATED}, headers)[0] == 200
        return next(l for l in capsys.readouterr().out.splitlines() if "maia3 fen=" in l)

    def test_tier_and_client_headers_are_logged(self, capsys):
        headers = {"x-tier": "premium", "x-client": "extension"}
        assert self._line(capsys, headers).endswith("tier=premium client=extension")

    def test_defaults_when_the_headers_are_missing(self, capsys):
        assert self._line(capsys, {}).endswith("tier=free client=app")
