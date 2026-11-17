"""
Maia-3 Lambda handler.

Predicts the move a human player at a given Elo would most likely play, using
the Maia-3 23M model (CSSLab, https://github.com/CSSLab/maia3, AGPL-3.0), then
chains predictions to build a PV (both sides predicted at their own Elo).

Unlike Maia-2, Maia-3 reads up to 8 positions of history. Callers that know how
the game reached the position send the moves, which is worth about 3 points of
move-match accuracy. Without them the current position is repeated as history.

Request body (JSON):
  fen          string    position to predict (optional when `moves` is sent)
  moves        [string]  UCI moves that led to `fen` (optional)
  startFen     string    position `moves` start from (default: standard start)
  elo          int       Elo of the side to move (clamped, see _ELO_MIN/_ELO_MAX)
  opponentElo  int       opponent Elo (optional; defaults to `elo`)
  numMoves     int       alternative first moves to return (default 5, max 10)
  pvDepth      int       half-moves in the PV continuation (default 10, max 20)

When both `fen` and `moves` are sent and the moves do not arrive at `fen`, the
history is dropped and `fen` alone is used (`historyPlies` reports 0).

Response body (JSON):
  topMove       string|null         most likely first move (UCI)
  topMoves      [{move, prob}]      first-ply moves by probability, descending
  pv            [string]            predicted continuation (UCI)
  elo           int                 clamped Elo actually used
  wdl           {win, draw, loss}   Maia-3 value head for the side to move: the
                                    outcome of a human game at these ratings,
                                    not an engine evaluation
  historyPlies  int                 earlier positions fed to the model (0-7)
"""

import base64
import json
import os
import re
from collections import deque

import chess

_ELO_MIN = 600
_ELO_MAX = 2800
_HISTORY = 8          # positions the model reads, current one included
_MAX_MOVES = 1000     # bound on the replay work a request can ask for
_CHECKPOINT = os.environ.get("MAIA3_CHECKPOINT", "/opt/maia3/maia3-23m.pt")


def _caller(headers) -> str:
    """`tier=free client=app` tag for the request's log line, as every Lambda logs it."""
    def word(value, default):
        value = re.sub(r"[^a-z0-9_-]", "", (value or "").lower())[:32]
        return value or default
    return f"tier={word(headers.get('x-tier'), 'free')} client={word(headers.get('x-client'), 'app')}"


# Squares where the knights start. Returning a knight there is almost never the
# move, but a chained argmax PV falls into Nf3-Nf6-Ng1-Ng8 loops. Same filter
# as the Maia-2 Lambda so the two answer alike.
_WHITE_KNIGHT_HOME = {"b1", "g1"}
_BLACK_KNIGHT_HOME = {"b8", "g8"}


def _is_knight_return_home(board, uci: str) -> bool:
    try:
        from_sq = chess.parse_square(uci[:2])
    except ValueError:
        return False
    piece = board.piece_at(from_sq)
    if piece is None or piece.piece_type != chess.KNIGHT:
        return False
    home = _WHITE_KNIGHT_HOME if piece.color == chess.WHITE else _BLACK_KNIGHT_HOME
    return uci[2:4] in home


# ---------------------------------------------------------------------------
# Model loading, once per container.
# ---------------------------------------------------------------------------

_model_error: str | None = None
_MODEL = None

try:
    import torch

    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

    from maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
    from maia3.uci import load_model, parse_args
    from maia3.utils import get_all_possible_moves, mirror_move

    _CFG = parse_args(["--model", "maia3-23m", "--checkpoint-path", _CHECKPOINT, "--device", "cpu"])
    _MODEL = load_model(_CFG)
    _ALL_MOVES = get_all_possible_moves()
    _MOVE_INDEX = {m: i for i, m in enumerate(_ALL_MOVES)}
except Exception as exc:
    _model_error = f"Failed to load Maia-3 model: {type(exc).__name__}: {exc}"
    print(_model_error)


def _predict(history: list, elo: int, opponent_elo: int):
    """Policy over legal moves plus the value head for the last board in `history`.

    `history` holds python-chess boards oldest first; the model reads the last
    _HISTORY of them and pads with the earliest when there are fewer.
    """
    board = history[-1]
    tokens = deque((tokenize_board(b) for b in history[-_HISTORY:]), maxlen=_HISTORY)
    x = get_historical_tokens(tokens, _CFG, base=0.0, inc=0.0, clk_left_before=0.0, clk_ponder=0.0)
    with torch.no_grad():
        logits_move, logits_value, _ = _MODEL(
            x.unsqueeze(0), torch.tensor([elo]), torch.tensor([opponent_elo])
        )
    mask = get_legal_moves_mask(board, _MOVE_INDEX)
    probs = torch.softmax(logits_move[0].float().masked_fill(~mask, float("-inf")), dim=-1)
    out = {}
    for idx in torch.nonzero(mask).flatten().tolist():
        uci = _ALL_MOVES[idx]
        # The model sees the board from the side to move, so black's moves come back mirrored.
        if board.turn == chess.BLACK:
            uci = mirror_move(uci)
        out[uci] = float(probs[idx])
    loss, draw, win = torch.softmax(logits_value[0].float(), dim=-1).tolist()
    return out, {"win": round(win, 4), "draw": round(draw, 4), "loss": round(loss, 4)}


def _same_position(a: chess.Board, b: chess.Board) -> bool:
    # Placement and side to move only. Positions stored by a client can disagree
    # on castling or en passant (a scanned board guesses them), and the model
    # reads neither; the caller's `fen` then supplies them for the final board.
    return a.board_fen() == b.board_fen() and a.turn == b.turn


def _build_history(body: dict) -> tuple[list | None, str | None]:
    """Boards oldest first ending at the position to predict, or (None, error)."""
    fen = body.get("fen")
    moves = body.get("moves") or []
    if not isinstance(moves, list) or len(moves) > _MAX_MOVES:
        return None, "moves must be a list of at most %d UCI moves" % _MAX_MOVES
    try:
        target = chess.Board(fen) if fen else None
    except ValueError:
        return None, "Invalid FEN"

    if moves:
        try:
            board = chess.Board(body.get("startFen") or chess.STARTING_FEN)
            boards = [board.copy(stack=False)]
            for uci in moves:
                move = chess.Move.from_uci(str(uci))
                if move not in board.legal_moves:
                    raise ValueError(f"illegal move {uci}")
                board.push(move)
                boards.append(board.copy(stack=False))
                del boards[:-_HISTORY]
            if target is None:
                return boards, None
            if _same_position(board, target):
                return boards[:-1] + [target], None
            print(f"history does not reach fen, dropping it: {fen}")
        except ValueError as exc:
            if target is None:
                return None, f"Invalid moves: {exc}"
            print(f"history unusable, dropping it: {exc}")

    return [target or chess.Board()], None


# ---------------------------------------------------------------------------

def handler(event, context):
    # Scheduled warmup ping. Exits before model use.
    if event.get("warmup") is True:
        return {"statusCode": 200, "body": "warm"}

    headers = event.get("headers") or {}
    trace_id = headers.get("x-trace-id", "")

    if _MODEL is None:
        return _response(503, {"error": _model_error or "Model not loaded"})

    try:
        raw = event.get("body") or "{}"
        # Function URLs base64 the body when the request has no text Content-Type.
        if event.get("isBase64Encoded"):
            raw = base64.b64decode(raw).decode()
        body = json.loads(raw)
        elo = max(_ELO_MIN, min(_ELO_MAX, int(body.get("elo", 1500))))
        opponent_elo = max(_ELO_MIN, min(_ELO_MAX, int(body.get("opponentElo", elo))))
        num_moves = min(max(1, int(body.get("numMoves", 5))), 10)
        pv_depth = min(max(1, int(body.get("pvDepth", 10))), 20)
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
        return _response(400, {"error": "Invalid JSON body"})

    history, error = _build_history(body)
    if error:
        return _response(400, {"error": error})
    board = history[-1]
    history_plies = len(history) - 1

    print(f"[{trace_id}] maia3 fen={board.fen()} history={history_plies} elo={elo} "
          f"pvDepth={pv_depth} {_caller(headers)}")

    if not board.is_valid():
        return _response(400, {"error": "Invalid FEN"})
    if board.is_game_over():
        return _response(200, {"topMove": None, "topMoves": [], "pv": [], "elo": elo,
                               "wdl": None, "historyPlies": history_plies})

    try:
        # First ply: the top-N display.
        move_probs, wdl = _predict(history, elo, opponent_elo)
        filtered = {m: p for m, p in move_probs.items() if not _is_knight_return_home(board, m)}
        if not filtered:
            filtered = move_probs
        top = sorted(filtered.items(), key=lambda x: -x[1])[:num_moves]
        total = sum(p for _, p in top)
        normalised = [(m, round(p / total, 4) if total > 0 else 0) for m, p in top]

        # PV by chaining argmax predictions, the history growing along the
        # chain. Same anti-shuffle rules as the Maia-2 Lambda: no knight first
        # moves from the starting layout on plies 0-1, no move that reverses an
        # earlier move by the same side, no repeated position. Falls back to
        # the top move when everything is filtered so the PV never truncates.
        pv: list[str] = []
        chain = list(history)
        pv_board = board.copy(stack=False)
        elos = [elo, opponent_elo]
        is_starting_layout = pv_board.board_fen() == chess.Board().board_fen()
        seen_epds = {pv_board.epd()}
        prev_moves_by_side = {chess.WHITE: [], chess.BLACK: []}

        for i in range(pv_depth):
            if pv_board.is_game_over():
                break
            if i == 0:
                sorted_probs = sorted(filtered.items(), key=lambda x: -x[1])
            else:
                probs, _ = _predict(chain, elos[i % 2], elos[(i + 1) % 2])
                sorted_probs = sorted(
                    ((m, p) for m, p in probs.items() if not _is_knight_return_home(pv_board, m)),
                    key=lambda x: -x[1],
                ) or sorted(probs.items(), key=lambda x: -x[1])

            side = pv_board.turn
            if i < 2 and is_starting_layout:
                banned = ({"g1f3", "g1h3", "b1a3", "b1c3"} if side == chess.WHITE
                          else {"g8f6", "g8h6", "b8a6", "b8c6"})
            else:
                banned = set()

            best_move = None
            for cand, _p in sorted_probs:
                if cand in banned:
                    continue
                if any(cand[:2] == m[2:4] and cand[2:4] == m[:2] for m in prev_moves_by_side[side]):
                    continue
                trial = pv_board.copy(stack=False)
                trial.push_uci(cand)
                if trial.epd() in seen_epds:
                    continue
                best_move = cand
                break
            if best_move is None and sorted_probs:
                best_move = sorted_probs[0][0]
            if best_move is None:
                break

            pv.append(best_move)
            pv_board.push_uci(best_move)
            chain.append(pv_board.copy(stack=False))
            del chain[:-_HISTORY]
            seen_epds.add(pv_board.epd())
            prev_moves_by_side[side].append(best_move)

    except Exception as exc:
        print(f"[{trace_id}] maia3 failed: {type(exc).__name__}: {exc}")
        return _response(500, {"error": "Inference failed"})

    return _response(200, {
        "topMove": normalised[0][0] if normalised else None,
        "topMoves": [{"move": m, "prob": p} for m, p in normalised],
        "pv": pv,
        "elo": elo,
        "wdl": wdl,
        "historyPlies": history_plies,
    })


def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }
