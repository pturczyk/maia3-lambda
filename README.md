# maia3-lambda

The AWS Lambda that serves human-like move predictions to the
[Chess Revision](https://chess-revision.com) app. It runs
[Maia-3](https://github.com/CSSLab/maia3) by the CSSLab at the University of
Toronto: given a position, the moves that led to it and a rating, it predicts
the move a human of that rating would most likely play.

This repository is published to meet the AGPL-3.0, which covers Maia-3 and so
this service built on it.

## What it does

- Loads the Maia-3 23M model once per container and runs it on CPU.
- Replays up to the last 8 positions of the game, because Maia-3 predicts
  better when it knows how the board got there.
- Returns the most likely moves with their probabilities, a predicted
  continuation for both sides at their own ratings, and the model's
  win/draw/loss estimate for a human game.

The request and response format is documented at the top of
[`handler.py`](handler.py).

## Build and run

```sh
docker build -t maia3-lambda .
docker run -p 9000:8080 maia3-lambda
curl -s localhost:9000/2015-03-31/functions/function/invocations \
  -d '{"body": "{\"moves\": [\"e2e4\", \"e7e5\"], \"elo\": 1500}"}'
```

The image pins the Maia-3 source to a commit and the weights to a Hugging Face
revision with a checksum, so a build is reproducible.

## Tests

```sh
pip install chess pytest
pytest tests
```

The tests stub out torch and the model, so they run offline in a second.

## Credits and license

Maia-3 and its weights are the work of the CSSLab, University of Toronto,
released under the AGPL-3.0. This service is released under the same license,
see [LICENSE](LICENSE).
