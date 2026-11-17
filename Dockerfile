# Maia-3 23M (CSSLab, AGPL-3.0) on AWS Lambda, arm64, CPU inference.
FROM public.ecr.aws/lambda/python:3.12-arm64

RUN dnf install -y git && dnf clean all

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-deps \
      "maia3 @ git+https://github.com/CSSLab/maia3.git@1e13597c42d4858b7cfd7cfdae01e297263364b2"

# Weights baked into the image, pinned to a Hugging Face revision and checked
# against a known digest, so a cold start never downloads anything.
ARG MAIA3_REVISION=51a0145a8178046f7de23119160b136672deeb2b
ARG MAIA3_SHA256=bce6cd1af5f0399ac7eed33fabb7a6a2ef6193662c2740f262bf93af7bfb3569
RUN mkdir -p /opt/maia3 && python -c "\
import hashlib, urllib.request; \
url = 'https://huggingface.co/UofTCSSLab/Maia3-23M/resolve/${MAIA3_REVISION}/maia3-23m.pt'; \
data = urllib.request.urlopen(url).read(); \
assert hashlib.sha256(data).hexdigest() == '${MAIA3_SHA256}', 'weights digest mismatch'; \
open('/opt/maia3/maia3-23m.pt', 'wb').write(data)"

COPY handler.py ${LAMBDA_TASK_ROOT}/

# Fail the build, not the first cold start, if the model cannot load and answer.
RUN cd ${LAMBDA_TASK_ROOT} && python -c "\
import json, handler; \
r = handler.handler({'body': json.dumps({'moves': ['e2e4'], 'pvDepth': 3})}, None); \
assert r['statusCode'] == 200, r; print('Maia-3 OK', r['body'])"

CMD ["handler.handler"]
