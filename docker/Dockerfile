ARG FROM_IMAGE="bitorch/engine"
FROM ${FROM_IMAGE} as bitorch-engine-base

FROM bitorch-engine-base as requirements-installed
COPY "../requirements.txt" "/green-bit-llm-req.txt"
RUN pip install packaging -r "/green-bit-llm-req.txt" && \
  rm "/green-bit-llm-req.txt" && \
  pip install flash-attn --no-build-isolation && \
  pip cache purge

# clone instead of mounting makes the code in the image independent from local changes
# to mount your code before building, use the target above and mount your local code
FROM requirements-installed as code-cloned
ARG GIT_URL="https://github.com/GreenBitAI/green-bit-llm.git"
ARG GIT_BRANCH="main"
ARG BUILD_TARGET="."
RUN git clone \
    --depth 1 \
    --branch "${GIT_BRANCH}" \
    "${GIT_URL}" \
    /green-bit-llm && \
  cd /green-bit-llm && \
  pip install -e .
