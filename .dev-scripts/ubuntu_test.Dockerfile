FROM ubuntu:latest
RUN apt-get update -q -y \
    && apt-get install -q -y wget

COPY "install_aio.sh" "/install_aio.sh"
RUN INSTALL_DIR="/gbai_workspace" \
    BUILDKIT_PROGRESS=plain \
    bash "/install_aio.sh" \
    && rm "/install_aio.sh"
