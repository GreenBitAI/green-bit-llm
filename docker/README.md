# Project Setup with Docker

In the following, we show how to build a [Docker](https://www.docker.com/) image for this project and explain additional options.

## Build Docker Image

1. Build the bitorch engine image according to [these instructions](https://github.com/GreenBitAI/bitorch-engine/blob/HEAD/docker/README.md).
2. Now you should be able to build the image by running the following commands:
```bash
# cd docker
# you should be in this `docker` directory
cp -f ../requirements.txt .
docker build -t gbai/green-bit-llm .
```
3. You can now run the container, for example with this:
```bash
docker run -it --rm --gpus all gbai/green-bit-llm
```
4. Alternatively, you can mount the directory `/root/.cache/huggingface/hub` which will save the downloaded model cache locally,
e.g. you could use your users cache directory:
```bash
docker run -it --rm --gpus all -v "${HOME}/.cache/huggingface/hub":"/root/.cache/huggingface/hub" gbai/green-bit-llm
```

## Build Options

Depending on your setup, you may want to adjust some options through build arguments:
- repository URL, e.g. add `--build-arg GIT_URL="https://github.com/MyFork/green-bit-llm.git"`
- green-bit-llm branch or tag, e.g. add `--build-arg GIT_BRANCH="v1.2.3"`
- if there is a problem, set the environment variable `BUILDKIT_PROGRESS=plain` to see all output

## For Development

A docker image without the code cloned, e.g. for mounting a local copy of the code, can be made easily with the target `requirements-installed`:
```bash
# cd docker
# you should be in this `docker` directory
cp -f ../requirements.txt .
docker build -t gbai/green-bit-llm:no-code --target requirements-installed .
docker run -it --rm --gpus all --volume "$(pwd)/..":/green-bit-llm gbai/green-bit-llm:no-code
# in the docker container:
cd /green-bit-llm
pip install -e .
```
However, this means the build results will not be persisted in the image, so you probably want to mount the same directory every time.
