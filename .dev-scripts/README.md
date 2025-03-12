## Development scripts

Currently, only the installation script resides here. To test it on MacOS, the docker image can be used:

```bash
docker build --platform linux/amd64 -t gbai_test -f ubuntu_test.Dockerfile .
docker run --platform linux/amd64 gbai_test:latest
```
