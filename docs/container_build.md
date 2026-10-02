# LFM container builds

`.github/workflows/container.yml` builds the ARM64 container on every branch
push and supports manual runs from the Actions tab. The workflow follows the
Buildx/login/build-and-push pattern from the supplied `pytorch-caney` example.
It publishes to `ghcr.io/nasa-nccs-hpda/lfm` using GitHub's `GITHUB_TOKEN`:

- `sha-<full commit SHA>` identifies the source commit for each build.
- `latest` is updated only by builds of the repository's default branch.

Builds are serialized per branch. A new push does not cancel an active build;
GitHub retains the newest pending run for that branch, replacing any older
pending run. This lets native dependency compilation finish while development
continues.

Commit the workflow, Dockerfile, `.dockerignore`, definition file,
shared installation script, and `requirements_container.txt` before pushing.

## GitHub setup

Enable GitHub Actions and allow the workflow to write packages. The workflow
requests `contents: read` and `packages: write`; it needs no Docker Hub secrets.
Organization policy must permit the referenced `actions/*` and `docker/*`
actions and package publication. If an existing GHCR package denies the push,
grant this repository Actions access in that package's settings.

The default runner is `ubuntu-24.04-arm`. These NVIDIA images and their source
builds need substantial temporary disk space. The workflow removes a few unused
SDK directories only on GitHub-hosted runners and reports available disk space.
If the standard runner runs out of space or memory, set the repository Actions
variable `CONTAINER_RUNNER` to a larger ARM64 runner label. A self-hosted runner
must have a current Actions agent and a working Docker daemon accessible to the
runner user. No GPU is required for the build-time checks.

The workflow pulls the public NGC base anonymously. If your environment requires
NGC authentication, configure a registry login before the build using an NGC API
key stored as a GitHub secret.

## Build inputs and validation

Both `Dockerfile` and `lfm_container-latest.def` use
`nvcr.io/nvidia/pytorch:26.06-py3` and run
`scripts/shell/install_container_dependencies.sh`. Keep their base image and
runtime environment variables aligned when changing them. The shared script
installs native GDAL/PROJ, builds their Python bindings, protects the
container-provided Python builds, and installs `requirements_container.txt`.

The combined requirements select the highest recorded versions from historical
freezes; this merged environment has not yet passed a full container build.
Dependency resolution, `pip check`, or the native-binding smoke checks can fail
if that combination is incompatible. These failures stop publication. The smoke
checks exercise GDAL/NumPy array I/O, the PROJ database, and PyTorch/torchvision
CPU NMS. They do not test GPU execution or model training.

The image retains the input requirements, installed native constraints, and a
fresh `pip freeze` in `/opt/requirements_container.txt`,
`/opt/lfm-native-constraints.txt`, and `/opt/lfm-installed-requirements.txt`.
Repository code, model weights, and datasets are mounted separately at runtime.

For a local Docker build on ARM64, run from the repository root:

```bash
docker build --platform linux/arm64 -t lfm:local .
```

The original Apptainer build remains available from the repository root:

```bash
sudo apptainer build lfm.sif lfm_container-latest.def
```

## Build an Apptainer sandbox on Explore

Docker runs only on the GitHub runner. Your HPC system needs Apptainer and
network access to GHCR; it does not need Docker installed. On an ARM64 host:

```bash
apptainer build --sandbox lfm-sandbox docker://ghcr.io/nasa-nccs-hpda/lfm:latest
apptainer exec --nv lfm-sandbox python -c 'import torch; print(torch.cuda.is_available())'
```

This downloads and unpacks the already-built image into a sandbox; it does not
repeat the dependency compilation on HPC. For a single-file image instead, use
`apptainer pull lfm.sif docker://ghcr.io/nasa-nccs-hpda/lfm:latest`.

Use the commit tag or the digest shown in the Actions run summary when you need
to identify a specific published image. Commit tags identify source inputs but
can be replaced by a rerun; a digest identifies the exact published contents.
If the package is private, authenticate with `apptainer registry login
--username YOUR_GITHUB_USERNAME docker://ghcr.io` using a token authorized to
read that package, or configure public visibility in the GHCR package settings.

References: [Docker's registry publication example](https://docs.docker.com/build/ci/github-actions/push-multi-registries/),
[GitHub package publication](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images),
and [Apptainer sandbox builds](https://apptainer.org/docs/user/main/build_a_container.html).
