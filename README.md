# Teams Meeting

Inspired by [fleeting](https://iminafleeting.com/?v=fr-FR), I love it! This is my new hobby.

## Getting Started

Activate nix shell:

```bash
nix-shell
```

When you in nix shell for the first time, the venv will not be created and entered.

```
uv sync
```

This should install the correct python version and create a venv with packages installed.

Then, you can manually enter the venv:

```
source .venv/bin/Activate
```

This is only required for the first time. Subsequent times, you can just run `nix-shell` and it will automatically enter the venv.

