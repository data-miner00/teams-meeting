{
  # Pinned to nixos-26.05. To update: pick a new commit, then get its hash with
  #   nix-prefetch-url --unpack https://github.com/NixOS/nixpkgs/archive/<commit>.tar.gz
  pkgs ? import (fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/1bc55b9def8165e82073919945c3239903fe4dc2.tar.gz";
    sha256 = "1csnzj9gyjz25xbmmmqvgm8pbkn681by405wqm3xlzasxd0rlrhg";
  }) {},
}:

let
  projectName = "teams-meeting";
in
pkgs.mkShellNoCC {
  name = "${projectName}-dev";

  packages = [
    pkgs.uv
  ];

  shellHook = ''
    echo "Nix environment for ${projectName} is ready! (uv $(uv --version))"
  '';
}
