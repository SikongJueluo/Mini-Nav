{
  pkgs,
  lib,
  config,
  inputs,
  ...
}: let
  pkgs-nixgl =
    (import inputs.nixpkgs {
      system = pkgs.stdenv.hostPlatform.system;
      config.allowUnfree = true;
      overlays = [inputs.nixgl.overlay];
    }).nixgl.override {
      nvidiaVersionFile = "/proc/driver/nvidia/version";
      nvidiaVersion = "580.126.09";
    };
in {
  packages = [
  ];

  enterShell = ''
    export UV_PROJECT_ENVIRONMENT=$HOME/local/share/mamba/envs/mini-nav/bin/

    eval "$(micromamba shell hook --shell bash)"
    micromamba activate mini-nav

    which uv
    which python
  '';

  # See full reference at https://devenv.sh/reference/options/
}
