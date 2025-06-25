{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        mpkgs = import inputs.nixpkgs_master {
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = true;
        };

        libList =
          [
            # Add needed packages here
            pkgs.stdenv.cc.cc
            pkgs.stdenv.cc
            pkgs.libGL
            pkgs.gcc
            pkgs.gcc.cc.lib
            pkgs.glib
            pkgs.libz
            pkgs.glibc
            pkgs.glibc.dev
          ]
          ++ pkgs.lib.optionals pkgs.stdenv.isLinux (
            with pkgs;
            [
              cudatoolkit

              # This is required for most app that uses graphics api
              # linuxPackages.nvidia_x11
            ]
          );
      in
      with pkgs;
      {
        devShells = {
          default =
            let
              python_with_pkgs = pkgs.python312.withPackages (pp: [
                # Add python pkgs here that you need from nix repos
              ]);
            in
            mkShell {
              NIX_LD = runCommand "ld.so" { } ''
                ln -s "$(cat '${pkgs.stdenv.cc}/nix-support/dynamic-linker')" $out
              '';
              NIX_LD_LIBRARY_PATH = lib.makeLibraryPath libList;
              packages = [
                python_with_pkgs
                python312Packages.venvShellHook
                mpkgs.uv
                tmux
                duckdb
                pkgs.stdenv.cc
                pkgs.gcc
                pkgs.glibc.dev
              ] ++ libList;
              venvDir = "./.venv";
              postVenvCreation = ''
                unset SOURCE_DATE_EPOCH
              '';
              postShellHook = ''
                unset SOURCE_DATE_EPOCH
              '';
              shellHook = ''
                export OPENBLAS_NUM_THREADS=16
                export OMP_NUM_THREADS=8
                export MKL_NUM_THREADS=8
                export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH:$LD_LIBRARY_PATH
                export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                runHook venvShellHook
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
# Things one might need for debugging or adding compatibility
# export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
# export LD_LIBRARY_PATH=${pkgs.cudaPackages.cuda_nvrtc}/lib
# export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
# export EXTRA_CCFLAGS="-I/usr/include"

# Data syncthing commands
# syncthing cli show system | jq .myID
# syncthing cli config devices add --device-id $DEVICE_ID_B
# syncthing cli config folders $FOLDER_ID devices add --device-id $DEVICE_ID_B
# syncthing cli config devices $DEVICE_ID_A auto-accept-folders set true
