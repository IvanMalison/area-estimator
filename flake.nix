{
  description = "Application packaged using poetry2nix";
  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        inherit (poetry2nix.legacyPackages.${system}) mkPoetryApplication;
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages = {
          myapp = mkPoetryApplication { projectDir = self; };
          default = self.packages.${system}.myapp;
        };

        devShells.default = pkgs.mkShell rec {
          packages = [ poetry2nix.packages.${system}.poetry ];
          buildInputs = with pkgs; [
            python311
            git
            poetry
            curl

            glib
            libGL
            boost
            cimg
            opencv
            stdenv.cc.cc
            zlib
          ];


          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath buildInputs;

          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_TOOLKIT"
            poetry env use $(which python)
          '';
        };
      });
}
