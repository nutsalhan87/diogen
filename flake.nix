{
  outputs = { self, nixpkgs }:
    let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in {
      devShell.x86_64-linux = pkgs.mkShell {
        packages = with pkgs; [ python3 poetry ];
        shellHook = with pkgs; ''
          export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib.makeLibraryPath [ stdenv.cc.cc zlib libGL glib ]}
          export POETRY_VIRTUALENVS_IN_PROJECT=true
        '';
      };
    };
}
