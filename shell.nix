let
  here = toString ./.;
  moz_overlay = import (builtins.fetchTarball
    "https://github.com/mozilla/nixpkgs-mozilla/archive/master.tar.gz");
  pkgs = import <nixpkgs> { overlays = [ moz_overlay ]; };
  rust = (pkgs.rustChannelOf {
    channel = "stable";
  }).rust.override {
    extensions = [ "rust-src" "rust-analysis" ];
  };
  rustPlatform = pkgs.makeRustPlatform {
    rustc = rust;
    cargo = rust;
  };



  systemDeps = with pkgs; [
    udev fontconfig
  ];
  systemLibStr = pkgs.lib.makeLibraryPath systemDeps;

in pkgs.mkShell {
  buildInputs = [
    rust
  ] ++ systemDeps;

  nativeBuildInputs = [
    pkgs.pkg-config
  ];

  LIBCLANG_PATH = "${pkgs.llvmPackages.libclang}/lib";
  LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib64:$LD_LIBRARY_PATH:${systemLibStr}";
  CARGO_INCREMENTAL = 1;
}
