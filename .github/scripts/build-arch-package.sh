#!/bin/bash
# Build the Arch package from the current commit instead of the release tarball.
# Usage: build-arch-package.sh [OUTPUT_DIR] [extra makepkg args…]
set -euo pipefail
root=$(git rev-parse --show-toplevel)
ver=$(sed -n 's/^pkgver=//p' "$root/packaging/arch/PKGBUILD")
out=$(realpath -m "${1:-$root/build/arch}")
shift || true
rm -rf "$out"
mkdir -p "$out"
git -C "$root" archive --format=tar.gz --prefix="webcam-blur-$ver/" -o "$out/webcam-blur-$ver.tar.gz" HEAD
cp "$root/packaging/arch/PKGBUILD" "$root/packaging/arch/webcam-blur.install" "$out/"
sed -i -e 's#^source=("$pkgname-$pkgver.tar.gz::[^"]*"#source=("$pkgname-$pkgver.tar.gz"#' \
       -e "0,/^sha256sums=('[0-9a-f]*'/s//sha256sums=('SKIP'/" "$out/PKGBUILD"
cd "$out"
makepkg --noconfirm "$@"
