#!/usr/bin/env bash
#
# Builds the macOS installer: a signed, notarised .pkg carrying the CLI, the server and the
# Admin Control as a proper .app bundle.
#
# Gatekeeper is stricter here than on any other platform LocalGen ships to. An unsigned package
# will not open by double-click at all, and a signed but un-notarised one is refused just the
# same on a machine that has never seen it. Both steps below are therefore required for anything
# a user is expected to download, and both need an Apple Developer account — see
# docs/installers.md.
#
#   ./packaging/macos/build.sh --version 0.1.0 [--arch x64]

set -euo pipefail

VERSION="0.1.0"
BACKEND="cpu"
CONFIGURATION="Release"
ARCH="arm64"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --configuration) CONFIGURATION="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

RUNTIME="osx-$ARCH"
BUNDLE_ID="studio.gravicode.localgen"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="$REPO_ROOT/artifacts"
STAGING="$OUTPUT/staging/macos-$ARCH"
PAYLOAD="$STAGING/root/usr/local/localgen"
APP="$STAGING/root/Applications/LocalGen Admin Control.app"

echo "LocalGen $VERSION ($BACKEND, $RUNTIME)"

rm -rf "$STAGING"
mkdir -p "$PAYLOAD" "$APP/Contents/MacOS" "$APP/Contents/Resources" "$OUTPUT" "$STAGING/scripts"

for project in \
    src/LocalGen.Cli/LocalGen.Cli.csproj \
    src/LocalGen.Server/LocalGen.Server.csproj
do
    echo "  publishing $project"
    dotnet publish "$REPO_ROOT/$project" \
        --configuration "$CONFIGURATION" \
        --runtime "$RUNTIME" \
        --self-contained true \
        --output "$PAYLOAD" \
        -p:LlamaBackend="$BACKEND" \
        -p:Version="$VERSION" \
        -p:DebugType=None \
        --nologo --verbosity quiet
done

# The desktop app is published into the bundle rather than beside the others: macOS will not give
# a plain executable a Dock icon, a menu bar or the ability to be launched from Finder.
echo "  publishing src/LocalGen.Desktop into the app bundle"
dotnet publish "$REPO_ROOT/src/LocalGen.Desktop/LocalGen.Desktop.csproj" \
    --configuration "$CONFIGURATION" \
    --runtime "$RUNTIME" \
    --self-contained true \
    --output "$APP/Contents/MacOS" \
    -p:LlamaBackend="$BACKEND" \
    -p:Version="$VERSION" \
    -p:DebugType=None \
    --nologo --verbosity quiet

find "$PAYLOAD" "$APP" -name '*.pdb' -delete

sed -e "s/@VERSION@/$VERSION/g" -e "s/@BUNDLE_ID@/$BUNDLE_ID.desktop/g" \
    "$SCRIPT_DIR/Info.plist.in" > "$APP/Contents/Info.plist"

chmod +x "$PAYLOAD/localgen" "$PAYLOAD/localgen-server" "$APP/Contents/MacOS/LocalGen.Desktop"

# ── Signing ─────────────────────────────────────────────────────────────────────────────────
#
# Every Mach-O in the payload is signed, not just the entry points: the hardened runtime refuses
# to load an unsigned dylib into a signed process, and llama.cpp arrives as several of them.
SIGN_IDENTITY="${LOCALGEN_MACOS_SIGN_IDENTITY:-}"
INSTALLER_IDENTITY="${LOCALGEN_MACOS_INSTALLER_IDENTITY:-}"

if [ -n "$SIGN_IDENTITY" ]; then
    echo "  signing with $SIGN_IDENTITY"

    # Inside out: a bundle's signature covers its contents, so anything signed after the bundle
    # invalidates it.
    while IFS= read -r -d '' binary; do
        codesign --force --timestamp --options runtime \
            --entitlements "$SCRIPT_DIR/LocalGen.entitlements" \
            --sign "$SIGN_IDENTITY" "$binary"
    done < <(find "$PAYLOAD" "$APP" -type f \( -name '*.dylib' -o -name '*.so' -o -perm +111 \) -print0)

    codesign --force --timestamp --options runtime \
        --entitlements "$SCRIPT_DIR/LocalGen.entitlements" \
        --sign "$SIGN_IDENTITY" "$APP"

    codesign --verify --deep --strict --verbose=2 "$APP"
else
    echo "  LOCALGEN_MACOS_SIGN_IDENTITY is not set — building an unsigned package."
    echo "  Gatekeeper will refuse it on any machine but this one."
fi

# ── Package ─────────────────────────────────────────────────────────────────────────────────
cp "$SCRIPT_DIR/postinstall" "$STAGING/scripts/postinstall"
chmod +x "$STAGING/scripts/postinstall"

COMPONENT="$STAGING/component.pkg"
PKG="$OUTPUT/LocalGen-$VERSION-osx-$ARCH${BACKEND:+-$BACKEND}.pkg"
PKG="${PKG/-cpu.pkg/.pkg}"

pkgbuild \
    --root "$STAGING/root" \
    --scripts "$STAGING/scripts" \
    --identifier "$BUNDLE_ID" \
    --version "$VERSION" \
    --install-location / \
    "$COMPONENT"

sed -e "s/@VERSION@/$VERSION/g" -e "s/@BUNDLE_ID@/$BUNDLE_ID/g" \
    "$SCRIPT_DIR/distribution.xml.in" > "$STAGING/distribution.xml"

productbuild \
    --distribution "$STAGING/distribution.xml" \
    --package-path "$STAGING" \
    --resources "$SCRIPT_DIR/resources" \
    "$PKG"

if [ -n "$INSTALLER_IDENTITY" ]; then
    # A different certificate from the one above: Apple issues "Developer ID Application" for code
    # and "Developer ID Installer" for packages, and each is rejected in the other's place.
    productsign --sign "$INSTALLER_IDENTITY" --timestamp "$PKG" "$PKG.signed"
    mv "$PKG.signed" "$PKG"
fi

# ── Notarisation ────────────────────────────────────────────────────────────────────────────
#
# Stapling matters as much as submitting: without the ticket attached to the file, a machine that
# is offline when the package is opened has no way to learn it was approved.
if [ -n "${LOCALGEN_NOTARY_PROFILE:-}" ]; then
    echo "  notarising (this waits on Apple, typically a few minutes)"
    xcrun notarytool submit "$PKG" --keychain-profile "$LOCALGEN_NOTARY_PROFILE" --wait
    xcrun stapler staple "$PKG"
    xcrun stapler validate "$PKG"
else
    echo "  LOCALGEN_NOTARY_PROFILE is not set — the package is not notarised."
fi

cd "$OUTPUT"
shasum -a 256 "$(basename "$PKG")" > "SHA256SUMS-macos-$ARCH.txt"

echo "✓ $(basename "$PKG")"
cat "SHA256SUMS-macos-$ARCH.txt"
