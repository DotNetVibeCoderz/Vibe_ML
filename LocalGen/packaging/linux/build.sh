#!/usr/bin/env bash
#
# Builds the Linux packages: a .deb for Debian and Ubuntu, and a .tar.gz for everything else.
#
# The payload is self-contained, so neither package depends on a .NET runtime being present. What
# it does depend on is the pair of system libraries llama.cpp links against, which are declared in
# the control file rather than discovered at first run.
#
#   ./packaging/linux/build.sh --version 0.1.0 [--backend cuda12]

set -euo pipefail

VERSION="0.1.0"
BACKEND="cpu"
CONFIGURATION="Release"
RUNTIME="linux-x64"
ARCH="amd64"
SKIP_PUBLISH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --configuration) CONFIGURATION="$2"; shift 2 ;;
        --runtime) RUNTIME="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        # Repackages the payload already in artifacts/staging. Publishing three self-contained
        # applications takes minutes and changing the packaging around them takes seconds.
        --skip-publish) SKIP_PUBLISH=true; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$REPO_ROOT/artifacts"
STAGING="$OUTPUT/staging/linux-$BACKEND"
PAYLOAD="$STAGING/opt/localgen"

echo "LocalGen $VERSION ($BACKEND, $RUNTIME)"

if [ "$SKIP_PUBLISH" = true ]; then
    [ -x "$PAYLOAD/localgen" ] || { echo "no payload at $PAYLOAD — run without --skip-publish first" >&2; exit 1; }

    # Only the packaging metadata is rebuilt, so the previous run's DEBIAN directory has to go:
    # dpkg-deb would otherwise package a stale control file alongside the new one.
    rm -rf "$STAGING/DEBIAN" "$STAGING/usr" "$STAGING/lib"
    echo "  reusing the payload in $PAYLOAD"
else
    rm -rf "$STAGING"
    mkdir -p "$PAYLOAD" "$OUTPUT"

    for project in \
        src/LocalGen.Cli/LocalGen.Cli.csproj \
        src/LocalGen.Server/LocalGen.Server.csproj \
        src/LocalGen.Desktop/LocalGen.Desktop.csproj
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

    find "$PAYLOAD" -name '*.pdb' -delete
    chmod +x "$PAYLOAD/localgen" "$PAYLOAD/localgen-server" "$PAYLOAD/LocalGen.Desktop"
fi

mkdir -p "$OUTPUT"
echo "  payload: $(find "$PAYLOAD" -type f | wc -l) files, $(du -sm "$PAYLOAD" | cut -f1) MB"

# ── Tarball ─────────────────────────────────────────────────────────────────────────────────
# For distributions without dpkg, and for anyone who would rather not install system-wide.
TARBALL="$OUTPUT/localgen-$VERSION-$RUNTIME${BACKEND:+-$BACKEND}.tar.gz"
TARBALL="${TARBALL/-cpu.tar.gz/.tar.gz}"

tar -czf "$TARBALL" -C "$STAGING/opt" localgen
echo "✓ $(basename "$TARBALL")"

# ── Debian package ──────────────────────────────────────────────────────────────────────────
mkdir -p "$STAGING/DEBIAN" \
         "$STAGING/usr/bin" \
         "$STAGING/usr/share/applications" \
         "$STAGING/lib/systemd/system"

# Symlinks rather than copies: the binaries are hundreds of megabytes and a wrapper script would
# break `localgen serve` reading its own path.
ln -sf /opt/localgen/localgen "$STAGING/usr/bin/localgen"
ln -sf /opt/localgen/localgen-server "$STAGING/usr/bin/localgen-server"

cp "$(dirname "${BASH_SOURCE[0]}")/localgen.service" "$STAGING/lib/systemd/system/localgen.service"
cp "$(dirname "${BASH_SOURCE[0]}")/localgen.desktop" "$STAGING/usr/share/applications/localgen.desktop"

INSTALLED_SIZE=$(du -sk "$STAGING/opt" | cut -f1)

# The CUDA build is a separate package rather than a variant of the same one: both install the
# same paths, so they must not be co-installable, and apt can only express that between packages
# with different names.
if [ "$BACKEND" = "cpu" ]; then
    PACKAGE="localgen"
    CONFLICTS=""
else
    PACKAGE="localgen-$BACKEND"
    CONFLICTS="Conflicts: localgen
Replaces: localgen
Provides: localgen"
fi

# Blank lines are stripped: an empty ${CONFLICTS} would otherwise leave one, and dpkg reads a
# blank line as the end of the stanza — everything after it would be silently dropped. The
# Description's continuation lines begin with a space, so none of them are empty.
sed '/^$/d' > "$STAGING/DEBIAN/control" <<EOF
Package: $PACKAGE
Version: $VERSION
Section: science
Priority: optional
Architecture: $ARCH
Installed-Size: $INSTALLED_SIZE
Maintainer: Gravicode Studios <hello@gravicode.com>
Depends: libc6, libstdc++6, libgomp1, ca-certificates
${CONFLICTS}
Homepage: https://github.com/gravicode/LocalGen
Description: Local AI inference engine with an OpenAI-compatible API
 LocalGen runs open language models on your own hardware and serves them over an
 OpenAI-compatible HTTP API, so existing clients work unchanged against a local
 endpoint. It ships a command-line interface, a server and a desktop admin
 application.
 .
 Models are downloaded separately and stored under ~/.localgen. Inference needs
 no network access at any point.
EOF

# The service is installed but not started. LocalGen listens on loopback and loads gigabytes of
# weights on demand; enabling that on every machine that installs the package would be a
# surprising thing for a package manager to do on the user's behalf.
cat > "$STAGING/DEBIAN/postinst" <<'EOF'
#!/bin/sh
set -e

if [ "$1" = "configure" ]; then
    if ! getent passwd localgen >/dev/null; then
        adduser --system --group --home /var/lib/localgen --shell /usr/sbin/nologin localgen
    fi

    mkdir -p /var/lib/localgen
    chown localgen:localgen /var/lib/localgen

    if [ -d /run/systemd/system ]; then
        systemctl daemon-reload || true
    fi

    echo "LocalGen installed. Pull a model and try it:"
    echo "  localgen pull huggingface:bartowski/SmolLM2-135M-Instruct-GGUF"
    echo "  localgen run smollm2-135m-instruct:q4_k_m 'Say hello'"
    echo
    echo "To run the API as a service:  systemctl enable --now localgen"
fi
EOF

cat > "$STAGING/DEBIAN/prerm" <<'EOF'
#!/bin/sh
set -e

if [ "$1" = "remove" ] && [ -d /run/systemd/system ]; then
    systemctl stop localgen || true
    systemctl disable localgen || true
fi
EOF

# Models are not ours to delete: /var/lib/localgen may hold tens of gigabytes the user pulled,
# and purge is the only path that removes it.
cat > "$STAGING/DEBIAN/postrm" <<'EOF'
#!/bin/sh
set -e

if [ "$1" = "purge" ]; then
    rm -rf /var/lib/localgen
    if getent passwd localgen >/dev/null; then
        deluser --system localgen || true
    fi
fi

if [ -d /run/systemd/system ]; then
    systemctl daemon-reload || true
fi
EOF

chmod 0755 "$STAGING/DEBIAN/postinst" "$STAGING/DEBIAN/prerm" "$STAGING/DEBIAN/postrm"

# <package>_<version>_<arch>.deb, which is what every Debian tool expects to find — the backend
# is already part of the package name rather than glued onto the end of the file name.
DEB="$OUTPUT/${PACKAGE}_${VERSION}_${ARCH}.deb"

dpkg-deb --build --root-owner-group "$STAGING" "$DEB" >/dev/null
echo "✓ $(basename "$DEB")"

# ── Checksums and signatures ────────────────────────────────────────────────────────────────
#
# A .deb is normally trusted through the repository that served it. These artifacts are served
# from a release page instead, so the checksum file — and its signature, when a key is available —
# is the only thing standing between a download and a substitution.
cd "$OUTPUT"
sha256sum "$(basename "$TARBALL")" "$(basename "$DEB")" > SHA256SUMS-linux.txt

if [ -n "${LOCALGEN_GPG_KEY:-}" ]; then
    echo "  signing with $LOCALGEN_GPG_KEY"

    # A passphrase has to arrive over the loopback pinentry: gpg would otherwise try to open a
    # dialog, which on a CI runner means hanging until the job times out.
    passphrase_args=()
    if [ -n "${GPG_PASSPHRASE:-}" ]; then
        passphrase_args=(--pinentry-mode loopback --passphrase "$GPG_PASSPHRASE")
    fi

    gpg --batch --yes --local-user "$LOCALGEN_GPG_KEY" "${passphrase_args[@]}" \
        --armor --detach-sign --output SHA256SUMS-linux.txt.asc SHA256SUMS-linux.txt
else
    echo "  LOCALGEN_GPG_KEY is not set — artifacts are unsigned."
fi

echo
cat SHA256SUMS-linux.txt
