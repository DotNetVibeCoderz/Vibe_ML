# Packaging

Installers for the three desktop platforms. Each script publishes the same three executables —
the `localgen` CLI, the `localgen-server` service and the Admin Control — self-contained, so an
installed LocalGen does not need the .NET runtime present on the machine.

| Platform | Script | Produces |
| --- | --- | --- |
| Windows | `windows/build.ps1` | `LocalGen-<version>-win-x64.msi` |
| macOS | `macos/build.sh` | `LocalGen-<version>-osx-<arch>.pkg` |
| Linux | `linux/build.sh` | `localgen_<version>_amd64.deb` and a `.tar.gz` |

Everything lands in `artifacts/` at the repository root.

```bash
pwsh packaging/windows/build.ps1 -Version 0.1.0
./packaging/macos/build.sh   --version 0.1.0
./packaging/linux/build.sh   --version 0.1.0
```

Add `-Backend cuda12` / `--backend cuda12` for an NVIDIA build. The CUDA payload is several
hundred megabytes larger, which is why the CPU backend is the default here as it is everywhere
else in the project.

## Signing

Every script builds an unsigned package when no credentials are present, and says so rather than
failing — an unsigned installer is exactly what you want while iterating on the packaging itself,
and exactly what you must not publish. The release workflow passes the credentials from repository
secrets, so a release build is signed without anyone handling a certificate by hand.

| Platform | Variables | What they do |
| --- | --- | --- |
| Windows | `LOCALGEN_WINDOWS_CERT`, `LOCALGEN_WINDOWS_CERT_PASSWORD` | Authenticode signature over the executables and the MSI, timestamped so it outlives the certificate. |
| macOS | `LOCALGEN_MACOS_SIGN_IDENTITY`, `LOCALGEN_MACOS_INSTALLER_IDENTITY`, `LOCALGEN_NOTARY_PROFILE` | Developer ID signature with the hardened runtime, then notarisation and stapling. |
| Linux | `LOCALGEN_GPG_KEY` | Detached signatures over the artifacts and `SHA256SUMS`. |

What signing is for differs by platform, and it is worth being clear about which problem is being
solved:

- **Windows and macOS refuse to run unsigned software comfortably.** SmartScreen warns; Gatekeeper
  on macOS blocks outright unless the package is both signed *and* notarised — a signature alone
  is not enough there.
- **Linux has no such gate.** A `.deb` is trusted because of the repository it came from, so what
  is signed here is the artifact itself, for someone downloading it directly.

Neither certificate can be committed, generated locally, or worked around. See
[`docs/installers.md`](../docs/installers.md) for what to obtain and where each one goes.
