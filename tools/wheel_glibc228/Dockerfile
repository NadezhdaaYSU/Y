# glibc 2.28 (AlmaLinux 8, RHEL 8 era): build.sh compiles liby.so and
# ysu_gpu_probe here and tests the wheel here. Rust is the host's rustup,
# mounted, so nothing Rust is downloaded.
FROM almalinux:8
# `llvm-libs` alone resolved to LLVM 17's compatibility library, on which the
# CPU JIT fails (docs/known_issues.md); 21 runs it.
RUN dnf -y install gcc python3.11 'llvm-libs >= 21' && dnf clean all
