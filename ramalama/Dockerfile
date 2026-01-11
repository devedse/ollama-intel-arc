FROM quay.io/ramalama/intel-gpu:latest

RUN --mount=type=cache,target=/var/cache/dnf \
    dnf install -y ramalama && \
    dnf clean all

CMD ["sleep", "infinity"]
