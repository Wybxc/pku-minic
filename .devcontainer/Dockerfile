FROM maxxing/compiler-dev

# Install additional packages
RUN apt-get update && apt-get install curl vim -y

# Update rust toolchain
RUN rustup update

# Install GitHub Cli
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
 && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
 && apt update \
 && apt install gh -y

# Install sharship
RUN (curl -sS https://starship.rs/install.sh > /tmp/starship.rs)\
 && sh /tmp/starship.rs --yes \
 && (echo 'eval "$(starship init bash)"' >> /root/.bashrc)
COPY ./starship.toml /root/.config/starship.toml

# Install tokei
RUN cargo install tokei

# Install cargo-bloat
RUN cargo install cargo-bloat

# Install kani
RUN cargo install --locked kani-verifier

CMD ["/bin/bash"]
