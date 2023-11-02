FROM maxxing/compiler-dev

# Install additional packages
RUN apt-get update && apt-get install curl vim valgrind -y

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

# Install cargo-binstall
RUN curl -L --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash

# Install useful tools
RUN cargo binstall --no-confirm tokei cargo-bloat just dufs

CMD ["/bin/bash"]
