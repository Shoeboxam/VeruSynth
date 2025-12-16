set -e

# beware: this script makes various linux assumptions
OS=x86-linux
TOOLCHAIN=x86_64-unknown-linux-gnu

# install verus
if [ ! -d /verus ]; then
    # https://github.com/verus-lang/verus/blob/main/INSTALL.md
    # I switched to an older version of verus that works with autoverus' benches
    curl -L -o verus.zip https://github.com/verus-lang/verus/releases/download/release%2F0.2025.03.28.892067f/verus-0.2025.03.28.892067f-x86-linux.zip
    # curl -L -o verus.zip "https://github.com/verus-lang/verus/releases/download/release%2F0.2025.12.14.a321fbe/verus-0.2025.12.14.a321fbe-${OS}.zip"
    unzip verus.zip
    mv "verus-${OS}" verus
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.82.0-${TOOLCHAIN}
    # curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.91.0-${TOOLCHAIN}
    echo "export PATH=\"$(pwd)/verus:\$PATH\"" >> ~/.bash_profile
    echo "Updated PATH in .bash_profile to include verus at $(pwd)/verus"
fi

# set up virtual environment
if [ ! -d /.venv-dev ]; then
    python3 -m venv .venv-dev
    source .venv-dev/bin/activate
    # for model training dependencies
    pip install -r requirements.txt
    # install verusynth as a dev (editable) package
    pip install -e .
fi

if [ ! -d /repositories ]; then
    mkdir repositories
    cd repositories
    git clone git@github.com:verus-lang/verus.git --depth 1
    git clone git@github.com:microsoft/verus-proof-synthesis.git --depth 1
    git clone git@github.com:secure-foundations/human-eval-verus.git --depth 1
fi
