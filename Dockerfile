FROM ubuntu:22.04

# Prevent automatic updates
RUN apt-get update && apt-get remove -y unattended-upgrades && \
    echo 'APT::Periodic::Update-Package-Lists "0";' > /etc/apt/apt.conf.d/10periodic && \
    echo 'APT::Periodic::Unattended-Upgrade "0";' > /etc/apt/apt.conf.d/20auto-upgrades

ENV DEBIAN_FRONTEND=noninteractive
RUN mkdir -p /etc/sudoers.d

# 필수 패키지 및 PyQt5 실행에 필요한 Qt5 라이브러리 설치
RUN apt-get update && apt-get install -y \
    sudo \
    python3 python3-pip \
    git \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    libx11-xcb1 \
    qtbase5-dev qttools5-dev-tools \
    libqt5gui5 libqt5widgets5 libqt5core5a \
    libgl1-mesa-glx \
    x11-apps \
    && rm -rf /var/lib/apt/lists/*

# 최신 pip 버전 설치
RUN pip install --upgrade pip
RUN pip install pip==20.0.1

# Install dependencies for pyenv
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev \
    wget curl llvm libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#    make build-essential libssl-dev zlib1g-dev \
#    libbz2-dev libreadline-dev libsqlite3-dev \
#    wget curl llvm libncursesw5-dev xz-utils tk-dev \
#    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# --- 사용자 생성 및 홈 디렉토리 설정 ---
#ARG USERNAME=msyu
#ARG USER_UID=1000
#ARG USER_GID=1000

# Allow build-time override via --build-arg
#ENV USERNAME=${USERNAME}
#ENV USER_UID=${USER_UID}
#ENV USER_GID=${USER_GID}

#RUN groupadd --gid $USER_GID $USERNAME && \
#    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
#    usermod -aG sudo $USERNAME && \
#    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
#    chmod 0440 /etc/sudoers.d/$USERNAME && \
#    mkdir -p /home/$USERNAME/rpmbuild/{BUILD,RPMS,SOURCES,SPECS,SRPMS} && \
#    chown -R $USERNAME:$USERNAME /home/$USERNAME



#RUN mkdir -p /home/msyu
# msyu 사용자 생성
RUN useradd -m -s /bin/bash msyu
RUN echo "msyu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/msyu && \
    chmod 0440 /etc/sudoers.d/msyu

# 런타임 디렉토리 생성
RUN mkdir -p /tmp/runtime-msyu && \
    chown msyu:msyu /tmp/runtime-msyu && \
    chmod 700 /tmp/runtime-msyu

# Install pyenv from stable release tag
RUN git clone --branch v2.3.28 https://github.com/pyenv/pyenv.git /opt/.pyenv

# Fix permissions
#RUN chown -R 1000:1000 /home/msyu /opt/.pyenv && chmod 2775 -R /opt/.pyenv

# Environment variables for pyenv
ENV PYENV_ROOT=/opt/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH

# Install Python 3.8 via pyenv
RUN bash -c 'export PATH="/opt/.pyenv/bin:$PATH" && eval "$(pyenv init --path)" && \
    pyenv install 3.8.18 && pyenv global 3.8.18'

# Upgrade pip for the new Python
RUN pip install --upgrade pip

# Install Python packages (CPU-only PyTorch)
COPY --chown=msyu:msyu requirements.txt .
#RUN pip install --no-deps --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Fix permissions
#RUN chown -R msyu:msyu /opt/.pyenv /home/msyu && chmod 2775 -R /opt/.pyenv
RUN chown -R 1000:1000 /opt/.pyenv /home/msyu && chmod 2775 -R /opt/.pyenv

# Copy entrypoint script
COPY --chown=msyu:msyu --chmod=700 ./entrypoint.sh /
ENTRYPOINT ["/entrypoint.sh"]

USER msyu
WORKDIR /workspace
