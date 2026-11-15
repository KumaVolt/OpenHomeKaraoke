# OpenHomeKaraoke Docker image
#
# Build (light image, no DNN vocal splitter):
#   docker build -t openhomekaraoke .
# Build with the CPU-only DNN vocal splitter (adds ~1-2 GB):
#   docker build --build-arg WITH_VOCAL_SPLITTER=true -t openhomekaraoke:vocal .
#
# See docker/README.md for how to run it.

FROM python:3.11-slim-bookworm

ARG WITH_VOCAL_SPLITTER=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    LANG=en_US.UTF-8 \
    TERM=xterm-256color

# System dependencies:
#  - vlc / ffmpeg: playback, pitch shifting, volume normalisation, screen streaming
#  - xvfb / x11-utils: virtual screen for headless mode (xwininfo is used by screencapture.sh)
#  - pulseaudio(-utils): virtual sound card for headless mode
#  - tmux / socat: used by screencapture.sh and the streamer restart logic in karaoke.py
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        vlc-bin vlc-plugin-base vlc-plugin-video-output \
        ffmpeg \
        xvfb x11-utils \
        pulseaudio pulseaudio-utils \
        tmux socat procps \
        fonts-dejavu-core \
        ca-certificates \
        locales \
 && sed -i 's/^# *en_US.UTF-8/en_US.UTF-8/' /etc/locale.gen \
 && locale-gen \
 && rm -rf /var/lib/apt/lists/*

# Intel GPU video encoding (Quick Sync through VAAPI) for the TV stream, on x86 only. It's used when the
# container is given /dev/dri, and frees the CPU on small machines such as a LattePanda or a NUC.
# The driver that can encode on Intel graphics is in Debian's non-free section.
RUN if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        sed -i 's/^Components: main$/Components: main non-free/' /etc/apt/sources.list.d/debian.sources \
     && apt-get update \
     && apt-get install -y --no-install-recommends intel-media-va-driver-non-free i965-va-driver-shaders vainfo \
     && rm -rf /var/lib/apt/lists/* ; \
    fi

WORKDIR /app

# Python dependencies. torch/librosa (only needed by the DNN vocal splitter) are
# installed separately so the default image stays small.
COPY requirements.txt /tmp/requirements.txt
RUN grep -v -i -E '^\s*(torch|librosa|soundfile|spec_utils|futures|pip|setuptools|wheel)\b' /tmp/requirements.txt > /tmp/requirements-base.txt \
 && pip install -r /tmp/requirements-base.txt \
 && if [ "$WITH_VOCAL_SPLITTER" = "true" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch \
     && pip install librosa==0.10.0 soundfile opencv-python-headless ; \
    fi \
 && rm /tmp/requirements*.txt

# Run as an unprivileged user: VLC refuses to run as root.
RUN useradd --create-home --uid 1000 --shell /bin/bash karaoke \
 && mkdir -p /songs \
 && chown karaoke:karaoke /songs /app

COPY --chown=karaoke:karaoke . /app
RUN chmod +x /app/docker/entrypoint.sh /app/screencapture.sh \
 # let VLC's subtitle renderer use the bundled Unicode font for synced lyrics in any language
 && mkdir -p /usr/local/share/fonts \
 && ln -s /app/font/arial-unicode-ms.ttf /usr/local/share/fonts/ \
 && fc-cache -f

USER karaoke

ENV WITH_VOCAL_SPLITTER=${WITH_VOCAL_SPLITTER} \
    OHK_MODE=headless \
    SCREEN_SIZE=1280x720 \
    STREAM_PORT=4000

# 5000: web UI (http), 5001: web UI (https, for browser microphone), 4000: TV stream
EXPOSE 5000 5001 4000
VOLUME ["/songs"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
