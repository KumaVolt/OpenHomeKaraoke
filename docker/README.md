# Running OpenHomeKaraoke with Docker

Docker installs VLC, ffmpeg, Python and every library for you, so you don't need conda, pip or system packages.
You only need [Docker](https://docs.docker.com/get-docker/) with Docker Compose v2.24 or newer.

There are two ways to run it:

| Mode | Where the karaoke screen appears | Works on |
|---|---|---|
| **Headless** (default) | Any web browser, usually the TV's, at `http://<host-ip>:4000` | Linux, macOS, Windows, NAS, Raspberry Pi 4/5 (64-bit OS) |
| **Host display** | The monitor or TV plugged into the machine (HDMI), with sound from its speakers | Linux with a desktop (X11, or Wayland with XWayland) |

## Quick start (headless)

```bash
git clone https://github.com/xuancong84/OpenHomeKaraoke.git
cd OpenHomeKaraoke
```

Open `docker-compose.yml` and change `HOST_IP` to the LAN IP address of this machine (for example `192.168.1.23`).
The QR code and on-screen URL use it, so phones need it to connect.

```bash
docker compose up -d
```

The first build takes a few minutes. After that:

- **Phones and computers**: open `http://<HOST_IP>:5000` to search YouTube and queue songs.
- **TV**: open `http://<HOST_IP>:4000` in the TV's web browser, then click or tap the video once to unmute it.
- Downloaded songs are saved to the `songs/` folder next to `docker-compose.yml`.

To stop it, run `docker compose down`. To see the logs, run `docker compose logs -f`.

> **macOS:** the AirPlay Receiver already uses port 5000. Turn it off in System Settings → General → AirDrop & Handoff,
> or change the port mapping to `"5050:5000"` and open `http://<HOST_IP>:5050` instead.

### How headless mode works

The container runs a virtual screen (Xvfb) and a virtual sound card (PulseAudio).
OpenHomeKaraoke and VLC draw on that virtual screen, and the project's own `screencapture.sh` encodes the
screen and sound with ffmpeg and serves them on port 4000. Expect a delay of a second or two on the TV.

## Host display mode (Linux)

Use this mode to show the karaoke screen full-screen on the monitor or TV connected to the machine, like running `app.py` directly.

```bash
xhost +local:        # allow the container to draw on your screen (run once per login)
docker compose -f docker-compose.yml -f docker-compose.host.yml up -d
```

This mode uses host networking, so the app finds your LAN IP by itself and `HOST_IP` is not needed.
Sound goes through your desktop's PulseAudio or PipeWire socket. The compose file assumes your user ID is 1000.
Check with `id -u`, and if it's different, edit the `user:` line and the `/run/user/1000` paths in `docker-compose.host.yml`.

## Options

### Environment variables (`environment:` in docker-compose.yml)

| Variable | Default | Meaning |
|---|---|---|
| `HOST_IP` | auto-detected | IP address shown in the QR code and on-screen URL. Set it in headless mode. |
| `SCREEN_SIZE` | `1280x720` | Resolution of the virtual screen in headless mode (for example `1920x1080`). |
| `OHK_MODE` | `headless` | `headless` or `host`. |
| `ENABLE_STREAM` | `1` headless, `0` host | Whether to stream the screen on port 4000. |
| `STREAM_PORT` | `4000` | Port of the TV stream inside the container. |

### app.py options

Put any `app.py` option in `command:`. For example:

```yaml
    command: ["--admin-password", "secret", "-L", "zh_CN", "--high-quality"]
```

Run `docker compose run --rm openhomekaraoke --help` to see all of them.

### DNN vocal splitter

The default image leaves out PyTorch to keep its size down. To include the CPU-only DNN vocal splitter
(it separates instrumental and vocal tracks), set `WITH_VOCAL_SPLITTER: "true"` under `build: args:` in `docker-compose.yml`, then rebuild:

```bash
docker compose up -d --build
```

This adds about 1–2 GB to the image. Splitting runs on the CPU in the background, so it's slow on small devices.

### YouTube cookies

If YouTube asks yt-dlp to sign in, export your browser cookies to a `cookies.txt` file, mount it,
and pass it with `-c`:

```yaml
    command: ["-c", "/cookies/cookies.txt"]
    volumes:
      - ./songs:/songs
      - ./cookies.txt:/cookies/cookies.txt:ro
```

## Troubleshooting

- **Phones can't connect, or the QR code shows a 172.x.x.x address**: set `HOST_IP` to this machine's LAN IP.
- **The TV page is black or silent**: click or tap the video once (browsers block autoplay with sound). Check that port 4000 is published.
  Run `docker compose exec openhomekaraoke cat /tmp/screencapture.log` for the streamer's log.
- **Host display mode: "cannot connect to X display"**: run `xhost +local:` on the host first, and check that `DISPLAY` is set (`echo $DISPLAY`).
- **Host display mode: no sound**: check that `/run/user/1000/pulse/native` exists on the host (PipeWire provides it through `pipewire-pulse`).
