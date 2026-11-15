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
> or change the port mapping to `"5050:5000"` and set `PUBLIC_URL: "http://<HOST_IP>:5050"` so the QR code uses the new port.
>
> **macOS firewall:** if your phone shows a blank page or keeps loading, allow incoming connections for Docker
> in System Settings → Network → Firewall → Options.

### How headless mode works

The container runs a virtual screen (Xvfb) and a virtual sound card (PulseAudio).
OpenHomeKaraoke and VLC draw on that virtual screen, and the project's own `screencapture.sh` encodes the
screen and sound with ffmpeg and serves them on port 4000. Expect a delay of a second or two on the TV.

## Prebuilt images (Portainer, NAS)

GitHub Actions builds the image for amd64 and arm64 and publishes it, so a server can pull it instead of building it:

| Image | Contents |
|---|---|
| `ghcr.io/kumavolt/openhomekaraoke:latest` | Light image |
| `ghcr.io/kumavolt/openhomekaraoke:latest-vocal` | With the DNN vocal splitter (about 1–2 GB bigger) |

Builds of other branches are tagged with the branch name, for example `:feature-synced-lyrics` and `:feature-synced-lyrics-vocal`.
`:latest` is built from `master`, so it only exists once Docker support is merged there.

In Portainer, go to **Stacks → Add stack → Web editor**, paste [`portainer-stack.yml`](portainer-stack.yml),
change `HOST_IP` (and the image tag if you want the vocal splitter), then deploy.
To update later, open the stack and choose **Update the stack** with **Re-pull image** turned on.

### Small machines (LattePanda, NUC, Intel mini PCs)

Two things keep playback smooth on a 2- or 4-core CPU:

- **The vocal splitter runs at the lowest CPU priority**, so it only uses time that playback and the TV stream leave
  over. Songs still get split, just more slowly while someone is singing.
- **The TV stream can be encoded on the Intel GPU** (Quick Sync) instead of the CPU. Give the container the GPU and
  the device's group, as in the commented lines of [`portainer-stack.yml`](portainer-stack.yml):

  ```yaml
      devices:
        - /dev/dri:/dev/dri
      group_add:
        - "993"   # the output of: stat -c %g /dev/dri/renderD128
  ```

  The streamer uses the GPU automatically when it works and falls back to the CPU otherwise. Check which one it picked
  with `docker exec openhomekaraoke grep Encoding /tmp/screencapture.log`. Set `STREAM_ENCODER: x264` to force the CPU.

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
| `PUBLIC_URL` | `http://<HOST_IP>:5000` | Full web UI URL for the QR code and screen, for use behind a reverse proxy (for example `https://karaoke.example.com`). |
| `STREAM_URL` | `http://<HOST_IP>:4000` | Full TV stream URL shown on screen, for use behind a reverse proxy (for example `https://karaoke-tv.example.com`). |
| `TRUST_PROXY` | off | Set to `1` behind a reverse proxy so the app sees each phone's real IP instead of the proxy's. Leave it off otherwise. |
| `SCREEN_SIZE` | `1280x720` | Resolution of the virtual screen in headless mode (for example `1920x1080`). |
| `OHK_MODE` | `headless` | `headless` or `host`. |
| `ENABLE_STREAM` | `1` headless, `0` host | Whether to stream the screen on port 4000. |
| `STREAM_PORT` | `4000` | Port of the TV stream inside the container. |
| `STREAM_ENCODER` | `auto` | `auto` uses the GPU's video encoder (VAAPI) when `/dev/dri` is passed in and works, else the CPU. `x264` or `vaapi` forces one. |

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

### Synced lyrics

Songs without lyrics of their own (not a karaoke video, no subtitles in the file) get time-synced lyrics from
[LRCLIB](https://lrclib.net), a free, open lyrics database. They're looked up during the "Get ready" countdown,
cached in `songs/.lyrics/`, and shown over the video with a karaoke colour fill.
Before they're saved, the app listens for where the singing is in the song (using the vocals-only track when the
vocal splitter has made one) and lines the lyrics up with it, which also handles music videos with an extra intro.
If they still run early or late, use the subtitle delay control in the web UI.
The subtitle switch in the web UI turns them off for a song, and `--no-lyrics` turns the lookup off completely.

### Pitch guide

With the vocal splitter image, songs it has processed can show the melody above the lyrics: each note is a bar
(higher bar, higher note), a playhead moves across with the music, and the notes light up as they're sung.
Turn it on with the **Pitch guide** switch on the phone's home tab; it's off by default and only appears for
songs that have a vocals-only track. The notes are read from that track, so they follow the recording's own
singer. Splitting is slow on a CPU (several minutes per song), so a newly downloaded song gets its guide later.

### YouTube cookies

If YouTube asks yt-dlp to sign in, export your browser cookies to a `cookies.txt` file, mount it,
and pass it with `-c`:

```yaml
    command: ["-c", "/cookies/cookies.txt"]
    volumes:
      - ./songs:/songs
      - ./cookies.txt:/cookies/cookies.txt:ro
```

## Behind a reverse proxy with your own domain

You can serve OpenHomeKaraoke on a domain through Nginx Proxy Manager, Traefik, Caddy or a similar proxy.
HTTPS also makes the browser microphone (speech search) work.

```yaml
services:
  openhomekaraoke:
    build: .
    container_name: openhomekaraoke
    restart: unless-stopped
    environment:
      PUBLIC_URL: "https://karaoke.example.com"
      STREAM_URL: "https://karaoke-tv.example.com"
      TRUST_PROXY: "1"   # use the client IP from the proxy's X-Forwarded-For header
    command: ["--admin-password", "change-me"]
    volumes:
      - ./songs:/songs
    shm_size: "256m"
    networks:
      - proxy          # the network your proxy is on; no ports need publishing

networks:
  proxy:
    external: true
```

Or serve both from one domain: set `STREAM_URL` to `https://karaoke.example.com/tv/` and add a custom location
`/tv/` that forwards to `openhomekaraoke:4000/` (the TV page works under a path).

Otherwise create two hosts in your proxy:

| Domain | Forward to | Notes |
|---|---|---|
| `karaoke.example.com` | `openhomekaraoke:5000` | Turn on WebSocket support. The web UI needs it. |
| `karaoke-tv.example.com` | `openhomekaraoke:4000` | The TV opens this. |

> **Security:** without `--admin-password`, every visitor is an admin and can delete or rename songs.
> Even with a password, anyone who can reach the site can make your server download from YouTube.
> Put both hosts behind your proxy's access control (for example an Nginx Proxy Manager Access List with basic auth),
> or reach the server over a VPN such as Tailscale or WireGuard instead of exposing it publicly.

> **YouTube on hosted servers:** YouTube often blocks downloads from datacenter or VPS IP addresses
> ("Sign in to confirm you're not a bot"). If that happens, use the cookies setup above.

The container always runs as uid 1000, so the songs folder must be writable by that user.

## Troubleshooting

- **Phones can't connect, or the QR code shows a 172.x.x.x address**: set `HOST_IP` to this machine's LAN IP.
- **The QR code opens the wrong port**: if you changed the left side of a port mapping (for example `"5050:5000"`),
  set `PUBLIC_URL` to the full address with that port, and `STREAM_URL` the same way if you changed `4000`.
- **The TV page is black or silent**: click or tap the video once (browsers block autoplay with sound). Check that port 4000 is published.
  Run `docker compose exec openhomekaraoke cat /tmp/screencapture.log` for the streamer's log.
- **Host display mode: "cannot connect to X display"**: run `xhost +local:` on the host first, and check that `DISPLAY` is set (`echo $DISPLAY`).
- **Host display mode: no sound**: check that `/run/user/1000/pulse/native` exists on the host (PipeWire provides it through `pipewire-pulse`).
