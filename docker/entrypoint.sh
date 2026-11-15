#!/bin/bash
# Entrypoint for the OpenHomeKaraoke container.
#
# OHK_MODE=headless (default): starts a virtual screen (Xvfb) and a virtual sound
#   card (PulseAudio null sink) inside the container, and streams both to
#   http://<host>:4000 with screencapture.sh so a TV/phone browser can show it.
# OHK_MODE=host: uses the host's X11 display and PulseAudio/PipeWire socket
#   (Linux only), so the karaoke screen shows on the monitor/TV plugged into the host.
#
# Any arguments given to the container are passed on to app.py.

set -euo pipefail
cd /app

OHK_MODE="${OHK_MODE:-headless}"
SCREEN_SIZE="${SCREEN_SIZE:-1280x720}"
STREAM_PORT="${STREAM_PORT:-4000}"
ENABLE_STREAM="${ENABLE_STREAM:-}"

log() { echo "[entrypoint] $*"; }

wait_for_x() {
	for _ in $(seq 1 50); do
		xwininfo -display "$DISPLAY" -root >/dev/null 2>&1 && return 0
		sleep 0.2
	done
	log "ERROR: cannot connect to X display $DISPLAY"
	exit 1
}

start_streamer() {
	# karaoke.py restarts the streamer via tmux pane PiKaraoke:0.3 and detects it by
	# the './screencapture.sh' command line, so mirror the layout used by run.sh.
	tmux new-session -d -s PiKaraoke -x 240 -y 60
	for _ in 1 2 3; do
		tmux split-window -t PiKaraoke
		tmux select-layout -t PiKaraoke tiled
	done
	tmux pipe-pane -t PiKaraoke:0.3 "cat >> /tmp/screencapture.log"
	# -t /tmp: Docker mounts /dev/shm noexec, but screencapture.sh executes a script from its temp dir
	tmux send-keys -t PiKaraoke:0.3 -l \
		"./screencapture.sh -p $STREAM_PORT -n $DISPLAY -s pulse -a ${PULSE_MONITOR:-default} -t /tmp"
	tmux send-keys -t PiKaraoke:0.3 Enter
	log "Streaming the karaoke screen on port $STREAM_PORT"
}

case "$OHK_MODE" in
headless)
	export DISPLAY=:0
	rm -f /tmp/.X0-lock /tmp/.X11-unix/X0
	log "Starting virtual screen $SCREEN_SIZE"
	Xvfb "$DISPLAY" -screen 0 "${SCREEN_SIZE}x24" -nolisten tcp >/tmp/xvfb.log 2>&1 &
	wait_for_x

	log "Starting virtual sound card"
	export PULSE_RUNTIME_PATH="/tmp/pulse-$(id -u)"
	# a restarted container keeps /tmp, and a stale pid file stops PulseAudio from starting
	rm -rf "$PULSE_RUNTIME_PATH" ~/.config/pulse
	pulseaudio -n --daemonize=yes --exit-idle-time=-1 --log-target=file:/tmp/pulseaudio.log \
		-L "module-native-protocol-unix" \
		-L "module-null-sink sink_name=karaoke sink_properties=device.description=Karaoke" \
		-L "module-always-sink"
	for _ in $(seq 1 25); do pactl info >/dev/null 2>&1 && break; sleep 0.2; done
	pactl set-default-sink karaoke
	PULSE_MONITOR=karaoke.monitor
	[ -z "$ENABLE_STREAM" ] && ENABLE_STREAM=1
	;;
host)
	if [ -z "${DISPLAY:-}" ]; then
		log "ERROR: OHK_MODE=host needs DISPLAY and the host's /tmp/.X11-unix mounted."
		log "       See docker/README.md."
		exit 1
	fi
	wait_for_x
	[ -z "$ENABLE_STREAM" ] && ENABLE_STREAM=0
	;;
*)
	log "ERROR: unknown OHK_MODE '$OHK_MODE' (use 'headless' or 'host')"
	exit 1
	;;
esac

if [ "$ENABLE_STREAM" = "1" ]; then
	GPU="${VAAPI_DEVICE:-/dev/dri/renderD128}"
	if [ -e "$GPU" ] && ! [ -r "$GPU" -a -w "$GPU" ]; then
		log "The GPU ($GPU) is passed in but this container can't use it, so the stream is encoded on the CPU."
		log "       Add  group_add: [\"$(stat -c %g "$GPU")\"]  to the service (see docker/README.md)."
	fi
	start_streamer
fi

ARGS=(-d /songs)
if [ "${SHOW_QR:-1}" != "0" ]; then
	ARGS+=(--show-overlay)  # QR code in the corner while songs play, so people can add songs any time
fi
if [ "${WITH_VOCAL_SPLITTER:-false}" = "true" ]; then
	# the vocal splitter only processes songs when these sub-folders exist
	mkdir -p /songs/vocal /songs/nonvocal
	ARGS+=(-V)
fi

log "Starting OpenHomeKaraoke: python3 app.py ${ARGS[*]} $*"
exec python3 app.py "${ARGS[@]}" "$@"
