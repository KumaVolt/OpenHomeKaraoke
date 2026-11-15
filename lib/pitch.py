"""The sung melody of a song as notes, from the vocals-only track made by the DNN vocal splitter.

1. Track the voice's pitch every 20 ms with YIN (numpy only): compare each short stretch of the
   waveform with itself shifted by every candidate period, and take the shortest period that repeats well.
2. Turn the pitch curve into notes: runs of frames that stay near one pitch, rounded to the nearest semitone.

Only the vocals-only track is used. On a full mix the tracker follows guitars and synths as often as the
voice, and a guide that shows the wrong melody is worse than none.
"""
import json, logging, os, subprocess
import numpy as np

SR = 16000
HOP = 320                    # 20 ms
FPS = SR / HOP
W = 512                      # samples compared per frame (32 ms)
FMIN, FMAX = 70.0, 1000.0    # singing range in Hz, low bass to high soprano
TMIN, TMAX = int(SR / FMAX), int(SR / FMIN)
N_FFT = 1024                 # >= W + TMAX
THRESH = 0.15                # YIN: a period whose normalised difference is below this counts as repeating
VOICED = 0.3                 # frames whose best period is worse than this are not pitched (breaths, consonants, noise)
MIN_NOTE = 0.12              # seconds
NOTES_VERSION = 1


def _decode(path):
	raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR),
	                               '-f', 's16le', '-'], timeout = 180)
	return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def track(x):
	"""(midi pitch per frame with NaN where nothing is sung, frame loudness in dB)"""
	L = W + TMAX
	n = max(0, (len(x) - L) // HOP + 1)
	midi = np.full(n, np.nan, np.float32)
	loud = np.full(n, -100.0, np.float32)
	taus = np.arange(TMAX + 1)
	for a in range(0, n, 1024):
		b = min(n, a + 1024)
		seg = x[np.arange(L)[None, :] + HOP * np.arange(a, b)[:, None]]
		head = seg[:, :W]
		corr = np.fft.irfft(np.fft.rfft(seg, N_FFT) * np.conj(np.fft.rfft(head, N_FFT)), N_FFT)[:, :TMAX + 1]
		cs = np.concatenate([np.zeros((b - a, 1), np.float32), np.cumsum(seg ** 2, axis = 1)], axis = 1)
		e0 = cs[:, W:W + 1]
		et = cs[:, W + taus] - cs[:, taus]
		d = np.maximum(0, e0 + et - 2 * corr)
		cum = np.cumsum(d[:, 1:], axis = 1)
		cmnd = np.ones_like(d)
		cmnd[:, 1:] = d[:, 1:] * taus[1:] / np.maximum(cum, 1e-12)
		loud[a:b] = 10 * np.log10(e0[:, 0] / W + 1e-10)
		c = cmnd[:, TMIN:TMAX + 1]
		cols = np.arange(c.shape[1])[None, :]
		below = c[:, :-1] < THRESH
		has = below.any(1)
		# the first period that repeats well, walked down to the bottom of its dip; else the best one overall
		first = np.where(has, below.argmax(1), 0)
		rising = (c[:, 1:] >= c[:, :-1]) & (cols[:, :-1] >= first[:, None])
		t = np.where(has, rising.argmax(1), c[:, :-1].argmin(1))
		t = np.clip(t, 1, c.shape[1] - 2)
		r = np.arange(len(t))
		y0, y1, y2 = c[r, t - 1], c[r, t], c[r, t + 1]
		den = y0 - 2 * y1 + y2
		tf = TMIN + t + np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / np.where(np.abs(den) > 1e-9, den, 1), 0)
		f = 69 + 12 * np.log2(SR / tf / 440)
		midi[a:b] = np.where(y1 <= VOICED, f, np.nan)
	return midi, loud


def notes_from_pitch(midi, loud):
	"""[(start s, end s, midi note), ...]"""
	m = midi.copy()
	m[loud < np.percentile(loud, 95) - 30] = np.nan  # quiet bleed from the backing track is not singing
	idx = np.flatnonzero(~np.isnan(m))
	if len(idx) == 0:
		return []
	# octave slips: fold each frame to within half an octave of the voice's local median
	for i in idx:
		near = m[max(0, i - 25):i + 26]
		med = np.nanmedian(near)
		while m[i] - med > 6:
			m[i] -= 12
		while med - m[i] > 6:
			m[i] += 12
	# 5-frame median filter over the pitched frames
	sm = m.copy()
	for i in idx:
		w = m[max(0, i - 2):i + 3]
		sm[i] = np.nanmedian(w)
	notes, start, vals = [], None, []

	def close(end):
		if start is not None and (end - start) / FPS >= MIN_NOTE:
			notes.append([start / FPS, end / FPS, int(round(float(np.median(vals))))])

	gap = 0
	for i in range(len(sm)):
		v = sm[i]
		if np.isnan(v):
			gap += 1
			if gap > 3 and start is not None:  # more than 60 ms unpitched ends the note
				close(i - gap + 1)
				start, vals = None, []
			continue
		gap = 0
		if start is not None and abs(v - np.median(vals[-15:])) > 0.8:  # moved to another note
			close(i)
			start, vals = None, []
		if start is None:
			start = i
		vals.append(v)
	close(len(sm))
	# join neighbours on the same semitone with a tiny gap between them (a vibrato wobble or a consonant)
	merged = []
	for n in notes:
		if merged and merged[-1][2] == n[2] and n[0] - merged[-1][1] < 0.1:
			merged[-1][1] = n[1]
		else:
			merged.append(n)
	return [(round(a, 2), round(b, 2), p) for a, b, p in merged]


def cache_path(song_path, cache_dir):
	return os.path.join(cache_dir, os.path.basename(song_path) + '.notes.json')


def load(song_path, cache_dir, vocal_track = None):
	"""The song's notes, worked out from the vocal track the first time and cached; None without a vocal track."""
	path = cache_path(song_path, cache_dir)
	have_vocal = vocal_track and os.path.isfile(vocal_track)
	if os.path.isfile(path) and (not have_vocal or os.path.getmtime(path) >= os.path.getmtime(vocal_track)):
		try:
			with open(path) as fp:
				d = json.load(fp)
			if d.get('version') == NOTES_VERSION:
				return [tuple(n) for n in d['notes']]
		except Exception:
			pass
	if not have_vocal:
		return None
	try:
		notes = notes_from_pitch(*track(_decode(vocal_track)))
	except Exception as e:
		logging.warning('Pitch guide: could not read the vocal track: %s' % e)
		return None
	os.makedirs(cache_dir, exist_ok = True)
	with open(path, 'w') as fp:
		json.dump({'version': NOTES_VERSION, 'notes': notes}, fp)
	logging.info('Pitch guide: %d notes for %s' % (len(notes), os.path.basename(song_path)))
	return notes
