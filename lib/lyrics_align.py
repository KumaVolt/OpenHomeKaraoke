"""Lines up LRC lyrics with when singing is actually heard in a song, using only ffmpeg and numpy.

1. Decode the audio at 16 kHz. Use the vocals-only track from the DNN vocal splitter when there is one.
   Otherwise estimate singing from the stereo mix: lead vocals are usually panned to the centre, so
   voice-band (250-3500 Hz) energy that is strong in the middle but not on the sides suggests singing.
2. Turn that into a 'singing now' on/off curve sampled every 20 ms.
3. Slide the lyrics' timeline along the curve to find the overall shift that fits best. This fixes music
   videos with an intro or skit that the recording the lyrics were timed to doesn't have.
4. Move each line to the nearby moment singing starts, and measure how long the singing lasts, so the
   colour fill follows the real phrase instead of an estimate.

Anything uncertain is left as it was: no shift unless it clearly fits better, no nudge without a nearby onset.
"""
import logging, subprocess
import numpy as np

SR = 16000
HOP = 320            # 20 ms frames
FPS = SR / HOP
WIN = 1024
NUDGE = 0.7          # seconds a line may move to meet the singing
MIN_GAP = 0.45       # silence that ends a sung phrase; shorter pauses (a comma, a breath) don't (seconds)
TAIL = 0.2           # fading word endings fall below the threshold; add them back (seconds)
EPS = 1e-10


def _decode(path, channels):
	raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', str(channels), '-ar', str(SR),
	                               '-f', 's16le', '-'], timeout = 120)
	return (np.frombuffer(raw, np.int16).astype(np.float32) / 32768).reshape(-1, channels)


def _band_energy(x):
	"""Voice-band energy of each 20 ms frame."""
	n = (len(x) - WIN) // HOP + 1
	if n <= 0:
		return np.zeros(0, np.float32)
	freqs = np.fft.rfftfreq(WIN, 1 / SR)
	band = (freqs >= 250) & (freqs <= 3500)
	window = np.hanning(WIN).astype(np.float32)
	out = np.empty(n, np.float32)
	for a in range(0, n, 2048):  # in chunks to keep memory small
		b = min(n, a + 2048)
		idx = np.arange(WIN)[None, :] + HOP * np.arange(a, b)[:, None]
		spec = np.abs(np.fft.rfft(x[idx] * window, axis = 1)) ** 2
		out[a:b] = spec[:, band].sum(1)
	return out


def _smooth(v, seconds):
	k = max(1, int(seconds * FPS))
	return np.convolve(v, np.ones(k) / k, mode = 'same')


def _clean(active):
	"""Drop blips shorter than 0.15 s and fill gaps shorter than 0.25 s."""
	a = active.copy()
	for value, min_len in ((False, int(0.25 * FPS)), (True, int(0.15 * FPS))):
		i, n = 0, len(a)
		while i < n:
			if a[i] == value:
				j = i
				while j < n and a[j] == value:
					j += 1
				if j - i < min_len and i > 0 and j < n:
					a[i:j] = not value
				i = j
			else:
				i += 1
	return a


def singing_curve(song_path, vocal_track = None):
	"""Boolean array, one value per 20 ms: is someone singing?"""
	if vocal_track:
		e = _band_energy(_decode(vocal_track, 1)[:, 0])
		score = 10 * np.log10(e + EPS)
	else:
		x = _decode(song_path, 2)
		mid, side = _band_energy((x[:, 0] + x[:, 1]) / 2), _band_energy((x[:, 0] - x[:, 1]) / 2)
		loud = 10 * np.log10(mid + EPS)
		if side.sum() < 1e-3 * mid.sum():  # mono recording: no stereo cue, loudness only
			score = loud
		else:
			score = 10 * np.log10((mid + EPS) / (side + EPS))
			score[loud < np.percentile(loud, 10) + 3] = score.min()  # silence is not singing
	score = _smooth(score, 0.25)
	lo, hi = np.percentile(score, 20), np.percentile(score, 90)
	return _clean(score > lo + 0.5 * (hi - lo))


def estimate_sing(words, gap):
	"""Usual time to sing a line: about 0.3 s per CJK character (one syllable each) and 0.42 s per other word."""
	units = 0.0
	for _, w in words:
		cjk = sum(1 for c in w if ord(c) > 0x2e80 and c.isalpha())
		units += 0.3 * cjk + (0.42 if any(c.isalnum() and ord(c) <= 0x2e80 for c in w) else 0)
	return min(gap * 0.92, max(1.0, units + 0.3))


_estimate_sing = estimate_sing


SECTION_BREAK = 3.5  # seconds without lyrics that start a new section (an instrumental break)
SECTION_SHIFT = 8    # how far a section may move on its own (seconds)


def _model(lines, offsets, n):
	"""What the lyrics say: singing from each line start (plus its offset) for an estimated time."""
	model = np.zeros(n, np.float32)
	for i, (t, words) in enumerate(lines):
		if words:
			gap = (lines[i + 1][0] if i + 1 < len(lines) else t + 6) - t
			a, b = int((t + offsets[i]) * FPS), int((t + offsets[i] + _estimate_sing(words, gap)) * FPS)
			model[max(0, a):max(0, min(n, b))] = 1
	return model


def _overlap(model, act, k):
	"""Fraction of the model's singing that lands on detected singing when the model moves by k frames."""
	n = len(act)
	if k >= 0:
		return float((model[:n - k] * act[k:]).sum() / max(1, model.sum()))
	return float((model[-k:] * act[:n + k]).sum() / max(1, model.sum()))


def _sections(lines):
	"""Line indexes grouped into sections split at instrumental breaks."""
	groups, last_end = [], None
	for i, (t, words) in enumerate(lines):
		if not words:
			continue
		if last_end is None or t - last_end >= SECTION_BREAK:
			groups.append([])
		groups[-1].append(i)
		gap = (lines[i + 1][0] if i + 1 < len(lines) else t + 6) - t
		last_end = t + _estimate_sing(words, gap)
	return groups


def align(song_path, lines, lrc_duration = None, song_duration = None, vocal_track = None, active = None):
	"""lines: [(start, words), ...] from lyrics.parse_lrc.
	Returns ([(start, words, sung_seconds or None), ...], summary for the log, fit score 0-1).
	Pass `active` (from singing_curve) to reuse it when trying several lyrics for the same song."""
	if active is None:
		active = singing_curve(song_path, vocal_track)
	n = len(active)
	if n == 0 or not active.any():
		return [(t, w, None) for t, w in lines], 'no singing detected', 0.0
	act = active.astype(np.float32)
	zero = [0.0] * len(lines)
	model = _model(lines, zero, n)
	if not model.any():
		return [(t, w, None) for t, w in lines], 'lyrics are outside the song', 0.0

	# 1. overall shift: where the recording lengths differ (e.g. a video intro) allow a shift of about that much
	diff = (song_duration - lrc_duration) if song_duration and lrc_duration else None
	lo, hi = (min(0, diff) - 5, max(0, diff) + 5) if diff is not None else (-15, 60)
	size = 1 << int(np.ceil(np.log2(2 * n)))
	corr = np.fft.irfft(np.fft.rfft(act - act.mean(), size) * np.conj(np.fft.rfft(model, size)), size)
	lags = np.arange(int(lo * FPS), int(hi * FPS) + 1)
	best = int(lags[np.argmax(corr[lags % size])])
	hit0, hit_best = _overlap(model, act, 0), _overlap(model, act, best)
	shift = best / FPS if hit_best - hit0 > 0.08 else 0.0
	summary = 'shift %+.2fs (lyrics on singing %d%% -> %d%%)' % (shift, hit0 * 100, (hit_best if shift else hit0) * 100)
	offsets = [shift] * len(lines)

	# 2. sections: a cover or a TV/radio edit can have a longer or shorter break; let each section move on its own
	moved_sections = 0
	groups = _sections(lines)
	for g, idx in enumerate(groups):
		sec = [(t, w) if i in idx else (t, []) for i, (t, w) in enumerate(lines)]
		m = _model(sec, offsets, n)
		if m.sum() < FPS:  # too short to judge
			continue
		base = _overlap(m, act, 0)
		k_range = range(-int(SECTION_SHIFT * FPS), int(SECTION_SHIFT * FPS) + 1, 2)
		scores = [(_overlap(m, act, k), -abs(k), k) for k in k_range]
		score, _, k = max(scores)
		if score - base < 0.12:
			continue
		first = lines[idx[0]][0] + offsets[idx[0]] + k / FPS
		prev_last = max((lines[i][0] + offsets[i] for i in groups[g - 1]), default = -1e9) if g else -1e9
		if first <= prev_last + 1.0:  # keep sections in order
			continue
		for i in range(idx[0], (groups[g + 1][0] if g + 1 < len(groups) else len(lines))):
			offsets[i] += k / FPS  # this section and its trailing pause
		moved_sections += 1
	if moved_sections:
		summary += ', %d of %d sections moved on their own' % (moved_sections, len(groups))

	# 3. per line: meet the nearest onset, and measure the phrase
	onsets = np.flatnonzero(np.diff(active.astype(np.int8)) == 1) + 1

	def place(offsets):
		out, deltas = [], []
		starts = [t + offsets[i] for i, (t, _) in enumerate(lines)]
		prev = -1e9
		for i, (t, words) in enumerate(lines):
			s = starts[i]
			nxt = starts[i + 1] if i + 1 < len(lines) else s + 6
			if not words:
				out.append((s, words, None))
				continue
			cand = onsets[np.abs(onsets / FPS - s) <= NUDGE] / FPS
			cand = cand[(cand > prev + 0.3) & (cand < nxt - 0.3)]
			if len(cand):
				new = float(cand[np.argmin(np.abs(cand - s))])
				deltas.append(new - s)
				s = new
			f = int(s * FPS)
			sung = None
			if 0 <= f < n and active[f:f + int(0.2 * FPS)].any():
				quiet = int(MIN_GAP * FPS)
				j = f
				while j < n and (active[j] or active[j:j + quiet].any()):
					j += 1
				# a breath, a quiet note or a busy mix can look like the end of the phrase, and a fill that ends early
				# races across the line; so measuring may only make a line slower than the usual pace, never faster
				floor = _estimate_sing(words, nxt - s)
				sung = min(max(floor, (j - f) / FPS + TAIL), max(0.3, (nxt - s) * 0.97))
			moved_by = s - t
			words = [(wt + moved_by if wt is not None else None, w) for wt, w in words]  # per-word times move along
			out.append((s, words, sung))
			prev = s
		return out, deltas

	out, deltas = place(offsets)
	# second pass: if most lines had to move the same way, the shifts were a bit off; move them all
	if len(deltas) >= 3 and abs(np.median(deltas)) > 0.05:
		med = float(np.median(deltas))
		offsets = [o + med for o in offsets]
		out, deltas = place(offsets)
		summary += ', refined %+.2fs' % med
	moved = sum(abs(d) > 0.05 for d in deltas)

	# fit score for choosing between lyrics: how much of the lyrics land on singing, and how much singing is covered
	final = _model([(s, w) for s, w, _ in out], [0.0] * len(out), n)
	both = float((final * act).sum())
	precision, recall = both / max(1, final.sum()), both / max(1, act.sum())
	score = 2 * precision * recall / max(1e-9, precision + recall)
	return out, summary + ', %d lines moved to the singing, fit %.2f' % (moved, score), score
