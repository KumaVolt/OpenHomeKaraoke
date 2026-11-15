"""Times every sung word, so the karaoke colour fill follows the singer instead of sweeping at an even pace.

lrclib.net lyrics are timed per line only. Once the DNN vocal splitter has made a vocals-only track, a
multilingual speech model (torchaudio's MMS forced aligner) lines the known lyric text up with that track and
reports when each word starts and ends. Japanese, Chinese and Korean are aligned through their romanization.

Each song is done once and the result is cached next to its lyrics (<song>.words.json); lines the model
can't place with confidence keep the per-line timing. This runs inside the vocal splitter process, which
already has torch and runs at the lowest CPU priority. The model (~1.2 GB) is downloaded on first use into
<songs>/.models so it survives container updates.
"""
import json, logging, os, re, subprocess, time

VERSION = 1
SR = 16000
CHUNK = 30           # seconds of audio per model pass
LEAD = 0.6           # a line's first word may start this much before its (singing-snapped) start
MAX_LINE = 12        # seconds; longer windows are cut so a missing line can't swallow the next one

_model = None
_retry_at = 0        # after a failed model load, wait before trying again
_lyrics_at = 0       # next time to look for split songs that still need lyrics


def words_path(song_path, cache_dir):
	return os.path.join(cache_dir, os.path.basename(song_path)) + '.words.json'


def load(song_path, cache_dir):
	"""[[(start, end) or None per word] per line], or None when the song hasn't been (or couldn't be) timed."""
	p = words_path(song_path, cache_dir)
	try:
		with open(p, encoding = 'utf-8') as fp:
			d = json.load(fp)
		return d['lines'] if d.get('version') == VERSION and d.get('lines') else None
	except (OSError, ValueError, KeyError):
		return None


def _decode(path):
	import numpy as np
	raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(SR),
	                               '-f', 's16le', '-'], timeout = 300)
	return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


def _get_model(model_dir):
	global _model
	if _model is None:
		import torch, torchaudio
		torch.hub.set_dir(model_dir)
		bundle = torchaudio.pipelines.MMS_FA
		_model = (bundle.get_model(with_star = True).eval(), bundle.get_dict(star = '*'))
	return _model


def release():
	"""Free the model's memory (about 1.2 GB) when there is nothing left to align."""
	global _model
	_model = None
	import gc
	gc.collect()


def _emissions(wav, model):
	"""Per-frame log-probabilities of each character for the whole track, computed in chunks."""
	import torch
	ems, pos, step, ov = [], 0, CHUNK * SR, SR
	with torch.inference_mode():
		while pos < len(wav):
			a = max(0, pos - ov)
			seg = torch.from_numpy(wav[a:pos + step].copy())
			e, _ = model(seg[None])
			e = torch.log_softmax(e[0], -1)
			fps = e.shape[0] / (len(seg) / SR)
			skip = int(round((pos - a) / SR * fps))
			ems.append(e[skip:skip + int(round(min(step, len(wav) - pos) / SR * fps))])
			pos += step
	em = torch.cat(ems)
	return em, em.shape[0] / (len(wav) / SR)


def _spell(word, lang):
	"""The letters the model knows for a word: romanized for ja/zh/ko, accents stripped otherwise."""
	if lang:
		from lib import romanize
		try:
			word = romanize.line(word, lang)
		except Exception:
			pass
	from unidecode import unidecode
	return re.sub(r"[^a-z']", '', unidecode(word).lower())


def _align_line(em, fps, a, b, spelled, D):
	"""(start, end) per word for one line's window [a, b) of frames, or None when the fit isn't believable."""
	import torch
	import torchaudio.functional as F
	toks, owner = [D['*']], [-1]
	for wi, s in enumerate(spelled):
		for ch in s:
			if ch in D:
				toks.append(D[ch])
				owner.append(wi)
	toks.append(D['*'])
	owner.append(-1)
	if len(toks) <= 2 or (b - a) < 2 * len(toks):
		return None
	ali, sc = F.forced_align(em[a:b][None], torch.tensor([toks], dtype = torch.int32), blank = 0)
	spans = F.merge_tokens(ali[0], sc[0].exp())
	if len(spans) != len(toks):
		return None
	out = [None] * len(spelled)
	for sp, o in zip(spans, owner):
		if o < 0:
			continue
		st, en = (a + sp.start) / fps, (a + sp.end) / fps
		out[o] = [out[o][0], en] if out[o] else [st, en]
	timed = [w for w in out if w]
	durs = sorted(e - s for s, e in timed)
	letters = sum(len(x) for x in spelled)
	if not timed or durs[len(durs) // 2] < 0.1 or timed[-1][1] - timed[0][0] < 0.05 * letters:
		return None  # words squeezed into a few frames: the model found no singing to match them to
	# words with nothing to align (a symbol, a lone dash) sit between their neighbours
	for i, w in enumerate(out):
		if w is None:
			prev = next((out[j][1] for j in range(i - 1, -1, -1) if out[j]), timed[0][0])
			nxt = next((out[j][0] for j in range(i + 1, len(out)) if out[j]), prev)
			out[i] = [prev, max(prev, nxt)]
	return [[round(s, 2), round(e, 2)] for s, e in out]


def align(song_path, cache_dir, vocal_track, model_dir):
	"""Time the words of a song's cached lyrics and cache the result. Returns how many lines were timed."""
	from lib import lyrics
	lines = lyrics.load(song_path, cache_dir, word_times = False)
	if not lines:
		return 0
	lang = lyrics.language(song_path, cache_dir)
	model, D = _get_model(model_dir)
	wav = _decode(vocal_track)
	em, fps = _emissions(wav, model)
	result, timed = [], 0
	for i, (t, words, _) in enumerate(lines):
		if not words:
			result.append(None)
			continue
		end = lines[i + 1][0] if i + 1 < len(lines) else t + 8
		a, b = int(max(0, t - LEAD) * fps), int(min(end, t + MAX_LINE) * fps)
		r = _align_line(em, fps, a, min(b, em.shape[0]), [_spell(w, lang) for _, w in words], D)
		result.append(r)
		timed += r is not None
	with open(words_path(song_path, cache_dir), 'w', encoding = 'utf-8') as fp:
		json.dump({'version': VERSION, 'lines': result}, fp)
	logging.info('Word timing: %d of %d lines timed for %s' % (timed, sum(1 for L in lines if L[1]), os.path.basename(song_path)))
	return timed


def pending(song_dir, cache_dir):
	"""Songs with lyrics and a vocals-only track whose words haven't been timed yet."""
	from lib import lyrics
	out = []
	try:
		names = os.listdir(song_dir)
	except OSError:
		return out
	for name in sorted(names):
		song = os.path.join(song_dir, name)
		vocal = os.path.join(song_dir, 'vocal', name + '.m4a')
		if name.startswith('.') or not os.path.isfile(song) or not os.path.isfile(vocal):
			continue
		if os.path.isfile(lyrics.cache_paths(song, cache_dir)[0]) and not os.path.isfile(words_path(song, cache_dir)):
			out.append((song, vocal))
	return out


def fetch_missing_lyrics(song_dir, cache_dir):
	"""Look up lyrics for split songs that haven't been played yet, so their words are timed before the first
	time they're sung. Each song is looked up once (lyrics.prepare caches a 'none' marker when there are none)."""
	from lib import lyrics
	try:
		names = sorted(os.listdir(song_dir))
	except OSError:
		return
	for name in names:
		song = os.path.join(song_dir, name)
		if name.startswith('.') or not os.path.isfile(song) or not os.path.isfile(os.path.join(song_dir, 'vocal', name + '.m4a')):
			continue
		if any(os.path.isfile(p) for p in lyrics.cache_paths(song, cache_dir)):
			continue
		title = os.path.splitext(name)[0].split('---')[0]
		try:
			lyrics.prepare(song, cache_dir, title, vocal_track = os.path.join(song_dir, 'vocal', name + '.m4a'))
		except Exception as e:
			logging.warning('Lyrics lookup failed for %s: %s' % (name, repr(e)[:200]))


def run_pending(song_dir):
	"""Time every song that is waiting; called by the vocal splitter when it has nothing to split."""
	global _retry_at
	cache_dir = os.path.join(song_dir, '.lyrics')
	global _lyrics_at
	if time.time() >= _lyrics_at:  # a failed lookup leaves no marker, so don't ask lrclib.net every 2 s
		_lyrics_at = time.time() + 300
		fetch_missing_lyrics(song_dir, cache_dir)
	todo = pending(song_dir, cache_dir)
	if not todo or time.time() < _retry_at:
		return
	try:
		_get_model(os.path.join(song_dir, '.models'))
	except Exception as e:  # no network for the first download, say; try again in a while
		logging.warning('Word timing: could not load the model: %s' % repr(e)[:300])
		_retry_at = time.time() + 600
		return
	for song, vocal in todo:
		try:
			align(song, cache_dir, vocal, os.path.join(song_dir, '.models'))
		except Exception as e:
			logging.warning('Word timing failed for %s: %s' % (os.path.basename(song), repr(e)[:300]))
			try:  # don't retry a song that fails every time
				with open(words_path(song, cache_dir), 'w', encoding = 'utf-8') as fp:
					json.dump({'version': VERSION, 'lines': None, 'error': repr(e)[:300]}, fp)
			except OSError:
				pass
	release()
