"""Times every sung word, so the karaoke colour fill follows the singer instead of sweeping at an even pace.

lrclib.net lyrics are timed per line only. Once the DNN vocal splitter has made a vocals-only track, a model
lines the known lyric text up with that track and reports when each word starts and ends:
 - English, French, German, Italian and Spanish: a phoneme recogniser trained on sung vocals (lib/singing_model.py,
   57 MB). On hand-timed songs it placed 92% of word starts within 0.3 s, and runs in seconds.
 - Any other language: a multilingual speech model (torchaudio's MMS forced aligner, ~1.2 GB), which knows more
   languages but is less sure of drawn-out sung words. Japanese, Chinese and Korean go through their romanization.

Each song is done once and the result is cached next to its lyrics (<song>.words.json); lines the model
can't place with confidence keep the per-line timing. This runs inside the vocal splitter process, which
already has torch and runs at the lowest CPU priority. Models are downloaded on first use into <songs>/.models
so they survive container updates.
"""
import hashlib, json, logging, os, re, subprocess, time

VERSION = 2          # 2: the singing model for the languages it knows
SR = 16000
CHUNK = 30           # seconds of audio per model pass
LEAD = 0.6           # a line's first word may start this much before its (singing-snapped) start
MAX_LINE = 12        # seconds; longer windows are cut so a missing line can't swallow the next one

SINGING_URL = ('https://raw.githubusercontent.com/jhuang448/LyricsAlignment-Multilingual/'
               'ca1a3923d6c8bf7d20eefa1080f3f104a6173d38/checkpoints/checkpoint_Baseline')
SINGING_SHA256 = 'd541bde0a0e2759c5eac1ee7b70e06169a2aada504bc2359a98c7de4344d4dfe'
SINGING_LANGS = {'en': 'en-us', 'fr': 'fr-fr', 'de': 'de', 'it': 'it', 'es': 'es'}  # espeak voices
# common short words, to tell the language of a song's lyrics
_STOP = {
	'en': 'the you i and to me my a it in your love is of be on that all we for so with what know oh baby no do can just like',
	'fr': 'je tu le la les et de des un une est pas que qui mon ma moi toi pour dans ne plus sur avec nous vous comme',
	'de': 'ich du und der die das nicht ist mein mich dich ein eine mit auf es wir sie so wie auch zu sich bin',
	'it': 'il di che e la non un una per mi ti sei io tu ma come con più amore mio sono questo nel del',
	'es': 'el la que de y no me te mi tu en un una es yo por con lo se más amor como para pero si',
	'nl': 'ik je jij de het een en van niet is mijn me dat wat op zijn met voor maar ook nog wel als er zo',
}
_STOP = {k: set(v.split()) for k, v in _STOP.items()}


class ModelUnavailable(Exception):
	"""A model couldn't be loaded or downloaded; try again later instead of giving up on the song."""


_model = None        # MMS
_singing = None
_espeak = {}
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


def _decode(path, sr = SR):
	import numpy as np
	raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr),
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


def _get_singing(model_dir):
	global _singing
	if _singing is None:
		from lib import singing_model
		path = os.path.join(model_dir, 'singing_baseline.pt')
		if not os.path.isfile(path):
			import requests
			os.makedirs(model_dir, exist_ok = True)
			data = requests.get(SINGING_URL, timeout = 300).content
			if hashlib.sha256(data).hexdigest() != SINGING_SHA256:
				raise ModelUnavailable('singing model download is incomplete or changed')
			with open(path + '.part', 'wb') as fp:
				fp.write(data)
			os.replace(path + '.part', path)
		_singing = singing_model.load(path)
	return _singing


def detect_language(words):
	"""'en', 'fr', 'de', 'it', 'es' or 'nl' when the lyrics are clearly in it, else None."""
	ws = [re.sub(r"[^\w']", '', w.lower()) for w in words]
	if len(ws) < 8:
		return None
	hits = {k: sum(w in v for w in ws) for k, v in _STOP.items()}
	best = max(hits, key = hits.get)
	second = max(v for k, v in hits.items() if k != best)
	return best if hits[best] >= max(4, 0.12 * len(ws)) and hits[best] >= 1.5 * second else None


def _phonemes(words, lang):
	"""Phoneme ids per word, in the singing model's phoneme set."""
	from lib import singing_model as sm
	if lang not in _espeak:
		from phonemizer.backend import EspeakBackend
		_espeak[lang] = EspeakBackend(SINGING_LANGS[lang])
	from phonemizer.separator import Separator
	clean = [re.sub(r"[^\w']", '', w.lower()) or 'a' for w in words]
	out = []
	for ph in _espeak[lang].phonemize(clean, separator = Separator(phone = ';', word = None), strip = True):
		out.append([sm.PHONE_ID.get(p, sm.UNKNOWN) for p in ph.split(';') if p])
	return out


def release():
	"""Free the models' memory when there is nothing left to align."""
	global _model, _singing
	_model = _singing = None
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


def _tokens_mms(spelled, D):
	"""Token ids and the word each belongs to (-1: none) for MMS: letters, with a 'star' around the line."""
	toks, owner = [D['*']], [-1]
	for wi, w in enumerate(spelled):
		for ch in w:
			if ch in D:
				toks.append(D[ch])
				owner.append(wi)
	return toks + [D['*']], owner + [-1]


def _tokens_singing(phones):
	"""Token ids and owners for the singing model: phonemes, with a word gap between words."""
	from lib import singing_model as sm
	toks, owner = [], []
	for wi, ph in enumerate(phones):
		if wi and toks:
			toks.append(sm.SPACE)
			owner.append(-1)
		toks += ph
		owner += [wi] * len(ph)
	return toks, owner


def _align_line_raw(em, fps, a, b, toks, owner, n, blank):
	"""(start, end) per word (None for words with nothing to align) for frames [a, b), or None if it can't run."""
	import torch
	import torchaudio.functional as F
	if sum(o >= 0 for o in owner) == 0 or (b - a) < len(toks) + sum(x == y for x, y in zip(toks, toks[1:])) + 2:
		return None
	try:
		ali, sc = F.forced_align(em[a:b][None], torch.tensor([toks], dtype = torch.int32), blank = blank)
	except RuntimeError:
		return None
	spans = F.merge_tokens(ali[0], sc[0].exp(), blank = blank)
	if len(spans) != len(toks):
		return None
	out = [None] * n
	for sp, o in zip(spans, owner):
		if o < 0:
			continue
		st, en = (a + sp.start) / fps, (a + sp.end) / fps
		out[o] = [out[o][0], en] if out[o] else [st, en]
	return out


def _align_line(em, fps, a, b, toks, owner, n, blank, letters):
	"""(start, end) per word for one line's window [a, b) of frames, or None when the fit isn't believable."""
	out = _align_line_raw(em, fps, a, b, toks, owner, n, blank)
	if out is None:
		return None
	timed = [w for w in out if w]
	durs = sorted(e - s for s, e in timed)
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
	roman = lyrics.language(song_path, cache_dir)          # ja / zh / ko
	lang = None if roman else detect_language([w for L in lines for _, w in L[1]])
	singing = lang in SINGING_LANGS
	try:
		if singing:
			from lib import singing_model as sm
			model, fps, blank = _get_singing(model_dir), sm.FPS, sm.BLANK
		else:
			(model, D), blank = _get_model(model_dir), 0
	except ModelUnavailable:
		raise
	except Exception as e:
		raise ModelUnavailable(repr(e)[:300])
	if singing:
		em = sm.emissions(model, _decode(vocal_track, sm.SR))
	else:
		em, fps = _emissions(_decode(vocal_track), model)
	result, timed = [], 0
	for i, (t, words, _) in enumerate(lines):
		if not words:
			result.append(None)
			continue
		ws = [w for _, w in words]
		if singing:
			phones = _phonemes(ws, lang)
			toks, owner = _tokens_singing(phones)
			letters = sum(len(re.sub(r'\W', '', w)) for w in ws)
		else:
			spelled = [_spell(w, roman) for w in ws]
			toks, owner = _tokens_mms(spelled, D)
			letters = sum(len(x) for x in spelled)
		end = lines[i + 1][0] if i + 1 < len(lines) else t + 8
		a, b = int(max(0, t - LEAD) * fps), int(min(end, t + MAX_LINE) * fps)
		r = _align_line(em, fps, a, min(b, em.shape[0]), toks, owner, len(ws), blank, letters)
		result.append(r)
		timed += r is not None
	with open(words_path(song_path, cache_dir), 'w', encoding = 'utf-8') as fp:
		json.dump({'version': VERSION, 'model': 'singing' if singing else 'mms', 'language': lang or roman,
		           'lines': result}, fp)
	logging.info('Word timing (%s, %s): %d of %d lines timed for %s' % ('singing model' if singing else 'MMS',
	             lang or roman or '?', timed, sum(1 for L in lines if L[1]), os.path.basename(song_path)))
	return timed


def _cached_version(song_path, cache_dir):
	try:
		with open(words_path(song_path, cache_dir), encoding = 'utf-8') as fp:
			return json.load(fp).get('version')
	except (OSError, ValueError, AttributeError):
		return None


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
		if os.path.isfile(lyrics.cache_paths(song, cache_dir)[0]) and _cached_version(song, cache_dir) != VERSION:
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
	for song, vocal in todo:
		try:
			align(song, cache_dir, vocal, os.path.join(song_dir, '.models'))
		except ModelUnavailable as e:  # no network for the first download, say; try again in a while
			logging.warning('Word timing: could not load the model: %s' % str(e)[:300])
			_retry_at = time.time() + 600
			break
		except Exception as e:
			logging.warning('Word timing failed for %s: %s' % (os.path.basename(song), repr(e)[:300]))
			try:  # don't retry a song that fails every time
				with open(words_path(song, cache_dir), 'w', encoding = 'utf-8') as fp:
					json.dump({'version': VERSION, 'lines': None, 'error': repr(e)[:300]}, fp)
			except OSError:
				pass
	release()
