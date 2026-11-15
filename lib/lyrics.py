"""Synced lyrics for songs that have none.

Looks the song up on LRCLIB (https://lrclib.net, a free, open lyrics database) and turns its
time-stamped LRC lyrics into an ASS subtitle file with a karaoke colour fill, which VLC shows
over the video (it needs VLC's libass plugin).

LRC usually only says when each line starts, so the fill across a line is spread over an
estimated singing time. When the lyrics carry per-word times (enhanced LRC, '<mm:ss.xx>' tags),
those are used instead. The phone's subtitle delay control shifts everything if a video has a
longer intro than the recording the lyrics were timed to.

Before saving, lib/lyrics_align.py lines the lyrics up with the singing it hears in the file: it fixes an
overall offset (such as a video intro), moves lines to where singing starts, and measures each sung phrase
so the fill runs as long as the singing does.
"""
import os, re, json, logging, subprocess
import requests
import numpy as np

API = 'https://lrclib.net/api/search'
USER_AGENT = 'OpenHomeKaraoke (https://github.com/xuancong84/OpenHomeKaraoke)'
ASS_VERSION = 5  # bump when the subtitle layout changes, so cached subtitle files are rebuilt
DURATION_TOLERANCE = 8  # seconds; a bigger difference usually means a different recording or a long video intro

BRACKETS = re.compile(r'[\(\[\{【（].*?[\)\]\}】）]')
NOISE = re.compile(r'\b(official|music|lyrics?|video|audio|hd|hq|4k|remaster(ed)?|m/?v|visuali[sz]er|version)\b', re.I)
LINE_TAG = re.compile(r'\[(\d+):(\d+(?:\.\d+)?)\]')
WORD_TAG = re.compile(r'<(\d+):(\d+(?:\.\d+)?)>')

# colours are &HAABBGGRR; they match the KTV theme (pink fill over light lilac, dark purple outline)
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sing,{font},58,&H00A53EFF,&H00FFEEF4,&H00301026,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,60,60,104,1
Style: SingHigh,{font},58,&H00A53EFF,&H00FFEEF4,&H00301026,&H00000000,-1,0,0,0,100,100,0,0,1,4,0,2,60,60,150,1
Style: Roman,{font},34,&H00A53EFF,&H00FFEEF4,&H00301026,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,60,60,106,1
Style: Next,{font},40,&H00C99BA9,&H00C99BA9,&H00301026,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,60,60,46,1
Style: Lane,{font},20,&H00FFFFFF,&H00FFFFFF,&H00A53EFF,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def clean_title(name):
	"""'Queen - Bohemian Rhapsody (Official Video) [4K]' -> 'Queen Bohemian Rhapsody'"""
	name = BRACKETS.sub(' ', name)
	name = NOISE.sub(' ', name)
	name = re.sub(r'[-–—|_~"“”]+', ' ', name)
	return ' '.join(name.split())


def media_info(path):
	"""(duration in seconds or None, has its own subtitle track)"""
	try:
		out = subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type',
		                               '-of', 'default=nw=1', path], timeout = 15).decode()
		dur = re.search(r'duration=([\d.]+)', out)
		return (float(dur.group(1)) if dur else None), 'codec_type=subtitle' in out
	except Exception:
		return None, False


def search(name, duration, limit = 6):
	"""LRCLIB entries with synced lyrics, best guesses first: same length as our file, then the rest.
	The caller picks between them by how well each fits the singing heard in the song."""
	r = requests.get(API, params = {'q': clean_title(name)}, headers = {'User-Agent': USER_AGENT}, timeout = 6)
	r.raise_for_status()
	hits = [h for h in r.json() if h.get('syncedLyrics')]
	seen, unique = set(), []
	for h in hits:  # the same lyrics are often uploaded several times
		key = h['syncedLyrics'].strip()
		if key not in seen:
			seen.add(key)
			unique.append(h)
	if duration:
		unique.sort(key = lambda h: abs((h.get('duration') or 1e9) - duration))
	return unique[:limit]


def parse_lrc(text):
	"""[(start, [(word_start or None, word), ...]), ...] sorted by start; an empty word list is a pause."""
	lines = []
	for raw in text.splitlines():
		stamps = LINE_TAG.findall(raw)
		body = LINE_TAG.sub('', raw)
		words = []
		parts = WORD_TAG.split(body)  # [text, m, s, text, m, s, text ...]
		if len(parts) > 1:
			for i in range(1, len(parts), 3):
				w = parts[i + 2].strip()
				if w:
					words.append((int(parts[i]) * 60 + float(parts[i + 1]), w))
		else:
			units = body.split() if ' ' in body.strip() else list(body.strip())  # words, or characters for CJK
			words = [(None, w) for w in units if w.strip()]
		for m, s in stamps:
			lines.append((int(m) * 60 + float(s), words))
	return sorted(lines, key = lambda L: L[0])


def _ts(t):
	t = max(0, t)
	return '%d:%02d:%05.2f' % (t // 3600, t % 3600 // 60, t % 60)


def _esc(s):
	return s.replace('\\', '/').replace('{', '(').replace('}', ')')


def _joiner(words):
	"""Lines without spaces (Chinese, Japanese) are split into single characters; put them back without spaces."""
	return '' if words and all(len(w) == 1 for _, w in words) and any(ord(w) > 0x2e80 for _, w in words) else ' '


def _roman_text(units, sep, lead, lang):
	"""The romanized line with its own karaoke fill. units: [(seconds, text), ...] of the original line.
	Each original character gets an equal share of its unit's time, and each romanized piece is filled
	over the time of the characters it was made from."""
	from lib import romanize
	text, char_t = '', []
	for k, (d, w) in enumerate(units):
		if k:
			text += sep
			char_t += [0.0] * len(sep)
		text += w
		char_t += [d / len(w)] * len(w)
	pieces = romanize.tokens(text, lang)
	rsep = romanize.separator(lang)
	out, pos = ('{\\k%d}' % round(lead * 100) if lead else ''), 0
	for orig, rom in pieces:
		d = sum(char_t[pos:pos + len(orig)])
		pos += len(orig)
		if not rom.strip():  # a space between words
			out += ' '
		elif d > 0.005:
			out += '{\\kf%d}%s%s' % (max(1, round(d * 100)), _esc(rom), rsep)
		else:
			out += _esc(rom) + rsep
	return ' '.join(out.split())


def to_ass(lines, font = 'Arial Unicode MS', roman_lang = None, notes = None):
	"""lines: [(start, words[, sung_seconds]), ...]. With roman_lang ('ja', 'zh', 'ko') a romanized line is
	shown under each sung line, filling in step with it. With notes ([(start, end, midi), ...]) a pitch guide
	lane is drawn above the lyrics."""
	events = pitch_events(lines, notes) if notes else []
	sing_style = 'SingHigh' if roman_lang else 'Sing'
	sung = [(L[0], L[1]) for L in lines if L[1]]
	if sung:  # show the first line a few seconds before it is sung
		t0, w0 = sung[0]
		events.append((max(0, t0 - 4), t0, 'Next', _joiner(w0).join(_esc(w) for _, w in w0)))
	for i, line in enumerate(lines):
		t, words = line[0], line[1]
		measured = line[2] if len(line) > 2 else None  # sung length from lyrics_align, when known
		if not words:
			continue
		nxt = lines[i + 1][0] if i + 1 < len(lines) else t + 6
		end = min(nxt, t + 12)
		gap = max(0.3, end - t)
		sep = _joiner(words)
		if all(wt is not None for wt, _ in words):  # enhanced LRC: real word times
			times = [wt for wt, _ in words] + [min(words[-1][0] + 0.8, end)]
			durs = [max(0.05, times[k + 1] - times[k]) for k in range(len(words))]
			lead = max(0, words[0][0] - t)
		else:  # line times only: spread the fill over the sung (or an estimated) time, weighted by word length
			from lib.lyrics_align import estimate_sing
			sing = estimate_sing(words, gap)  # never fill faster than the usual pace: a racing fill is worse than a slow one
			if measured:
				sing = min(gap * 0.97, max(sing, measured))
			total = sum(len(w) for _, w in words)
			durs = [sing * len(w) / total for _, w in words]
			lead = 0
		text = ('{\\k%d}' % round(lead * 100) if lead else '') + \
		       sep.join('{\\kf%d}%s' % (max(1, round(d * 100)), _esc(w)) for d, (_, w) in zip(durs, words))
		events.append((t, end, sing_style, text))
		if roman_lang:
			try:
				events.append((t, end, 'Roman', _roman_text([(d, w) for d, (_, w) in zip(durs, words)], sep, lead, roman_lang)))
			except Exception as e:
				logging.info('Romanization failed for a line: %s' % repr(e)[:200])
		upcoming = next((L for L in lines[i + 1:] if L[1]), None)  # (start, words, ...)
		if upcoming and upcoming[0] - end < 4:
			events.append((t, end, 'Next', _joiner(upcoming[1]).join(_esc(w) for _, w in upcoming[1])))
	body = ''.join('Dialogue: %d,%s,%s,%s,,0,0,0,,%s\n' % (e[4] if len(e) > 4 else 0, _ts(e[0]), _ts(e[1]), e[2], e[3]) for e in events)
	return ASS_HEADER.replace('{font}', font) + body


def cache_paths(song_path, cache_dir):
	"""(aligned lyrics .json, 'no lyrics' marker). Subtitle files are built from the .json when the song plays."""
	base = os.path.join(cache_dir, os.path.basename(song_path))
	return base + '.json', base + '.none'


def load(song_path, cache_dir):
	path = cache_paths(song_path, cache_dir)[0]
	if not os.path.isfile(path):
		return None
	with open(path, encoding = 'utf-8') as fp:
		return [(L[0], [tuple(w) for w in L[1]], L[2]) for L in json.load(fp)]


def language(song_path, cache_dir):
	"""'ja', 'zh' or 'ko' when the cached lyrics can be romanized, else None."""
	from lib import romanize
	lines = load(song_path, cache_dir)
	return romanize.detect(w for L in lines for _, w in L[1]) if lines else None


def subtitle_file(song_path, cache_dir, romanized = True, notes = None):
	"""Path of the subtitle file to give VLC, built from the cached lyrics and/or the pitch guide notes, or None."""
	lines = load(song_path, cache_dir)
	if lines is None and not notes:
		return None
	lang = language(song_path, cache_dir) if romanized and lines else None
	base = cache_paths(song_path, cache_dir)[0][:-5]
	ass = base + ('.' + lang if lang else '') + ('.pitch' if notes else '') + '.v%d.ass' % ASS_VERSION
	os.makedirs(cache_dir, exist_ok = True)
	with open(ass, 'w', encoding = 'utf-8') as fp:  # cheap to build, so always fresh
		fp.write(to_ass(lines or [], roman_lang = lang, notes = notes))
	return ass


# ---------- pitch guide lane ----------
LANE = (170, 40, 1110, 170)   # x0, y0, x1, y1 on the 1280x720 subtitle canvas, above the picture's centre
PAGE_MAX = 8.0                # seconds of melody on one page at most


def _rrect(x0, y0, x1, y1, r):
	"""ASS drawing commands for a rectangle with rounded corners."""
	r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
	x0, y0, x1, y1, r = (int(round(v)) for v in (x0, y0, x1, y1, r))
	return ('m %d %d l %d %d b %d %d %d %d %d %d l %d %d b %d %d %d %d %d %d l %d %d b %d %d %d %d %d %d l %d %d b %d %d %d %d %d %d '
	        % (x0 + r, y0, x1 - r, y0, x1, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1, y1, x1 - r, y1,
	           x0 + r, y1, x0, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0, x0, y0, x0 + r, y0))


def _pages(lines, notes):
	"""[(start, end)] spans shown one at a time: one lyric line each when there are lyrics, split at most every PAGE_MAX s."""
	first, last = notes[0][0], max(n[1] for n in notes)
	marks = sorted({round(L[0], 2) for L in lines if L[1]} | {round(first - 0.3, 2)})
	marks = [m for m in marks if m < last] + [round(last + 0.3, 2)]
	pages = []
	for a, b in zip(marks, marks[1:]):
		k = int(np.ceil((b - a) / PAGE_MAX)) if b - a > PAGE_MAX else 1
		pages += [(a + (b - a) * j / k, a + (b - a) * (j + 1) / k) for j in range(k)]
	return [(a, b) for a, b in pages if any(n[0] < b and n[1] > a for n in notes)]


def pitch_events(lines, notes):
	"""The pitch guide: each page shows the notes to sing as bars (higher bar = higher note), a playhead
	sweeps across with the music, and the bars light up as it passes them."""
	X0, Y0, X1, Y1 = LANE
	events = []
	for p0, p1 in _pages(lines, notes):
		page = [(max(a, p0), min(b, p1), m) for a, b, m in notes if a < p1 and b > p0 and min(b, p1) - max(a, p0) > 0.05]
		lo, hi = min(m for _, _, m in page), max(m for _, _, m in page)
		span = max(10, hi - lo + 3)  # a semitone of room above and below
		lo = (lo + hi) / 2 - span / 2
		row = (Y1 - Y0) / span
		xt = lambda t: X0 + (t - p0) / (p1 - p0) * (X1 - X0)
		bars = ''.join(_rrect(xt(a), Y1 - (m - lo + 0.5) * row - 6, xt(b) - 2, Y1 - (m - lo + 0.5) * row + 6, 6) for a, b, m in page)
		ms = int(round((p1 - p0) * 1000))
		pos = '\\an7\\pos(0,0)\\shad0'
		events.append((p0, p1, 'Lane', '{%s\\bord1.5\\1c&H2A0E22&\\1a&H28&\\3c&HA53EFF&\\3a&H90&\\p1}%s' % (pos, _rrect(X0 - 18, Y0 - 12, X1 + 18, Y1 + 12, 16)), 0))
		events.append((p0, p1, 'Lane', '{%s\\bord0\\1c&HFFEEF4&\\1a&H98&\\p1}%s' % (pos, bars), 1))
		events.append((p0, p1, 'Lane', '{%s\\bord2\\blur2\\1c&HA53EFF&\\3c&HA53EFF&\\3a&H80&\\clip(%d,0,%d,720)\\t(0,%d,\\clip(%d,0,%d,720))\\p1}%s'
		               % (pos, X0, X0, ms, X0, X1, bars), 2))
		playhead = _rrect(0, 0, 3, Y1 - Y0 + 16, 1.5)
		events.append((p0, p1, 'Lane', '{\\an7\\move(%d,%d,%d,%d)\\shad0\\bord0\\1c&HFFFFFF&\\1a&H30&\\p1}%s'
		               % (X0 - 1, Y0 - 8, X1 - 1, Y0 - 8, playhead), 3))
	return events


def pick(song_path, hits, duration, vocal_track = None):
	"""Line each candidate up with the singing heard in the file and keep the one that fits best.
	Covers and TV/radio edits of a song have different timing, and the closest length is often not the right one."""
	try:
		from lib import lyrics_align
		active = lyrics_align.singing_curve(song_path, vocal_track if vocal_track and os.path.isfile(vocal_track) else None)
		best = None
		for h in hits:
			lines, summary, score = lyrics_align.align(song_path, parse_lrc(h['syncedLyrics']), h.get('duration'), duration, active = active)
			logging.info('Lyrics candidate %.2f: "%s - %s" (%ss): %s' % (score, h.get('artistName'), h.get('trackName'), h.get('duration'), summary))
			if best is None or score > best[0] + 0.02:  # ties go to the earlier, closer-length candidate
				best = (score, h, lines)
		return best[1], best[2]
	except Exception as e:
		logging.info('Lyrics alignment skipped: %s' % repr(e)[:200])
		return hits[0], [(t, w, None) for t, w in parse_lrc(hits[0]['syncedLyrics'])]


def prepare(song_path, cache_dir, name, vocal_track = None):
	"""Fetch, align and cache lyrics for a song. Returns True when the song has lyrics."""
	cached, none = cache_paths(song_path, cache_dir)
	if os.path.isfile(cached):
		return True
	if os.path.isfile(none):
		return False
	reason = None
	if 'karaoke' in name.lower():
		reason = 'karaoke video, lyrics are in the picture'
	else:
		duration, has_subs = media_info(song_path)
		if has_subs:
			reason = 'the file has its own subtitles'
		else:
			try:
				hits = search(name, duration)
			except Exception as e:
				logging.info('Lyrics lookup failed, will retry next time: %s' % repr(e)[:200])
				return False
			if not hits:
				reason = 'not found on lrclib.net'
			else:
				hit, lines = pick(song_path, hits, duration, vocal_track)
				os.makedirs(cache_dir, exist_ok = True)
				with open(cached, 'w', encoding = 'utf-8') as fp:
					json.dump([[t, [list(x) for x in w], sung] for t, w, sung in lines], fp, ensure_ascii = False)
				logging.info('Lyrics: using "%s - %s" from lrclib.net for %s' % (hit.get('artistName'), hit.get('trackName'), name))
				return True
	logging.info('Lyrics: none for %s (%s)' % (name, reason))
	os.makedirs(cache_dir, exist_ok = True)
	open(none, 'w').close()
	return False
