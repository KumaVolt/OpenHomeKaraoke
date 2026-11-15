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
import os, re, logging, subprocess
import requests

API = 'https://lrclib.net/api/search'
USER_AGENT = 'OpenHomeKaraoke (https://github.com/xuancong84/OpenHomeKaraoke)'
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
Style: Next,{font},40,&H00C99BA9,&H00C99BA9,&H00301026,&H00000000,-1,0,0,0,100,100,0,0,1,3,0,2,60,60,46,1

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


def search(name, duration):
	"""Best LRCLIB match with synced lyrics, preferring the same length as our file."""
	r = requests.get(API, params = {'q': clean_title(name)}, headers = {'User-Agent': USER_AGENT}, timeout = 6)
	r.raise_for_status()
	hits = [h for h in r.json() if h.get('syncedLyrics')]
	if not hits:
		return None
	if duration:
		close = [h for h in hits if h.get('duration') and abs(h['duration'] - duration) <= DURATION_TOLERANCE]
		if close:
			return min(close, key = lambda h: abs(h['duration'] - duration))
		logging.info('Lyrics: no match within %ds of the song length, timing may be off: %s' % (DURATION_TOLERANCE, name))
	return hits[0]


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
	return '' if words and all(len(w) == 1 and ord(w) > 0x2e80 for _, w in words) else ' '


def to_ass(lines, font = 'Arial Unicode MS'):
	events = []
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
		else:  # line times only: spread the fill over an estimated singing time, weighted by word length
			per_unit = 0.42 if sep == ' ' else 0.3
			sing = min(gap * 0.97, measured) if measured else min(gap * 0.92, max(1.0, per_unit * len(words) + 0.3))
			total = sum(len(w) for _, w in words)
			durs = [sing * len(w) / total for _, w in words]
			lead = 0
		text = ('{\\k%d}' % round(lead * 100) if lead else '') + \
		       sep.join('{\\kf%d}%s' % (max(1, round(d * 100)), _esc(w)) for d, (_, w) in zip(durs, words))
		events.append((t, end, 'Sing', text))
		upcoming = next((L for L in lines[i + 1:] if L[1]), None)  # (start, words, ...)
		if upcoming and upcoming[0] - end < 4:
			events.append((t, end, 'Next', _joiner(upcoming[1]).join(_esc(w) for _, w in upcoming[1])))
	body = ''.join('Dialogue: 0,%s,%s,%s,,0,0,0,,%s\n' % (_ts(a), _ts(b), style, text) for a, b, style, text in events)
	return ASS_HEADER.replace('{font}', font) + body


def cache_paths(song_path, cache_dir):
	base = os.path.join(cache_dir, os.path.basename(song_path))
	return base + '.ass', base + '.none'


def prepare(song_path, cache_dir, name, vocal_track = None):
	"""Fetch and cache lyrics for a song. Returns the .ass path, or None when the song gets no lyrics."""
	ass, none = cache_paths(song_path, cache_dir)
	if os.path.isfile(ass):
		return ass
	if os.path.isfile(none):
		return None
	reason = None
	if 'karaoke' in name.lower():
		reason = 'karaoke video, lyrics are in the picture'
	else:
		duration, has_subs = media_info(song_path)
		if has_subs:
			reason = 'the file has its own subtitles'
		else:
			try:
				hit = search(name, duration)
			except Exception as e:
				logging.info('Lyrics lookup failed, will retry next time: %s' % repr(e)[:200])
				return None
			if hit is None:
				reason = 'not found on lrclib.net'
			else:
				lines = parse_lrc(hit['syncedLyrics'])
				try:  # line the lyrics up with the singing heard in this file
					from lib import lyrics_align
					lines, summary = lyrics_align.align(song_path, lines, hit.get('duration'), duration,
					                                    vocal_track if vocal_track and os.path.isfile(vocal_track) else None)
					logging.info('Lyrics alignment: ' + summary)
				except Exception as e:
					logging.info('Lyrics alignment skipped: %s' % repr(e)[:200])
				os.makedirs(cache_dir, exist_ok = True)
				with open(ass, 'w', encoding = 'utf-8') as fp:
					fp.write(to_ass(lines))
				logging.info('Lyrics: using "%s - %s" from lrclib.net for %s' % (hit.get('artistName'), hit.get('trackName'), name))
				return ass
	logging.info('Lyrics: none for %s (%s)' % (name, reason))
	os.makedirs(cache_dir, exist_ok = True)
	open(none, 'w').close()
	return None
