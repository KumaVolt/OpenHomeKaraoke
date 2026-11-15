"""Pronunciation in Latin letters for lyrics: Japanese romaji (Hepburn, via pykakasi), Chinese pinyin with
tone marks (via pinyin) and Korean Revised Romanization (built in).

tokens() splits a line into (original piece, romanized piece) pairs that cover the line in order, so the
karaoke fill on the romanized line can follow the timing of the original characters.
Readings come from dictionaries: kanji with an unusual or poetic reading in a song can come out wrong.
"""
import re

KANA = re.compile('[぀-ヿ]')
HANGUL = re.compile('[가-힣]')
HAN = re.compile('[㐀-䶿一-鿿]')


def detect(texts):
	"""'ja', 'zh', 'ko' or None for a song's lyric lines. Kana anywhere means Japanese, even for kanji-only lines."""
	s = ''.join(texts)
	if KANA.search(s):
		return 'ja'
	if HANGUL.search(s):
		return 'ko'
	if HAN.search(s):
		return 'zh'
	return None


def tokens(text, lang):
	if lang == 'ja':
		return _ja(text)
	if lang == 'zh':
		return _split(text, HAN, _pinyin)
	if lang == 'ko':
		return _ko(text)
	return [(text, text)]


def separator(lang):
	"""What goes between romanized pieces: Korean syllables join into words, others are spaced."""
	return '' if lang == 'ko' else ' '


def line(text, lang):
	sep = separator(lang)
	return ' '.join(sep.join(r for _, r in tokens(text, lang)).split())


# ---------- Japanese ----------
_kks = None


def _ja(text):
	global _kks
	if _kks is None:
		import pykakasi
		_kks = pykakasi.kakasi()
	out = [(item['orig'], item['hepburn']) for item in _kks.convert(text)]
	return out if ''.join(o for o, _ in out) == text else [(text, ' '.join(r for _, r in out))]


# ---------- Chinese ----------
def _pinyin(ch):
	import pinyin
	return pinyin.get(ch, format = 'diacritical')


def _split(text, pattern, convert):
	"""One token per matching character, runs of anything else kept as they are."""
	out = []
	for ch in text:
		if pattern.match(ch):
			out.append((ch, convert(ch)))
		elif out and not pattern.match(out[-1][0][-1]):
			out[-1] = (out[-1][0] + ch, out[-1][1] + ch)
		else:
			out.append((ch, ch))
	return out


# ---------- Korean (Revised Romanization) ----------
_INITIAL = ['g', 'kk', 'n', 'd', 'tt', 'r', 'm', 'b', 'pp', 's', 'ss', '', 'j', 'jj', 'ch', 'k', 't', 'p', 'h']
_VOWEL = ['a', 'ae', 'ya', 'yae', 'eo', 'e', 'yeo', 'ye', 'o', 'wa', 'wae', 'oe', 'yo', 'u', 'wo', 'we', 'wi', 'yu', 'eu', 'ui', 'i']
_FINAL = ['', 'k', 'k', 'k', 'n', 'n', 'n', 't', 'l', 'k', 'm', 'l', 'l', 'l', 'p', 'l', 'm', 'p', 'p', 't', 't', 'ng', 't', 't', 'k', 't', 'p', 't']
# before a syllable that starts with a vowel, a final consonant is sounded at the start of that syllable
_CARRY = {1: 'g', 2: 'kk', 4: 'n', 7: 'd', 8: 'r', 16: 'm', 17: 'b', 19: 's', 20: 'ss', 22: 'j', 23: 'ch', 24: 'k', 25: 't', 26: 'p', 27: ''}


def _ko(text):
	sylls = []
	for ch in text:
		c = ord(ch) - 0xAC00
		sylls.append([c // 588, (c % 588) // 28, c % 28] if 0 <= c < 11172 else None)
	parts = [[_INITIAL[s[0]], _VOWEL[s[1]], _FINAL[s[2]]] if s else None for s in sylls]
	for i in range(len(sylls) - 1):
		s, n = sylls[i], sylls[i + 1]
		if s and n and n[0] == 11 and s[2] in _CARRY:  # next syllable starts with silent ieung
			parts[i][2] = ''
			parts[i + 1][0] = _CARRY[s[2]]
	out = []
	for ch, p in zip(text, parts):
		if p:
			out.append((ch, ''.join(p)))
		elif out and not HANGUL.match(out[-1][0][-1]):
			out[-1] = (out[-1][0] + ch, out[-1][1] + ch)
		else:
			out.append((ch, ch))
	return out
