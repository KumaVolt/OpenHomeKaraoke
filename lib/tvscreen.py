"""Draws the karaoke TV screen with pygame: the welcome card, the up-next card and the get-ready card.

Layout is designed for a 1280x720 reference screen and scaled to the real screen size,
centred (letterboxed) on screens that are not 16:9.
"""
import math
import pygame, pygame.freetype
import arabic_reshaper
from bidi.algorithm import get_display
from unidecode import unidecode

ROOM = (26, 16, 48)
GLOW = (40, 24, 78)
BOOTH = (38, 26, 68)
VELVET = (58, 43, 94)
PINK = (255, 62, 165)
CYAN = (62, 231, 255)
LYRIC = (244, 238, 255)
MUTED = (169, 155, 201)
WHITE = (255, 255, 255)

REF_W, REF_H = 1280, 720
FONT_FILES = ['font/arial-unicode-ms.ttf', 'font/unifont.ttf']


def _blur(surf, factor):
	w, h = surf.get_size()
	small = pygame.transform.smoothscale(surf, (max(1, w // factor), max(1, h // factor)))
	return pygame.transform.smoothscale(small, (w, h))


class TVScreen:
	def __init__(self):
		self.fonts = None
		self.cache_key = None
		self.cache = None
		self.images = {}

	# ---------- text ----------
	def _load_fonts(self):
		if self.fonts is None:
			self.fonts = []
			for path in FONT_FILES:
				try:
					font = pygame.freetype.Font(path, 20)
					font.pad = True  # same line height for every string, so baselines line up
					self.fonts.append(font)
				except Exception:
					pass
			if not self.fonts:
				font = pygame.freetype.SysFont(None, 20)
				font.pad = True
				self.fonts.append(font)
		return self.fonts

	def _bold_font(self):
		# pygame ships FreeSans Bold, a real bold face for Latin text; other scripts fall back to synthetic bold
		if not hasattr(self, 'bold'):
			try:
				self.bold = pygame.freetype.Font(None, 20)
				self.bold.pad = True
			except Exception:
				self.bold = None
		return self.bold

	def _pick(self, text, size, bold = False):
		"""(font, text, style): the first font that has every character, otherwise transliterate."""
		fonts = self._load_fonts()
		strong = pygame.freetype.STYLE_STRONG if bold else pygame.freetype.STYLE_DEFAULT
		if bold and self._bold_font() and None not in self.bold.get_metrics(text, size = size):
			return self.bold, get_display(arabic_reshaper.reshape(text)), pygame.freetype.STYLE_DEFAULT
		for font in fonts:
			if None not in font.get_metrics(text, size = size):
				font.strength = 0.04
				return font, get_display(arabic_reshaper.reshape(text)), strong
		return fonts[0], unidecode(text), strong

	def measure(self, text, size, bold = False):
		font, text, style = self._pick(text, size, bold)
		return font.get_rect(text, size = size, style = style).width

	def text(self, text, size, color, bold = False, max_w = None, min_size = None):
		"""One line of text; shrinks down to min_size, then cuts with '…' to fit max_w."""
		font, text, style = self._pick(text or '', size, bold)
		render = lambda t, sz: font.render(t, color, style = style, size = sz)[0]
		surf = render(text, size)
		if max_w:
			while surf.get_width() > max_w and min_size and size > min_size:
				size = max(min_size, size * 0.9)
				surf = render(text, size)
			while surf.get_width() > max_w and len(text) > 1:
				text = text[:max(1, int(len(text) * max_w / surf.get_width()) - 1)].rstrip() + '…'
				surf = render(text, size)
		return surf

	def title(self, text, size, color, max_w):
		"""A song title on up to two lines, breaking at 'Song - Artist' when it does not fit on one."""
		if self.measure(text, size, True) <= max_w:
			return [self.text(text, size, color, True)]
		parts = text.split(' - ', 1)
		if len(parts) == 2 and all(parts):
			return [self.text(p, size, color, True, max_w) for p in parts]
		return self.wrap(text, size, color, max_w, 2, True)

	def wrap(self, text, size, color, max_w, max_lines, bold = False):
		"""Word-wrapped lines (character-wrapped for text without spaces, like Chinese)."""
		sep = ' ' if ' ' in text.strip() else ''
		tokens = text.split(' ') if sep else list(text)
		lines, cur = [], ''
		for tok in tokens:
			cand = (cur + sep + tok) if cur else tok
			if cur and self.measure(cand, size, bold) > max_w:
				lines.append(cur)
				cur = tok
			else:
				cur = cand
		lines.append(cur)
		if len(lines) > max_lines:
			lines = lines[:max_lines - 1] + [sep.join(lines[max_lines - 1:])]
		return [self.text(L, size, color, bold, max_w) for L in lines]

	# ---------- shapes ----------
	def _rrect(self, surf, color, rect, radius, width = 0):
		if len(color) == 4:
			tmp = pygame.Surface(rect.size, pygame.SRCALPHA)
			pygame.draw.rect(tmp, color, tmp.get_rect(), width, border_radius = radius)
			surf.blit(tmp, rect.topleft)
		else:
			pygame.draw.rect(surf, color, rect, width, border_radius = radius)

	def _sign(self, surf, rect):
		"""The neon sign: pink border with a soft glow. The only glowing thing on screen."""
		s = self.s
		radius = int(28 * s)
		pad = int(40 * s)
		glow = pygame.Surface((rect.w + 2 * pad, rect.h + 2 * pad), pygame.SRCALPHA)
		pygame.draw.rect(glow, PINK + (140,), pygame.Rect(pad, pad, rect.w, rect.h), max(3, int(12 * s)), border_radius = radius)
		glow = _blur(_blur(glow, 6), 3)
		surf.blit(glow, (rect.x - pad, rect.y - pad))
		self._rrect(surf, BOOTH + (215,), rect, radius)
		pygame.draw.rect(surf, PINK, rect, max(2, int(3 * s)), border_radius = radius)

	def _mic(self, surf, x, y, h, color):
		"""Microphone icon with its top-left at (x, y); returns its width."""
		w = h * 0.55
		lw = max(2, int(h * 0.1))
		pygame.draw.rect(surf, color, pygame.Rect(x + w * 0.27, y, w * 0.46, h * 0.62), border_radius = int(w * 0.23))
		pygame.draw.arc(surf, color, pygame.Rect(x, y + h * 0.18, w, h * 0.62), math.pi, 2 * math.pi, lw)
		pygame.draw.line(surf, color, (x + w / 2, y + h * 0.8), (x + w / 2, y + h), lw)
		return w

	def _singer(self, surf, name, x, y, size, right = False, max_w = 260):
		"""Mic icon + singer name; (x, y) is the top-left, or the top-right when right=True."""
		label = self.text(name, size, CYAN, True, max_w = int(max_w * self.s))
		icon_h = label.get_height() * 0.78
		gap = size * 0.35
		total = icon_h * 0.55 + gap + label.get_width()
		if right:
			x -= total
		self._mic(surf, x, y + (label.get_height() - icon_h) / 2, icon_h, CYAN)
		surf.blit(label, (x + icon_h * 0.55 + gap, y))
		return label.get_height()

	def _image(self, path, size, round_ = False):
		key = (path, size, round_)
		if key not in self.images:
			img = pygame.image.load(path)
			if round_:
				img = pygame.transform.smoothscale(img.convert_alpha(), (size, size))
				mask = pygame.Surface((size, size), pygame.SRCALPHA)
				pygame.draw.circle(mask, (255, 255, 255, 255), (size // 2, size // 2), size // 2)
				img.blit(mask, (0, 0), special_flags = pygame.BLEND_RGBA_MIN)
			else:
				img = pygame.transform.scale(img, (size, size))  # nearest neighbour keeps QR modules sharp
			self.images[key] = img
		return self.images[key]

	def _background(self, W, H):
		small = pygame.Surface((max(1, W // 16), max(1, H // 16)))
		small.fill(ROOM)
		sw, sh = small.get_size()
		pygame.draw.ellipse(small, GLOW, pygame.Rect(sw * -0.05, sh * 0.02, sw * 0.7, sh * 0.76))
		small = _blur(small, 4)
		return pygame.transform.smoothscale(small, (W, H))

	# ---------- layout ----------
	def R(self, x, y, w, h):
		s = self.s
		return pygame.Rect(int(self.ox + x * s), int(self.oy + y * s), int(w * s), int(h * s))

	def P(self, x, y):
		return int(self.ox + x * self.s), int(self.oy + y * self.s)

	def draw(self, screen, st):
		W, H = screen.get_size()
		self.s = s = min(W / REF_W, H / REF_H)
		self.ox, self.oy = (W - REF_W * s) / 2, (H - REF_H * s) / 2

		key = (W, H, repr({k: v for k, v in st.items() if k != 'progress'}))
		if key != self.cache_key:
			frame = self._background(W, H)
			getattr(self, '_draw_' + st['mode'])(frame, st)
			self.cache, self.cache_key = frame, key
		screen.blit(self.cache, (0, 0))

		if st['mode'] == 'ready':  # the countdown bar moves every frame, so it is not cached
			bar = self.bar_rect
			pygame.draw.rect(screen, VELVET, bar, border_radius = bar.h // 2)
			fill = bar.copy()
			fill.w = max(fill.h, int(bar.w * st.get('progress', 0)))
			pygame.draw.rect(screen, PINK, fill, border_radius = bar.h // 2)

	def _brand(self, surf, st):
		s = self.s
		d = int(52 * s)
		x, y = self.P(56, 36)
		if st.get('logo'):
			try:
				surf.blit(self._image(st['logo'], d, True), (x, y))
			except Exception:
				pass
		name = self.text(st['app_name'], 24 * s, MUTED, True)
		surf.blit(name, (x + d + int(14 * s), y + (d - name.get_height()) // 2))

	def _qr_card(self, surf, st, heading):
		s = self.s
		card = self.R(884, 164, 340, 392)
		self._rrect(surf, BOOTH, card, int(24 * s))
		pygame.draw.rect(surf, VELVET, card, max(1, int(s)), border_radius = int(24 * s))
		code = self.R(929, 192, 250, 250)
		self._rrect(surf, WHITE, code, int(14 * s))
		q = int(218 * s)
		try:
			surf.blit(self._image(st['qr'], q), (code.centerx - q // 2, code.centery - q // 2))
		except Exception:
			pass
		h = self.text(heading, 30 * s, LYRIC, True, max_w = int(300 * s), min_size = 20 * s)
		surf.blit(h, (card.centerx - h.get_width() // 2, int(self.oy + 462 * s)))
		u = self.text(st['url'], 20 * s, CYAN, True, max_w = int(300 * s), min_size = 14 * s)
		surf.blit(u, (card.centerx - u.get_width() // 2, int(self.oy + 506 * s)))

	def _draw_empty(self, surf, st):
		s = self.s
		self._brand(surf, st)
		wide = not st.get('qr')
		inner = (1168 if wide else 760) - 96
		label = self.text(st['label'], 24 * s, MUTED, True)
		head = self.wrap(st['headline'], 68 * s, LYRIC, inner * s, 3, True)
		para = self.wrap(st['body'], 28 * s, MUTED, min(inner, 640) * s, 3)
		head_lh, para_lh = 74 * s, 38 * s
		h = 48 * s + label.get_height() + 10 * s + len(head) * head_lh + 22 * s + len(para) * para_lh + 44 * s
		top = (REF_H * s - h) / 2 / s
		sign = self.R(56, top, inner + 96, h / s)
		self._sign(surf, sign)
		x, y = sign.x + int(48 * s), sign.y + int(48 * s)
		surf.blit(label, (x, y)); y += label.get_height() + 10 * s
		for L in head:
			surf.blit(L, (x, y + (head_lh - L.get_height()) / 2)); y += head_lh
		y += 22 * s
		for L in para:
			surf.blit(L, (x, y)); y += para_lh
		if st.get('qr'):
			self._qr_card(surf, st, st['qr_heading'])
		foot = self.text(st['footer_left'], 20 * s, MUTED)
		surf.blit(foot, self.P(56, 676 - foot.get_height() / s))
		if st.get('footer_right'):
			fr = self.text(st['footer_right'], 18 * s, MUTED)
			x, y = self.P(1224, 676)
			surf.blit(fr, (x - fr.get_width(), y - fr.get_height()))

	def _draw_list(self, surf, st):
		s = self.s
		self._brand(surf, st)
		sign_w = 760 if st.get('qr') else 1168
		inner = sign_w - 80
		label = self.text(st['label'], 24 * s, MUTED, True)
		title = self.title(st['title'], 62 * s, LYRIC, inner * s)
		title_lh = 68 * s
		singer_h = 42 * s if st.get('singer') else 0
		h = 34 * s + label.get_height() + 8 * s + len(title) * title_lh + (14 * s + singer_h) + 34 * s
		sign = self.R(56, 150, sign_w, h / s)
		self._sign(surf, sign)
		x, y = sign.x + int(40 * s), sign.y + int(34 * s)
		surf.blit(label, (x, y)); y += label.get_height() + 8 * s
		for L in title:
			surf.blit(L, (x, y + (title_lh - L.get_height()) / 2)); y += title_lh
		if st.get('singer'):
			self._singer(surf, st['singer'], x, y + 14 * s, 34 * s, max_w = inner - 60)

		rows = st.get('rows') or []
		y = sign.bottom + 22 * s
		row_h = 58 * s
		if rows:
			lab = self.text(st['rows_label'], 24 * s, MUTED, True)
			surf.blit(lab, (x, y)); y += lab.get_height() + 6 * s
		for n, (song, user) in enumerate(rows):
			if y + row_h > self.oy + (REF_H - 30) * s:
				break
			pygame.draw.line(surf, VELVET, (sign.x, y), (sign.right, y), max(1, int(s)))
			num = self.text(str(n + st.get('rows_start', 1)), 24 * s, MUTED)
			surf.blit(num, (x, y + (row_h - num.get_height()) / 2))
			user_w = 0
			if user:
				lab = self.text(user, 24 * s, CYAN, True, max_w = int(260 * s))
				user_w = lab.get_width() + 30 * s
				self._singer(surf, user, sign.right - 40 * s, y + (row_h - lab.get_height()) / 2, 24 * s, right = True)
			t = self.text(song, 28 * s, LYRIC, True, max_w = int(sign.right - 40 * s - user_w - 20 * s - (x + 44 * s)))
			surf.blit(t, (x + 44 * s, y + (row_h - t.get_height()) / 2))
			y += row_h

		if st.get('qr'):
			self._qr_card(surf, st, st['qr_heading'])

	def _draw_ready(self, surf, st):
		s = self.s
		maxw = 1060 * s
		label = self.text(st['label'], 24 * s, MUTED, True)
		who = self.text(st['singer'], 88 * s, CYAN, True, max_w = maxw, min_size = 56 * s) if st.get('singer') else None
		title = self.title(st['title'], 46 * s, LYRIC, maxw)
		count = self.text(st['countdown'], 26 * s, MUTED)
		parts = [label] + ([who] if who else []) + title + [count]
		w = max(max(p.get_width() for p in parts) + 160 * s, 640 * s)
		gaps = 14 * s + (26 * s if who else 0) + 34 * s + 26 * s
		h = 46 * s * 2 + sum(p.get_height() for p in parts) + gaps + 8 * s
		sign = pygame.Rect(0, 0, int(w), int(h))
		sign.center = (surf.get_width() // 2, surf.get_height() // 2)
		self._sign(surf, sign)
		y = sign.y + 46 * s
		cx = sign.centerx
		surf.blit(label, (cx - label.get_width() / 2, y)); y += label.get_height() + 14 * s
		if who:
			surf.blit(who, (cx - who.get_width() / 2, y)); y += who.get_height() + 26 * s
		for L in title:
			surf.blit(L, (cx - L.get_width() / 2, y)); y += L.get_height()
		y += 34 * s
		surf.blit(count, (cx - count.get_width() / 2, y)); y += count.get_height() + 18 * s
		self.bar_rect = pygame.Rect(int(cx - 210 * s), int(y), int(420 * s), max(4, int(8 * s)))
