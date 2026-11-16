import os, sys, io, math, random, time, json, datetime
import logging, socket, subprocess, threading
import multiprocessing as mp
import shutil, psutil, traceback, tarfile, requests
from subprocess import check_output
from collections import *

import numpy as np

from constants import media_types

import pygame
import qrcode
import arabic_reshaper
from bidi.algorithm import get_display
from unidecode import unidecode
from flask import request
from lib import omxclient, vlcclient
from lib.tvscreen import TVScreen
from lib import lyrics, pitch
from lib.get_platform import *
from lib.NLP import *
from app import getString

if get_platform() != "windows":
	from signal import SIGALRM, alarm, signal, SIGTERM
	signal(SIGTERM, lambda signum, stack_frame: os.K.stop())

STD_VOL = 65536/8/np.sqrt(2)
ip2websock, ip2pane = {}, {}

ws_send = lambda ip, msg: ip2websock[ip].send(msg) if ip in ip2websock else None

def flash(message: str, category: str = "message", client_ip = ''):
	ws_send(client_ip or request.remote_addr, f'showNotification("{message}", "{category}")')

def cleanse_modules(name):
	if name == 'yt_dlp':
		# yt-dlp wraps urllib3's _PERCENT_RE on import; re-importing it wraps the wrapper,
		# which breaks every `requests` call in this process (e.g. VLC status polling)
		try:
			import urllib3.util.url as u3url
			while type(u3url._PERCENT_RE).__name__ == 'Urllib3PercentREOverride':
				u3url._PERCENT_RE = u3url._PERCENT_RE.re
		except Exception:
			pass
	try:
		for module_name in sorted(sys.modules.keys()):
			if module_name.startswith(name):
				del sys.modules[module_name]
		del globals()[name]
	except:
		pass


def run_vocal_splitter(argv):
	"""The DNN vocal splitter, at the lowest CPU priority: on a small CPU it would otherwise take time from
	VLC and the TV stream and make the karaoke stutter. It only gets the CPU time playback leaves over."""
	try:
		os.nice(19)
	except (AttributeError, OSError):  # Windows has no nice
		pass
	import vocal_splitter
	vocal_splitter.main(argv)


class Karaoke:
	ref_W, ref_H = 1920, 1080      # reference screen size, control drawing scale

	queue = []
	queue_json = ''
	available_songs = []
	rename_history = {}
	songname_trans = {} # transliteration is used for sorting and initial letter search
	now_playing = None
	now_playing_filename = None
	now_playing_user = None
	now_playing_transpose = 0
	now_playing_slave = ''
	audio_delay = 0
	has_video = True
	has_subtitle = False
	lyrics_romanize = True    # show romaji/pinyin under Japanese/Chinese/Korean synced lyrics
	lyrics_pitch = False      # optional pitch guide lane; needs a vocals-only track from the DNN vocal splitter
	subtitle_delay = 0
	play_speed = 1.0
	show_subtitle = True
	last_vocal_info = 0
	last_vocal_time = 0
	use_DNN_vocal = True
	vocal_process = None
	vocal_wanted = False
	vocal_device = None
	vocal_mode = 'mixed'
	is_paused = True
	firstSongStarted = False
	switchingSong = False
	qr_code_path = None
	base_path = os.path.dirname(__file__)
	volume_offset = 0
	default_logo_path = os.path.join(base_path, "logo.jpg")
	logical_volume = None   # for normalized volume
	status_dirty = True
	event_dirty = threading.Event()

	def __init__(self, args):

		# override with supplied constructor args if provided
		self.__dict__.update(args.__dict__)
		self.omxplayer_adev = 'both'
		self.download_path = args.dl_path
		self.volume_offset = self.volume = args.volume
		self.logo_path = self.default_logo_path if args.logo_path == None else args.logo_path

		# other initializations
		self.platform = get_platform()
		self.vlcclient = None
		self.omxclient = None
		self.screen = None
		self.player_state = {}
		self.downloading_songs = {}
		self.log_level = int(args.log_level)

		logging.basicConfig(
			format = "[%(asctime)s] %(levelname)s: %(message)s",
			datefmt = "%Y-%m-%d %H:%M:%S",
			level = self.log_level,
		)

		logging.debug(vars(args))

		if self.save_delays:
			self.init_save_delays()

		# Generate connection URL and QR code, retry in case pi is still starting up
		# and doesn't have an IP yet (occurs when launched from /etc/rc.local)
		end_time = int(time.time()) + 30

		if self.platform == "raspberry_pi":
			while int(time.time()) < end_time:
				addresses_str = check_output(["hostname", "-I"]).strip().decode("utf-8")
				addresses = addresses_str.split(" ")
				self.ip = addresses[0]
				if not self.is_network_connected():
					logging.debug("Couldn't get IP, retrying....")
				else:
					break
		else:
			self.ip = self.get_ip()

		logging.debug("IP address (for QR code and splash screen): " + self.ip)

		self.url = "%s://%s:%s" % (('https' if self.ssl else 'http'), self.ip, self.port)
		# Behind a reverse proxy (e.g. https://karaoke.example.com) the URL shown in the
		# QR code and on screen differs from ip:port, so allow overriding it
		if os.environ.get('PUBLIC_URL'):
			self.url = os.environ['PUBLIC_URL'].rstrip('/')
		# screencapture.sh serves plain http on port 4000
		# keep a trailing slash: the stream page served under a path (e.g. /tv/) loads its video relative to it
		self.stream_url = os.environ.get('STREAM_URL', '') or "http://%s:4000" % self.ip

		# get songs from download_path
		self.get_available_songs()
		self.get_youtubedl_version()
		self.song2vol = Try(lambda: json.load(Open(self.download_path+'/.mp3_volume.json.gz')), {})
		
		# Automatically upgrade yt-dlp if using pip
		if not args.youtubedl_path:
			threading.Thread(target=self._upgrade_yt_dlp).start()

		# clean up old sessions
		self.kill_player()

		self.load_settings()
		self.generate_qr_code()
		if self.use_vlc:
			# the QR overlay is added per song (tv_qr_overlay), sized for each video, rather than by the client
			self.vlcclient = vlcclient.VLCClient(port = self.vlc_port, path = self.vlc_path, qrcode = None, url = self.url)
			lyrics.SIDE_MARGIN = 180 if self.show_overlay else 60  # keep sung lines clear of the QR code in the corner
		else:
			self.omxclient = omxclient.OMXClient(path = self.omxplayer_path, adev = self.omxplayer_adev,
			                                     dual_screen = self.dual_screen, volume_offset = self.volume_offset)

		if not self.hide_splash_screen:
			self.initialize_screen(not args.windowed)
			self.render_splash_screen()

		self.cloud = args.cloud
		if args.cloud:
			self.cloud_trigger = threading.Event()
			self.cloud_tasks = []
			threading.Thread(target=self._cloud_thread).start()

	def _upgrade_yt_dlp(self):
		import pip, yt_dlp
		fn = '.yt-dlp.last-update'
		date_today = datetime.datetime.today().isoformat()[:10]
		date_last = Try(lambda: open(fn).read().strip(), '')
		if date_today == date_last:
			logging.info(f"yt-dlp is up-to-date at {date_today}")
			return

		self.upgrade_youtubedl()
		self.get_youtubedl_version()
		with open(fn, 'w') as fp:
			print(date_today, file=fp)


	def _cloud_thread(self):
		while True:
			self.cloud_trigger.wait()
			self.cloud_trigger.clear()
			if not self.running: return
			while self.cloud_tasks:
				try:
					fn = self.cloud_tasks.pop(0)
					bn, dn = os.path.basename(fn), os.path.dirname(fn)
					if os.path.isfile(f'{self.download_path}nonvocal/{bn}.m4a') and os.path.isfile(f'{self.download_path}vocal/{bn}.m4a'):
						continue
					os.system(f'ffmpeg -y -i "{fn}" -vn -c copy {self.tmp_dir}/input.m4a')
					with open(f'{self.tmp_dir}/input.m4a', 'rb') as f:
						r = requests.post(self.cloud+'/split_vocal', files={'file': f})
					with open(f'{self.tmp_dir}/output.tar.gz', 'wb') as f:
						f.write(r.content)
					with tarfile.open(f'{self.tmp_dir}/output.tar.gz') as tar:
						tar.extract('nonvocal.m4a', f'{self.download_path}nonvocal')
						os.rename(f'{self.download_path}nonvocal/nonvocal.m4a', f'{self.download_path}nonvocal/{bn}.m4a')
						tar.extract('vocal.m4a', f'{self.download_path}vocal')
						os.rename(f'{self.download_path}vocal/vocal.m4a', f'{self.download_path}vocal/{bn}.m4a')
				except:
					traceback.print_exc()


	# Other ip-getting methods are unreliable and sometimes return 127.0.0.1
	# https://stackoverflow.com/a/28950776
	def get_ip(self):
		# Inside Docker the detected IP is the container's, so allow overriding it
		if os.environ.get('HOST_IP'):
			return os.environ['HOST_IP']
		s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		try:
			# doesn't even have to be reachable
			s.connect(("8.8.8.8", 1))
			IP = s.getsockname()[0]
		except Exception:
			IP = "127.0.0.1"
		finally:
			s.close()
		return IP

	def get_youtubedl_version(self):
		self.youtubedl_version = self.call_yt_dlp(['--version'], True).strip()
		return self.youtubedl_version

	def upgrade_youtubedl(self):
		logging.info("Upgrading youtube-dl, current version: %s" % self.youtubedl_version)
		if self.youtubedl_path:
			self.call_yt_dlp(['-U'])
		else:
			try:
				import pip
				pip.main(['install', 'yt-dlp', '-U'])
				cleanse_modules('yt_dlp')
				import yt_dlp
			except:
				pass
		logging.info("Done. New version: %s" % self.get_youtubedl_version())

	def is_network_connected(self):
		return not len(self.ip) < 7

	def tv_qr_overlay(self, file_path):
		"""VLC options that pin a "Scan to sing" QR code to the bottom right while a song plays, so people can
		add songs without waiting for the screen between songs. VLC draws it in the video's own pixels, so the
		card is made to measure for the video's height (and cached per height)."""
		if not self.show_overlay or self.hide_ip or not self.url:
			return []
		try:
			out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
			                               'stream=height', '-of', 'csv=p=0', file_path], timeout = 15, text = True)
			h = int(out.split()[0])
		except Exception:
			return []  # audio-only, or can't tell: no overlay
		path = os.path.join(self.base_path, 'qrcode_tv_%d.png' % h)
		if not os.path.isfile(path):
			try:
				from PIL import Image, ImageDraw, ImageFont
				qr = qrcode.QRCode(border = 2, error_correction = qrcode.constants.ERROR_CORRECT_M)
				qr.add_data(self.url)
				qr.make(fit = True)
				n = qr.modules_count + 4
				box = max(2, round(h * 0.17 / n))
				qr.box_size = box
				code = qr.make_image(fill_color = 'black', back_color = 'white').convert('RGBA')
				T = lambda ii: getString(ii) or os.langs['en_US'][ii]
				fs = max(10, round(h * 0.026))
				try:
					font = ImageFont.truetype(os.path.join(self.base_path, 'font', 'arial-unicode-ms.ttf'), fs)
				except OSError:
					font = ImageFont.load_default()
				caption = T(244)
				tw = ImageDraw.Draw(code).textlength(caption, font = font)
				pad, margin = round(h * 0.012), round(h * 0.03)
				cw = max(code.width, int(tw) + 2 * pad)
				ch = code.height + fs + 2 * pad
				img = Image.new('RGBA', (cw + margin, ch + margin), (0, 0, 0, 0))
				d = ImageDraw.Draw(img)
				d.rounded_rectangle((0, 0, cw - 1, ch - 1), radius = pad * 2, fill = (255, 255, 255, 255))
				img.paste(code, ((cw - code.width) // 2, 0))
				d.text((cw / 2, code.height + pad * 0.3), caption, font = font, fill = (40, 16, 48, 255), anchor = 'ma')
				img.save(path)
			except Exception:
				logging.warning('Could not make the TV QR code: ' + traceback.format_exc())
				return []
		return ['--sub-source', 'logo{file=%s,position=10,opacity=235}' % path]

	def generate_qr_code(self):
		logging.debug("Generating URL QR code")
		qr = qrcode.QRCode(version = 1, box_size = 1, border = 4, error_correction = qrcode.constants.ERROR_CORRECT_H)
		qr.add_data(self.url)
		qr.make()
		img = qr.make_image()
		self.qr_code_path = os.path.join(self.base_path, "qrcode.png")
		img.save(self.qr_code_path)

	def get_default_display_mode(self):
		if self.use_vlc:
			if self.platform == "raspberry_pi":
				# HACK apparently if display mode is fullscreen the vlc window will be at the bottom of pygame
				os.environ["SDL_VIDEO_CENTERED"] = "1"
				return pygame.NOFRAME
			else:
				return pygame.FULLSCREEN
		else:
			return pygame.FULLSCREEN

	def initialize_screen(self, fullscreen=True):
		if not self.hide_splash_screen:
			logging.debug("Initializing pygame")
			pygame.init()
			pygame.display.set_caption("pikaraoke")
			pygame.mouse.set_visible(0)
			self.fonts = {}
			self.WIDTH = pygame.display.Info().current_w
			self.HEIGHT = pygame.display.Info().current_h
			logging.debug("Initializing screen mode")

			if self.platform != "raspberry_pi":
				self.toggle_full_screen(fullscreen)
			else:
				# this section is an unbelievable nasty hack - for some reason Pygame
				# needs a keyboardinterrupt to initialise in some limited circumstances
				# source: https://stackoverflow.com/questions/17035699/pygame-requires-keyboard-interrupt-to-init-display
				class Alarm(Exception):
					pass

				def alarm_handler(signum, frame):
					raise Alarm

				signal(SIGALRM, alarm_handler)
				alarm(3)
				try:
					self.toggle_full_screen(fullscreen)
					alarm(0)
				except Alarm:
					raise KeyboardInterrupt
			logging.debug("Done initializing splash screen")

	def toggle_full_screen(self, fullscreen=None):
		if not self.hide_splash_screen:
			logging.debug("Toggling fullscreen...")
			self.full_screen = not self.full_screen if fullscreen is None else fullscreen
			if self.full_screen:
				self.screen = pygame.display.set_mode([self.WIDTH, self.HEIGHT], self.get_default_display_mode())
			else:
				self.screen = pygame.display.set_mode([self.WIDTH*3//4, self.HEIGHT*3//4], pygame.RESIZABLE)
			if self.is_file_playing():
				self.play_transposed(self.now_playing_transpose)
			else:
				self.render_splash_screen()

	def normalize(self, v):
		r = self.screen.get_width()/self.ref_W
		if type(v) is list:
			return [i*r for i in v]
		elif type(v) is tuple:
			return tuple(i * r for i in v)
		return v*r

	def render_splash_screen(self):
		if self.hide_splash_screen:
			return
		if not self.hide_ip and not self.is_network_connected():
			self.screen.fill((0, 0, 0))
			text = self.render_font(30, getString(48), (255, 255, 255))
			self.screen.blit(text[0], self.normalize((20, 20)))
			pygame.display.update()
			time.sleep(10)
			logging.info("No IP found. Network/Wifi configuration required. For wifi config, try: sudo raspi-config or the desktop GUI: startx")
			self.stop()
			return
		if not hasattr(self, 'tv'):
			self.tv = TVScreen()
		self.tv.draw(self.screen, self.tv_state())

	def tv_state(self):
		"""What the TV screen shows between songs (see lib/tvscreen.py)."""
		T = lambda ii: getString(ii) or os.langs['en_US'][ii]
		short = lambda url: url.split('://', 1)[-1]
		st = {'app_name': T(136), 'logo': self.logo_path,
		      'qr': None if self.hide_ip else self.qr_code_path, 'url': short(self.url)}
		remaining = getattr(self, 'ready_until', 0) - time.time()
		if remaining > 0 and self.queue:
			st.update(mode = 'ready', label = T(250), singer = self.queue[0]['user'], title = self.queue[0]['title'],
			          countdown = T(251) % math.ceil(remaining), progress = 1 - remaining / max(self.splash_delay, 0.1))
		elif self.now_playing and not self.has_video:  # audio-only song: nothing else draws on the screen
			st.update(mode = 'list', label = T(242), title = self.now_playing, singer = self.now_playing_user,
			          rows_label = T(240), rows = [(q['title'], q['user']) for q in self.queue[:3]], rows_start = 1, qr_heading = T(243))
		elif self.queue:
			st.update(mode = 'list', label = T(240), title = self.queue[0]['title'], singer = self.queue[0]['user'],
			          rows_label = T(241), rows = [(q['title'], q['user']) for q in self.queue[1:4]], rows_start = 2, qr_heading = T(243))
		else:
			st.update(mode = 'empty', label = T(245), headline = T(246) if st['qr'] else T(252), body = T(247),
			          qr_heading = T(244), footer_left = T(248) % len(self.available_songs),
			          footer_right = (T(249) + short(self.stream_url)) if self.streamer_alive() else '')
		return st

	def render_font(self, sizes, text, *kargs):
		if type(sizes) != list:
			sizes = [sizes]

		# normalize font size
		sizes = [s*self.screen.get_width()/self.ref_W for s in sizes]

		# initialize fonts if not found
		for size in sizes:
			if size not in self.fonts:
				self.fonts[size] = [pygame.freetype.SysFont(pygame.freetype.get_default_font(), size)] \
						+ [pygame.freetype.Font(f'font/{name}', size) for name in ['arial-unicode-ms.ttf', 'unifont.ttf']]

		# find a font that contains all characters of the song title, if cannot find, then display transliteration instead
		found = None
		for ii, font in enumerate(self.fonts[size]):
			if None not in font.get_metrics(text):
				found = ii
				break
		if found is None:
			text = unidecode(text)
			found = 0

		# reshape Arabic text
		text = get_display(arabic_reshaper.reshape(text))

		# draw the font, if too wide, half the string
		width = self.screen.get_width()
		for size in sorted(sizes, reverse = True):
			font = self.fonts[size][found]
			render = font.render(text, *kargs)
			# reduce font size if text too long
			if render[1].width > width and size != min(sizes):
				continue
			while render[1].width >= width:
				text = text[:int(len(text) * min(width / render[1].width, 0.618))] + '…'
				del render
				render = font.render(text, *kargs)
			break
		return render

	def call_yt_dlp(self, argv, get_stdout = False):
		if self.youtubedl_path:
			if get_stdout:
				return subprocess.check_output([self.youtubedl_path]+argv).decode("utf-8")
			else:
				return subprocess.call([self.youtubedl_path]+argv)
		ret_code = 0
		if get_stdout:
			old_stdout = sys.stdout
			sys.stdout = io.StringIO()
		try:
			import yt_dlp
			yt_dlp.main(argv)
		except SystemExit as e:
			ret_code = e.code
		if get_stdout:
			ret_stdout = sys.stdout
			sys.stdout = old_stdout
			return ret_stdout.getvalue()
		return ret_code

	def get_search_results(self, textToSearch):
		logging.info("Searching YouTube for: " + textToSearch)
		num_results = 10
		yt_search = 'ytsearch%d:%s' % (num_results, textToSearch)
		cmd = ["-j", "--no-playlist", "--flat-playlist", "--js-runtimes", "node", "--remote-components", "ejs:github"] + self.cookies_opt + [yt_search]
		logging.debug("Youtube-dl search command: " + " ".join(cmd))
		try:
			# output = subprocess.check_output(cmd).decode("utf-8")
			output = self.call_yt_dlp(cmd, True)
			logging.debug("Search results: " + output)
			rc = []
			for each in output.split("\n"):
				if len(each) > 2:
					j = json.loads(each)
					if (not "title" in j) or (not "url" in j):
						continue
					rc.append([j["title"], j["url"], j["id"], sec2hhmmss(j.get("duration", 0))])
			return rc
		except Exception as e:
			logging.debug("Error while executing search: " + str(e))
			raise e

	def get_yt_dlp_json(self, url):
		# out_json = subprocess.check_output([self.youtubedl_path, '-j', url])
		out_json = self.call_yt_dlp(['-j', url], True)
		return json.loads(out_json)

	def get_downloaded_file_basename(self, url):
		try:
			youtube_id = url.split("watch?v=")[1].split('&')[0]
		except:
			try:
				info_json = self.get_yt_dlp_json(url)
				youtube_id = info_json['id']
			except:
				logging.error("Error parsing video id from url: " + url)
				return None

		try:
			return [i for i in os.listdir(self.download_path+'tmp/') if youtube_id in i][0]
		except:
			pass

		filename = f"{info_json['title']}---{info_json['id']}.{info_json['ext']}"
		return filename if os.path.isfile(self.download_path+'tmp/'+filename) else None

	def download_video(self, client_lang='', client_ip='', song_url = '', enqueue = False, song_added_by = "Pikaraoke", include_subtitles = False, high_quality = False):
		logging.info("Downloading video: " + song_url)
		getString2 = lambda ii: os.langs.get(client_lang, os.langs['en_US'])[ii]
		self.downloading_songs[song_url] = 1
		dl_path = "%(title)s---%(id)s.%(ext)s"
		# Prefer H.264: YouTube's default AV1 or VP9 at 1080p60 needs more CPU to decode than small machines have, and the
		# TV stream is 720p anyway. VIDEO_FORMAT_SORT (yt-dlp -S) overrides the preference, e.g. "res:1080" on a fast machine.
		sort = os.environ.get('VIDEO_FORMAT_SORT', 'vcodec:h264,res:1080,fps,acodec:m4a' if high_quality else 'vcodec:h264,res:720,fps:30,acodec:m4a')
		opt_quality = ['-f', 'bv*[height<=1080]+ba[abr<=160]/b' if high_quality else 'bv*+ba/b', '-S', sort]
		opt_sub = ['--sub-langs', 'all', '--embed-subs'] if include_subtitles else []
		cmd = ['--fixup', 'force', '--socket-timeout', '3', '-R', 'infinite', '--remux-video', 'mp4', '--js-runtimes', 'node', '--remote-components', 'ejs:github'] + self.cookies_opt + opt_quality +\
			["-o", self.download_path+'tmp/'+dl_path] + opt_sub + [song_url]
		logging.info("Youtube-dl command: " + " ".join(cmd))
		rc = self.call_yt_dlp(cmd)
		if rc != 0:
			logging.error("Error code while downloading, retrying without format options ...")
			cmd = ['--fixup', 'force', '--socket-timeout', '3', '-R', 'infinite', '--remux-video', 'mp4', '--js-runtimes', 'node', '--remote-components', 'ejs:github'] + self.cookies_opt + \
				["-o", self.download_path + 'tmp/' + dl_path] + opt_sub + [song_url]
			logging.debug("Youtube-dl command: " + " ".join(cmd))
			rc = self.call_yt_dlp(cmd)
		if rc == 0:
			logging.debug("Song successfully downloaded: " + song_url)
			self.downloading_songs[song_url] = 0
			bn = self.get_downloaded_file_basename(song_url)
			if bn:
				shutil.move(self.download_path+'tmp/'+bn, self.download_path+bn)
				self.get_available_songs()
				if enqueue:
					self.enqueue(self.download_path+bn, song_added_by)
					self.downloading_songs[song_url] = '00'
					flash(getString2(189)+' '+getString2(191), client_ip = client_ip)
				else:
					flash(getString2(189), client_ip = client_ip)
			else:
				logging.error("Error queueing song: " + song_url)
				self.downloading_songs[song_url] = '01'
				flash(getString2(189)+' '+getString2(192), client_ip = client_ip)
		else:
			logging.error("Error downloading song: " + song_url)
			self.downloading_songs[song_url] = -1
			flash(getString2(190), client_ip = client_ip)
		return ws_send(client_ip, 'download_ended()')

	def get_available_songs(self):
		logging.info("Fetching available songs in: " + self.download_path)
		files_grabbed = []
		self.songname_trans = {}
		for bn in os.listdir(self.download_path):
			fn = self.download_path + bn
			if not bn.startswith('.') and os.path.isfile(fn):
				if os.path.splitext(fn)[1].lower() in media_types:
					files_grabbed.append(fn)
					trans = unidecode(self.filename_from_path(fn)).lower()
					# strip leading non-transliterable symbols
					while trans and not trans[0].islower() and not trans[0].isdigit():
						trans = trans[1:]
					self.songname_trans[fn] = trans

		# self.available_songs = sorted(files_grabbed, key = lambda f: str.lower(os.path.basename(f)))
		self.available_songs = sorted(self.songname_trans, key = self.songname_trans.get)

	def get_all_assoc_files(self, song_path):
		basename = os.path.basename(song_path)
		basestem = os.path.splitext(basename)
		return [self.download_path + basename,
				self.download_path + basestem[0] + '.cdg',
				self.download_path + 'nonvocal/' + basename + '.m4a',
				self.download_path + 'nonvocal/.' + basename + '.m4a',
				self.download_path + 'vocal/' + basename + '.m4a',
				self.download_path + 'vocal/.' + basename + '.m4a']

	def delete_if_exist(self, filename):
		if os.path.isfile(filename):
			try:
				os.remove(filename)
			except:
				pass

	def delete(self, song_path):
		logging.info("Deleting song: " + song_path)

		# delete all associated cdg/vocal/nonvocal files if exist
		for fn in self.get_all_assoc_files(song_path):
			self.delete_if_exist(fn)

		self.get_available_songs()

	def rename_if_exist(self, old_path, new_path):
		if os.path.isfile(old_path):
			try:
				shutil.move(old_path, new_path)
			except:
				pass

	def rename(self, song_path, new_basestem):
		logging.info("Renaming song: '" + song_path + "' to: " + new_basestem)
		ext = os.path.splitext(song_path)
		if len(ext) < 2:
			ext += ['']
		new_basename = new_basestem + ext[1]

		# can handle the case while the file is being processed by vocal splitter, it has been renamed multiple times
		old_basename = os.path.basename(song_path)
		self.rename_history[old_basename] = new_basename
		for k, v in self.rename_history.items():
			if v == old_basename:
				self.rename_history[k] = new_basename

		# rename all associated cdg/vocal/nonvocal files if exist
		for src, tgt in zip(self.get_all_assoc_files(song_path), self.get_all_assoc_files(new_basename)):
			self.rename_if_exist(src, tgt)

		# rename queue entry if inside queue
		for item in self.queue:
			if item['file'] == song_path:
				item['file'] = self.download_path + new_basename
				item['title'] = self.filename_from_path(item['file'])
				break

		self.get_available_songs()

	def filename_from_path(self, file_path):
		rc = os.path.basename(file_path)
		rc = os.path.splitext(rc)[0]
		rc = rc.split("---")[0]  # removes youtube id if present
		return rc

	def kill_player(self):
		if self.use_vlc:
			logging.debug("Killing old VLC processes")
			if self.vlcclient != None:
				self.vlcclient.kill()
		elif self.omxclient != None:
				self.omxclient.kill()

	def play_file(self, file_path, extra_params = []):
		self.switchingSong = True
		if self.use_vlc:
			if self.save_delays:
				saved_delays = self.delays.get(os.path.basename(file_path), {})
				self.audio_delay = self.audio_delay if self.audio_delay else saved_delays.get('audio_delay', 0)
				self.subtitle_delay = self.subtitle_delay if self.subtitle_delay else saved_delays.get('subtitle_delay', 0)
				self.show_subtitle = False if self.show_subtitle==False else saved_delays.get('show_subtitle', True)
			extra_params1 = []
			logging.info("Playing video in VLC: " + file_path)
			if self.platform != 'osx':
				extra_params1 += ['--drawable-hwnd' if self.platform == 'windows' else '--drawable-xid',
				                  hex(pygame.display.get_wm_info()['window'])]
			self.now_playing_slave = self.try_set_vocal_mode(self.vocal_mode, file_path)
			if os.path.isfile(self.now_playing_slave):
				extra_params1 += [f'--input-slave={self.now_playing_slave}', '--audio-track=1']
			if self.audio_delay:
				extra_params1 += [f'--audio-desync={self.audio_delay * 1000}']
			if self.subtitle_delay:
				extra_params1 += [f'--sub-delay={self.subtitle_delay * 10}']
			lyrics_file = self.lyrics_file(file_path)
			if self.show_subtitle:
				extra_params1 += [f'--sub-track=0']
				if lyrics_file:
					extra_params1 += [f'--sub-file={lyrics_file}']
			if self.play_speed != 1:
				extra_params1 += [f'--rate={self.play_speed}']
			extra_params1 += self.tv_qr_overlay(file_path)
			self.now_playing = self.filename_from_path(file_path)
			self.now_playing_filename = file_path
			self.is_paused = ('--start-paused' in extra_params1)
			if self.normalize_vol and self.logical_volume is not None:
				self.volume = self.logical_volume / np.sqrt(self.get_mp3_volume(file_path))
			if self.now_playing_transpose == 0:
				xml = self.vlcclient.play_file(file_path, self.volume, extra_params + extra_params1)
			else:
				xml = self.vlcclient.play_file_transpose(file_path, self.now_playing_transpose, self.volume, extra_params + extra_params1)
			if not xml:	# VLC could not play this file, the run loop moves on to the next song
				self.switchingSong = False
				return
			# VLC can list an added subtitle file a moment after the video; our own lyrics count even while switched off
			self.has_subtitle = "<info name='Type'>Subtitle</info>" in xml or bool(lyrics_file)
			self.has_video = "<info name='Type'>Video</info>" in xml
			self.volume = round(float(self.vlcclient.get_val_xml(xml, 'volume')))
			if self.normalize_vol:
				self.media_vol = self.get_mp3_volume(self.now_playing_filename)
				self.logical_volume = self.volume * np.sqrt(self.media_vol)
		else:
			logging.info("Playing video in omxplayer: " + file_path)
			self.omxclient.play_file(file_path)

		self.switchingSong = False
		self.status_dirty = True
		self.render_splash_screen()  # remove old previous track

	def play_transposed(self, semitones):
		if self.use_vlc:
			self.now_playing_transpose = semitones
			status_xml = self.vlcclient.command().text if self.is_paused else self.vlcclient.pause(False).text
			info = self.vlcclient.get_info_xml(status_xml)
			posi = info['position']*info['length']
			self.play_file(self.now_playing_filename, [f'--start-time={posi}'] + (['--start-paused'] if self.is_paused else []))
		else:
			logging.error("Not using VLC. Can't transpose track.")

	def is_file_playing(self):
		client = self.vlcclient if self.use_vlc else self.omxclient
		if client is not None and client.is_running():
			return True
		elif self.now_playing_filename:
			self.now_playing = self.now_playing_filename = None
		return False

	def is_song_in_queue(self, song_path):
		return song_path in map(lambda t: t['file'], self.queue)

	def enqueue(self, song_path, user = "Pikaraoke"):
		if (self.is_song_in_queue(song_path)):
			logging.warn("Song is already in queue, will not add: " + song_path)
			return False
		else:
			logging.info("'%s' is adding song to queue: %s" % (user, song_path))
			self.queue.append({"user": user, "file": song_path, "title": self.filename_from_path(song_path)})
			self.update_queue()
			return True

	def queue_add_random(self, amount):
		logging.info("Adding %d random songs to queue" % amount)
		songs = list(self.available_songs)  # make a copy
		if len(songs) == 0:
			logging.warn("No available songs!")
			return False
		i = 0
		while i < amount:
			r = random.randint(0, len(songs) - 1)
			if self.is_song_in_queue(songs[r]):
				logging.warn("Song already in queue, trying another... " + songs[r])
			else:
				self.queue.append({"user": "Random", "file": songs[r], "title": self.filename_from_path(songs[r])})
				i += 1
			songs.pop(r)
			if len(songs) == 0:
				self.update_queue()
				logging.warn("Ran out of songs!")
				return False
		self.update_queue()
		return True

	def update_queue(self):
		self.queue_json = json.dumps(self.queue)
		self.status_dirty = True

	def queue_clear(self):
		logging.info("Clearing queue!")
		self.queue = []
		self.update_queue()
		self.skip()

	def queue_edit(self, song_file, action, **kwargs):
		if action == "move":
			try:
				src, tgt, size = [int(kwargs[n]) for n in ['src', 'tgt', 'size']]
				if size > len(self.queue):
					# new songs have started while dragging the list
					diff = size - len(self.queue)
					src -= diff
					tgt -= diff
				song = self.queue.pop(src)
				self.queue.insert(tgt, song)
			except:
				logging.error("Invalid move song request: " + str(kwargs))
				return False
		else:
			match = [(ii,each) for ii,each in enumerate(self.queue) if song_file in each["file"]]
			index, song = match[0] if match else (-1, None)
			if song == None:
				logging.error("Song not found in queue: " + song["file"])
				return False
			if action == "up":
				if index < 1:
					logging.warn("Song is up next, can't bump up in queue: " + song["file"])
					return False
				else:
					logging.info("Bumping song up in queue: " + song["file"])
					del self.queue[index]
					self.queue.insert(index - 1, song)
			elif action == "down":
				if index == len(self.queue) - 1:
					logging.warn("Song is already last, can't bump down in queue: " + song["file"])
					return False
				else:
					logging.info("Bumping song down in queue: " + song["file"])
					del self.queue[index]
					self.queue.insert(index + 1, song)
			elif action == "delete":
				logging.info("Deleting song from queue: " + song["file"])
				del self.queue[index]
			else:
				logging.error("Unrecognized direction: " + action)
				return False
		self.update_queue()
		return True

	def skip(self):
		if self.is_file_playing():
			logging.info("Skipping: " + self.now_playing)
			if self.use_vlc:
				self.vlcclient.stop()
			else:
				self.omxclient.stop()
			self.reset_now_playing()
			return True
		logging.warning("Tried to skip, but no file is playing!")
		return False

	def seek(self, seek_sec):
		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.seek(seek_sec)
			else:
				logging.warning("OMXplayer cannot seek track!")
			return True
		logging.warning("Tried to seek, but no file is playing!")
		return False

	def set_delays_dict(self, filename, key, val, dft_val=0):
		basename = os.path.basename(filename)
		delays = self.delays.get(basename, {})
		if val == dft_val:
			delays.pop(key, None)
		else:
			delays[key] = val
		if delays:
			self.delays[basename] = delays
		else:
			self.delays.pop(basename, {})
		self.delays_dirty = True

	def set_audio_delay(self, delay):
		if delay == '+':
			self.audio_delay += 0.1
		elif delay == '-':
			self.audio_delay -= 0.1
		elif delay == '':
			self.audio_delay = 0
		else:
			try:
				self.audio_delay = float(delay)
			except:
				logging.warning(f"Tried to set audio delay to an invalid value {delay}, ignored!")
				return False

		if self.save_delays:
			self.set_delays_dict(self.now_playing_filename, 'audio_delay', self.audio_delay)

		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.command(f"audiodelay&val={self.audio_delay}")
			else:
				logging.warning("OMXplayer cannot set audio delay!")
			self.status_dirty = True
			return self.audio_delay
		logging.warning("Tried to set audio delay, but no file is playing!")
		return False

	def set_subtitle_delay(self, delay):
		if delay == '+':
			self.subtitle_delay += 0.1
		elif delay == '-':
			self.subtitle_delay -= 0.1
		elif delay == '':
			self.subtitle_delay = 0
		else:
			try:
				self.subtitle_delay = float(delay)
			except:
				logging.warning(f"Tried to set subtitle delay to an invalid value {delay}, ignored!")
				return False

		if self.save_delays:
			self.set_delays_dict(self.now_playing_filename, 'subtitle_delay', self.subtitle_delay)

		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.command(f"subdelay&val={self.subtitle_delay}")
			else:
				logging.warning("OMXplayer cannot set subtitle delay!")
			self.status_dirty = True
			return self.subtitle_delay
		logging.warning("Tried to set subtitle delay, but no file is playing!")
		return False

	def toggle_subtitle(self):
		self.show_subtitle = not self.show_subtitle
		if self.save_delays:
			self.set_delays_dict(self.now_playing_filename, 'show_subtitle', self.show_subtitle, True)
		self.play_vocal(force=True)

	def pause(self):
		if self.is_file_playing():
			logging.info("Toggling pause: " + self.now_playing)
			if self.use_vlc:
				if self.vlcclient.is_playing():
					self.vlcclient.pause()
					self.is_paused = True
				else:
					self.vlcclient.play()
					self.is_paused = False
			else:
				if self.omxclient.is_playing():
					self.omxclient.pause()
					self.is_paused = True
				else:
					self.omxclient.play()
					self.is_paused = False
			self.status_dirty = True
			return True
		else:
			logging.warning("Tried to pause, but no file is playing!")
			return False

	def vol_up(self):
		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.vol_up()
				xml = self.vlcclient.command().text
				self.volume = int(self.vlcclient.get_val_xml(xml, 'volume'))
			else:
				self.volume = self.omxclient.vol_up()
			self.update_logical_vol()
			return self.volume
		else:
			logging.warning("Tried to volume up, but no file is playing!")
			return False

	def vol_down(self):
		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.vol_down()
				xml = self.vlcclient.command().text
				self.volume = int(self.vlcclient.get_val_xml(xml, 'volume'))
			else:
				self.volume = self.omxclient.vol_down()
			self.update_logical_vol()
			return self.volume
		else:
			logging.warning("Tried to volume down, but no file is playing!")
			return False

	def vol_set(self, volume):
		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.vol_set(volume)
				xml = self.vlcclient.command().text
				self.volume = int(self.vlcclient.get_val_xml(xml, 'volume'))
			else:
				logging.warning("Only VLC player can set volume, ignored!")
				self.volume = self.omxclient.volume_offset
			self.update_logical_vol()
			return self.volume
		else:
			logging.warning("Tried to set volume, but no file is playing!")
			return False

	def play_speed_set(self, speed):
		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.playspeed_set(speed)
				xml = self.vlcclient.command().text
				self.play_speed = float(self.vlcclient.get_val_xml(xml, 'rate'))
				logging.info(f"Playback speed set to {self.play_speed}")
			else:
				logging.warning("Only VLC player can set playback speed, ignored!")
			return self.play_speed
		else:
			logging.warning("Tried to set play speed, but no file is playing!")
			return False

	def try_set_vocal_mode(self, mode, now_playing_filename):
		if mode not in ['mixed', 'vocal', 'nonvocal']:
			mode = {1: 'nonvocal', 2: 'mixed', 3: 'vocal'}[self.get_vocal_mode()]
		play_slave = '' if mode == 'mixed' else self.download_path + mode + '/' + ('' if self.use_DNN_vocal else '.') \
		                                       + os.path.basename(now_playing_filename) + '.m4a'
		if os.path.isfile(play_slave):
			self.vocal_mode = mode
		else:
			play_slave = ''
			self.vocal_mode = 'mixed'
		return play_slave

	def play_vocal(self, mode = None, force = False):
		# mode=vocal/nonvocal/mixed, or else (use current)
		if self.use_vlc:
			play_slave = self.try_set_vocal_mode(mode, self.now_playing_filename)
			if not force and self.now_playing_slave == play_slave:
				return
			status_xml = self.vlcclient.command().text if self.is_paused else self.vlcclient.pause(False).text
			info = self.vlcclient.get_info_xml(status_xml)
			posi = info['position']*info['length']
			self.play_file(self.now_playing_filename, [f'--start-time={posi}'] + (['--start-paused'] if self.is_paused else []))
			self.get_vocal_info(True)
		else:
			logging.error("Not using VLC. Can't play vocal/nonvocal.")

	def get_vocal_mode(self):
		if '/nonvocal/' in self.now_playing_slave.replace('\\', '/'):
			return 1
		elif '/vocal/' in self.now_playing_slave.replace('\\', '/'):
			return 3
		return 2

	def get_vocal_info(self, force_update=False):
		tm = time.time()
		if not force_update and tm-self.last_vocal_time < 2:
			return self.last_vocal_info
		if not self.now_playing_filename:
			return 0
		mask = 0
		bn = os.path.basename(self.now_playing_filename)
		if os.path.isfile(f'{self.download_path}nonvocal/{bn}.m4a'):
			mask |= 0b00000001
		if os.path.isfile(f'{self.download_path}vocal/{bn}.m4a'):
			mask |= 0b00000010
		if os.path.isfile(f'{self.download_path}nonvocal/.{bn}.m4a'):
			mask |= 0b00000100
		if os.path.isfile(f'{self.download_path}vocal/.{bn}.m4a'):
			mask |= 0b00001000
		if 'vocal/.' in self.now_playing_slave:
			mask |= 0b10000000
		if self.use_DNN_vocal:
			mask |= 0b01000000
		mask |= (self.get_vocal_mode() << 4)
		self.last_vocal_info = mask
		self.last_vocal_time = tm
		return mask

	def get_state(self):
		if self.use_vlc and self.vlcclient.is_transposing:
			return defaultdict(lambda: None, self.player_state)
		if not self.is_file_playing():
			self.player_state['now_playing'] = None
			return defaultdict(lambda: None, self.player_state)
		new_state = self.vlcclient.get_info_xml() if self.use_vlc else {
			'volume': self.omxclient.volume_offset,
			'state': ('paused' if self.omxclient.paused else 'playing')
		}
		self.player_state.update(new_state)
		return defaultdict(lambda: None, self.player_state)

	def restart(self):
		if self.is_file_playing():
			if self.use_vlc:
				self.vlcclient.restart()
			else:
				self.omxclient.restart()
			self.is_paused = False
			return True
		else:
			logging.warning("Tried to restart, but no file is playing!")
			return False

	def stop(self):
		self.running = False

	def handle_run_loop(self):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				logging.warn("Window closed: Exiting pikaraoke...")
				self.running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_ESCAPE:
					logging.warn("ESC pressed: Exiting pikaraoke...")
					self.running = False
				if event.key == pygame.K_f:
					self.toggle_full_screen()
		if not self.is_file_playing() or not self.has_video:
			self.render_splash_screen()
			pygame.display.update()
		pygame.time.wait(100)

	# Use this to reset the screen in case it loses focus
	# This seems to occur in windows after playing a video
	def pygame_reset_screen(self):
		if not self.hide_splash_screen:
			logging.debug("Resetting pygame screen...")
			pygame.display.quit()
			self.initialize_screen()
			self.render_splash_screen()

	def reset_now_playing(self):
		self.auto_save_delays()
		self.now_playing = None
		self.now_playing_filename = None
		self.now_playing_user = None
		self.is_paused = True
		self.now_playing_transpose = 0
		self.now_playing_slave = ''
		self.audio_delay = 0
		self.subtitle_delay = 0
		self.show_subtitle = True
		self.has_subtitle = False
		self.has_video = True
		self.last_vocal_info = 0
		self.play_speed = 1

	def lyrics_cache_dir(self):
		return os.path.join(self.download_path, '.lyrics')

	def lyrics_file(self, song_path):
		"""Subtitle file with the cached synced lyrics for a song, if any (see lib/lyrics.py).
		Japanese, Chinese and Korean lyrics get a romanized line under them unless it is switched off,
		and the pitch guide lane is added when it is switched on and the song has a vocals-only track."""
		notes = self.pitch_notes(song_path) if self.lyrics_pitch else None
		if getattr(self, 'no_lyrics', False) and not notes:
			return None
		try:
			if notes and not lyrics.load(song_path, self.lyrics_cache_dir()) and lyrics.media_info(song_path)[1]:
				notes = None  # the file has its own subtitles; keep them rather than replacing them with the lane
			return lyrics.subtitle_file(song_path, self.lyrics_cache_dir(), self.lyrics_romanize, notes)
		except Exception:
			logging.warning('Could not build the lyrics subtitle: ' + traceback.format_exc())
			return None

	def vocal_track(self, song_path):
		"""The DNN vocal splitter's vocals-only track for a song (it may not exist yet)."""
		return self.download_path + 'vocal/' + os.path.basename(song_path) + '.m4a'

	def has_pitch_source(self):
		return bool(self.now_playing_filename) and os.path.isfile(self.vocal_track(self.now_playing_filename))

	def pitch_notes(self, song_path):
		try:
			return pitch.load(song_path, self.lyrics_cache_dir(), self.vocal_track(song_path))
		except Exception:
			logging.warning('Pitch guide failed: ' + traceback.format_exc())
			return None

	def settings_file(self):
		return os.path.join(self.download_path, '.settings.json')

	def load_settings(self):
		"""Choices made in the Settings page that should survive a restart (kept with the songs)."""
		try:
			with open(self.settings_file()) as fp:
				saved = json.load(fp)
		except (OSError, ValueError):
			return
		if isinstance(saved.get('show_qr'), bool):
			self.show_overlay = saved['show_qr']

	def save_setting(self, key, value):
		try:
			with open(self.settings_file()) as fp:
				saved = json.load(fp)
		except (OSError, ValueError):
			saved = {}
		saved[key] = value
		try:
			with open(self.settings_file(), 'w') as fp:
				json.dump(saved, fp)
		except OSError:
			logging.warning('Could not save the setting %s' % key)

	def set_qr(self, on):
		"""Show or hide the "Scan to sing" QR code that is pinned to the corner while songs play."""
		self.save_setting('show_qr', on)
		if on == self.show_overlay:
			return
		self.show_overlay = on
		lyrics.SIDE_MARGIN = 180 if on else 60
		if self.is_file_playing():
			self.play_vocal(force = True)  # restarts VLC at the same position, with or without the overlay

	def toggle_pitch(self):
		self.lyrics_pitch = not self.lyrics_pitch
		if self.is_file_playing() and self.show_subtitle and self.has_pitch_source():
			self.play_vocal(force = True)  # restarts VLC at the same position with the other subtitle file

	def lyrics_language(self):
		"""'ja', 'zh' or 'ko' when the playing song's lyrics can be romanized."""
		if not self.now_playing_filename or getattr(self, 'no_lyrics', False):
			return None
		try:
			return lyrics.language(self.now_playing_filename, self.lyrics_cache_dir())
		except Exception:
			return None

	def toggle_romanize(self):
		self.lyrics_romanize = not self.lyrics_romanize
		if self.is_file_playing() and self.show_subtitle and self.lyrics_language():
			self.play_vocal(force = True)  # restarts VLC at the same position with the other subtitle file

	def fetch_lyrics(self, song_path):
		"""Look up lyrics in the background while the "Get ready" card counts down."""
		if getattr(self, 'no_lyrics', False):
			return None
		def job():
			try:
				lyrics.prepare(song_path, self.lyrics_cache_dir(), self.filename_from_path(song_path),
				               vocal_track = self.vocal_track(song_path))
				if self.lyrics_pitch:
					self.pitch_notes(song_path)  # work out the notes now rather than when the song starts
			except Exception:
				logging.warning('Lyrics lookup crashed: ' + traceback.format_exc())
		th = threading.Thread(target = job, daemon = True)
		th.start()
		return th

	def streamer_alive(self):
		for p in psutil.process_iter():
			try:
				if './screencapture.sh' in p.cmdline():
					return True
			except psutil.Error:  # zombies left by a streamer restart, or processes that just exited
				pass
		return False

	def streamer_restart(self, delay=0):
		if self.platform in ['windows', 'osx']:
			return
		os.system(f"sleep {delay} && tmux send-keys -t PiKaraoke:0.3 C-c && tmux send-keys -t PiKaraoke:0.3 Up Enter")

	def streamer_stop(self, delay=0):
		if self.platform in ['windows', 'osx']:
			return
		os.system(f"sleep {delay} && tmux send-keys -t PiKaraoke:0.3 C-c")

	def vocal_alive(self):
		try:
			return bool(self.vocal_process and self.vocal_process.is_alive())\
					or bool([1 for p in psutil.process_iter() if 'vocal_splitter.py' in p.cmdline()])
		except:
			return None

	def vocal_watchdog(self):
		"""Start the vocal splitter again if it died (for example killed when memory ran out), at most every 2 minutes.
		Without it, songs downloaded after a crash are never split and their words never timed."""
		p = self.vocal_process
		if p is None or p.is_alive() or not self.vocal_wanted or time.time() < getattr(self, 'vocal_retry_at', 0):
			return
		self.vocal_retry_at = time.time() + 120
		logging.warning('Vocal splitter stopped (exit code %s); starting it again' % p.exitcode)
		Try(lambda: self.vocal_restart())

	def vocal_restart(self):
		self.vocal_wanted = True
		if self.platform == 'windows' or self.run_vocal:
			import vocal_splitter
			if self.vocal_process is not None and self.vocal_process.is_alive():
				self.vocal_process.kill()
			if shutil.which('ffmpeg'):
				self.vocal_process = mp.Process(target=run_vocal_splitter, args=(['-p', '-d', self.download_path],))
				self.vocal_process.start()
		else:
			os.system(f"tmux send-keys -t PiKaraoke:0.4 C-c && tmux send-keys -t PiKaraoke:0.4 Up Enter")

	def vocal_stop(self):
		self.vocal_wanted = False
		if self.vocal_process is not None and self.vocal_process.is_alive():
			self.vocal_process.kill()
		elif self.platform != 'windows':
			os.system(f"tmux send-keys -t PiKaraoke:0.4 C-c")

	def get_mp3_volume(self, filename):
		try:
			basename, md5, fsize = os.path.basename(filename), md5sum(filename), os.stat(filename).st_size
			vol_fsize_md5 = self.song2vol.get(basename, [0]*3)
			if fsize == vol_fsize_md5[1] and md5 == vol_fsize_md5[2]:
				return vol_fsize_md5[0]
			pcm_data = subprocess.check_output(['ffmpeg', '-i', filename, '-vn', '-f', 's16le', '-acodec', 'pcm_s16le', '-'], stderr = subprocess.DEVNULL)
			volume_val = np.clip(np.sqrt(np.std(np.frombuffer(pcm_data, dtype = np.int16))/STD_VOL), 1/16, 16)
			self.song2vol[basename] = [volume_val, fsize, md5]
			with Open(self.download_path+'/.mp3_volume.json.gz', 'wb') as fp:
				json.dump(self.song2vol, fp, indent=1)
			return volume_val
		except:
			self.normalize_vol = False
			return 1

	def update_logical_vol(self):
		if hasattr(self, 'media_vol'):
			self.logical_volume = self.volume * self.media_vol

	def enable_vol_norm(self, enable):
		self.normalize_vol = enable
		if enable and shutil.which('ffmpeg') is None:
			self.normalize_vol = enable = False
		if enable and self.now_playing_filename:
			self.volume = self.vlcclient.get_info_xml()['volume']
			self.media_vol = self.get_mp3_volume(self.now_playing_filename)
			self.update_logical_vol()
		return str(self.logical_volume)

	def init_save_delays(self):
		self.delays_dirty = False
		try:
			self.delays = eval(open(self.save_delays).read())
		except:
			self.delays = {}
			with open(self.save_delays, 'w') as fp:
				fp.write(str(self.delays))

	def set_save_delays(self, state):
		if state != bool(self.save_delays):
			if state:
				self.save_delays = self.dft_delays_file
				self.init_save_delays()
			else:
				self.save_delays = None
				self.delete_if_exist(self.dft_delays_file)

	def auto_save_delays(self):
		if self.save_delays and self.delays_dirty:
			self.delays_dirty = False
			with open(self.save_delays, 'w') as fp:
				fp.write(str(self.delays))

	def run(self):
		logging.info("Starting PiKaraoke!")
		self.running = True

		# Windows does not have tmux, vocal splitter can only be invoked from the main program
		if self.platform == 'windows' or self.run_vocal:
			Try(lambda: self.vocal_restart())

		while self.running:
			try:
				if not self.is_file_playing() and self.now_playing != None:
					self.reset_now_playing()
				if self.queue:
					if not self.is_file_playing():
						self.reset_now_playing()
						self.ready_until = time.time() + self.splash_delay  # shows the "Get ready" card
						lyrics_job = self.fetch_lyrics(self.queue[0]['file'])
						self.render_splash_screen()
						while time.time() < self.ready_until or (lyrics_job and lyrics_job.is_alive() and time.time() < self.ready_until + 3):
							self.handle_run_loop()
						self.ready_until = 0
						head = self.queue.pop(0)
						self.play_file(head['file'])
						if self.cloud:
							self.cloud_tasks += [head['file']]
							self.cloud_trigger.set()
						if not self.firstSongStarted:
							if self.streamer_alive():
								self.streamer_restart(1)
							self.firstSongStarted = True
						self.now_playing_user = head["user"]
						self.update_queue()
				self.handle_run_loop()
				if time.time() >= getattr(self, 'vocal_check_at', 0):
					self.vocal_check_at = time.time() + 15
					self.vocal_watchdog()
			except KeyboardInterrupt:
				logging.warn("Keyboard interrupt: Exiting pikaraoke...")
				self.running = False

		# Clean up before quit
		self.streamer_stop()
		self.vocal_stop()
		vplayer = self.vlcclient if self.use_vlc else self.omxclient
		if vplayer is not None: vplayer.stop()
		self.auto_save_delays()
		time.sleep(1)
		if vplayer is not None: vplayer.kill()
