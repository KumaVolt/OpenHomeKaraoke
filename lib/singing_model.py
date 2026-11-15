"""Phoneme recogniser trained on sung vocals, used to time the words of lyrics (see lib/word_align.py).

The network and its phoneme set come from Jiawen Huang and Emmanouil Benetos, "LyricsAlignment-Multilingual"
(https://github.com/jhuang448/LyricsAlignment-Multilingual), trained on separated vocals in English, French,
German, Italian and Spanish. Only what inference needs is kept here.

MIT License, Copyright (c) 2022 Jiawen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 22050
FPS = SR / 256 / 3           # output frames per second (~28.7)
PHONES = ['a', 'aɪ', 'aʊ', 'b', 'd', 'dʒ', 'e', 'ee', 'eɪ', 'eː', 'f', 'h', 'i', 'iː', 'j',
          'k', 'l', 'm', 'n', 'o', 'oʊ', 'oː', 'p', 'r', 's', 'ss', 't', 'ts', 'tʃ', 'tː',
          'u', 'uː', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ŋ', 'ɐ', 'ɑː', 'ɑːɹ', 'ɑ̃',
          'ɔ', 'ɔː', 'ɔ̃', 'ə', 'ɚ', 'ɛ', 'ɛ̃', 'ɜ', 'ɜː', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʁ',
          'ʃ', 'ʊ', 'ʊɹ', 'ʌ', 'ʒ', 'ʝ', 'β', 'θ', ' ']
PHONE_ID = {p: i for i, p in enumerate(PHONES)}
SPACE, UNKNOWN, BLANK = PHONE_ID[' '], 70, 71


class _CNNLayerNorm(nn.Module):
	def __init__(self, n_feats):
		super().__init__()
		self.layer_norm = nn.LayerNorm(n_feats)

	def forward(self, x):
		return self.layer_norm(x.transpose(2, 3).contiguous()).transpose(2, 3).contiguous()


class _ResidualCNN(nn.Module):
	def __init__(self, ch, kernel, dropout, n_feats):
		super().__init__()
		self.cnn1 = nn.Conv2d(ch, ch, kernel, 1, padding = kernel // 2)
		self.cnn2 = nn.Conv2d(ch, ch, kernel, 1, padding = kernel // 2)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.layer_norm1 = _CNNLayerNorm(n_feats)
		self.layer_norm2 = _CNNLayerNorm(n_feats)

	def forward(self, x):
		r = x
		x = self.cnn1(self.dropout1(F.gelu(self.layer_norm1(x))))
		x = self.cnn2(self.dropout2(F.gelu(self.layer_norm2(x))))
		return x + r


class _BiLSTM(nn.Module):
	def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
		super().__init__()
		self.BiLSTM = nn.LSTM(input_size = rnn_dim, hidden_size = hidden_size, num_layers = 1,
		                      batch_first = batch_first, bidirectional = True)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		return self.dropout(self.BiLSTM(x)[0])


class AcousticModel(nn.Module):
	def __init__(self, n_cnn_layers = 1, rnn_dim = 256, n_class = 72, n_feats = 32, dropout = 0.1):
		super().__init__()
		self.cnn_layers = nn.Sequential(nn.Conv2d(1, n_feats, 3, stride = 1, padding = 1), nn.ReLU())
		self.rescnn_layers = nn.Sequential(*[_ResidualCNN(n_feats, 3, dropout, 128) for _ in range(n_cnn_layers)])
		self.maxpooling = nn.MaxPool2d(kernel_size = (2, 3))
		self.fully_connected = nn.Linear(n_feats * 64, rnn_dim)
		self.bilstm = nn.Sequential(
			_BiLSTM(rnn_dim, rnn_dim, dropout, True),
			_BiLSTM(rnn_dim * 2, rnn_dim, dropout, False),
			_BiLSTM(rnn_dim * 2, rnn_dim, dropout, False))
		self.classifier = nn.Sequential(nn.Linear(rnn_dim * 2, n_class))

	def forward(self, x):
		x = self.maxpooling(self.rescnn_layers(self.cnn_layers(x)))
		s = x.size()
		x = self.fully_connected(x.view(s[0], s[1] * s[2], s[3]).transpose(1, 2))
		return self.classifier(self.bilstm(x))


def load(path):
	"""The model with the weights of the published checkpoint (read without running any pickled code)."""
	import numpy as np
	try:
		ck = torch.load(path, map_location = 'cpu', weights_only = True)
	except Exception:  # the checkpoint also stores a few numpy scalars next to the weights
		torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtype, type(np.dtype('float64'))])
		ck = torch.load(path, map_location = 'cpu', weights_only = True)
	m = AcousticModel()
	m.load_state_dict(ck['model_state_dict'])
	return m.eval()


def emissions(model, wav):
	"""Per-frame log-probabilities of each phoneme for a mono 22050 Hz float32 track."""
	import torchaudio
	mel = torchaudio.transforms.MelSpectrogram(sample_rate = SR, n_mels = 128, n_fft = 512)
	with torch.inference_mode():
		x = mel(torch.from_numpy(wav)[None])[None]      # (1, 1, mels, T)
		out = F.log_softmax(model(x), dim = 2)[0]
	return out[:int(len(wav) / SR * FPS)]
