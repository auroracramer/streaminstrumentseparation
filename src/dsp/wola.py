import numpy as np

class WolaSTFT(object):
    """
    A class that implements STFT using WOLA for chunk by chunk processing.
    """



    def __init__(self, chunk_size=1024, window_size=256, overlap=128):
        assert overlap >= 0, "Cannot have negative overlap!"
        assert overlap < window_size, "Cannot have a larger overlap than " + \
                                      "the window size!"
        assert window_size < chunk_size, "Cannot have a larger window size " + \
                                         "than the chunk size!"
        assert (chunk_size-overlap) % (window_size-overlap) == 0, \
                "Window size and overlap values don't fit cleanly into chunk."

        self._chunk_size = chunk_size
        self._window_size = window_size
        self._overlap = overlap
        self._hopsize = window_size - overlap
        self._num_windows = (chunk_size-overlap)/self._hopsize
        self._rfft_size = window_size/2 + 1 if window_size%2==0 \
                          else (window_size+1)/2

        # Use the sqrt-Hamming window for the analysis and synthesis windows
        self._window = np.sqrt(np.hamming(window_size+1))[:-1]
        # Normalize window
        self._window = self._window/np.linalg.norm(self._window)

        # Set up magnitude correction
        self._window_corr = np.zeros(chunk_size)
        for i in xrange(self._num_windows):
            start = i*self._hopsize
            self._window_corr[start:start+window_size] += self._window**2

        self._nonzero_corr = self._window_corr != 0



    def correct_magnitude(self, x):
        """
        Corrects magnitude when taking inverse STFT.
        """
        x = x.copy()
        x[self._nonzero_corr] /= self._window_corr[self._nonzero_corr]
        return x



    def analysis(self, chunk):
        """
        Computes the STFT of the input, in a WOLA fashion.
        """
        assert chunk.ndim == 1 and chunk.size == self._chunk_size, \
                "Invalid chunk size!"


        stft = np.zeros((self._rfft_size, self._num_windows), dtype='cfloat')
        for i in xrange(self._num_windows):
            start = i*self._hopsize
            windowed = self._window * chunk[start:start+self._window_size]
            stft[:,i] = np.fft.rfft(windowed)

            if i == self._num_windows - 1:
                num_proc = start + self._window_size
                assert num_proc == self._chunk_size, \
                        "Number of processed samples does not add up to " + \
                        "chunk size. Got %d, but should be %d" \
                        % (num_proc,self._chunk_size)

        num_proc = (self._num_windows-1)*self._hopsize + self._window_size
        assert num_proc == self._chunk_size, "Number of processed samples " + \
                "does not add up to chunk size. Got %d, but should be %d" \
                % (num_proc,self._chunk_size)

        return stft



    def synthesis(self, stft):
        """
        Computes the inverse STFT, and performs WOLA to produce an output.
        """
        assert stft.shape == (self._rfft_size, self._num_windows), \
                "Invalid STFT size for this filter bank."

        self._output = np.zeros(self._chunk_size, dtype='float')
        i= np.zeros(self._chunk_size, dtype='float')

        for i in xrange(self._num_windows):
            start = i*self._hopsize
            windowed = np.real(self._window * np.fft.irfft(stft[:,i]))
            self._output[start:start+self._window_size] += windowed

        self._output = self.correct_magnitude(self._output)
        assert self._output.size == self._chunk_size

        return self._output
