import numpy as np

# Credit due to:
# http://scimusing.wordpress.com/2013/10/25/ring-buffers-in-pythonnumpy/

class RingBuffer():
    """
    A implementation of a ring buffer. A buffer that uses a pointer to
    keep track of the beginning of the buffer, in order to avoid allocating
    new memory when pushing elements out of the buffer.

    This class uses numpy arrays.
    """

    def __init__(self, length):
        self._length = length
        self._data = np.zeros(length, dtype=float)
        self._index = 0



    @property
    def length(self):
        """
        Returns the length of the buffer.
        """
        return self._length



    def push(self, x):
        """
        Pushes the elements in the input array x into the buffer.
        """
        assert x.ndim == 1, "Input array must be a 1-dimensional array!"
        assert x.size <= self._length, "Input array is large than the buffer!"
        assert x.size == 0, "Input cannot be empty!"

        x_indices = (self._index + np.arange(x.size)) % self._length
        self._data[x_indices] = x
        self._index = x_indices[-1] + 1



    @property
    def data(self):
        """
        Returns a copy of the data in the buffer.
        """
        indices = (self._index + np.arange(self._length)) % self._length
        return self._data[indices]



    def advance_ptr(self, adv):
        """
        Advances the pointer forward by adv.
        """
        assert adv < self._length, "Advancing the pointer an amount as long"  \
                                 + " or longer than the buffer length is not" \
                                 + " allowed."
        self._index = (self._index + adv) % self._length
