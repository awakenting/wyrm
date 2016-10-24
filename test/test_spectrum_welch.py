from __future__ import division

import unittest

import numpy as np

from wyrm.types import Data
from wyrm.processing import spectrum_welch
from wyrm.processing import swapaxes


class TestSpectrumWelch(unittest.TestCase):

    def setUp(self):
        # create some data
        fs = 100
        dt = 5
        self.freqs = [2, 7, 15]
        self.amps = [30, 10, 2]
        t = np.linspace(0, dt, fs*dt)
        data = np.sum([a * np.sin(2*np.pi*t*f) for a, f in zip(self.amps, self.freqs)], axis=0)
        data = data[:, np.newaxis]
        data = np.concatenate([data, data], axis=1)
        channel = np.array(['ch1', 'ch2'])
        self.dat = Data(data, [t, channel], ['time', 'channel'], ['s', '#'])
        self.dat.fs = fs

    def test_spectrum_welch(self):
        """Calculate the spectrum."""
        nperseg = 100
        dat = spectrum_welch(self.dat, nperseg=nperseg)
        # check that the amplitudes are almost correct
        for idx, freq in enumerate(self.freqs):
            for chan in range(dat.data.shape[1]):
                self.assertAlmostEqual(dat.data[np.argmin(np.abs(dat.axes[0]-freq)), chan], self.amps[idx]**2/2, delta=.15)
        # check that the max freq is <= self.dat.fs / 2, and min freq >= 0
        self.assertGreaterEqual(min(dat.axes[0]), 0)
        self.assertLessEqual(max(dat.axes[0]), self.dat.fs / 2)

    def test_spectrum_has_no_fs(self):
        """A spectrum has no sampling freq."""
        dat = spectrum_welch(self.dat)
        self.assertFalse(hasattr(dat, 'fs'))

    def test_spectrum_copy(self):
        """spectrum must not modify argument."""
        cpy = self.dat.copy()
        spectrum_welch(self.dat)
        self.assertEqual(cpy, self.dat)

    def test_spectrum_swapaxes(self):
        """spectrum must work with nonstandard timeaxis."""
        dat = spectrum_welch(swapaxes(self.dat, 0, 1), timeaxis=1)
        dat = swapaxes(dat, 0, 1)
        dat2 = spectrum_welch(self.dat)
        self.assertEqual(dat, dat2)


if __name__ == '__main__':
    unittest.main()


