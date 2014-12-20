#!/usr/bin/env python2
import numpy as np
import numpy.linalg as npla
import argparse
import scikits.audiolab as skal

EPSILON = np.finfo(float).eps

def calculate_SNR(source, approx):
    return 10*np.log10(npla.norm(source)/(npla.norm(source - approx) + EPSILON))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculates the SNR between the original audio and the reconstructed audio.')
    parser.add_argument('original')
    parser.add_argument('reconstructed')
    args = parser.parse_args()

    original, fs_o, enc_o = skal.wavread(args.original)
    recon, fs_r, enc_r = skal.wavread(args.reconstructed)

    assert fs_o == fs_r

    print "SNR: %f" % calculate_SNR(original, recon)


