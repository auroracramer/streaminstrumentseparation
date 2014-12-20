#!/usr/bin/env python2

import sys, os, os.path
# UGLY HACK

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = "%s\\..\\..\\samples" % SCRIPT_PATH
MODEL_PATH = "%s\\..\\..\\model\\instrument_templates_plca_singles_club.p" % SCRIPT_PATH

#sys.path.insert(0, '%s\\..\\..\\thirdparty\\plca' % SCRIPT_PATH)

import re
import cPickle
import sklearn.decomposition as skde
import nmf_simple
import numpy as np
import plca.plca as plca
import librosa
import scikits.audiolab as skal
import util

EPS = np.finfo(np.float32).eps

canonical_chroma = { "Ab" : "G#", "A"  : "A", "As" : "A#", \
                     "Bb" : "A#", "B"  : "B", "Bs" : "C",  \
                     "Cb" : "B",  "C"  : "C", "Cs" : "C#", \
                     "Db" : "C#", "D"  : "D", "Ds" : "D#", \
                     "Eb" : "D#", "E"  : "E", "Es" : "F",  \
                     "Fb" : "E",  "F"  : "F", "Fs" : "F#", \
                     "Gb" : "F#", "G"  : "G", "Gs" : "G#"  }



def extract_training_feature(datum, n_fft=2048, hop_length=512):
    """
    pad_size = 0
    if len(datum) != n_fft:
        pad_size += (len(datum) + n_fft - 1)/n_fft*n_fft  - len(datum)

    print "Adding padding..."
    pad_size += 2048
    datum = np.pad(datum, pad_size/2, mode='constant')
    """

    harmonic, percussive = librosa.effects.hpss(datum)

    S_h = librosa.core.stft(harmonic, n_fft=2048, hop_length=512)
    S_p = librosa.core.stft(percussive, n_fft=2048, hop_length=512)
    W_h, _, _, _, _, _ = plca.PLCA.analyze(abs(S_h), 1)
    W_p, _, _, _, _, _ = plca.PLCA.analyze(abs(S_h), 1)

    return np.concatenate((W_h, W_p), axis=1)



def learn_templates(instr_samples):
    templates = dict()

    i = 1
    for instr, samples in instr_samples.items():
        instr_templates = []
        j = 1
        for data, chroma, octave in samples:
            print "Instr %d/%d --- Sample %d/%d" % (i, len(instr_samples), j, len(samples))
            feature = extract_training_feature(data)
            instr_templates.append( (feature, chroma, octave) )
            j += 1
        templates[instr] = instr_templates
        i += 1

    return templates



def gather_training_data(path=SAMPLE_PATH):
    instr_names = os.walk(path).next()[1]
    samples = dict()

    pitch_pattern = re.compile("([A-G][sb]?)(\d+)")

    # NOTE: Could potentially make subdirs for different qualities

    for instr in instr_names:
        #if instr not in ('guitar', 'trumpet'): continue
        instr_samples = []
        instr_sample_dir = "%s\%s" % (SAMPLE_PATH, instr)
        for samp in [f for f in os.listdir(instr_sample_dir) \
                if os.path.isfile(os.path.join(instr_sample_dir, f)) \
                and os.path.splitext(f)[1].lower() == ".wav"]:
            data, fs, enc = skal.wavread("%s\%s" % (instr_sample_dir, samp))

            matches = pitch_pattern.search(samp)
            assert matches is not None

            chroma, octave = matches.groups()
            chroma = canonical_chroma[chroma]

            # NOTE: It's quite possible that using a dictionary
            #       instead of a list will be helpful, but we'll
            #       cross that bridge when we get to it
            instr_samples.append( (data, chroma, octave) )

        samples[instr] = instr_samples

    return samples



def train():
    instr_samples = gather_training_data()
    instr_templates = learn_templates(instr_samples)

    with open(MODEL_PATH, 'wb') as output:
        cPickle.dump(instr_templates, output)



if __name__ == "__main__":
    train()
