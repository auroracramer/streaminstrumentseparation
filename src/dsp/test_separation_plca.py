import os.path
import numpy as np
import pyaudio as pa
import scipy.signal as sig
import wave
import sys
import scikits.audiolab as skal
import struct
from separation_old import separate
import librosa
import cPickle
import util
import plca_learn

CHUNK_SIZE = 1024

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "%s\\..\\..\\model\\instrument_templates_plca_singles_club.p" % SCRIPT_PATH

def decode(in_data, channels):
    """
    Convert a byte stream into a 2D numpy array with
    shape (chunk_size, channels)

    Samples are interleaved, so for a stereo stream with left channel
    of [L0, L1, L2, ...] and right channel of [R0, R1, R2, ...], the output
    is ordered as [L0, R0, L1, R1, ...]
    """
    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    result = np.fromstring(in_data, dtype=np.float32)

    chunk_length = len(result) / channels
    assert chunk_length == int(chunk_length)

    result = np.reshape(result, (chunk_length, channels))
    return result

def encode(signal):
    """
    Convert a 2D numpy array into a byte stream for PyAudio

    Signal should be a numpy array with shape (chunk_size, channels)
    """
    interleaved = signal.flatten()

    # TODO: handle data type as parameter, convert between pyaudio/numpy types
    out_data = interleaved.astype(np.float32).tostring()
    return out_data

def run():
    # NOTE: Replace this with the name of the test example to perform SS on
    test_name = "grace_short"
    wf = skal.Sndfile("%s\\..\\..\\test\\%s.wav" % (SCRIPT_PATH, test_name), "r")
    p = pa.PyAudio()

    width = int(wf.encoding[-2:])/8
    fs = wf.samplerate

    # NOTE: Replace this with the names of the instruments that are in the audio
    user_instr_names = ['piano', 'trumpet']



    stream = p.open(format=p.get_format_from_width(4),
                    channels=1,
                    rate=wf.samplerate,
                    output=True)

    with open(MODEL_PATH, 'rb') as template_file:
        templates = cPickle.load(template_file)
        instr_to_cols = dict()

        cols = []

        idx = 0
        for instr in user_instr_names:
            instr_templates = templates[instr]
            L = 0
            # For now, we're just discarding the pitch info
            for feature, chroma, octave in instr_templates:
                cols.append(feature)
                L += feature.shape[1]
            instr_to_cols[instr] = (idx, idx + L)
            idx += L

        W = np.concatenate(cols, axis=1)


    #encoded_frames = dict(zip(user_instr_names, [[] for _ in range(len(user_instr_names))]))
    raw_frames = dict(zip(user_instr_names, [[] for _ in range(len(user_instr_names))]))
    divergence = []

    # NOTE: Change this to use the adaptive algorithm or not
    adaptive = False

    threshold = 3
    src_sep = plca_learn.AdaptiveSourceSeparator(W, user_instr_names, instr_to_cols, threshold, fs, CHUNK_SIZE, adaptive)

    i = 0
    frames_left = wf.nframes
    while frames_left > 0:
        frames_requested = min(CHUNK_SIZE, frames_left)
        data = wf.read_frames(frames_requested, dtype=np.float32)
        processed_frames, div = src_sep.process_segment(data)
        for instr in processed_frames:
            raw_source = processed_frames[instr]
            #encoded_frames[instr].append(encoded_source)
            raw_frames[instr].append(raw_source)
        divergence.append(div)
        i += 1
        frames_left -= frames_requested
        print "========= Frames Completed: %d/%d ==========" % (wf.nframes - frames_left, wf.nframes)

    print "Done processing frames."

    stream.stop_stream()
    stream.close()

    p.terminate()

    format = skal.Format('wav')
    mode_desc = "-adapt" if adaptive else "-noadapt"
    for instr, audio_data in raw_frames.items():
        output = skal.Sndfile("%s\\..\\..\\%s-separated-%s%s.wav" % (SCRIPT_PATH, test_name, instr, mode_desc),\
                'w', format, 1, fs)

        for frame in audio_data:
            output.write_frames(frame)
        output.close()

    wf.close()

    with open("%s\\..\\..\\divergence_results-%s%s.txt" % (SCRIPT_PATH, test_name, mode_desc), 'w+') as f:
        f.write("Divergence Results\n")
        f.write("====================\n\n")
        for i, div in enumerate(divergence):
            f.write("    -- Segment %d: %f\n" % (i, div))

if __name__ == "__main__":
    run()
    sys.exit(0)

