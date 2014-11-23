import pyaudio as pa
import time
from time import sleep

WIDTH = 2
CHANNELS = 1
RATE = 44100

def process(in_data, frame_count, time_info, status):
    if status:
        print "Playback Error: %i" % status
    return (in_data, pa.paContinue)


def run():
    p = pa.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH), \
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    stream_callback=process)

    stream.start_stream()

    while stream.is_active():
        time.sleep(0.1)

    stream.stop_stream()
    stream.close()

    p.terminate()

if __name__ == "__main__":
    run()






