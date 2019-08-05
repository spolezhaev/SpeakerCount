import numpy
import scipy.io.wavfile as wf
import sys
import os
## number of ms of silence before selecting a new segment 
ms = 600

class VoiceActivityDetection:

    def __init__(self, sr, ms, channel):
        self.sr = sr
        self.channel = channel
        self.step = int(sr/50)
        self.buffer_size = int(sr/50) 
        self.buffer = numpy.array([],dtype=numpy.int16)
        self.out_buffer = numpy.array([],dtype=numpy.int16)
        self.n = 0
        self.VADthd = 0.
        self.VADn = 0.
        self.silence_counter = 0
        self.segment_count = 0
        self.voice_detected = False
        self.silence_thd_ms = ms

    # Voice Activity Detection
    # Adaptive threshold
    def vad(self, _frame):
        frame = numpy.array(_frame) ** 2.
        result = True
        threshold = 0.1
        thd = numpy.min(frame) + numpy.ptp(frame) * threshold
        self.VADthd = (self.VADn * self.VADthd + thd) / float(self.VADn + 1.)
        self.VADn += 1.

        if numpy.mean(frame) <= self.VADthd:
            self.silence_counter += 1
        else:
            self.silence_counter = 0

        if self.silence_counter > self.silence_thd_ms*self.sr/(1000*self.buffer_size):
            result = False
        return result

    # Push new audio samples into the buffer.
    def add_samples(self, data):
        self.buffer = numpy.append(self.buffer, data)
        result = len(self.buffer) >= self.buffer_size
        # print('buffer size %i'%self.buffer.size)
        return result

    # Pull a portion of the buffer to process
    # (pulled samples are deleted after being
    # processed
    def get_frame(self):
        window = self.buffer[:self.buffer_size]
        self.buffer = self.buffer[self.step:]
        # print('buffer size %i'%self.buffer.size)
        return window

    # Adds new audio samples to the internal
    # buffer and process them
    def process(self, data, filename, original_filename):
        if self.add_samples(data):
            while len(self.buffer) >= self.buffer_size:
                # Framing
                window = self.get_frame()
                # print('window size %i'%window.size)
                if self.vad(window):  # speech frame
                    #print('voiced')
                    self.out_buffer = numpy.append(self.out_buffer, window)
                    self.voice_detected = True
                elif self.voice_detected:
                    #print('unvoiced')
                    self.voice_detected = False
                    self.segment_count = self.segment_count + 1
                    wf.write('%s.%i.wav'%(filename, self.segment_count),self.sr,self.out_buffer)
                    if os.path.isfile(str(original_filename)):
                        os.remove(original_filename)
                    self.out_buffer = numpy.array([],dtype=numpy.int16)
                    #print(self.segment_count)

                # print('out_buffer size %i'%self.out_buffer.size)

    def get_voice_samples(self):
        return self.out_buffer


# wav = wf.read(sys.argv[1])
# ch = 1
# if len(wav[1].shape) > 1:
#     ch = wav[1].shape[1]
# sr = wav[0]

# if len(wav[1].shape) > 1:
#     c0 = wav[1][:,0]
# else:
#     c0 = wav[1][:]

# print('c0 %i'%c0.size)

# vad = VoiceActivityDetection(sr, ms, 1)
# vad.process(c0)

# if ch==1:
#     exit()
    
# vad = VoiceActivityDetection(sr, ms, 2)
# c1 = wav[1][:,1]
# vad.process(c1)