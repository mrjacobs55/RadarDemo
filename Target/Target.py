## Simple Target 

import uhd
import matplotlib.pyplot as plt
import numpy as np
import sys
import signal
import time
import threading

# Class Target:
# {
#     def __init__(self, center_freq, samp_rate, min_distance, max_distance, min_velocity, max_velocity, min_acceleration, max_acceleration, start_distance):
#         self.center_freq = center_freq
#         self.samp_rate = samp_rate
#         self.min_distance = min_distance
#         self.max_distance = max_distance
#         self.min_velocity = min_velocity
#         self.max_velocity = max_velocity
#         self.min_acceleration = min_acceleration
#         self.max_acceleration = max_acceleration
#         self.distance = start_distance
# }

def exit_gracefully(signum, frame):
    # Stop Stream
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    streamer.issue_stream_cmd(stream_cmd)
    print("Exiting Gracefully")
    sys.exit(0)

# Receive Samples
def rx(num_samps):
    samples = np.zeros(num_samps, dtype=np.complex64)
    for i in range(num_samps//buffer_size):
        streamer.recv(recv_buffer, metadata)
        samples[i*buffer_size:(i+1)*buffer_size] = recv_buffer[0]
    return samples


def plot_spectrogram(samples, num_samps, samp_rate, center_freq, binLen = 128):
    
    shaped = np.reshape(samples, (binLen, int(num_samps/binLen)) )
    ffts = np.abs(np.fft.fftshift(np.fft.fft(shaped, axis=1),axes=1))

    f = np.fft.fftshift(np.fft.fftfreq(int(num_samps/binLen), 1/samp_rate))
    f = f + center_freq
    t = np.arange(0, num_samps/samp_rate, num_samps/samp_rate/binLen)

    plt.clf()
    plt.pcolormesh(f, t, ffts, shading="gouraud")
    plt.title("Spectrogram")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Time [sec]')
    plt.show()


oversamp = 5
num_samps = oversamp * (2**11) #2**12 #4096 # number of samples received
center_freq = 1900e6 # Hz
samp_rate = oversamp * 2e6 # Hz
gain = 31.5 # dB
duration = num_samps/samp_rate # seconds
buffer_size = 2048

usrp = uhd.usrp.MultiUSRP("serial=30D2DAA")
usrp.set_rx_antenna("RX2")
usrp.set_tx_antenna("TX/RX")

usrp.set_rx_rate(samp_rate, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(center_freq), 0)
usrp.set_rx_gain(gain, 0)
usrp.set_rx_dc_offset(True)
usrp.set_rx_bandwidth(samp_rate, 0)
# Set up the stream and receive buffer
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)
#buffer_size = streamer.get_max_num_samps()
recv_buffer = np.zeros((1, buffer_size), dtype=np.complex64)
# Start Stream
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)

signal.signal(signal.SIGINT, exit_gracefully)


pow = []
pow_thresh = 0.1 #Function Generator is 0.1
signal_max_len = 2**20 * oversamp
signal_exists = 0
sig_buffer = np.zeros((signal_max_len), dtype=np.complex64)

while(True):
    streamer.recv(recv_buffer, metadata)
    if np.max(np.abs(recv_buffer[0])) > pow_thresh:
        if signal_exists + buffer_size > signal_max_len:
            print("Signal Too Long")
            exit_gracefully(0,0)    
        sig_buffer[signal_exists:signal_exists+buffer_size] = recv_buffer[0]
        signal_exists += buffer_size
    else:
        if signal_exists > 0:
            print("Signal Detected of Len " + str(signal_exists))
            #plt.clf()
            #plt.plot(np.abs(sig_buffer[0:signal_exists]))
            #plt.show()
            plot_spectrogram(sig_buffer[0:signal_exists], signal_exists, samp_rate, center_freq, binLen=2048)
        signal_exists = 0    
    #pow.append(np.max(np.abs(recv_buffer[0])))
    #samples = rx(num_samps)
    #pow.append(10*np.log10(np.mean(np.abs(samples)**2)))
    #plot_spectrogram(samples, num_samps, samp_rate, center_freq)
    # if(len(pow) % 2000 == 0):
    #     plt.clf()
    #     plt.plot(pow)
    #     plt.show()

    #usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)
