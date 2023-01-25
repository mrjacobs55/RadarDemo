## Simple Target 

import uhd


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
num_samps = 16384 # number of samples received
center_freq = 915e6 # Hz
sample_rate = 1e6 # Hz
gain = 31.5 # dB
duration = num_samps/sample_rate # seconds

usrp = uhd.usrp.MultiUSRP("serial=E4R22N4UP")
usrp.set_rx_antenna("RX2")
usrp.set_tx_antenna("TX/RX")

while(True):
    samples = usrp.recv_num_samps(num_samps, center_freq, sample_rate, [0], gain) # units: N, Hz, Hz, list of channel IDs, dB
    usrp.send_waveform(samples, duration, center_freq, sample_rate, [0], gain)
