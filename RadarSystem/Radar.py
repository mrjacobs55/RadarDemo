
import uhd
import numpy as np
import scipy.constants as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import time


samp_rate = 2e6  #hz
prf = 100 #Hz
duty_factor = 100 #0.15 #.05
β = 1e6 #400e3; #Pulse Bandwidth
repetitions = 128
fc = 1.900e9; #Hz

tau = duty_factor * (1/prf) # Length of transmit time in seconds
lam = (1/fc) * sc.speed_of_light

def POW(sig):
    return 10*np.log10(np.sum(np.power(np.abs(sig),2))/len(sig))

def single_pulse(samp_rate, prf, tau, β, target_pow = 0, envelope_type="Rectangular", pulse_type="Increasing"):
    t = np.arange(0, tau, 1/samp_rate)
    if envelope_type == "Rectangular": a = np.ones(len(t))
    elif envelope_type == "HalfSin": a = np.sin(1/(tau/np.pi)*t)
    else: 
        LookupError("Envelope type not found")
        return  
    
    if pulse_type == "Increasing": pulse = a * np.exp(1j*np.pi*(β/tau)*np.power(t,2))
    elif pulse_type == "Decreasing": pulse = a * np.exp(-1j*np.pi*(β/tau)*(np.power(t,2) - 2*tau*t))    
    elif pulse_type == "NLFM": 
        base = 1.5
        t1 = np.arange(0, tau, 1/(samp_rate * 4))
        lfm = np.exp(1j*np.pi*(β/tau)*np.power(t1,2))
        idx = np.power(np.linspace(0,np.power(len(t1), 1/base)-1,num=len(t)), base).astype(int)
        #np.rint(np.logspace(0, np.log(len(t))/np.log(base), num=len(t), base=base)).astype(int)
        pulse = a * lfm[idx]
        # plt.clf()
        # plt.plot(idx)
        # plt.savefig("idx.png")
    else:
        LookupError("Pulse type not found")
        
    pulse = pulse * 10**((target_pow - POW(pulse))/20)
    
    return t, pulse
    
    
    
t, p = single_pulse(samp_rate, prf, tau, β, pulse_type="Increasing", envelope_type="Rectangular") #, envelope_type="HalfSin")            
    
    


num_samps = len(p) # number of samples received
gain = 20# 31# dB
duration = num_samps/samp_rate # seconds

usrp = uhd.usrp.MultiUSRP("serial=E2R22N0UP")
usrp.set_rx_antenna("RX2")
usrp.set_tx_antenna("TX/RX")
usrp.set_tx_bandwidth(samp_rate, 0)

# usrp.set_tx_rate(samp_rate)
# usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(fc),0)
# usrp.set_tx_gain(gain,0)
# usrp.set_tx_dc_offset(True, 0)

# st_args = uhd.usrp.StreamArgs("fc32", "sc16")
# st_args.channels = [0]
# metadata = uhd.types.TXMetadata()
# streamer = usrp.get_tx_stream(st_args)



print(duration)

binLen = 25
shaped = np.reshape(p, (binLen, int(len(p)/binLen) ))
ffts = np.abs(np.fft.fftshift(np.fft.fft(shaped, axis=1),axes=1))

f = np.fft.fftshift(np.fft.fftfreq(int(len(p)/binLen), 1/samp_rate))
t = np.arange(0, len(p)/samp_rate, len(p)/samp_rate/binLen)

plt.clf()
plt.pcolormesh(f, t, ffts, shading="gouraud")
plt.title("Spectrogram")
plt.xlabel('Frequency [Hz]')
plt.ylabel('Time [sec]')
plt.savefig("Spectrogram.png")

plt.clf()
plt.title("Pulse Power Envelope")
plt.plot(np.abs(p))
plt.savefig("Pulse.png")

plt.clf()
plt.title("FFT of Pulse")
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(p[0:int(len(p))]))))
plt.savefig("FFT.png")

usrp.set_tx_dc_offset(True, 0)

while True:
    usrp.send_waveform(p, duration, fc, samp_rate, [0], gain)
    time.sleep(1.5)
    # streamer.send(p, metadata)


