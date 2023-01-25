
import uhd
import numpy as np
import scipy.constants as sc
import scipy.signal as sig
import matplotlib.pyplot as plt

samp_rate = 1e6  #hz
prf = 100 #Hz
duty_factor = 1.5 #.05
β = 500e3; #Pulse Bandwidth
repetitions = 128
fc = 915e6; #Hz

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
    #elif pulse_type == "Decreasing": pulse = a * np.exp(1j*np.pi*(β/tau)*(np.power(t,2) - 2*tau*t))    
    else:
        LookupError("Pulse type not found")
        
    pulse = pulse * 10**((target_pow - POW(pulse))/20)
    
    return t, pulse
    
    
    
t, p = single_pulse(samp_rate, prf, tau, β, pulse_type="Increasing", envelope_type="HalfSin") #, envelope_type="HalfSin")            
    
    


num_samps = len(p) # number of samples received
gain = 31.5 # dB
duration = num_samps/samp_rate # seconds

usrp = uhd.usrp.MultiUSRP("serial=E2R22N0UP")
usrp.set_rx_antenna("RX2")
usrp.set_tx_antenna("TX/RX")

print(duration)

#while(True):
usrp.send_waveform(p, duration, fc, samp_rate, [0], gain)

binLen = 50
shaped = np.reshape(p, (binLen, int(len(p)/binLen) ))
ffts = np.abs((np.fft.fft(shaped, axis=1)))

f = np.fft.fftshift(np.fft.fftfreq(int(len(p)/binLen), 1/samp_rate))
t = np.arange(0, len(p)/samp_rate, len(p)/samp_rate/binLen)
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




