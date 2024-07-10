import numpy as np
import matplotlib.pyplot as plt

# first method 
def show(current, duration, title):
    ts = np.arange(0, duration, bm.get_dt())
    plt.plot(ts, current)
    plt.title(title)
    plt.xlabel('Time [ms]')
    plt.ylabel('Current Value')
    plt.show()
duration = 900
current9 = bp.inputs.square_input(amplitude=1., frequency=71.43,
                                  duration=duration, t_start=0)
show(current9, duration, 'Square Input')

# second method
import numpy as np
import matplotlib.pyplot as plt

def generate_perturbation(t, perturbation_amplitude=1.0, interval=14.0, pulse_width=1.0):
    # Calculate the phase within each interval
    phase = t % interval
    # print('phase=', perturbation_amplitude * bm.less(phase, pulse_width))
    
    # Create a pulse that's high for pulse_width ms at the start of each interval
    
    return perturbation_amplitude * bm.less(phase, pulse_width)

# Create a time array from 0 to 100 ms with 0.1 ms steps
t = np.arange(0, 900, 0.05)

# Calculate the perturbation values
perturbation = generate_perturbation(t, pulse_width=1.0)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(t, perturbation)
plt.title('Perturbation Function Over Time (1ms pulse every 14ms)')
plt.xlabel('Time (ms)')
plt.ylabel('Perturbation Amplitude')
plt.grid(True)

# Add vertical lines every 14 ms to highlight the periodicity
for i in range(0, 901, 14):
    plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


a =np.where(perturbation==1)
b = np.where(current9==1)
len(a[0]), len(set(a[0]).intersection(set(b[0])))
# result 1300, 1300
# perfectly matches 
