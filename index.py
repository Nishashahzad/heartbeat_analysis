import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

cap = cv2.VideoCapture(0)

def get_mean_color(frame, x, y, w, h):
    roi = frame[y:y+h, x:x+w]
    mean_color = np.mean(roi)
    return mean_color

x, y, w, h = 300, 200, 100, 100

color_data = deque(maxlen=300)
heart_rates = []

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
sampling_rate = 30

plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], color='green', lw=1)
ax.set_xlim(0, 300)
ax.set_ylim(0, 255)
ax.set_xlabel('Time')
ax.set_ylabel('Intensity')
ax.set_title('Heartbeat Visualization')

last_beat_time = time.time()

def update_plot():
    ax.set_xlim(max(0, len(color_data) - 300), len(color_data))
    if len(color_data) > 1:
        xdata = np.arange(len(color_data))
        ydata = np.array(color_data)
        line.set_data(xdata, ydata)
    fig.canvas.draw()
    fig.canvas.flush_events()

def calculate_heart_rate():
    if len(color_data) >= fps * sampling_rate:
        recent_color_data = np.array(list(color_data))[-int(fps * sampling_rate):]
        detrended = recent_color_data - np.mean(recent_color_data)
        fft_results = np.fft.rfft(detrended)
        fft_freq = np.fft.rfftfreq(detrended.shape[0], 1.0 / fps)
        peak_freq_index = np.argmax(np.abs(fft_results))
        peak_freq = fft_freq[peak_freq_index]
        heart_rate = peak_freq * 60  
        heart_rates.append(heart_rate)
        print(f"Heart Rate: {heart_rate:.2f} BPM")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    mean_color = get_mean_color(frame, x, y, w, h)
    
    color_data.append(mean_color)
    
    cv2.imshow('Webcam', frame)
    
    update_plot()
    
    current_time = time.time()
    if current_time - last_beat_time >= 2:
        calculate_heart_rate()
        last_beat_time = current_time
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
