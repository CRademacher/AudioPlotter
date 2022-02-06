# import argparse
import queue
import sys
import sounddevice as sd
import time

import numpy as np

from scipy.io import wavfile
from scipy.fft import fft, fftfreq

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg


font = {'family' : 'Arial',
        'weight' : 'bold',
        'size'   : 22}
matplotlib.rc('font', **font)

matplotlib.rcParams['toolbar'] = 'None'

#Hz of notes
NOTES = np.array([27.5,29.13524,30.86771,32.7032,34.64783,36.7081,38.89087,41.20344,43.65353,46.2493,48.99943,51.91309,\
55,58.27047,61.73541,65.40639,69.29566,73.41619,77.78175,82.40689,87.30706,92.49861,97.99886,103.8262,110,116.5409,123.4708,\
130.8128,138.5913,146.8324,155.5635,164.8138,174.6141,184.9972,195.9977,207.6523,220,233.0819,246.9417,261.6256,277.1826,293.6648,\
311.127,329.6276,349.2282,369.9944,391.9954,415.3047,440,466.1638,493.8833,523.2511,554.3653,587.3295,622.254,659.2551,698.4565,\
739.9888,783.9909,830.6094,880,932.3275,987.7666,1046.502,1108.731,1174.659,1244.508,1318.51,1396.913,1479.978,1567.982,1661.219,\
1760,1864.655,1975.533,2093.005,2217.461,2349.318,2489.016,2637.02,2793.826,2959.955,3135.963,3322.438,3520,3729.31,3951.066,4186.009])

KEY_NAMES = ['A0', 'A#0', 'B0', 'C1', 'C#1', 'D1', 'D#1', 'E1', 'F1', 'F#1', 'G1', 'G#1', 'A1', 'A#1', 'B1', 'C2', 'C#2', 'D2', 'D#2'\
, 'E2', 'F2', 'F#2', 'G2', 'G#2', 'A2', 'A#2', 'B2', 'C3', 'C#3', 'D3', 'D#3', 'E3', 'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3', 'C4',\
 'C#4', 'D4', 'D#4', 'E4', 'F4', 'F#4', 'G4', 'G#4', 'A4', 'A#4', 'B4', 'C5', 'C#5', 'D5', 'D#5', 'E5', 'F5', 'F#5', 'G5', 'G#5', 'A5',\
  'A#5', 'B5', 'C6', 'C#6', 'D6', 'D#6', 'E6', 'F6', 'F#6', 'G6', 'G#6', 'A6', 'A#6', 'B6', 'C7', 'C#7', 'D7', 'D#7', 'E7', 'F7', 'F#7',\
   'G7', 'G#7', 'A7', 'A#7', 'B7', 'C8']

class AudioPlotter:
    def __init__(self,channels,device,window,interval,downsample,downsample_plot,blocksize,samplerate):
        self.channels = channels
        self.device = device
        self.window = window
        self.interval = interval
        self.downsample = downsample
        self.downsample_plot = downsample_plot
        self.blocksize = blocksize
        self.samplerate = samplerate

        self.maxRawData_y = 0
        self.maxFreq_y = 0
        self.q = queue.Queue()

        self.rawDataLength = int(self.window* self.samplerate / (1000 * self.downsample * self.downsample_plot))
        self.setupFigure()

        #plot initial data
        self.rawAudioData = np.zeros(self.rawDataLength)
        freqxf,freqyf = self.findFFT(self.rawAudioData[:])

        self.rawDataLine, = self.ax1.plot(self.rawAudioData, linewidth = '2',color = 'black')
        self.freqLine, = self.ax2.plot(freqxf,freqyf, linewidth = '2',color = 'red')
        self.fig.tight_layout(pad=0.4)

        self.times = np.zeros(20) # for keeping track of framerate
        self.times[-1] = time.time()
        stream = sd.InputStream(
            device=self.device, blocksize = self.blocksize,channels=self.channels[0],
            samplerate=self.samplerate, callback=self.audio_callback,latency = 'low')
        ani = FuncAnimation(self.fig, self.update_plot, interval=self.interval, blit=False,frames = 60)
        with stream:
            plt.show()

    def setupFigure(self):
        self.fig, ((self.ax1,self.ax2),(self.ax3,self.ax4)) = plt.subplots(2,2)
        self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()
        img = mpimg.imread("A:\\Downloads\\p515.jpg")
        self.ax3.imshow(img)

        self.ax1.set_title("Raw Input",fontweight='bold')
        self.ax2.set_title("Frequency Domain",fontweight='bold')
        self.ax2.set_xlabel("Hz",fontweight='bold')

        for i,ax in enumerate([self.ax1,self.ax2,self.ax3,self.ax4]):
            ax.set_yticks([0])
            ax.yaxis.grid(True)
            if i == 0:
                ax.tick_params(bottom=False, top=False, labelbottom=False,
                           right=False, left=False, labelleft=False)
                ax.set_xlim(0,self.rawDataLength)
            elif i == 1:
                ax.tick_params(bottom=True, top=False, labelbottom=True,
                           right=False, left=False, labelleft=False)
                ax.set_xlim([0,4300])
            else:
                ax.tick_params(bottom=False, top=False, labelbottom=False,
                            right=False, left=False, labelleft=False)
                ax.axis('off')

        self.noteText = plt.gcf().text(0.02,0.012,'Now Playing:\nA0',fontsize = 32)
        self.FPSText = plt.gcf().text(0.01,0.95,"FPS:",fontsize = 32, color = 'red')
        self.circle = plt.Circle(self.getCoordsOfNote('C3'),8,color='red')
        self.ax3.add_patch(self.circle)

    def audio_callback(self, indata, frames, time, status):
        """This is called (from a separate thread) for each audio block."""
        if status:
            print(status, file=sys.stderr)
        # Fancy indexing with mapping creates a (necessary!) copy:
        self.q.put(indata[::self.downsample, self.channels[0]-1])


    def update_plot(self, frame):
        """This is called by matplotlib for each plot update.
        Typically, audio callbacks happen more frequently than plot updates,
        therefore the queue tends to contain multiple blocks of audio data.
        """

        changed = False
        while True:
            try:
                data = self.q.get_nowait()
                changed = True
            except queue.Empty:
                break

            newRawAudioData = data[::self.downsample_plot]
            shift = len(newRawAudioData)
            self.rawAudioData = np.roll(self.rawAudioData, -shift, axis=0)
            self.rawAudioData[-shift:] = newRawAudioData

            newfreqx,newfreqy = self.findFFT(data)
            thisNote = self.findNote(newfreqx[np.argmax(newfreqy)])
            self.noteText.set_text("Now Playing:\n" + thisNote)
            self.circle.center = self.getCoordsOfNote(thisNote)

            self.rawDataLine.set_ydata(self.rawAudioData[:])
            self.freqLine.set_xdata(newfreqx)
            self.freqLine.set_ydata(newfreqy)

        if changed:
            if max(self.rawAudioData[:]) > self.maxRawData_y:
                self.maxRawData_y = max(self.rawAudioData[:])
                self.ax1.set_ylim(-self.maxRawData_y,self.maxRawData_y)
            if max(newfreqy) > self.maxFreq_y:
                self.maxFreq_y = max(newfreqy[:])
                self.ax2.set_ylim(0,self.maxFreq_y)

        self.times = np.roll(self.times,-1,axis=0)
        self.times[-1] = time.time()
        self.FPSText.set_text("FPS: " + str(round(20/(self.times[-1]-self.times[0]),1)))

        return None

    def findFFT(self, data):
        T = 1/(self.samplerate/self.downsample)
        N = len(data)
        xf = fftfreq(N,T)[:N//2]
        yf = fft(data)
        yf = (2.0/N)*np.abs(yf[0:N//2])
        return xf,yf

    def findNote(self, freq):
        return KEY_NAMES[(np.abs(NOTES-freq)).argmin()]

    def getCoordsOfNote(self, note):
        y = 300
        upper = 1430
        lower = 66
        step = (upper-lower)/89
        idx = KEY_NAMES.index(note)
        x = lower + (idx+1)*step
        return (x,y)

    def onclick(self, event):
        print(event.x,event.y)

def main():
    channels = [1] # just choose first channel in a mic with multiple channels
    device = None # uses default microphone input
    window = 10000 # ms, total time shown
    interval = 10 # ms, time between updates
    downsample = 2 # every Nth sample is used for calculation
    downsample_plot = 100 # every Mth sample in every Nth sample (above) is used in the raw audio plot
    blocksize = 4000 # I think this is the amount of information that gets passed each stream tick
    samplerate = sd.query_devices(device,'input')['default_samplerate']

    audioplotter = AudioPlotter(channels,device,window,interval,downsample,downsample_plot,blocksize,samplerate)

if __name__ == '__main__':
    main()
