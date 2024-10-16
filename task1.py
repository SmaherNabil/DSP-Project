import tkinter as tk
from signal import signal
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


class SignalProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Signal Processing GUI")
        self.signals = []

        # GUI Elements
        self.load_button = tk.Button(master, text="Load Signal", command=self.load_signal)
        self.load_button.pack()

        self.plot_cont_button = tk.Button(master, text="Plot Continuous Signal ", command=self.plot_cont_signal)
        self.plot_cont_button.pack()

        self.plot_disc_button = tk.Button(master, text="Plot Discrete Signal", command=self.plot_disc_signal)
        self.plot_disc_button.pack()

        self.add_button = tk.Button(master, text="Add Signals", command=self.add_signals)
        self.add_button.pack()

        self.multiply_button = tk.Button(master, text="Multiply Signal by Constant", command=self.multiply_signal)
        self.multiply_button.pack()

        self.subtract_button = tk.Button(master, text="Subtract Signals", command=self.subtract_signals)
        self.subtract_button.pack()

        self.delay_button = tk.Button(master, text="Delay/Advance Signal", command=self.delay_signal)
        self.delay_button.pack()

        self.fold_button = tk.Button(master, text="Fold/Reverse Signal", command=self.fold_signal)
        self.fold_button.pack()

        # Create a Menubutton
        menubutton = tk.Menubutton(master, text="Generate Signal", relief=tk.RAISED)

        # Create the menu for the menubutton
        menubutton.menu = tk.Menu(menubutton, tearoff=0)
        menubutton["menu"] = menubutton.menu

        # Add checkbuttons for sine and cosine generation
        menubutton.menu.add_command(label="Generate Sine Signal", command=self.generate_sine_signal)
        menubutton.menu.add_command(label="Generate Cosine Signal", command=self.generate_cosine_signal)

        # Pack the Menubutton
        menubutton.pack()
#done
    def load_signal(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if filepath:
            try:
                # Read the file and split lines into index-value pairs
                with open(filepath, 'r') as file:
                    data = file.readlines()

                # Parse the signal data (ignoring the first 3 lines which gives the number of samples and 2 zeros )
                NumOfSam=data[2]
                signal = np.array([list(map(float, line.strip().split())) for line in data[3:]])

                # Append the signal to the signals list
                self.signals.append(signal)

                # Display success message
                messagebox.showinfo("Success", f"Loaded signal with {NumOfSam} samples.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load signal: {e}")
#done
    def plot_cont_signal(self):
        if not self.signals:
            messagebox.showwarning("No Signal", "Please load a signal first.")
            return

        plt.figure()

        for signal in self.signals:
            plt.plot(signal[:, 0], signal[:, 1], label='Signal')

        # Set labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Signal Plot')

        # Add grid lines for better visualization
        plt.grid(True)

        # Get current axes and set limits to show both positive and negative regions
        plt.axhline(0, color='red', linewidth=0.9)  # X-axis (horizontal line)
        plt.axvline(0, color='red', linewidth=0.9)  # Y-axis (vertical line)

        # Adjust the limits dynamically based on the signal range
        plt.xlim(np.min([np.min(signal[:, 0]) for signal in self.signals]),
                 np.max([np.max(signal[:, 0]) for signal in self.signals])) #for indices
        plt.ylim(np.min([np.min(signal[:, 1]) for signal in self.signals]),
                 np.max([np.max(signal[:, 1]) for signal in self.signals])) #for values

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
#done
    def plot_disc_signal(self):
        if not self.signals:
            messagebox.showwarning("No Signal", "Please load a signal first.")
            return

        plt.figure()

        for signal in self.signals:
            plt.scatter(signal[:, 0], signal[:, 1],color='magenta', label='Signal')
            plt.vlines(signal[:, 0], ymin=0, ymax= signal[:, 1], color='purple', linestyle='solid', label='Lines to x-axis')

        # Set labels and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Signal Plot')

        # Add grid lines for better visualization
        plt.grid(True)

        # Get current axes and set limits to show both positive and negative regions
        plt.axhline(0, color='cyan', linewidth=0.9)  # X-axis (horizontal line)
        plt.axvline(0, color='cyan', linewidth=0.9)  # Y-axis (vertical line)

        # Adjust the limits dynamically based on the signal range
        plt.xlim(np.min([np.min(signal[:, 0]) for signal in self.signals]),
                 np.max([np.max(signal[:, 0]) for signal in self.signals]))  # for indices
        plt.ylim(np.min([np.min(signal[:, 1]) for signal in self.signals]),
                 np.max([np.max(signal[:, 1]) for signal in self.signals]))  # for values

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()
#done
    def add_signals(self):
        if len(self.signals) < 2:
            messagebox.showwarning("Error", "Need at least two signals to add.")
            return

        # Create a defaultdict to store the sum of values for each index
        result_dict = defaultdict(float)

        # Iterate through each signal and sum the values by index
        for signal in self.signals:
            for index, value in signal:
                result_dict[index] += value

        # Sort the result by index and convert it to a numpy array
        result = np.array(sorted(result_dict.items()))
        self.plot_signal_helper(result, 'Resultant Signal (Addition)', 'Added Signals')
#done
    def multiply_signal(self):
        if not self.signals:
            messagebox.showwarning("No Signal", "Please load a signal first.")
            return

        constant = float(tk.simpledialog.askstring("Input", "Enter the constant:"))

        # Ask for the signal number (1-based index)
        while True:
            SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
            if 1 <= SigNumber <= len(self.signals):  # Check if the signal number is valid
                break  # Exit the loop if the number is valid
            else:
                messagebox.showwarning("Invalid Signal",
                                       "Hnhzr Ya M3lm? Signal is not in the loaded signals!")

        # Access the signal (adjusting for 0-based indexing)
        signal = self.signals[SigNumber - 1]

        # Multiply the signal values (y-values) by the constant
        result = signal.copy()
        result[:, 1] *= constant
        self.plot_signal_helper(result, f'Signal {SigNumber} multiplied by {constant}', 'Amplified/Reduced Signal')
#done
    def subtract_signals(self):
        if len(self.signals) < 2:
            messagebox.showwarning("Error", "Need at least two signals to subtract.")
            return

        # Create a defaultdict to store the sum of values for each index
        result_dict = defaultdict(float)

        # Iterate through each signal and sum the values by index
        for signal in self.signals:
            for index, value in signal:
                result_dict[index] -= value

        # Sort the result by index and convert it to a numpy array
        result = np.array(sorted(result_dict.items()))
        self.plot_signal_helper(result, 'Resultant Signal (Subtraction)', 'Subtracted Signal')
#done
    def delay_signal(self):
        if not self.signals:
            messagebox.showwarning("No Signal", "Please load a signal first.")
            return
        k = int(tk.simpledialog.askstring("Input", "Enter delay/advance value (negative for advance):"))

        # Ask for the signal number (1-based index)
        while True:
            SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
            if 1 <= SigNumber <= len(self.signals):  # Check if the signal number is valid
                break  # Exit the loop if the number is valid
            else:
                messagebox.showwarning("Invalid Signal",
                                       "Hnhzr Ya M3lm? Signal is not in the loaded signals!")

        # Access the signal (adjusting for 0-based indexing)
        signal = self.signals[SigNumber - 1]
        result = signal.copy()
        result[:, 0] += k #on x axis "time"
        self.plot_signal_helper(result, f'Signal delayed/advanced by {k}', 'Delayed/Advanced Signal')
#done
    def fold_signal(self):
        if not self.signals:
            messagebox.showwarning("No Signal", "Please load a signal first.")
            return
        # Ask for the signal number (1-based index)
        while True:
            SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
            if 1 <= SigNumber <= len(self.signals):  # Check if the signal number is valid
                break  # Exit the loop if the number is valid
            else:
                messagebox.showwarning("Invalid Signal",
                                       "Hnhzr Ya M3lm? Signal is not in the loaded signals!")

        # Access the signal (adjusting for 0-based indexing)
        signal = self.signals[SigNumber - 1]
        result = signal.copy()
        result[:, 0] = -result[:, 0]

        self.plot_signal_helper(result, 'Folded Signal', 'Folded/Reversed Signal')
#done
    def plot_signal_helper(self, result, label, title):
        plt.plot(result[:, 0], result[:, 1], label=label)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(title)

        # Set the origin lines (x-axis and y-axis) in red
        plt.axhline(0, color='red', linewidth=0.9)  # x-axis (horizontal line)
        plt.axvline(0, color='red', linewidth=0.9)  # y-axis (vertical line)

        plt.legend()
        plt.show()

    def generate_sine_signal(self):
        self.input_window('sin')
    def generate_cosine_signal(self):
        self.input_window('cos')
    def input_window(self, signal_type):
        input_win = tk.Toplevel(self.master)
        input_win.title(f"Generate {signal_type} Signal")

        tk.Label(input_win, text="Amplitude:").grid(row=0)
        amplitude_entry = tk.Entry(input_win)
        amplitude_entry.grid(row=0, column=1)

        tk.Label(input_win, text="Analog Frequency (Hz):").grid(row=1)
        Analog_freq_entry = tk.Entry(input_win)
        Analog_freq_entry.grid(row=1, column=1)

        tk.Label(input_win, text="Theta (Radians):").grid(row=2)
        theta_entry = tk.Entry(input_win)
        theta_entry.grid(row=2, column=1)

        tk.Label(input_win, text="Sampling Frequency (Hz):").grid(row=3)
        Sampling_freq_entry = tk.Entry(input_win)
        Sampling_freq_entry.grid(row=3, column=1)

        self.generate_button = tk.Button(input_win, text="Generate")
        self.generate_button.grid(row=4, column=1)


        amplitude = float(amplitude_entry.get())
        analog_freq = float(Analog_freq_entry.get())
        theta = float(theta_entry.get())
        sampling_freq = float(Sampling_freq_entry.get())

        def fire_generate():
            if fire==True:
                  self.Generate_signal(signal_type, sampling_freq, theta, amplitude)
            else :
                self.Generate_signal(signal_type, FS, theta, amplitude)

        x= self.Check_Frequency(analog_freq,sampling_freq)
        if x == True :
            self.generate_button.config(command=fire_generate)
            fire=True

        else :
            while True:
                FS = int(tk.simpledialog.askstring("Input", f'Please Enter Valid Sampling Frequency greater than or equal ,{2*analog_freq}, : '))
                if FS>= 2*analog_freq:
                    fire=False
                    self.generate_button.config(command=fire_generate)

    def Check_Frequency(self,F_analog,F_sampling):
        if(F_sampling>= 2*F_analog):
            return True
        else:
            return False

    def Generate_signal(self,SignalType,F_sampling,Theta,Amplitude):
        t = np.linspace(0, 2 * np.pi, 1000)
        if SignalType== "Sin":
            signal = Amplitude * np.sin(F_sampling * t + Theta)
            self.plot_signal_helper(signal, f'Signal Generated by Sin ', 'Sin Signal Generated')
        elif SignalType=="Cos":
            signal = Amplitude * np.cos(F_sampling * t + Theta)
            self.plot_signal_helper(signal, f'Signal Generated by Cos', 'Cos Signal Generated')








#testing functions


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessor(root)
    root.mainloop()
