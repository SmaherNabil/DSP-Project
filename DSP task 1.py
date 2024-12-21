import cmath
import math
import tkinter as tk
from signal import signal
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from numpy.ma.core import power
import os
import numpy as np
from tkinter import messagebox

import numpy as np
from tkinter import filedialog, messagebox

import math


class SignalProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Signal Processing GUI")
        self.signals = []
        self.class_1_signals=[]
        self.class_2_signals=[]
        self.test_signals=[]
        self.NumOfSam=0
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


        menu2button = tk.Menubutton(master, text="Quantize Signal", relief=tk.RAISED)
        # Create the menu for the menubutton
        menu2button.menu = tk.Menu(menu2button, tearoff=0)
        menu2button["menu"] = menu2button.menu
        # Add checkbuttons enter levels or enter number of bits
        menu2button.menu.add_command(label="Enter Number Of Levels", command=self.using_number_of_levels)
        menu2button.menu.add_command(label="Enter Number Of Bits", command=self.using_number_of_bits)
        # Pack the Menubutton
        menu2button.pack()

        self.convolve_button = tk.Button(master, text="Convolve", command=self.convolve_signals)
        self.convolve_button.pack()

        self.smooth_button = tk.Button(master, text="Smooth", command=self.smooth_signal)
        self.smooth_button.pack()

        self.sharpening_button = tk.Button(master, text="Sharpening", command=self.sharpening_signal)
        self.sharpening_button.pack()

        menu2button = tk.Menubutton(master, text="Fourier", relief=tk.RAISED)
        # Create the menu for the menubutton
        menu2button.menu = tk.Menu(menu2button, tearoff=0)
        menu2button["menu"] = menu2button.menu
        # Add checkbuttons enter levels or enter number of bits
        menu2button.menu.add_command(label="DFT", command=self.DFT_transform)
        menu2button.menu.add_command(label="IDFT", command=self.IDFT_transform)
        # Pack the Menubutton
        menu2button.pack()

        self.cross_correlation_button = tk.Button(master, text="Cross Correlation", command=self.cross_correlation)
        self.cross_correlation_button.pack()

        self.class_correlation_button = tk.Button(master, text="Class Correlation", command=self.correlation_for_classes)
        self.class_correlation_button.pack()

        menu3button = tk.Menubutton(master, text="Filter", relief=tk.RAISED)
        # Create the menu for the menubutton
        menu3button.menu = tk.Menu(menu3button, tearoff=0)
        menu3button["menu"] = menu3button.menu
        # Add checkbuttons enter levels or enter number of bits
        menu3button.menu.add_command(label="Low pass", command=self.low_pass_filter)
        menu3button.menu.add_command(label="High pass", command=self.high_pass_filter)
        menu3button.menu.add_command(label="Band pass", command=self.band_pass_filter)
        menu3button.menu.add_command(label="Band Reject", command=self.band_reject_filter)
        # Pack the Menubutton
        menu3button.pack()
        self.apply_filter_direct = tk.Button(master, text="apply filter(direct)", command=self.apply_filter_direct)
        self.apply_filter_direct.pack()
        self.apply_filter_fast= tk.Button(master, text="apply filter(fast)", command=self.apply_filter_fast)
        self.apply_filter_fast.pack()

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
            self.plot_signal_helper(signal, label='Signal', title='Analog signal')
# done
    def plot_disc_signal(self):
        if not self.signals:
            messagebox.showwarning("No Signal", "Please load a signal first.")
            return
        plt.figure()
        for signal in self.signals:
            self.plot_sample_signal(signal, label='Signal', title='Discrete signal')
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
    # for plotting continuous signals
    def plot_signal_helper(self, result, label, title):
        plt.plot(result[:, 0], result[:, 1], label=label)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(title)

        # Set the origin lines (x-axis and y-axis) in red
        plt.axhline(0, color='red', linewidth=0.9)  # x-axis (horizontal line)
        plt.axvline(0, color='red', linewidth=0.9)  # y-axis (vertical line)

        plt.legend()
        plt.show()
#done
    # for plotting discrete signals
    def plot_sample_signal(self, result, label, title):
        plt.scatter(result[:, 0], result[:, 1], label=label,color='magenta')
        plt.vlines(result[:, 0], ymin=0, ymax=result[:, 1], color='purple', linestyle='solid',
                   label='Lines to x-axis')

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(title)

        plt.grid(True)

        plt.axhline(0, color='cyan', linewidth=0.9)  # X-axis (horizontal line)
        plt.axvline(0, color='cyan', linewidth=0.9)  # Y-axis (vertical line)

        plt.legend()
        plt.show()
# done
    def generate_sine_signal(self):
        self.input_window('sin')
# done
    def generate_cosine_signal(self):
        self.input_window('cos')
# done
    def input_window(self, signal_type):
        input_win = tk.Toplevel(self.master)
        input_win.title(f"Generate {signal_type.capitalize()} Signal")

        tk.Label(input_win, text="Amplitude:").grid(row=0, column=0)
        amplitude_entry = tk.Entry(input_win)
        amplitude_entry.grid(row=0, column=1)

        tk.Label(input_win, text="Analog Frequency (Hz):").grid(row=1, column=0)
        Analog_freq_entry = tk.Entry(input_win)
        Analog_freq_entry.grid(row=1, column=1)

        tk.Label(input_win, text="Theta (Radians):").grid(row=2, column=0)
        theta_entry = tk.Entry(input_win)
        theta_entry.grid(row=2, column=1)

        tk.Label(input_win, text="Sampling Frequency (Hz):").grid(row=3, column=0)
        Sampling_freq_entry = tk.Entry(input_win)
        Sampling_freq_entry.grid(row=3, column=1)

        def generate_analog_signal():
                amplitude = float(amplitude_entry.get())
                analog_freq = float(Analog_freq_entry.get())
                theta = float(theta_entry.get())
                sampling_freq = float(Sampling_freq_entry.get())
                thetaIn_Deg=np.deg2rad(theta)
                if sampling_freq >= 2 * analog_freq:
                    # points in all time
                    t = np.linspace(0, 1, 1000)
                    if signal_type == 'sin':
                        # x(t) = A sin(2*pi*F*T + thetaIn_Deg)
                        signal = amplitude * np.sin(2 * np.pi * analog_freq * t + thetaIn_Deg)
                    else:
                        # x(t) = A cos(2*pi*F*T + theta)
                        signal = amplitude * np.cos(2 * np.pi * analog_freq * t + thetaIn_Deg)

                    result = np.column_stack((t, signal))  # Combine time and signal
                    self.plot_signal_helper(result, f"Generated {signal_type.capitalize()} Signal", f"{signal_type.capitalize()} Signal")
                else:
                    messagebox.showerror("Error", "Sampling frequency must be at least twice the analog frequency.")

        # Generate button to create the signal
        tk.Button(input_win, text="Generate Analog", command=generate_analog_signal).grid(row=4, column=0, columnspan=2)
        def generate_discrete_signal():
            amplitude = float(amplitude_entry.get())
            analog_freq = float(Analog_freq_entry.get())
            theta = float(theta_entry.get())
            sampling_freq = float(Sampling_freq_entry.get())
            # x(n) = A sin( 2*pi*fn/Fs + theta)
            if sampling_freq >= 2 * analog_freq:
                t = np.arange(0, 1, 1 / sampling_freq)  # Discrete time points with 1/sampling_rate --> n/fs

                if signal_type == 'sin':
                    signal = amplitude * np.sin(2 * np.pi * analog_freq * t + theta)
                else:
                    signal = amplitude * np.cos(2 * np.pi * analog_freq * t + theta)

                # Combine time and signal for plotting
                result = np.column_stack((t, signal))
                self.plot_sample_signal(result, f"Generated Discrete {signal_type.capitalize()} Signal",
                                        f"Discrete {signal_type.capitalize()} Signal")
            else:
                messagebox.showerror("Error", "Sampling frequency must be at least twice the analog frequency.")

        # Create a new button to generate the discrete signal
        tk.Button(input_win, text="Generate Discrete", command=generate_discrete_signal).grid(row=5, column=0,
                                                                                              columnspan=2)
#done
    def plot_quantized_signal(self,result , label, title,mid_points,quantization_errors):

        plt.scatter(result[:, 0], result[:, 1], label=label, color='magenta')
        plt.vlines(result[:, 0], ymin=0, ymax=result[:, 1], color='purple', linestyle='solid',
                   label='Lines to x-axis')

        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title(title)

        plt.grid(True)

        plt.axhline(0, color='cyan', linewidth=0.9)  # X-axis (horizontal line)
        plt.axvline(0, color='cyan', linewidth=0.9)  # Y-axis (vertical line)
        # Add horizontal lines for each level on the plot
        for level in mid_points:
            plt.axhline(y=level, color='black', linestyle='--', linewidth=0.7, label=f'Level: {level}')
            # Calculate average quantization error
        avg_error = np.mean(np.square(quantization_errors))
        # Display average error on the plot
        plt.text(0.05, 0.95, f'Average Error: {avg_error:.4f}', transform=plt.gca().transAxes,
                 fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

        plt.legend()
        plt.show()
#done
    def using_number_of_levels(self):
        SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
        SigNumber -= 1
        amplitude_values = self.signals[SigNumber][:, 1]

        # Find the minimum and maximum values in the amplitude column
        min_val = np.min(amplitude_values)
        max_val = np.max(amplitude_values)
        levels = int(tk.simpledialog.askstring("Input", "Enter the number of levels:"))
        delta = (max_val - min_val) / levels

        print(min_val)
        print(max_val)
        print(delta)

        new_ranges = []
        temp = min_val

        # Create the new ranges using a for loop
        for i in range(levels):
            pair = (temp, temp + delta)  # Create a pair for the range
            new_ranges.append(pair)
            temp += delta
        print(new_ranges)

        mid_points = []
        for i in new_ranges:
            mid_point = (i[0] + i[1]) / 2
            mid_points.append(mid_point)
        mid_points = [round(i, 4) for i in mid_points]
        print("Midpoints:", mid_points)

        # Determine the number of bits required
        num_bits = int(np.ceil(np.log2(levels)))
        print("Number of Bits Required:", num_bits)

        # Create a dictionary to map each midpoint's level to its binary representation
        level_to_binary = {
            i: format(i, f'0{num_bits}b') for i in range(levels)
        }
        print("Level to Binary Mapping:", level_to_binary)

        # Create arrays to store quantized values, quantization errors, and binary representations
        quantized_array = []
        quantization_errors = []
        binary_representations = []
        Interval_level=[]

        # Map each value in amplitude_values to the nearest midpoint and corresponding level's binary representation
        for value in amplitude_values:
            nearest_mid = min(mid_points, key=lambda x: abs(x - value))
            quantized_array.append(nearest_mid)  # Append the nearest midpoint to quantized_array
            quantization_errors.append(nearest_mid-value)  # Calculate quantization error

            level_index = mid_points.index(nearest_mid)  # Find the index of the nearest midpoint
            Interval_level.append(level_index+1)
            binary_representations.append(level_to_binary[level_index])  # Get binary representation of the level

        print("Quantized Array:", quantized_array)
        print("Binary Representations:", binary_representations)

        # Combine x values and quantized values for plotting
        x = self.signals[SigNumber][:, 0]
        result = np.column_stack((x, quantized_array))
        self.plot_quantized_signal(result, "Quantized Signal", "Using number of levels", mid_points,
                                   quantization_errors)


        QuantizationTest2("D:\\d\\fcis 2025\\pythonProject\\Quan2_Out.txt", Interval_level,binary_representations,
                          quantized_array,quantization_errors)
    #done
    def using_number_of_bits(self):
        SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
        SigNumber-=1
        amplitude_values = self.signals[SigNumber][:, 1]

        # Find the minimum and maximum values in the amplitude column
        min_val = np.min(amplitude_values)
        max_val = np.max(amplitude_values)
        bits = int(tk.simpledialog.askstring("Input", "Enter the number of bits:"))
        levels = 2 ** bits  # Correct way to calculate levels
        delta = (max_val - min_val) / levels

        print(levels)
        print(min_val)
        print(max_val)
        print(delta)

        new_ranges = []
        temp = min_val
        # Create the new ranges using a for loop
        for i in range(levels):
            pair = (temp, temp + delta)  # Create a pair for the range
            new_ranges.append(pair)
            temp += delta
        print(new_ranges)
        mid_points = []
        for i in new_ranges:
            mid_point = (i[0] + i[1]) / 2
            mid_points.append(mid_point)
        mid_points = [round(i, 4) for i in mid_points]
        print(mid_points)

        # Create a dictionary to map each midpoint's level to its binary representation
        level_to_binary = {
            i: format(i, f'0{bits}b') for i in range(levels)
        }
        print("Level to Binary Mapping:", level_to_binary)

        # Create a new array to store quantized values
        quantized_array = []
        quantization_errors=[]
        binary_representations = []

        # Map each value in amplitude_values to the nearest midpoint
        for value in amplitude_values:
            # Find the nearest midpoint using a l1mbda function to calculate the distance
            nearest_mid = min(mid_points, key=lambda x: round(abs(x -  value),4))
            quantized_array.append(nearest_mid)  # Append the nearest midpoint to quantized_array
            quantization_errors.append(value - nearest_mid)  # Calculate quantization error
            level_index = mid_points.index(nearest_mid)  # Find the index of the nearest midpoint
            binary_representations.append(level_to_binary[level_index])  # Get binary representation of the level

        print("Quantized Array:", quantized_array)
        print("Binary Representations:", binary_representations)

        x = self.signals[SigNumber][:, 0]
        result = np.column_stack((x, quantized_array))
        self.plot_quantized_signal(result, "Quantized Signal", "Using number of levels", mid_points,quantization_errors)
        QuantizationTest1("D:\\d\\fcis 2025\\pythonProject\\Quan1_Out.txt", binary_representations,
                          quantized_array)
#done
    def convolve_signals(self):
        FilterNumber = int(tk.simpledialog.askstring("Input", "Enter the Filter Number :"))
        FilterNumber -= 1

        SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
        SigNumber -= 1

        print("in if in convolve")
        range_of_out = (min(self.signals[SigNumber][:, 0]) + min(self.signals[FilterNumber][:, 0]),
                        max(self.signals[SigNumber][:, 0]) + max(self.signals[FilterNumber][:, 0]))
        x = self.signals[SigNumber][:, 1]
        h = self.signals[FilterNumber][:, 1]

        # The length of the output signal y[n]
        len_y = len_x + len_h - 1
        y = [0] * len_y  # Initialize output signal to zeros

        # Iterate through each index in y
        for n in range(len_y):
            # For the current position in y, calculate the sum of products
            for k in range(len_x):
                # Calculate the corresponding index in x
                h_index = n - k
                if 0 <= h_index < len_h:  # Ensure the x_index is valid
                    y[n] += h[h_index] * x[k]
        # Prepare result as a 2D array with indices
        output_indices = np.arange(range_of_out[0], range_of_out[1] + 1)
        result = np.column_stack((output_indices, y))  # Combine indices and y into a 2D array

        self.plot_sample_signal(result, "convolution", "convolved signal")
        print("in convlove after signal convolved ###################### ",result)
#done
    def smooth_signal(self):

        window_size = int(tk.simpledialog.askstring("Input", "Enter the window size:"))
        len_y=len(self.signals[0][:, 0])-window_size+1
        range_of_out = (min(self.signals[0][:, 0]) ,max(self.signals[0][:, 0])-( window_size-1))

        x=self.signals[0][:,1]
        sum_x=0
        y = [0] *len_y # Initialize output signal to zeros
        # Perform the moving average
        for i in range(len_y):
            sum_x = 0
            for j in range(window_size):
                sum_x += x[i + j]
            y[i] = sum_x / window_size
        output_indices = np.arange(range_of_out[0],range_of_out[1]+1)
        result = np.column_stack((output_indices, y))  # Combine indices and y into a 2D array
        self.plot_sample_signal(result, "Smoothed signal", "Smoothing")
#done
    def sharpening_signal(self):
        der_type = int(tk.simpledialog.askstring("Input", "Enter 1 for 1st derivative, 2 for 2nd derivative:"))
        x = self.signals[0][:, 1]
        len_x = len(x)

        # First derivative
        len_y = len_x - 1
        y = [0] * len_y
        for n in range(1, len_x):
            y[n - 1] = x[n] - x[n - 1]
        if der_type == 1:
            range_of_out = (min(self.signals[0][:, 0]) , max(self.signals[0][:, 0])-(1))
            output_indices = np.arange(range_of_out[0], range_of_out[1] + 1)
            result = np.column_stack((output_indices, y))
            self.plot_sample_signal(result, "First Derivative", "Sharpen Signal")

        elif der_type == 2:
            len_z = len_y - 1
            z = [0] * len_z
            for n in range(1, len_y):
                z[n - 1] = y[n] - y[n - 1]

            range_of_out = (min(self.signals[0][:, 0]) , max(self.signals[0][:, 0])-(2))
            output_indices = np.arange(range_of_out[0], range_of_out[1] + 1)
            result = np.column_stack((output_indices, z))
            self.plot_sample_signal(result, "Second Derivative", "Sharpen Signal")
#done
    def DFT_transform(self):
        SigNumber = int(tk.simpledialog.askstring("Input", "Enter the Signal Number:"))
        SigNumber -= 1

        frequency = int(tk.simpledialog.askstring("Input", "Enter frequency in HZ:"))

        x = self.signals[SigNumber][:, 1]
        signLen = int(len(x))
        y = [0] * signLen
        Amp = [0] * signLen
        X_axis = [0] * signLen
        PhaseShift = [0] * signLen
        elnatta = (2 * (math.pi) * frequency) / signLen

        for K in range(signLen):
            for N in range(signLen):
                exponent = -1j * 2 * cmath.pi * K * N / signLen
                y[K] += x[N] * cmath.exp(exponent)
            # Extract real and imaginary parts
            real_part = round(y[K].real,10)
            imag_part = round(y[K].imag,10)
            print(real_part)
            print(imag_part)
            tempPhase=imag_part/real_part
            PhaseShift[K]= math.atan2(imag_part,real_part)
            Amp[K]=math.sqrt(real_part**2+imag_part**2)
            X_axis[K]=K*elnatta
        y_cleaned = [complex(round(val.real, 10), round(val.imag, 10)) for val in y]
        print(y_cleaned)
        print(Amp)
        print(PhaseShift)
        print(X_axis)

        resultAmp = np.column_stack((X_axis, Amp))
        self.plot_sample_signal(resultAmp, "Amplitude graph", "DFT output 1")
        resultpha = np.column_stack((X_axis, PhaseShift))
        self.plot_sample_signal(resultpha, "Phase Shift graph", "DFT output 2")
        return Amp,PhaseShift
#done
    def IDFT_transform(self,amp,phase):
        if amp is None and phase is None:
            amp = self.signals[0][:, 0]
            phase = self.signals[0][:, 1]

        signLen=len(amp)
        y = [0] * signLen
        out= [0] * signLen
        imag_part = [0] * signLen
        real_part = [0] * signLen
        for k in range(signLen):
            imag_part[k]=amp[k]*math.sin(phase[k])
            real_part[k]=amp[k]*math.cos(phase[k])
            y[k]=complex(real_part[k],imag_part[k])

        y_cleaned = [complex(round(val.real, 10), round(val.imag, 10)) for val in y]


        print("y : ",y_cleaned)
        for N in range(signLen):
            for K in range(signLen):
                exponent = 1j * 2 * cmath.pi * K * N / signLen
                out[N] += y[K] * cmath.exp(exponent)
            out[N]/=signLen
        out_cleaned = [complex(round(val.real, 10), round(val.imag, 10)) for val in out]
        indices = list(range(signLen))  # Generate indices from 0 to len(signal)-1
        amplitudes = [round(abs(val), 10) for val in out]
        print(amplitudes)
        result = np.column_stack((indices, amplitudes))
        self.plot_sample_signal(result, "Resulted Signal", "IDFT output 1")
        print("result in IDFT : ",result)
#point 1 and 2 in task 8
    def cross_correlation(self):
            messagebox.showinfo("Success", f"Time Delay is equal to  {timeDelay} .")
#point 3 in task 8
    def correlation(self, signal_a, signal_b):
        return r

    def classify_signal(self, test_signal):
            return 2  # Class 2

    def process_and_classify_all_test_signals(self):
        """
        Process all test signals and classify them based on maximum correlation.
        """
        classified_signals = {}

        for filename, test_signal in self.test_signals:
            predicted_class = self.classify_signal(test_signal)
            classified_signals[filename] = predicted_class

        return classified_signals

    def load_all_signals(self, base_dir):
        # Directories for class 1, class 2, and test signals

        class_1_dir = os.path.join(base_dir, 'Class 1')
        class_2_dir = os.path.join(base_dir, 'Class 2')
        test_signals_dir = os.path.join(base_dir, 'Test Signals')

        # Helper function to load files from a given directory
        def load_files_from_directory(directory):
            signals = []
            for filename in os.listdir(directory):
                if filename.endswith('.txt'):
                    filepath = os.path.join(directory, filename)
                    signal = self.load_signal_from_file(filepath)  # Use the function to load a single file
                    signals.append((filename, signal))
            return signals

        # Load signals from all directories
        class_1_signals = load_files_from_directory(class_1_dir)
        class_2_signals = load_files_from_directory(class_2_dir)
        test_signals = load_files_from_directory(test_signals_dir)

        # Add signals to the respective lists
        self.class_1_signals = class_1_signals
        self.class_2_signals = class_2_signals
        self.test_signals = test_signals

        # Show a success message
        messagebox.showinfo("Success", "All signals from Class 1, Class 2, and Test Signals have been loaded.")

    def load_signal_from_file(self, filepath):
        """ Load a signal from a single file and return it as a numpy array """
        try:
            with open(filepath, 'r') as file:
                data = file.readlines()

            # Parse the signal data starting from the fourth line
            signal_values = [float(line.strip()) for line in data[0:]]

            # If you need time indices (assuming uniform sampling), generate them
            time_indices = np.arange(len(signal_values))

            # Combine time indices and signal values into a 2D array
            signal = np.column_stack((time_indices, signal_values))

            return signal

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while loading the signal from {filepath}: {e}")
            return None

    def correlation_for_classes(self):
        base_directory = r"D:\d\fcis 2025\pythonProject\Correlation Task Files\point3 Files"
        self.load_all_signals(base_directory)
        classified=self.process_and_classify_all_test_signals()
        print(classified)
        messagebox.showinfo("Success", f"After classification here is the output {classified} .")
    def low_pass_filter(self):
        self.filtering('LowPass')
    def high_pass_filter(self):
        self.filtering('HighPass')
    def band_pass_filter(self):
        self.filtering('BandPass')
    def band_reject_filter(self):
        self.filtering('BandReject')

    def get_window(self,window_type, index, N):
        if window_type == "rectangular":
            return 1
        elif window_type == "hanning":
            return 0.5 + (0.5 * math.cos((2 * math.pi * index) / N))
        elif window_type == "hamming":
            return 0.54 + (0.46 * math.cos((2 * math.pi * index) / N))
        elif window_type == "blackman-harris":
            return (0.42+ (0.5 * math.cos((2 * math.pi * index) / (N - 1)))
                    + 0.08 * (math.cos((4 * math.pi * index) / (N - 1)))
            )

    def get_filter_params(self,fs, stop, transition_band):

        if stop <= 21:
            window_type = "rectangular"
            N = math.ceil(0.9 * fs / transition_band)
        elif stop <= 44:
            window_type = "hanning"
            N = math.ceil(3.1 * fs / transition_band)
        elif stop <= 53:
            window_type = "hamming"
            N = math.ceil(3.3 * fs / transition_band)
        elif stop <= 74:
            window_type = "blackman-harris"
            N = math.ceil(5.5 * fs / transition_band)

        # Ensure odd filter length
        if N % 2 == 0:
            N += 1
        return window_type, N

    def filtering(self,filterType):

        input_win = tk.Toplevel(self.master)
        input_win.title(f"Generate {filterType.capitalize()} Signal")

        tk.Label(input_win, text="Sampling Frequency:").grid(row=0, column=0)
        FS_entry = tk.Entry(input_win)
        FS_entry.grid(row=0, column=1)

        tk.Label(input_win, text="Cut Off Frequency:").grid(row=1, column=0)
        Cut_off_entry = tk.Entry(input_win)
        Cut_off_entry.grid(row=1, column=1)

        tk.Label(input_win, text="Stop Attenuation:").grid(row=2, column=0)
        Stop_entry = tk.Entry(input_win)
        Stop_entry.grid(row=2, column=1)

        tk.Label(input_win, text="Transition Band:").grid(row=3, column=0)
        Transition_Band_Entry = tk.Entry(input_win)
        Transition_Band_Entry.grid(row=3, column=1)

        if(filterType=='LowPass'):
            def LowPass():
                # Retrieve values from Entry widgets using .get()
                fs = float(FS_entry.get())
                fc = float(Cut_off_entry.get())
                stop = float(Stop_entry.get())
                transition_band = float(Transition_Band_Entry.get())

                newFc = fc + transition_band / 2
                newFc /= fs

                window_type, N = self.get_filter_params(fs, stop, transition_band)
                n = (N - 1) // 2

                # initialize output signal
                h = [0] * N
                for val in range(-n, n + 1):
                    print("iteration # ", val)
                    window_value = self.get_window(window_type, val, N)
                    if val != 0:
                        h[val] = (2 * newFc) * (math.sin(val * 2 * math.pi * newFc) / (val * 2 * math.pi * newFc))
                    else:
                        h[val] = (2 * newFc)
                    h[val] *= window_value
                    print("h : ", h[val])
                output_indices = np.arange(-n, n + 1)
                k = n
                rotated_arr = h[-k:] + h[:-k]

                result = np.column_stack((output_indices, rotated_arr))  # Combine indices and y into a 2D array

                self.plot_sample_signal(result, " Low Pass Filter ", "Filtered signal")
                self.signals.append(result)
            # Add the filtering button
            tk.Button(input_win, text="Filtering", command=LowPass).grid(row=4, column=0, columnspan=2)
        if (filterType == 'HighPass'):
            def HighPass():
                # Retrieve values from Entry widgets using .get()
                fs = float(FS_entry.get())
                fc = float(Cut_off_entry.get())
                stop = float(Stop_entry.get())
                transition_band = float(Transition_Band_Entry.get())
                newFc = fc - transition_band / 2
                newFc /= fs

                window_type, N = self.get_filter_params(fs, stop, transition_band)
                n = (N - 1) // 2

                # initialize output signal
                h = [0] * N
                for val in range(-n, n + 1):
                    print("iteration # ", val)
                    window_value = self.get_window(window_type, val, N)
                    if val != 0:
                        h[val] = (-2 * newFc) * (
                                    math.sin(val * 2 * math.pi * newFc) / (val * 2 * math.pi * newFc))
                    else:
                        h[val] = 1-(2 * newFc)
                    h[val] *= window_value
                    print("h : ",h[val])

                output_indices = np.arange(-n, n + 1)
                k = n
                rotated_arr = h[-k:] + h[:-k]

                result = np.column_stack((output_indices, rotated_arr))  # Combine indices and y into a 2D array

                self.plot_sample_signal(result, " High Pass Filter ", "Filtered signal")

            # Add the filtering button
            tk.Button(input_win, text="Filtering", command=HighPass).grid(row=4, column=0, columnspan=2)
        if (filterType == 'BandPass'):
            tk.Label(input_win, text="Cut Off Frequency2:").grid(row=4, column=0)
            Cut_off2_entry = tk.Entry(input_win)
            Cut_off2_entry.grid(row=4, column=1)

            def BandPass():
                # Retrieve values from Entry widgets using .get()
                fs = float(FS_entry.get())
                fc1= float(Cut_off_entry.get())
                fc2 = float(Cut_off2_entry.get())
                stop = float(Stop_entry.get())
                transition_band = float(Transition_Band_Entry.get())

                newFc1 = fc1 - transition_band / 2
                newFc1 /= fs
                newFc2 = fc2 + transition_band / 2
                newFc2 /= fs

                window_type, N = self.get_filter_params(fs, stop, transition_band)
                n = (N - 1) // 2

                # initialize output signal
                h = [0] * N
                for val in range(-n, n + 1):
                    print("iteration # ", val)
                    window_value = self.get_window(window_type, val, N)
                    if val != 0:
                        if val != 0:
                            h[val] = (2 * newFc2 * (math.sin(val * 2 * math.pi * newFc2) / (val * 2 * math.pi * newFc2))
                                      - 2 * newFc1 * (
                                                  math.sin(val * 2 * math.pi * newFc1) / (val * 2 * math.pi * newFc1)))
                        else:
                            h[val] = 2 * (newFc2 - newFc1)
                    h[val] *= window_value
                    print("h : ",h[val])

                output_indices = np.arange(-n, n + 1)
                k = n
                rotated_arr = h[-k:] + h[:-k]

                result = np.column_stack((output_indices, rotated_arr))  # Combine indices and y into a 2D array

                self.plot_sample_signal(result, " Band Pass Filter ", "Filtered signal")

            # Add the filtering button
            tk.Button(input_win, text="Filtering", command=BandPass).grid(row=5, column=0, columnspan=2)
        if (filterType == 'BandReject'):
            tk.Label(input_win, text="Cut Off Frequency2:").grid(row=4, column=0)
            Cut_off2_entry = tk.Entry(input_win)
            Cut_off2_entry.grid(row=4, column=1)

            def BandReject():
                # Retrieve values from Entry widgets using .get()
                fs = float(FS_entry.get())
                fc1= float(Cut_off_entry.get())
                fc2 = float(Cut_off2_entry.get())
                stop = float(Stop_entry.get())
                transition_band = float(Transition_Band_Entry.get())

                newFc1 = fc1 + transition_band / 2
                newFc1 /= fs
                newFc2 = fc2 - transition_band / 2
                newFc2 /= fs

                window_type, N = self.get_filter_params(fs, stop, transition_band)
                n = (N - 1) // 2

                # initialize output signal
                h = [0] * N
                for val in range(-n, n + 1):
                    print("iteration # ", val)
                    window_value = self.get_window(window_type, val, N)
                    if val != 0:
                        if val != 0:
                            h[val] = (2 * newFc1 * (math.sin(val * 2 * math.pi * newFc1) / (val * 2 * math.pi * newFc1))
                                      - 2 * newFc2 * (
                                                  math.sin(val * 2 * math.pi * newFc2) / (val * 2 * math.pi * newFc2)))
                        else:
                            h[val] = 1-(2 * (newFc2 - newFc1))
                    h[val] *= window_value
                    print("h : ",h[val])

                output_indices = np.arange(-n, n + 1)
                k = n
                rotated_arr = h[-k:] + h[:-k]

                result = np.column_stack((output_indices, rotated_arr))  # Combine indices and y into a 2D array

                self.plot_sample_signal(result, " Band Reject Filter ", "Filtered signal")

            # Add the filtering button
            tk.Button(input_win, text="Filtering", command=BandReject).grid(row=5, column=0, columnspan=2)
    def apply_filter_direct(self):
        self.convolve_signals()
    def apply_filter_fast(self):

        ampFilter, phaseFilter = self.DFT_transform()
        ampSig,phaseSig=self.DFT_transform()

        new_length = len(ampFilter) + len(ampSig) - 1

        # Pad the amplitude
        ampFilter_padded = np.pad(ampFilter, (0, new_length - len(ampFilter)), mode='constant')
        ampSig_padded = np.pad(ampSig, (0, new_length - len(ampSig)), mode='constant')

        # Pad the phase
        phaseFilter_padded = np.pad(phaseFilter, (0, new_length - len(phaseFilter)), mode='constant')
        phaseSig_padded = np.pad(phaseSig, (0, new_length - len(phaseSig)), mode='constant')

        amp_result = [a1 * a2 for a1, a2 in zip(ampFilter_padded, ampSig_padded)]
        phase_result = [p1 + p2 for p1, p2 in zip(phaseFilter_padded, phaseSig_padded)]

        print("amplitude result in filter",amp_result)
        print("################################################################################")
        print(phase_result)
        self.IDFT_transform(amp_result,phase_result)


if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessor(root)
    root.mainloop()