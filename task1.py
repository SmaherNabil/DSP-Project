import tkinter as tk
from signal import signal
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from numpy.ma.core import power

from QuanTest1 import QuantizationTest1


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


        menu2button = tk.Menubutton(master, text="Quantize Signal", relief=tk.RAISED)
        # Create the menu for the menubutton
        menu2button.menu = tk.Menu(menu2button, tearoff=0)
        menu2button["menu"] = menu2button.menu
        # Add checkbuttons enter levels or enter number of bits
        menu2button.menu.add_command(label="Enter Number Of Levels", command=self.using_number_of_levels)
        menu2button.menu.add_command(label="Enter Number Of Bits", command=self.using_number_of_bits)
        # Pack the Menubutton
        menu2button.pack()

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
    def plot_quantized_signal(self, result, label, title,mid_points,quantization_errors):
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

        # Map each value in amplitude_values to the nearest midpoint and corresponding level's binary representation
        for value in amplitude_values:
            nearest_mid = min(mid_points, key=lambda x: abs(x - value))
            quantized_array.append(nearest_mid)  # Append the nearest midpoint to quantized_array
            quantization_errors.append(value - nearest_mid)  # Calculate quantization error
            level_index = mid_points.index(nearest_mid)  # Find the index of the nearest midpoint
            binary_representations.append(level_to_binary[level_index])  # Get binary representation of the level

        print("Quantized Array:", quantized_array)
        print("Binary Representations:", binary_representations)

        # Combine x values and quantized values for plotting
        x = self.signals[SigNumber][:, 0]
        result = np.column_stack((x, quantized_array))
        self.plot_quantized_signal(result, "Quantized Signal", "Using number of levels", mid_points,
                                   quantization_errors)


        QuantizationTest1("D:\\d\\fcis 2025\\pythonProject\\Quan1_Out.txt", binary_representations,
                          quantized_array)
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
            nearest_mid = min(mid_points, key=lambda x: abs(x -  value))
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
if __name__ == "__main__":
    root = tk.Tk()
    app = SignalProcessor(root)
    root.mainloop()
