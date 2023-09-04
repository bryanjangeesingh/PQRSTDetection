import numpy as np
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import neurokit2 as nk
import numpy as np
from scipy.signal import butter, filtfilt
import pywt


class ECG_Analyzer:
    def __init__(self, header, dat, lead_index):
        self.header = header
        self.dat = dat
        self.lead_index = lead_index
        self.pPeaks, self.qPeaks, self.rPeaks, self.sPeaks, self.tPeaks = (
            [],
            [],
            [],
            [],
            [],
        )

    def find_closest_value(self, a, b):
        """
        Find the value in array b that is closest to a value in array a and return the corresponding value from array a.

        Parameters:
        a (np.array): The array from which to return the value.
        b (np.array): The array in which to find the closest value.

        Returns:
        float: The value from array a that is closest to any value in array b.
        """
        # Find the index of the closest value in a for each value in b
        a = np.array(a)
        b = np.array(b)

        idx = (np.abs(a[:, None] - b)).argmin(axis=0)

        # Get the closest values in a
        closest_values = a[idx]

        # Find the value from a that is closest to any value in b
        result = closest_values[np.abs(closest_values - b).argmin()]

        return result

    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        return b, a

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def filter_data(self, data, cutoff=6, fs=128.00, order=6):
        # Extract x and y from the list of pairs
        x, y = zip(*data)

        # Filter the data.
        y_filtered = self.butter_lowpass_filter(y, cutoff, fs, order)

        # Zip the filtered y data with the same x data
        filtered_data = list(zip(x, y_filtered))

        return filtered_data

    def extractRPeaks(self):
        # Construct the file names
        hea_file = self.header
        dat_file = self.dat

        # Read the header file
        with open(hea_file, "r") as f:
            header = f.readlines()

        # Parse the header file to get necessary metadata
        num_leads = int(header[0].split()[1])
        num_samples = int(header[0].split()[3])
        sampling_frequency = float(header[0].split()[2])
        data_format = header[1].split()[1]

        # Extract the scaling factor and offset from the header file
        scaling_factor, offset = map(
            float, re.findall(r"[-+]?\d*\.\d+|\d+", header[1].split()[2])
        )

        # Read the data file
        if data_format == "16":
            dtype = np.int16
        elif data_format == "32":
            dtype = np.float32
        else:
            raise ValueError("Unknown data format")

        data = np.fromfile(dat_file, dtype=dtype)

        # Reshape the data
        data = data.reshape((num_samples, num_leads))

        # Convert the data to microvolts
        data = (data - offset) / scaling_factor

        # Get the index of Lead II
        lead_ii_index = self.lead_index  # adjust this if necessary

        # Extract the data for Lead II
        lead_ii_data = data[:, lead_ii_index]

        # Detect R-peaks using nk.ecg_peaks
        _, rpeaks = nk.ecg_peaks(lead_ii_data, sampling_rate=sampling_frequency)

        # Extract R-peaks indices
        rpeaks_indices = rpeaks["ECG_R_Peaks"]

        # Create a list of tuples of (time, amplitude) pairs for the R-peaks
        rpeak_times = rpeaks_indices / sampling_frequency
        rpeak_amplitudes = lead_ii_data[rpeaks_indices]
        rpeak_points = list(zip(rpeak_times, rpeak_amplitudes))

        return rpeak_points

    def extractData(self):
        # Construct the file names
        hea_file = self.header
        dat_file = self.dat

        # Read the header file
        with open(hea_file, "r") as f:
            header = f.readlines()

        # Parse the header file to get necessary metadata
        num_leads = int(header[0].split()[1])
        num_samples = int(header[0].split()[3])
        sampling_frequency = float(header[0].split()[2])
        data_format = header[1].split()[1]

        scaling_factor, offset = map(
            float, re.findall(r"[-+]?\d*\.\d+|\d+", header[1].split()[2])
        )

        # Read the data file
        if data_format == "16":
            dtype = np.int16
        elif data_format == "32":
            dtype = np.float32
        else:
            raise ValueError("Unknown data format")

        data = np.fromfile(dat_file, dtype=dtype)

        # Reshape the data
        data = data.reshape((num_samples, num_leads))

        data = (data - offset) / scaling_factor

        # Create a time array in seconds
        time = np.arange(num_samples) / sampling_frequency

        # Get the index of Lead II
        lead_ii_index = self.lead_index  # adjust this if necessary

        # Extract the data for Lead II
        lead_ii_data = data[:, lead_ii_index]

        # Create a list of tuples of (time, amplitude) pairs
        data_points = list(zip(time, lead_ii_data))

        return data_points

    def extractQPeaks(self, data_points, r_peaks):
        # Start getting the QPeaks
        qPeaksArr = []
        minVolt = float("inf")
        for _, voltage in data_points:
            minVolt = min(minVolt, voltage)

        # Get all the data point to the left of the Rpeaks - we will do a gradient descent approach
        for rPeakTime, rPeakVoltage in r_peaks:
            for idx, (time, voltage) in enumerate(data_points):
                if time == rPeakTime:
                    left_of_r_peak = data_points[0 : idx + 1]

                    left_of_r_peak_sorted = sorted(
                        left_of_r_peak, key=lambda x: x[0], reverse=True
                    )

                    # we need to rescale the data by getting the min voltage and adding it to all the voltages
                    rescaled = [
                        (time, voltage + abs(minVolt))
                        for time, voltage in left_of_r_peak_sorted
                    ]

                    for j, (currTime, currVoltage) in enumerate(rescaled[2:]):
                        # if voltage of the prev value is less than the current voltage, we found the q peak
                        if j > 0:
                            prevTime, prevVolt = rescaled[j - 1][0], rescaled[j - 1][1]
                            if prevVolt < currVoltage:
                                qPeak = (prevTime, prevVolt - abs(minVolt))
                                qPeaksArr.append(qPeak)
                                break
        return qPeaksArr

    def extractSPeaks(self, data_points, r_peaks):
        # Start getting the QPeaks
        sPeaksArr = []
        minVolt = float("inf")
        for _, voltage in data_points:
            minVolt = min(minVolt, voltage)

        # Get all the data point to the left of the Rpeaks - we will do a gradient descent approach
        for rPeakTime, rPeakVoltage in r_peaks:
            for idx, (time, voltage) in enumerate(data_points):
                if time == rPeakTime:
                    right_of_r_peak = data_points[idx:]

                    # we need to rescale the data by getting the min voltage and adding it to all the voltages
                    rescaled = [
                        (time, voltage + abs(minVolt))
                        for time, voltage in right_of_r_peak
                    ]

                    for j, (currTime, currVoltage) in enumerate(rescaled):
                        # if voltage of the prev value is less than the current voltage, we found the q peak
                        if j > 0:
                            prevTime, prevVolt = rescaled[j - 1][0], rescaled[j - 1][1]
                            if prevVolt < currVoltage:
                                sPeak = (prevTime, prevVolt - abs(minVolt))
                                sPeaksArr.append(sPeak)
                                break
        return sPeaksArr

    def fit_polynomial(self, coords, degree):
        """
        Fit a polynomial of specified degree to a list of coordinates.

        Parameters:
        coords (list): List of coordinates as tuples (x, y)
        degree (int): Degree of the polynomial

        Returns:
        np.poly1d: Fitted polynomial
        """
        # Unpack the coordinates
        x, y = zip(*coords)

        # Fit the polynomial
        p = np.polyfit(x, y, degree)
        polynomial = np.poly1d(p)

        derivative = polynomial.deriv()

        # Find the root(s) of the derivative
        roots = np.roots(derivative)

        # Filter the real roots
        real_roots = [root.real for root in roots if root.imag == 0]

        return polynomial, real_roots

    def find_local_maxima(self, data_points):
        """
        Find the local maxima in a list of data points.

        Parameters:
        data_points (list): List of data points as tuples (x, y)

        Returns:
        list: List of local maxima as tuples (x, y)
        """
        # Initialize the list of local maxima
        local_maxima = []

        # Iterate over the data points
        for i in range(1, len(data_points) - 1):
            # Get the current point and its neighbors
            prev_point = data_points[i - 1]
            curr_point = data_points[i]
            next_point = data_points[i + 1]

            # Check if the current point is a local maximum
            if curr_point[1] > prev_point[1] and curr_point[1] > next_point[1]:
                local_maxima.append(curr_point)

        return local_maxima

    def extractPPeaks(self, data_points, q_peaks, windowDistanceAwayFromP):
        lookup = {round(t, 2): v for t, v in data_points}
        reverseLookupVoltage = {v: t for t, v in data_points}
        arr = []
        data_points_rounded = [(round(x, 2), round(y, 2)) for x, y in data_points]
        data_points_rounded4 = [(round(x, 4), round(y, 4)) for x, y in data_points]

        for time, voltage in q_peaks:
            time, voltage = round(time, 4), round(voltage, 4)
            startTime = round(time - windowDistanceAwayFromP, 2)
            startVoltage = round(lookup[startTime], 2)

            windowStartIdx = data_points_rounded.index((startTime, startVoltage))
            windowEndIdx = data_points_rounded4.index((time, voltage))
            windowOfDataPoints = data_points[windowStartIdx : windowEndIdx + 1]

            filteredWindow = self.filter_data(windowOfDataPoints)

            # waveletDenoising = apply_wavelet_denoising(windowOfDataPoints)

            arr.append(filteredWindow)

        pPeaksArr = []
        for window in arr:
            localMax = self.find_local_maxima(window)
            maxTime = round(localMax[0][0] if localMax else 0, 2)
            pPeak = (maxTime, lookup[maxTime])
            pPeaksArr.append(pPeak)
            # p, roots = fit_polynomial(window, 2)
            # roots = round(roots[0], 2)
            # print(roots)
            # pPeak = (roots, lookup[roots])
            # pPeaksArr.append(pPeak)

        return pPeaksArr

    def extractTPeaks(self, data_points, s_peaks, windowDistanceAwayFromS):
        lookup = {round(t, 2): v for t, v in data_points}
        reverseLookupVoltage = {v: t for t, v in data_points}

        arr = []
        data_points_rounded = [(round(x, 2), round(y, 2)) for x, y in data_points]
        data_points_rounded4 = [(round(x, 4), round(y, 4)) for x, y in data_points]

        for time, voltage in s_peaks:
            time, voltage = round(time, 4), round(voltage, 4)
            endTime = round(time + windowDistanceAwayFromS, 2)
            if endTime >= 120:
                break
            endVoltage = round(lookup[endTime], 2)

            windowStartIdx = data_points_rounded4.index((time, voltage))
            windowEndIdx = data_points_rounded.index((endTime, endVoltage))
            windowOfDataPoints = data_points[windowStartIdx : windowEndIdx + 1]

            filteredWindow = self.filter_data(windowOfDataPoints)
            arr.append(filteredWindow)

        tPeaksArr = []
        for window in arr:
            # call the local max and fit_polynomial function.
            # find the two points that agree with each other and return the one generated by the localMax

            localMax = self.find_local_maxima(window)
            p, roots = self.fit_polynomial(
                window, 2
            )  # this can only ever have one stationary point since it's a quadratic
            # print(f'local max is{localMax} and roots is {roots}')
            if len(localMax) > 1:
                maxTime = self.find_closest_value([x for x, y in localMax], roots)
            else:
                maxTime = localMax[0][0] if localMax else 0

            maxTime = round(maxTime, 2)
            # print(f'the max time was chosen to be {maxTime}')

            # maxTime = round(localMax[0][0] if len(localMax) > 1 else localMax[0][0], 2)
            tPeak = (maxTime, lookup[maxTime])
            tPeaksArr.append(tPeak)

            # sometimes the ^ localMax function returns multiple local maxes and we can't decide which is the correct one. maybe we can use the method below to get a consenus of which value
            # to choose if the localMax functions returns more than one local max

            # p, roots = fit_polynomial(window, 2)
            # roots.sort()
            # maxTime = round(roots[0], 2)
            # tPeak = (maxTime, lookup[maxTime])
            # tPeaksArr.append(tPeak)

            # p, roots = fit_polynomial(window, 2)
            # print('roots', roots, 'localMax', localMax)

        return tPeaksArr

    def makePlot(
        self,
        saveImg=False,
        xStart=0,
        xEnd=5,
    ):
        # Construct the file names
        hea_file = self.header
        dat_file = self.dat

        # Read the header file
        with open(hea_file, "r") as f:
            header = f.readlines()

        # Parse the header file to get necessary metadata
        num_leads = int(header[0].split()[1])
        num_samples = int(header[0].split()[3])
        sampling_frequency = float(header[0].split()[2])
        data_format = header[1].split()[1]

        # Extract the scaling factor and offset from the header file
        scaling_factor, offset = map(
            float, re.findall(r"[-+]?\d*\.\d+|\d+", header[1].split()[2])
        )

        # Read the data file
        if data_format == "16":
            dtype = np.int16
        elif data_format == "32":
            dtype = np.float32
        else:
            raise ValueError("Unknown data format")

        data = np.fromfile(dat_file, dtype=dtype)

        # Reshape the data
        data = data.reshape((num_samples, num_leads))

        # Convert the data to microvolts
        data = (data - offset) / scaling_factor

        # Create a time array in seconds
        time = np.arange(num_samples) / sampling_frequency

        lead_ii_index = self.lead_index  # adjust this if necessary

        # Plot the ECG data
        plt.figure(figsize=(12, 6))
        plt.plot(time, data[:, lead_ii_index], label="Lead II")

        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude (uV)")
        plt.title(f"ECG Data")
        plt.legend()
        plt.xlim(xStart, xEnd)

        plt.grid(True)

        x, y = zip(*self.rPeaks)
        plt.scatter(x, y, color="red")
        for i, txt in enumerate(y):
            plt.annotate("R", (x[i], y[i]))

        x1, y1 = zip(*self.qPeaks)
        plt.scatter(x1, y1, color="Green")
        for i, txt in enumerate(y1):
            plt.annotate("Q", (x1[i], y1[i]))

        x2, y2 = zip(*self.sPeaks)
        plt.scatter(x2, y2, color="purple")
        for i, txt in enumerate(y2):
            plt.annotate("S", (x2[i], y2[i]))

        x3, y3 = zip(*self.tPeaks)
        plt.scatter(x3, y3, color="orange")
        for i, txt in enumerate(y3):
            plt.annotate("T", (x3[i], y3[i]))

        x4, y4 = zip(*self.pPeaks)
        plt.scatter(x4, y4, color="orange")
        for i, txt in enumerate(y4):
            plt.annotate("P", (x4[i], y4[i]))

        if saveImg:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            new_directory = "myECGPlots"
            path = os.path.join(current_directory, new_directory)
            # check if the directory already exists
            if not os.path.exists(path):
                os.mkdir(path)
                print(f"Directory '{new_directory}' created")

            plt.savefig(os.path.join(output_dir))

        plt.show()

    def getAllPeaks(self, plot=False):
        r_peaks = self.extractRPeaks()
        data_points = self.extractData()
        qPeaksArr = self.extractQPeaks(data_points, r_peaks)
        sPeaksArr = self.extractSPeaks(data_points, r_peaks)
        pPeaksArr = self.extractPPeaks(data_points, qPeaksArr, 0.2)
        tPeaksArr = self.extractTPeaks(
            data_points, sPeaksArr, 0.3
        )  # <- we need to figure out how to adjust this based on the waveform (0.4)

        self.pPeaks, self.qPeaks, self.rPeaks, self.sPeaks, self.tPeaks = (
            pPeaksArr,
            qPeaksArr,
            r_peaks,
            sPeaksArr,
            tPeaksArr,
        )

        if plot:
            self.makePlot()

        output = {
            "P_Peaks": pPeaksArr,
            "Q_Peaks": qPeaksArr,
            "R_Peaks": r_peaks,
            "S_Peaks": sPeaksArr,
            "T_Peaks": tPeaksArr,
        }

        return output


header = "/Users/bryanjangeesingh/Downloads/brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0/31.hea"
dat = "/Users/bryanjangeesingh/Downloads/brno-university-of-technology-ecg-signal-database-with-annotations-of-p-wave-but-pdb-1.0.0/31.dat"
lead = 0
myEcg = ECG_Analyzer(header, dat, lead)

myDict = myEcg.getAllPeaks(plot=True)
