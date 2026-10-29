import mne

import moabb
from moabb.datasets import Cho2017

import numpy as np
import matplotlib.pyplot as plt


def mu_power(epochs, tmin, tmax):
    spectrum = epochs.copy().crop(tmin, tmax).compute_psd(
        method="welch",
        fmin=8,
        fmax=13,
        n_fft=512,
        n_overlap=256
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    return np.trapezoid(psd, freqs, axis=-1)


if __name__ == "__main__":
    dataset = Cho2017()

    print(dataset.subject_list)  # list(range(1, 53))

    # files = dataset.data_path(subject=1) # downloads from s3
    # print(files)


    data = dataset.get_data(subjects=[1, 2, 3])
    # data = {"subject_id": {"session_id": {"run_id": run}}}

    print(data.keys())  # dict_keys([1])
    print(data[1].keys())  # dict_keys(['0']) # 1 session
    print(data[1]["0"].keys())  # dict_keys(['0']) # 6 combined runs

    raw = data[3]["0"]["0"]

    print(raw)  # <RawArray | 69 x 717300 (1401.0 s), ~377.7 MiB, data loaded>
    print(raw.get_data().shape)  # (69, 717300)

    for name, channel_type in zip(raw.ch_names, raw.get_channel_types()):
        print(name, channel_type)  # eeg, emg, stim


    # find event onsets by delta(Stim) > 0: [0, 0, 1, 0, 0, 0, 2, 0, 0, 0]
    events = mne.find_events(
        raw,
        stim_channel="Stim",
        initial_event=True
    )
    print(events.shape)  # [200, 3]: [[sample_index, previous_value, event_id]...]
    print(events[:10])


    epochs = mne.Epochs(
        raw,
        events,
        event_id={
            "left_hand": 1,
            "right_hand": 2
        },
        tmin=0,
        tmax=3 - 1 / raw.info["sfreq"],  # exclude last sample, 3 - 1/512 = 2.998s
        picks="eeg",
        baseline=None,
        preload=True
    )

    print(epochs) # 200 events: 100 left, 100 right hand
    print(epochs.get_data().shape) # [200, 64, 512*3=1536]


    spectrum = epochs.compute_psd(
        method="welch",
        picks=["C3", "C4"],
        fmin=1,
        fmax=35,
        n_fft=512,
        n_per_seg=512,
        n_overlap=256 # 5 windows average
    )
    psd, freqs = spectrum.get_data(return_freqs=True)
    print("PSD:", psd.shape) # (200, 2, 35)
    print("Frequencies:", freqs.shape) # (35,)


    labels = epochs.events[:, 2] # [[sample_index, previous_value, event_id]...]

    left_psd = psd[labels == 1] # all trials where label=1 (left)
    right_psd = psd[labels == 2]

    print(left_psd.shape) # (100, 2, 35)
    print(right_psd.shape) # (100, 2, 35)

    mu = (freqs >= 8) & (freqs <= 13) # boolean mask, freq in [8; 13]

    for i, channel in enumerate(["C3", "C4"]):
        # C3 - left, C4 - right

        # e.g. on C3 average amplitude of right hand trials should be lower,
        # i.e. amplitude of mu on C3 on right hand trials should be lower,
        # and amplitude of mu on C4 on left hand trials should be lower
        left_mean = left_psd[:, i, :].mean(axis=0)
        right_mean = right_psd[:, i, :].mean(axis=0)

        left_mu = np.trapezoid(left_mean[mu], freqs[mu])
        right_mu = np.trapezoid(right_mean[mu], freqs[mu])

        print(f"{channel}:")
        print(f"  Left Mu: {left_mu * 1e12:.2f} µV²")
        print(f"  Right Mu: {right_mu * 1e12:.2f} µV²")

        plt.figure(figsize=(10, 4))

        plt.plot(freqs, left_mean * 1e12, label="Left Hand")
        plt.plot(freqs, right_mean * 1e12, label="Right Hand")

        plt.axvspan(8, 13, alpha=0.15, color="green")
        plt.title(f"{channel} — Motor Imagery PSD")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (µV²/Hz)")
        plt.legend()
        plt.grid()
        plt.show()


    left = epochs["left_hand"]

    signal = left.get_data(picks=["C3"])
    print(signal.shape)  # (100, 1, 1536)
    signal = signal[0, 0]
    print(signal.shape)  # (1536)
    time = left.times  # [0.0, ..., 2.998]
    print(time)

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title("Left Hand Motor Imagery (C3)")
    plt.grid()
    plt.show()

    # Filter raw EEG
    filtered = raw.copy()
    filtered.set_eeg_reference("average")
    filtered.filter(l_freq=1, h_freq=35, picks="eeg")

    # Create filtered epochs
    filtered_epochs = mne.Epochs(
        filtered,
        events,
        event_id={"left_hand": 1, "right_hand": 2},
        tmin=-1.5,
        tmax=3 - 1 / raw.info["sfreq"],
        picks="eeg",
        baseline=None,
        preload=True
    )

    # First left-hand trial, C3
    before = signal
    after = filtered_epochs["left_hand"].get_data(picks=["C3"])[0, 0]

    plt.figure(figsize=(12, 4))

    plt.plot(time, before, label="Original")
    plt.plot(time, after, label="Filtered (1–35 Hz)")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    plt.title("Left Hand Motor Imagery (C3)")
    plt.legend()
    plt.grid()
    plt.show()
