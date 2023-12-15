import numpy as np
from admm_ss import admm_ss
from scipy import signal
import funs
import scipy.fftpack as pyfft
import matplotlib.pyplot as plt
from scipy.signal import stft

# Data folder
data_folder = "3-9-23/"

# ###################################### #
# ########### scene settings ########### #
# ###################################### #

# scene settings
num_ula_mics = 16
num_sma_spheres = 5
sphere_subdivisions = 9
num_sma_mics = num_sma_spheres*sphere_subdivisions
selected_source = 1
speed_of_sound = 343.0

# ir settings
samplerate_raw = 16000
ir_length = 7200  # num samples in time

# mics' configurations: 'a', 'b', 'c', 'd', 'e'
configuration = 'd'
mics_to_plot = [2, 6, 10, 14, 20, 29, 38, 47, 56]  # arbitrary

# ######################################## #
# ########### acquire the data ########### #
# ######################################## #

# Get room_dims, source pos, mics pos, IRs
[room_dims, walls_pos] = funs.dimensions_and_walls(data_folder)
pos_src = funs.get_source_pos(data_folder, selected_source)
pos_all = funs.get_all_mics_pos(data_folder, num_ula_mics, num_sma_mics, num_sma_spheres,
                                sphere_subdivisions)
ir_all = funs.get_all_irs(data_folder, num_ula_mics, num_sma_mics, ir_length, selected_source,
                          num_sma_spheres, sphere_subdivisions)

# Translate the center of the room to (0,0,0)
new_center = np.array([0, 0, 0])
new_pos_src, new_pos_all, new_walls_pos = funs.translate_points_from_room_center_to_origin(
    room_dims, new_center, pos_src, pos_all, walls_pos)

# Microphone positions of all signals
pos_eval = new_pos_all
# Microphone positions of mics input
pos_mic = funs.build_pos_or_sig_mic(configuration, pos_eval, num_ula_mics, num_sma_spheres, sphere_subdivisions)

# ############################################ #
# ########### setup D and W spaces ########### #
# ############################################ #

# Region Omega: 1x1m around the source
# with 13x13 grid points (aka origins of the monopoles)
# each point is 0.077m from each other
num_of_monopoles_per_row_column = 13
side_length_m = 1
monopoles_positions = funs.setup_omega_region_monopole_pos(
    new_pos_src, num_of_monopoles_per_row_column, num_of_monopoles_per_row_column, side_length_m)

# uniformly sampled plane waves from 0 to 2pi rad are used as W
radius = max(room_dims)/2 + 1
num_of_pws = 127
plane_waves_pos = funs.setup_plane_waves_pos(new_center, radius, num_of_pws)

# ######################################### #
# ########### Matrix dimensions ########### #
# ######################################### #

# M is the number of mic in analysis, rows for D
M = (np.shape(pos_mic))[0]
# N is the number of monopoles. columns for D and W. much larger than M
N = (np.shape(monopoles_positions))[0]
# L is the number of plane waves to use, columns for W
L = (np.shape(plane_waves_pos))[0]
# T
T = 1
# M_new number of mics in synthesis
M_synth = (np.shape(mics_to_plot))[0]

# ##################################################################### #
# ########### 3D plot of room, src, mics, monopoles and pws ########### #
# ##################################################################### #

# plot of the scene
funs.plot_scene(pos_eval, pos_mic, new_pos_src, mics_to_plot, new_walls_pos,
                configuration, monopoles_positions, plane_waves_pos, new_center)

# ############################################################## #
# ########## build input signals from given mic recs ########### #
# ############################################################## #

# Down sampling of input signals
downSampling = 8  # 2
ir_eval = signal.resample_poly(ir_all, up=1, down=downSampling, axis=-1)
samplerate = samplerate_raw // downSampling

# Analysis signal
sig_mic = funs.build_pos_or_sig_mic(configuration, ir_eval, num_ula_mics, num_sma_spheres, sphere_subdivisions)
num_time_samples = (np.shape(sig_mic))[1]  # = ir_length

# Build the stft matrix of all analysis mics
num_of_freqs = 129  # must be len(freqs_stft) - 1
stft_of_sigs_freq_dom = np.zeros((M, num_of_freqs), dtype=complex)
freqs_stft = 0
time_stft = 0

# Time axis for plots
dt_plot = 1.0 / samplerate
t_plots = np.linspace(0.0, num_time_samples * dt_plot, num_time_samples)

for ir_idx in range(M):
    b_time_plot = sig_mic[ir_idx, :]
    freqs_stft, time_stft, Zxx = stft(b_time_plot, samplerate, nperseg=256)
    freqs_at_time_zero = Zxx[:, 0]
    freq_content_of_curr_mic = np.asarray(freqs_at_time_zero.reshape((T, num_of_freqs)))
    stft_of_sigs_freq_dom[ir_idx, :] = freq_content_of_curr_mic

    # visualize the analysis signals
    funs.plot_time_and_freq_domains_of_a_rir_version_2(t_plots, b_time_plot, time_stft, freqs_stft, Zxx, ir_idx, 'a')


# ######################################## #
# ########## execute the method ########## #
# ######################################## #

mu = 0.6  # at page 7, near (36)
MAX_ITER = 200

# Initialize the matrix of built signals
# Freq axis for the method
k_bins = freqs_stft[1:]
num_of_freqs = (np.shape(k_bins))[0]
synthesized_sig_freq_dom = np.zeros((M_synth, num_of_freqs), dtype=complex)
X = np.zeros((N, num_of_freqs), dtype=complex)

# pos of synth signals to plot
pos_mic_to_plot = np.zeros((M_synth, 3))
for idx_pos, idx_mic in enumerate(mics_to_plot):
    pos_mic_to_plot[idx_pos, :] = pos_eval[idx_mic]

# method loop
for bin_num_idx, k_bin_value in enumerate(k_bins):
    print("curr bin: " + str(bin_num_idx))

    # ### ANALYSIS ### #
    dict_of_greens_fun_analysis = funs.populate_greens_function_dictionary(
                                  M, N, monopoles_positions, pos_mic, speed_of_sound, k_bin_value)
    dict_of_plane_waves_analysis = funs.populate_plane_waves_dictionary(
                                   M, L, plane_waves_pos, pos_mic, new_center, speed_of_sound, k_bin_value)
    b = np.asarray((stft_of_sigs_freq_dom[:, bin_num_idx]).reshape((M, T)))

    # ### CALL THE METHOD FOR THE CURR FREQ COLUMN ### #
    x_res, u_res, num_it = admm_ss(dict_of_greens_fun_analysis, dict_of_plane_waves_analysis, b, mu, MAX_ITER)

    # ### CHECK THAT ONLY ONE MONOPOLE IS ACTIVE AT ALL FREQUENCIES ### #
    monopoles_at_curr_freq = x_res.flatten()
    X[:, bin_num_idx] = monopoles_at_curr_freq
    active_monopole = np.argmax(monopoles_at_curr_freq)
    print("active monopole: " + str(active_monopole))

    # ### SYNTHESIS ### #
    dict_of_greens_fun_synthesis = funs.populate_greens_function_dictionary(
                               M_synth, N, monopoles_positions, pos_mic_to_plot, speed_of_sound, k_bin_value)
    dict_of_plane_waves_synthesis = funs.populate_plane_waves_dictionary(
                                M_synth, L, plane_waves_pos, pos_mic_to_plot, new_center, speed_of_sound, k_bin_value)
    resulting_weights_freq_dom = (dict_of_greens_fun_synthesis @ x_res +
                                  dict_of_plane_waves_synthesis @ u_res)  # resulting_weights_freq_dom is an M_new x T

    synthesized_sig_freq_dom[:, bin_num_idx] = resulting_weights_freq_dom.flatten()

    # if bin_num_idx == 30:
    #     break

monopoles_contribution = np.sum(np.abs(X), axis=1)
active_monopole = np.argmax(monopoles_contribution)
print("activ mon " + str(active_monopole))

plt.imshow(np.abs(X), cmap='viridis', interpolation='nearest', aspect='auto')
cbar = plt.colorbar()
cbar.set_label('active monopole per freq')
plt.xlabel('k bin')
plt.ylabel('monopoles')
plt.show()

# return to the time domain with ifft by row
synthesized_signals_at_new_mics = pyfft.ifft(synthesized_sig_freq_dom, num_time_samples)

print("matrix of synthesized signals (freq dom): " + str(np.shape(synthesized_sig_freq_dom)))
print("matrix of synthesized signals (time dom): " + str(np.shape(synthesized_signals_at_new_mics)))

for ir_idx, actual_mic_id in enumerate(mics_to_plot):
    y_time_plot = np.real(synthesized_signals_at_new_mics[ir_idx, :])
    # y_freq_plot = synthesized_sig_freq_dom[ir_idx, :]
    freqs_stft, time_stft, Zxx = stft(y_time_plot, samplerate, nperseg=256)
    funs.plot_time_and_freq_domains_of_a_rir_version_2(
        t_plots, y_time_plot, time_stft, freqs_stft, Zxx, ir_idx, 's')

# todo: check which monopole is the active one
# todo: anechoic
# todo: remove down sampling (higher freq cap and more bins) -> move to the machines
# todo: solve amplitude escalation
# todo: put pw in 3D (sphere around the room), increase the population of pw, make the pw 3d
# todo: mse check with ground signals

print("end")
