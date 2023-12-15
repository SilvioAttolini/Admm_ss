import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
from scipy import signal
sys.path.append('../')


# ######################################## #
# ########## scene elems. setup ########## #
# ######################################## #


def get_source_pos(data_folder, selected_source):
    sources_data = data_folder + "sourcePositionsSimAttolini.mat"
    sources_positions = scipy.io.loadmat(sources_data)
    sources_pos = sources_positions['src_pos']
    selected_source_pos = sources_pos[selected_source, :]

    return selected_source_pos


def get_all_mics_pos(data_folder, num_ula_mics, num_sma_mics, num_sma_spheres, sphere_subdivisions):

    tot_num_mics = num_ula_mics + num_sma_mics
    all_mics_pos = np.zeros((tot_num_mics, 3))
    curr_mic = 0

    ulas_data = data_folder + "ULAmicsPositionsSimAttolini.mat"
    ulas_mics_positions = scipy.io.loadmat(ulas_data)
    ulas_mics_pos = ulas_mics_positions['mic_array']

    for i in range(0, num_ula_mics):
        all_mics_pos[i, :] = ulas_mics_pos[i, :]
        curr_mic += 1

    smas_data = data_folder + "SMAmicsPositionsSimAttolini.mat"
    smas_mics_positions = scipy.io.loadmat(smas_data)
    smas_mics_pos = smas_mics_positions['all_SMA_mic_positions']

    for i in range(0, num_sma_spheres):
        for j in range(0, sphere_subdivisions):
            all_mics_pos[curr_mic, :] = smas_mics_pos[i, j, :]
            curr_mic = curr_mic + 1

    return all_mics_pos


def get_all_irs(data_folder, num_ula_mics, num_sma_mics, ir_length, selected_source,
                num_sma_spheres, sphere_subdivisions):

    tot_num_mics = num_ula_mics + num_sma_mics
    all_irs = np.zeros((tot_num_mics, ir_length))
    curr_rir = 0

    rirs_data = data_folder + "ULArirsSimAttolini.mat"
    ula_rirs_full = scipy.io.loadmat(rirs_data)
    ula_rirs = ula_rirs_full['h_rir']

    for i in range(0, num_ula_mics):
        all_irs[i, :] = ula_rirs[i, :, selected_source]
        curr_rir += 1

    smirs_data = data_folder + "SMAsmirsSimAttolini.mat"
    sma_smirs_full = scipy.io.loadmat(smirs_data)
    sma_smirs = sma_smirs_full['all_h_smir']

    for i in range(0, num_sma_spheres):
        for j in range(0, sphere_subdivisions):
            all_irs[curr_rir, :] = sma_smirs[i, j, :, selected_source]
            curr_rir += 1

    return all_irs


def translate_points_from_room_center_to_origin(room_dims, new_center, old_src_pos, old_mic_pos, old_walls_pos):
    old_center = np.array([room_dims[0]/2, room_dims[1]/2, 1])  # 1 is the z coord of the mics

    # Calculate the translation vector
    translation_vector = new_center - old_center

    # Apply the translation to each point
    moved_src_pos = old_src_pos + translation_vector
    moved_mic_pos = old_mic_pos + translation_vector
    moved_walls_pos = old_walls_pos + translation_vector

    return moved_src_pos, moved_mic_pos, moved_walls_pos


def build_pos_or_sig_mic(configuration, pos_or_sig_eval, num_ula_mics, num_sma_spheres, sphere_subdivisions):
    if configuration == 'a':
        mic_index_1 = 0
        number_of_input_mics = num_ula_mics
        mic_index_2 = mic_index_1 + number_of_input_mics
        pos_or_sig_mic = np.append(pos_or_sig_eval[:mic_index_1, :], pos_or_sig_eval[mic_index_2:, :], axis=0)
    elif configuration == 'b':
        mic_index_1 = num_ula_mics
        number_of_input_mics = sphere_subdivisions * num_sma_spheres
        mic_index_2 = mic_index_1 + number_of_input_mics
        pos_or_sig_mic = np.append(pos_or_sig_eval[:mic_index_1, :], pos_or_sig_eval[mic_index_2:, :], axis=0)
    elif configuration == 'c':
        mic_index_1 = num_ula_mics + sphere_subdivisions * 2
        number_of_input_mics = sphere_subdivisions
        mic_index_2 = mic_index_1 + number_of_input_mics
        pos_or_sig_mic = np.append(pos_or_sig_eval[:mic_index_1, :], pos_or_sig_eval[mic_index_2:, :], axis=0)
    elif configuration == 'd':
        step = 4
        pos_or_sig_mic = pos_or_sig_eval[::step]
    elif configuration == 'e':
        step = 7
        pos_or_sig_mic = pos_or_sig_eval[::step]
    else:
        pos_or_sig_mic = pos_or_sig_eval
        print("config error")

    return pos_or_sig_mic


def dimensions_and_walls(data_folder):
    room_data = data_folder + "roomDimensionsSimAttolini.mat"
    room_d = scipy.io.loadmat(room_data)
    room_dims = (room_d["room_dim"])[0]

    step = 1  # m
    xs = np.arange(0, room_dims[0] + step, step)
    ys = np.arange(0, room_dims[1] + step, step)
    zs = np.arange(0, room_dims[2] + step, step)
    walls_pos = np.zeros((len(xs), len(ys), len(zs), 3))

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                walls_pos[i, j, k, :] = np.array([x, y, z])

    walls_pos[1:-1, 1:-1] = np.array([0, 0, 0])

    return room_dims, walls_pos


# ############################################ #
# ########### setup D and W spaces ########### #
# ############################################ #


def setup_omega_region_monopole_pos(pos_src, num_of_grid_points_x, num_of_grid_points_y, side_length):
    step_x = side_length / num_of_grid_points_x
    step_y = side_length / num_of_grid_points_y
    top_left_corner = pos_src + np.array([-(side_length/2-step_x/2), side_length/2-step_y/2, 0])
    monopole_pos_matrix = np.zeros((num_of_grid_points_x, num_of_grid_points_y, 3))

    for row in range(num_of_grid_points_x):
        for col in range(num_of_grid_points_y):
            monopole_pos_matrix[row, col, :] = top_left_corner + np.array([step_x*row, -step_y*col, 0])
    monopole_pos_array = monopole_pos_matrix.reshape(-1, monopole_pos_matrix.shape[-1])
    # print(monopole_pos_array)

    return monopole_pos_array


def setup_plane_waves_pos(center, radius, num_of_pw):
    angles = np.linspace(0, 2 * np.pi * (1 - 1/num_of_pw), num_of_pw)  # extend to 3D

    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    z = center[2]
    pw_pos = np.zeros((num_of_pw, 3))
    for circle_p in range(num_of_pw):
        pw_pos[circle_p, :] = np.array([x[circle_p], y[circle_p], z])
    return pw_pos


# ##################################################################### #
# ########### 3D plot of room, src, mics, monopoles and pws ########### #
# ##################################################################### #


def plot_scene(pos_eval, pos_mic, pos_src, mics_to_plot, walls_pos, config, mon_pos, pw_pos, center):

    pos_mic_to_plot = np.zeros((len(mics_to_plot), 3))
    for idx_pos, idx_mic in enumerate(mics_to_plot):
        pos_mic_to_plot[idx_pos, :] = pos_eval[idx_mic]

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    ax_3d.scatter(pos_eval[:, 0], pos_eval[:, 1], pos_eval[:, 2], marker='.', color='cyan', label="pos eval")
    ax_3d.scatter(pos_mic[:, 0], pos_mic[:, 1], pos_mic[:, 2], marker='o', color='green', label="pos mic in")
    ax_3d.scatter(pos_mic_to_plot[:, 0], pos_mic_to_plot[:, 1], pos_mic_to_plot[:, 2],
                  marker='*', color='black', label="plotted mics")
    ax_3d.scatter(pos_src[0], pos_src[1], pos_src[2], marker='s', color='red', label="source pos")
    ax_3d.scatter(walls_pos[:, :, :, 0], walls_pos[:, :, :, 1], walls_pos[:, :, :, 2],
                  marker='x', color='black', label="walls")
    ax_3d.scatter(mon_pos[:, 0], mon_pos[:, 1], mon_pos[:, 2], marker='.', color='orange', label="monopoles")
    ax_3d.scatter(pw_pos[:, 0], pw_pos[:, 1], pw_pos[:, 2], marker='.', color='grey', label="plane waves")
    for i, p in enumerate(pw_pos):
        plt.plot([center[0], p[0]], [center[1], p[1]], [center[2], p[2]], linestyle='-', color='black', alpha=0.2)
    ax_3d.set_xlabel('X-axis')
    ax_3d.set_ylabel('Y-axis')
    ax_3d.set_zlabel('Z-axis')
    ax_3d.legend()
    ax_3d.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_3d.set_title('Room set-up, configuration: \'' + config + '\'')
    plt.show()

    # Plot in 2D (XY-plane)
    fig, ax_xy = plt.subplots()
    ax_xy.scatter(pos_eval[:, 0], pos_eval[:, 1], marker='.', color='cyan', label="pos eval")
    ax_xy.scatter(pos_mic[:, 0], pos_mic[:, 1], marker='o', color='green', label="pos mic in")
    ax_xy.scatter(pos_mic_to_plot[:, 0], pos_mic_to_plot[:, 1],  marker='*', color='black', label="plotted mics")
    ax_xy.scatter(pos_src[0], pos_src[1], marker='s', color='red', label="source pos")
    ax_xy.scatter(walls_pos[:, :, :, 0], walls_pos[:, :, :, 1], marker='x', color='black', label="walls")
    ax_xy.scatter(mon_pos[:, 0], mon_pos[:, 1], marker='.', color='orange', label="monopoles")
    ax_xy.scatter(pw_pos[:, 0], pw_pos[:, 1], marker='.', color='grey', label="plane waves")
    for i, p in enumerate(pw_pos):
        plt.plot([center[0], p[0]], [center[1], p[1]], linestyle='-', color='black', alpha=0.2)
    ax_xy.set_xlabel('X-axis')
    ax_xy.set_ylabel('Y-axis')
    ax_xy.set_aspect('equal')
    ax_xy.legend()
    ax_xy.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    ax_xy.set_title('View from Z-axis')
    plt.show()

    # Plot in 2D mic only
    fig, ax_xy = plt.subplots()
    ax_xy.scatter(pos_eval[:, 0], pos_eval[:, 1], marker='.', color='cyan', label="pos eval")
    ax_xy.scatter(pos_mic[:, 0], pos_mic[:, 1], marker='o', color='green', label="pos mic in")
    ax_xy.scatter(pos_mic_to_plot[:, 0], pos_mic_to_plot[:, 1], marker='*', color='black', label="plotted mics")
    ax_xy.scatter(pos_src[0], pos_src[1], marker='s', color='red', label="source pos")
    # ax_xy.scatter(walls_pos[:, :, :, 0], walls_pos[:, :, :, 1], marker='x', color='black', label="walls")
    ax_xy.scatter(mon_pos[:, 0], mon_pos[:, 1], marker='.', color='orange', label="monopoles")
    # ax_xy.scatter(pw_pos[:, 0], pw_pos[:, 1], marker='.', color='grey', label="plane waves")
    # for i, p in enumerate(pw_pos):
    #     plt.plot([center[0], p[0]], [center[1], p[1]], linestyle='-', color='black', alpha=0.2)
    ax_xy.set_xlabel('X-axis')
    ax_xy.set_ylabel('Y-axis')
    ax_xy.set_aspect('equal')
    ax_xy.legend()
    ax_xy.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    ax_xy.set_title('mics')
    plt.show()

    # Plot in 2D (YZ-plane)
    # fig_yz = plt.figure()
    # ax_yz = fig_yz.add_subplot(111)
    # ax_yz.scatter(pos_eval[:, 1], pos_eval[:, 2], marker='.', color='cyan', label="pos eval")
    # ax_yz.scatter(pos_mic[:, 1], pos_mic[:, 2], marker='o', color='green', label="pos mic in")
    # ax_yz.scatter(pos_mic_to_plot[:, 1], pos_mic_to_plot[:, 2], marker='*', color='black', label="plotted mics")
    # ax_yz.scatter(pos_src[1], pos_src[2], marker='s', color='red', label="source pos")
    # ax_yz.scatter(walls_pos[:, :, :, 1], walls_pos[:, :, :, 2], marker='x', color='black', label="walls")
    # ax_yz.scatter(mon_pos[:, 1], mon_pos[:, 2], marker='.', color='orange', label="monopoles")
    # ax_yz.scatter(pw_pos[:, 1], pw_pos[:, 2], marker='.', color='grey', label="plane waves")
    # for i, p in enumerate(pw_pos):
    #     plt.plot([center[1], p[1]], [center[2], p[2]], linestyle='-', color='black', alpha=0.2)
    # ax_yz.set_xlabel('Y-axis')
    # ax_yz.set_ylabel('Z-axis')
    # ax_yz.legend()
    # ax_yz.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # ax_yz.set_title('View from X-axis')
    # plt.show()


# ############################################################### #
# ########### build input signals from given mic recs ########### #
# ############################################################### #


def filter_irs_eval(ir_eval, max_freq, samplerate):
    h = signal.firwin(numtaps=64, cutoff=max_freq, fs=samplerate)
    sig_eval = signal.filtfilt(h, 1, ir_eval, axis=-1)

    return sig_eval


def plot_time_and_freq_domains_of_a_rir(time, sig_in_time, freq_from_bins, sig_in_freq, ir_num, a_or_s):
    if a_or_s == 'a':
        N = len(sig_in_freq) // 2
    elif a_or_s == 's':
        N = len(sig_in_freq)  # iffts are already half as long
    else:
        N = 0
        print("a or s")

    plt.subplot(121)
    plt.plot(time, sig_in_time)
    plt.title('Time Domain of IR ' + str(ir_num) + "[" + a_or_s + "]")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(122)
    # plt.stem(freq_bins, np.abs(sig_in_freq))
    plt.plot(freq_from_bins, 20 * np.log10(np.abs(sig_in_freq[0:N])))
    plt.title('Freq Domain of IR ' + str(ir_num) + "[" + a_or_s + "]")
    plt.xlabel('Frequency Hz')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # freq plot in db
    # plt.subplot(133)
    # plt.semilogy(freq_from_bins, 2.0 / N * np.abs(sig_in_freq[0:N]))
    # plt.title('Freq Domain of IR ' + str(ir_num))
    # plt.xlabel('Frequency Hz')
    # plt.ylabel('Amplitude')
    # plt.xlim(0, 1300)
    # plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_time_and_freq_domains_of_a_rir_version_2(time_ax, sig_in_time, time_stft, freqs_stft, Zxx, ir_num, a_or_s):
    plt.subplot(121)
    plt.plot(time_ax, sig_in_time)
    plt.title('Time Domain of IR ' + str(ir_num) + "[" + a_or_s + "]")
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(122)
    # Plot the magnitude of the STFT
    plt.pcolormesh(time_stft, freqs_stft, np.abs(Zxx), shading='auto')
    plt.title('STFT of IR ' + str(ir_num) + "[" + a_or_s + "]")
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude')

    plt.tight_layout()
    plt.show()

# ######################################## #
# ########## execute the method ########## #
# ######################################## #


def populate_greens_function_dictionary(M, N, monopoles_positions, pos_mic, speed_of_sound, frequency):
    # D is the dictionary matrix of Green's functions
    # G(r|r') = e^(j*k*||r-r'||_2) / 4*pi*||r-r'||_2

    wavelength = speed_of_sound / frequency
    k = 2 * np.pi / wavelength
    dictionary_gf = np.zeros((M, N), dtype=complex)

    for mic_idx, mic_pos in enumerate(pos_mic):
        for mon_idx, mon_pos in enumerate(monopoles_positions):
            distance = np.linalg.norm(mic_pos - mon_pos)
            dictionary_gf[mic_idx, mon_idx] = np.exp(1j * k * distance) / (4 * np.pi * distance)

    return dictionary_gf


def create_array_of_k_l(L, plane_waves_pos, center, speed_of_sound, freq):  # switch to 3d
    # k is a vector that is made up of kx, ky and kz to specify the direction
    k_mag = 2 * np.pi * freq / speed_of_sound

    vector_angles = np.zeros(L)
    # angle of the vector from each point to the center -> inward going pw
    for i, pw_origin in enumerate(plane_waves_pos):
        vector_angles[i] = np.arctan2(center[1] - pw_origin[1], center[0] - pw_origin[0])

    # the plane waves are parallel to the z axis
    pw_propagation_vector = np.zeros((L, 2))
    for i, angle in enumerate(vector_angles):
        # print(f"Point {i}: Angle = {angle:.2f} radians, cos: {(np.cos(angle)):.2f}, sin: {(np.sin(angle)):.2f}")
        pw_propagation_vector[i, :] = k_mag * np.array([np.cos(angle), np.sin(angle)])

    return pw_propagation_vector


def populate_plane_waves_dictionary(M, L, plane_waves_pos, mic_pos, center, speed_of_sound, freq):
    # todo: update to 3d pws
    # W is the dictionary matrix of plane waves
    # u_H(r) = e^(j*k_l*r), l in L, r_m in plane_waves_pos

    pw_propagation_vector = create_array_of_k_l(L, plane_waves_pos, center, speed_of_sound, freq)

    pos_all_2d = mic_pos[:, :-1]  # remove the z coordinate of the mics -> keep it for 3D pws

    dictionary_pw = np.zeros((M, L), dtype=complex)
    for i, m_pos in enumerate(pos_all_2d):
        for j, pw_k in enumerate(pw_propagation_vector):
            dictionary_pw[i, j] = np.exp(1j * pw_k @ m_pos)

    return dictionary_pw


def plot_overlaying_irs(ground, estimated, t, ir_number, samplerate, max_freq):

    plt.plot(t, ground, label="ground")
    plt.plot(t, estimated, alpha=0.7, linewidth=1, label="estimated")
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title("ground (filtered) vs estimated IR (" + str(ir_number) + ") \n "
              "@ samplerate " + str(samplerate) + " Hz, max freq: " + str(max_freq) + "Hz")
    plt.show()
