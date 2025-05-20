from ABCD import *
from utils import *
import json

GHz = 1e9
MHz = 1e6
kHz = 1e3
fF = 1e-15
um = 1e-6
us = 1e-6
ms = 1e-3
nH = 1e-9

pi = np.pi
vp = 1.2e8
z0 = 50

w_scale = 2*pi*MHz

#-------------------------------------------------------------------------------
# Exercise 1
#-------------------------------------------------------------------------------


C = 30*fF
ind = 1*nH
l = 4000*um

test_w_arr = 2*pi*np.linspace(2, 60, 20_000) * GHz

#-- Exercise 1a -----------------------------------------------------------------

def test_s21_helper(abcd_arr):
    a = abcd_arr[:, 0, 0]
    b = abcd_arr[:, 0, 1]
    c = abcd_arr[:, 1, 0]
    d = abcd_arr[:, 1, 1]
    return 2 / (a + b/z0 + c*z0 + d)

def test_get_ABCD_series_inductance(
        w_array: np.ndarray, # array of frequencies, shape (N,)
        ind: float # inductance
        ) -> np.ndarray: # array of ABCD matrices, one per frequency, shape (2,2,N)

    # abcd_arr needed in shape of w_array
    abcd_arr = np.zeros((len(w_array), 2, 2), dtype=np.complex128)

    for i in range(len(abcd_arr)):
        abcd_arr[i] = np.array([
            [1    ,       1j*w_array[i]*ind],   
            [0   ,                        1]
        ])

    return abcd_arr


def test_series_inductance(abcd_func):
    sol_abcd = ABCDSeriesInductance(ind)
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])


    att_abcd_arr = abcd_func(test_w_arr, ind)
    att_s21 = np.abs(test_s21_helper(att_abcd_arr))

    if np.mean(np.abs(sol_s21-att_s21)) < 1e-12:
        print("ABCD_series_inductance CORRECT.")
    else:
        print("ABCD_series_inductance INCORRECT.")


def test_series_capacitance(abcd_func):
    sol_abcd = ABCDSeriesCapacitance(C)
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])


    att_abcd_arr = abcd_func(test_w_arr, C)
    att_s21 = np.abs(test_s21_helper(att_abcd_arr))

    if np.mean(np.abs(sol_s21-att_s21)) < 1e-12:
        print("ABCD_series_capacitance CORRECT.")
    else:
        print("ABCD_series_capacitance INCORRECT.")


def test_transmission_line(abcd_func):
    sol_abcd = ABCDTEMTransmissionLine(l, z0, vp)
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])


    att_abcd_arr = abcd_func(test_w_arr, l, z0, vp)
    att_s21 = np.abs(test_s21_helper(att_abcd_arr))

    if np.mean(np.abs(sol_s21-att_s21)) < 1e-12:
        print("ABCD_transmission_line CORRECT.")
    else:
        print("ABCD_transmission_line INCORRECT.")


def test_parallel_inductance(abcd_func):
    sol_abcd = ABCDParallelInductance(ind)
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])


    att_abcd_arr = abcd_func(test_w_arr, ind)
    att_s21 = np.abs(test_s21_helper(att_abcd_arr))

    if np.mean(np.abs(sol_s21-att_s21)) < 1e-12:
        print("ABCD_parallel_inductance CORRECT.")
    else:
        print("ABCD_parallel_inductance INCORRECT.")


#-- Exercise 1b -----------------------------------------------------------------

def test_s21(s21_func):
    example_sol_abcd = ABCDSeriesInductance(ind)
    example_att_abcd = test_get_ABCD_series_inductance(test_w_arr, ind)


    sol_s21 = np.abs(example_sol_abcd.sparams(test_w_arr, z0)[2])
    att_s21 = np.abs(s21_func(example_att_abcd, z0))

    if np.mean(np.abs(sol_s21 - att_s21)) < 1e-12:
        print("s21_from_abcd CORRECT.")
    else:
        print("s21_from_abcd INCORRECT.")


#-- Exercise 1c -----------------------------------------------------------------

def test_abcd_total_1c(att_abcd_total):
    sol_abcd = ABCDSeriesCapacitance(C) * ABCDTEMTransmissionLine(l, z0, vp) * ABCDParallelInductance(ind)
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])

    att_s21 = np.abs(test_s21_helper(att_abcd_total))

    if np.mean(np.abs(np.abs(sol_s21) - att_s21)) < 1e-12:
        print("abcd_total_1c CORRECT.")
    else:
        print("abcd_total_1c INCORRECT.")
    

#-- Exercise 1d -----------------------------------------------------------------

# def test_T_junction(abcd_func):
#     example_sol_abcd = ABCDSeriesInductance(ind)
#     example_att_abcd = test_get_ABCD_series_inductance(test_w_arr, ind)

#     sol_abcd_t_junction = ABCDTJunction(example_sol_abcd)
#     sol_s21 = np.abs(sol_abcd_t_junction.sparams(test_w_arr, z0)[2])

#     att_abcd_t_junction = abcd_func(example_att_abcd)
#     att_s21 = np.abs(test_s21_helper(att_abcd_t_junction))

#     if np.mean(np.abs(sol_s21 - att_s21)) < 1e-12:
#         print("get_ABCD_T_junction CORRECT.")
#     else:
#         print("get_ABCD_T_junction INCORRECT.")

def test_abcd_1d(att_abcd_total_t_junction):
    sol_abcd = ABCDSeriesCapacitance(C) * ABCDTEMTransmissionLine(l, z0, vp) * ABCDParallelInductance(ind)
    sol_abcd_t_junction = ABCDTJunction(sol_abcd)
    sol_s21 = np.abs(sol_abcd_t_junction.sparams(test_w_arr, z0)[2])

    att_s21 = np.abs(test_s21_helper(att_abcd_total_t_junction))

    if np.mean(np.abs(sol_s21 - att_s21)) < 1e-12:
        print("get_ABCD_T_junction CORRECT.")
    else:
        print("get_ABCD_T_junction INCORRECT.")


#-------------------------------------------------------------------------------
# Exercise 2
#-------------------------------------------------------------------------------

#-- Exercise 2a -----------------------------------------------------------------


def test_get_f_n_quarter_wave_helper(
        n, # index of the harmonic
        C_c, # coupling capacitance
        l, # length of quarter wave reosnator
        z0, # characteristic impedance
        vp # phase velocity
    ) -> float: # frequency of the nth harmonic in GHz

    l_eff = l + vp*z0 * C_c
    f_n = vp / (4*l_eff) * (2*n - 1)

    return f_n

def test_get_f_n_2a(f_n_fun):
    sol_f_n = test_get_f_n_quarter_wave_helper(1, C, l, z0, vp)
    att_f_n = f_n_fun(1, C, l, z0, vp)

    if np.abs(sol_f_n - att_f_n) < 1e-12:
        print("get_f_n_quarter_wave CORRECT.")
    else:
        print("get_f_n_quarter_wave INCORRECT.")

def test_resonance_frequencies_2a(att_f_n_wo_cap_list, att_f_n_w_cap_list):
    
    implementation_correct = True

    assert len(att_f_n_wo_cap_list) == 4, "f_n_wo_cap_list must have length 4"
    assert len(att_f_n_w_cap_list) == 4, "f_n_w_cap_list must have length 4"
    
    sol_f_n_wo_cap_list = [test_get_f_n_quarter_wave_helper(n, 0, l, z0, vp)/GHz for n in range(1, 5)]
    if np.mean(np.abs(np.array(sol_f_n_wo_cap_list) - np.array(att_f_n_wo_cap_list))) < 1e-12:
        print("f_n_without_cap_correction_list CORRECT.")
    else:
        print("f_n_without_cap_correction_list INCORRECT.")
        implementation_correct = False

    sol_f_n_w_cap_list = [test_get_f_n_quarter_wave_helper(n, C, l, z0, vp)/GHz for n in range(1, 5)]
    if np.mean(np.abs(np.array(sol_f_n_w_cap_list) - np.array(att_f_n_w_cap_list))) < 1e-12:
        print("f_n_with_cap_correction_list CORRECT.")
    else:
        print("f_n_with_cap_correction_list INCORRECT.")
        implementation_correct = False


    if implementation_correct:
        print("\n")
        print(f"{'Mode n':<8} | {'f_n (without cap) [GHz]':<25} | {'f_n (with cap) [GHz]':<25}")
        print("-" * 64)

        for n in range(1, 5):
            f_without = att_f_n_wo_cap_list[n-1]
            f_with = att_f_n_w_cap_list[n-1]
            print(f"{n:<8} | {f_without:<25.5f} | {f_with:<25.5f}")

#-- Exercise 2b -----------------------------------------------------------------

def test_abcd_2b_lumped(att_abcd):
    w0 = 2*pi* 7.5 *GHz
    C_r = np.pi / (4 * w0 * z0)
    ind_r = 1 / (w0**2 * C_r)

    sol_abcd = ABCDTJunction(
                        ABCDSeriesCapacitance(C) * 
                        ABCDParallelInductance(ind_r) *
                        ABCDSeriesCapacitance(C_r)
                        )
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])
    att_s21 = np.abs(att_abcd.sparams(test_w_arr, z0)[2])

    if np.mean(np.abs(sol_s21 - att_s21)) < 1e-12:
        print("abcd_2b_lumped CORRECT.")
    else:
        print("abcd_2b_lumped INCORRECT.")

def test_abcd_2b_half_wave(att_abcd):
    l_half_wave = 2* 4000*um #* 1.01
    sol_abcd= ABCDTJunction(
                                ABCDSeriesCapacitance(C) *
                                ABCDTEMTransmissionLine(l_half_wave, z0, vp) *
                                ABCDSeriesCapacitance(C))
    sol_s21 = np.abs(sol_abcd.sparams(test_w_arr, z0)[2])
    att_s21 = np.abs(att_abcd.sparams(test_w_arr, z0)[2])

    if np.mean(np.abs(sol_s21 - att_s21)) < 1e-12:
        print("abcd_2b_lumped CORRECT.")
    else:
        print("abcd_2b_lumped INCORRECT.")

#-- Exercise 2c -----------------------------------------------------------------
test_2c_w_arr = 2*pi* np.linspace(7.1, 7.26, 10_000) * GHz # frequency array in rad/s
test_2c_wr = 2*pi* 7.18 *GHz # resonance frequency in rad/s

test_2c_kappa_list = 2*pi* np.array([5, 15, 25]) * MHz # coupling rates in rad/s
test_2c_phi_list = np.array([-pi/4, 0, pi/4]) # phase in radians

def test_get_s21_lorentzian(
        w_arr: np.ndarray, # array of frequencies in rad/s, shape (N,)
        wr: float, # resonance frequency in rad/s
        kappa: float, # coupling rate in rad/s
        phi: float, # phase in radians
) -> np.ndarray: # array of s21 values, one per frequency, shape (N,)
    
    s21_lorentzian = np.cos(phi) - np.exp(1j*phi) * kappa / (kappa + 2j*(wr - w_arr))

    return s21_lorentzian


def test_get_s21_2c(s21_func):
    sol_s21 = test_get_s21_lorentzian(test_2c_w_arr, test_2c_wr, test_2c_kappa_list[0], test_2c_phi_list[0])
    att_s21 = s21_func(test_2c_w_arr, test_2c_wr, test_2c_kappa_list[0], test_2c_phi_list[0])

    if np.mean(np.abs(sol_s21 - att_s21)) < 1e-12:
        print("get_s21_lorentzian CORRECT.")
    else:
        print("get_s21_lorentzian INCORRECT.")

def test_s21_kappa_phi_list_2c(att_s21_kappa_list, att_s21_phi_list):

    assert len(att_s21_kappa_list) == 3, "att_s21_kappa_list must have length 3"
    assert len(att_s21_phi_list) == 3, "att_s21_phi_list must have length 3"

    sol_s21_kappa_list = []
    sol_s21_phi_list = []

    phi = 0 # Use phi=0 for this list.
    for kappa in test_2c_kappa_list:
        sol_s21_kappa = test_get_s21_lorentzian(test_2c_w_arr, test_2c_wr, kappa, phi)
        sol_s21_kappa_list.append(sol_s21_kappa)

    kappa = 2*pi*5*MHz # Use kappa= 2*pi* 5*MHz for this list.
    for phi in test_2c_phi_list:
        sol_s21_phi = test_get_s21_lorentzian(test_2c_w_arr, test_2c_wr, kappa, phi)
        sol_s21_phi_list.append(sol_s21_phi)

    if np.mean(np.abs(np.array(sol_s21_kappa_list) - np.array(att_s21_kappa_list))) < 1e-12:
        print("s21_kappa_list CORRECT.")
    else:
        print("s21_kappa_list INCORRECT.")

    if np.mean(np.abs(np.array(sol_s21_phi_list) - np.array(att_s21_phi_list))) < 1e-12:
        print("s21_phi_list CORRECT.")
    else:
        print("s21_phi_list INCORRECT.")


#-- Exercise 2d -----------------------------------------------------------------


def test_kappa_2d(att_kappa):
    sol_kappa = 18.239342290205407
    if np.abs(sol_kappa - att_kappa) < 1e-6:
        print("kappa_2d CORRECT.")
    else:
        print("kappa_2d INCORRECT.")

#-- Exercise 2d -----------------------------------------------------------------

def test_kappa_list_2e(att_kappa_list):
    with open('json_results/kappa_list_2e.json', 'r') as f:
        sol_kappa_list = json.load(f)
    sol_kappa_list = np.array(sol_kappa_list)
    att_kappa_list = np.array(att_kappa_list)
    if np.mean(np.abs(sol_kappa_list - att_kappa_list)) < 1e-6:
        print("kappa_list_2e CORRECT.")
    else:
        print("kappa_list_2e INCORRECT.")

#-------------------------------------------------------------------------------
# Exercise 3
#-------------------------------------------------------------------------------

def test_paper_trace_3(att_paper_trace):
    with open('json_results/paper_trace_3.json', 'r') as f:
        sol_paper_trace = json.load(f)
    sol_paper_trace = np.abs(sol_paper_trace)
    att_paper_trace = np.abs(att_paper_trace)
    
    if np.mean(np.abs(sol_paper_trace - att_paper_trace)) < 1e-12:
        print("paper_trace CORRECT.")

        print_large_text("")
        print_large_text("U R AWESOME")
        print_large_text("")
        print_large_text("SQDEW <3 YOU")
    else:
        print("paper_trace INCORRECT.")


