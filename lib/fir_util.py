
import numpy as np
import scipy.signal

def Ry(theta):
    """
    Ry(theta) matrix

    Ry is a rotation on the bloch sphere, so theta is double
    that of a normal Givens rotation
    """
    return np.array(
        [
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2), np.cos(theta/2)]
        ]        
    )

def power_complementary_fir(H0):
    """
    Find a power complementary FIR filter for the given direct form coefficients
    """

    # 1 - |H0(z)|^2
    comp_squared = -np.convolve(H0, np.conj(np.flip(H0)))
    comp_squared[len(H0)-1] += 1

    # Find minimum phase spectral factor
    return scipy.signal.minimum_phase(comp_squared, half=True)

def direct_pair_to_lattice_real(H0, H1):
    """
    Given a power complementary pair of order M direct form FIRs, find the
    lattice coefficients theta_0, ..., theta_M

    The coefficients are given in the convention used for Ry gates, i.e., double cover

    Real coefficients only.

    See Sec. 4.3 in P. P. Vaidyanathan, Multirate Systems and Filter Banks, Prentice Hall, 1993
    """

    def head_angle(A):
        head = A[:, 0]
        return np.arctan2(head[1], head[0])


    def shift_H1(A):
        B = np.copy(A)
        B[1] = np.roll(B[1], -1)
        B[1, -1] = 0
        return B
    
    M = len(H0) - 1
    
    mat = np.stack((H0, H1))

    coeffs = []

    for i in range(M):
        theta = 2 * head_angle(mat)
        coeffs.append(theta)
        mat = shift_H1(Ry(-theta) @ mat)
    
    theta = 2 * head_angle(mat)
    coeffs.append(theta)

    return np.flip(coeffs)

def lattice_freq_response(theta, freqs):
    z_ = np.exp(-1j * freqs)
    response = np.repeat(
        np.array([1, 0], dtype=np.complex128)[np.newaxis, :, np.newaxis],
        len(freqs), axis=0
    )
    for i, th in enumerate(theta):
        if i > 0:
            response[:, 1, 0] *= z_
        response = Ry(th) @ response
    
    return response[:, 0, 0], response[:, 1, 0]
