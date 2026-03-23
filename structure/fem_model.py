"""OpenSeesPy FE model builder, dynamic analysis, eigen analysis."""
import os
import sys
import math
import ctypes
import numpy as np
import openseespy.opensees as op
from .properties import opensees_constants, MDOFCantil_Property


# ---------------------------------------------------------------------------
# Context manager to suppress C-level stdout/stderr (OpenSeesPy warnings)
# ---------------------------------------------------------------------------
if sys.platform == 'win32':
    _libc = ctypes.cdll.msvcrt
    _kernel32 = ctypes.windll.kernel32
    _STD_OUTPUT_HANDLE = ctypes.c_ulong(-11 & 0xFFFFFFFF)
    _STD_ERROR_HANDLE = ctypes.c_ulong(-12 & 0xFFFFFFFF)
else:
    _libc = ctypes.CDLL(None)
    _kernel32 = None


class _SuppressCOutput:
    """Temporarily redirect C-level stdout and stderr to devnull."""

    def __enter__(self):
        # Flush Python and C buffers
        sys.stdout.flush()
        sys.stderr.flush()
        _libc.fflush(None)
        # Save original file descriptors
        self._stdout_fd = os.dup(1)
        self._stderr_fd = os.dup(2)
        # Open devnull
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 1)
        os.dup2(self._devnull, 2)
        # On Windows, also redirect kernel32 console handles
        if _kernel32 is not None:
            self._orig_stdout_handle = _kernel32.GetStdHandle(_STD_OUTPUT_HANDLE)
            self._orig_stderr_handle = _kernel32.GetStdHandle(_STD_ERROR_HANDLE)
            devnull_handle = _kernel32.CreateFileW(
                "NUL", 0x40000000, 0, None, 3, 0, None)  # GENERIC_WRITE, OPEN_EXISTING
            self._devnull_handle = devnull_handle
            _kernel32.SetStdHandle(_STD_OUTPUT_HANDLE, devnull_handle)
            _kernel32.SetStdHandle(_STD_ERROR_HANDLE, devnull_handle)
        return self

    def __exit__(self, *args):
        # Flush before restoring
        _libc.fflush(None)
        # Restore kernel32 console handles first (Windows)
        if _kernel32 is not None:
            _kernel32.SetStdHandle(_STD_OUTPUT_HANDLE, self._orig_stdout_handle)
            _kernel32.SetStdHandle(_STD_ERROR_HANDLE, self._orig_stderr_handle)
            _kernel32.CloseHandle(self._devnull_handle)
        # Restore original descriptors
        os.dup2(self._stdout_fd, 1)
        os.dup2(self._stderr_fd, 2)
        # Close saved copies
        os.close(self._stdout_fd)
        os.close(self._stderr_fd)
        os.close(self._devnull)


def _assign_mass(nDOF_Prop):
    """Assign nodal mass to the OpenSees model."""
    m = nDOF_Prop.nodal_mass
    if not isinstance(m, list):
        for i in range(nDOF_Prop.ndof):
            op.mass(i + 1, m)
    else:
        if len(m) == nDOF_Prop.ndof:
            for i in range(nDOF_Prop.ndof):
                op.mass(i + 1, m[i])
        elif len(m) == 1:
            for i in range(nDOF_Prop.ndof):
                op.mass(i + 1, m[0])
        else:
            raise ValueError("nodal_mass length must equal ndof or 1")


def ZeroLengthMDoF(nDOF_Prop, dirfolder="."):
    """Create multi-DoF shear building with zeroLength elements and Steel01 material."""
    os.makedirs(dirfolder, exist_ok=True)

    op.wipe()
    op.model('basic', '-ndm', 1, '-ndf', 1)

    op.node(0, 0.0)
    op.fix(0, 1)
    for i in range(nDOF_Prop.ndof):
        op.node(i + 1, 0.0)

    _assign_mass(nDOF_Prop)

    op.uniaxialMaterial('Steel01', 11, nDOF_Prop.fy, nDOF_Prop.Ecolumn, nDOF_Prop.bb)

    for i in range(nDOF_Prop.ndof):
        op.element('zeroLength', 100 * (i + 1) + 1, i, i + 1,
                    '-mat', 11, '-dir', 1, '-doRayleigh', 1)

    # Rayleigh damping
    if nDOF_Prop.ndof == 1:
        eigen_1 = op.eigen('-fullGenLapack', 1)
        angular_freq = eigen_1[0] ** 0.5
        beta_k = 2 * nDOF_Prop.rayleigh_xi / angular_freq
        op.rayleigh(0.0, beta_k, 0.0, 0.0)
    else:
        eigs = op.eigen('-fullGenLapack', 2)
        omega1, omega2 = eigs[0] ** 0.5, eigs[1] ** 0.5
        alpha_m = 2.0 * nDOF_Prop.rayleigh_xi * (omega1 * omega2) / (omega1 + omega2)
        beta_k = 2.0 * nDOF_Prop.rayleigh_xi / (omega1 + omega2)
        op.rayleigh(alpha_m, beta_k, 0.0, 0.0)

    return nDOF_Prop.modelname


def ZeroLengthMDoFDynamicAnalysis(nDOF_Prop, GMfilename, dirfolder, GMfact=1.0,
                                   recordnodelist=None, recorddoflist=None,
                                   acc_dsp_rct='acc', dt_analysis=0.01):
    """Run transient dynamic analysis with Newmark-beta integrator."""
    if recordnodelist is None:
        recordnodelist = list(range(1, nDOF_Prop.ndof + 1))
    if recorddoflist is None:
        recorddoflist = [1]

    op.wipe()
    modelname = ZeroLengthMDoF(nDOF_Prop, dirfolder)

    GMdirset = GMfilename.split('/')
    record = [GMdirset[-1]]
    recordpath = '/'.join(GMdirset[:-1]) + '/'
    eqname = '_'.join(GMdirset[-2:])

    dt, nPts = ReadRecord(recordpath + record[0] + '.AT2', recordpath + record[0] + '.dat')

    with _SuppressCOutput():
        op.system('BandGeneral')
        op.numberer('RCM')
        op.constraints('Plain')
        op.integrator('Newmark', 0.5, 0.25)
        op.algorithm('Newton')
        op.analysis('Transient')

        IDgmSeries = 500
        count = 2
        record_single = recordpath + record[0]
        op.timeSeries('Path', count, '-dt', dt, '-filePath', record_single + '.dat', '-factor', 9.81 * GMfact)
        op.pattern('UniformExcitation', IDgmSeries + count, 1, '-accel', count)

        responsetype = {'acc': 'accel', 'vel': 'vel', 'dsp': 'disp', 'rct': 'reaction'}.get(acc_dsp_rct, 'accel')
        outfile = os.path.join(dirfolder, 'model({})_inp({})_{}.txt'.format(modelname, eqname, acc_dsp_rct))
        if acc_dsp_rct != 'rct':
            op.recorder('Node', '-file', outfile,
                         '-time', '-node', *recordnodelist, '-dof', *recorddoflist, responsetype)
        else:
            op.recorder('Element', '-file', outfile,
                         '-time', '-ele', *[100 * (i + 1) + 1 for i in range(nDOF_Prop.ndof)], 'force')

        tFinal = nPts * dt
        Nsteps = int(tFinal / dt_analysis)
        result = op.analyze(Nsteps, dt_analysis)
        op.wipe()
    return result


def ReadRecord(inFilename, outFilename):
    """Parse PEER NGA-West2 AT2 file format and write .dat file."""
    dt = 0.0
    npts = 0
    with open(inFilename, 'r') as inFile:
        with open(outFilename, 'w') as outFile:
            flag = 0
            for line in inFile:
                if line == '\n':
                    continue
                elif flag == 1:
                    outFile.write(line)
                else:
                    words = line.split()
                    if len(words) >= 4:
                        if words[0] == 'NPTS=':
                            for word in words:
                                if word != '':
                                    if flag == 1:
                                        dt = float(word)
                                        break
                                    if flag == 2:
                                        npts = int(word.strip(','))
                                        flag = 0
                                    if word == 'DT=' or word == 'dt':
                                        flag = 1
                                    if word == 'NPTS=':
                                        flag = 2
                        elif words[-1] == 'DT':
                            count = 0
                            for word in words:
                                if word != '':
                                    if count == 0:
                                        npts = int(word)
                                    elif count == 1:
                                        dt = float(word)
                                    elif word == 'DT':
                                        flag = 1
                                        break
                                    count += 1
    return dt, npts


def get_mass_matrix(nDOF_Prop):
    """Build diagonal mass matrix from structure properties."""
    op.wipe()
    ZeroLengthMDoF(nDOF_Prop)
    M = np.zeros((nDOF_Prop.ndof, nDOF_Prop.ndof))
    m = nDOF_Prop.nodal_mass
    if not isinstance(m, list):
        np.fill_diagonal(M, m)
    elif len(m) == nDOF_Prop.ndof:
        np.fill_diagonal(M, m)
    elif len(m) == 1:
        np.fill_diagonal(M, m[0])
    return M


def get_mode_shapes(nDOF_Prop, plot_fig=False):
    """Compute normalized mode shapes via OpenSeesPy eigen analysis."""
    op.wipe()
    # Use simple truss model for eigen analysis
    op.model('basic', '-ndm', 1, '-ndf', 1)
    for i in range(nDOF_Prop.ndof + 1):
        op.node(i, nDOF_Prop.Lcolumn * i)
    op.fix(0, 1)
    _assign_mass(nDOF_Prop)
    op.uniaxialMaterial('Steel01', 11, nDOF_Prop.fy, nDOF_Prop.Ecolumn, nDOF_Prop.bb)
    for i in range(nDOF_Prop.ndof):
        op.element('truss', 100 + i, i, i + 1, nDOF_Prop.Acolumn, 11)

    op.eigen('-fullGenLapack', nDOF_Prop.ndof)
    mode_shapes = []
    for i in range(nDOF_Prop.ndof):
        mode = [op.nodeEigenvector(j, i + 1)[0] for j in range(1, nDOF_Prop.ndof + 1)]
        mode_shapes.append(mode)

    modeshapes = np.array(mode_shapes)
    modeshapes_n = np.zeros_like(modeshapes)
    for i in range(nDOF_Prop.ndof):
        modeshapes_n[i, :] = modeshapes[i, :] / np.linalg.norm(modeshapes[i, :])

    if plot_fig:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i in range(nDOF_Prop.ndof):
            plt.plot(np.insert(mode_shapes[i], 0, 0.0), range(nDOF_Prop.ndof + 1),
                     marker='o', label=f'Mode {i + 1}')
        plt.xlabel('Mode Shape')
        plt.ylabel('Floor Level')
        plt.title(f'Mode Shapes of {nDOF_Prop.ndof}DOF Shear Building')
        plt.legend()
        plt.grid()
        plt.show()

    return modeshapes_n


def get_natural_frequencies(nDOF_Prop):
    """Compute natural frequencies (Hz) via eigen analysis."""
    op.wipe()
    op.model('basic', '-ndm', 1, '-ndf', 1)
    for i in range(nDOF_Prop.ndof + 1):
        op.node(i, nDOF_Prop.Lcolumn * i)
    op.fix(0, 1)
    _assign_mass(nDOF_Prop)
    op.uniaxialMaterial('Steel01', 11, nDOF_Prop.fy, nDOF_Prop.Ecolumn, nDOF_Prop.bb)
    for i in range(nDOF_Prop.ndof):
        op.element('truss', 100 + i, i, i + 1, nDOF_Prop.Acolumn, 11)

    eigvals = op.eigen('-fullGenLapack', nDOF_Prop.ndof)
    omegas = np.sqrt(np.array(eigvals))
    return omegas / (2 * np.pi)
