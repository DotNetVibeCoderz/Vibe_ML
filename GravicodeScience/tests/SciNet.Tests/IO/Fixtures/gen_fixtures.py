"""Generates the SciNet IO fixtures (scinet_*, mm_*, hb_*, nc_*, wav_*, fortran_*) with scipy 1.17.1 / numpy 2.4.4.

scinet.arff and sparse.arff are hand-written; scipy/* are copied verbatim from SciPy's own test data (BSD-3-Clause).
"""
import os, struct, sys
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.io import wavfile, FortranFile, netcdf_file

OUT = os.path.dirname(os.path.abspath(__file__))  # this Fixtures folder
os.makedirs(OUT, exist_ok=True)
P = lambda n: os.path.join(OUT, n)

# ------------------------------------------------------------------ MAT v5
def mat_dict():
    st = {'a': 1.0, 'b': 'text', 'c': np.array([[1.0, 2.0], [3.0, 4.0]])}
    starr = np.zeros((1, 2), dtype=[('x', object), ('y', object)])
    starr[0, 0] = (np.array([[1.0]]), 'first')
    starr[0, 1] = (np.array([[2.0, 3.0]]), 'second')
    cell = np.empty((1, 3), dtype=object)
    cell[0, 0] = 1.0; cell[0, 1] = 'two'; cell[0, 2] = np.arange(3.0)
    nest = np.empty((2, 1), dtype=object)
    inner = np.empty((1, 2), dtype=object); inner[0, 0] = 'deep'; inner[0, 1] = np.int16(-7)
    nest[0, 0] = {'p': np.array([[5.0, 6.0]]), 'q': inner}
    nest[1, 0] = inner
    spm = sp.csc_matrix(np.array([[1.0, 0, 0, 2.5, 0], [0, 0, -3.0, 0, 0], [0, 4.0, 0, 0, 0], [0, 0, 0, 0, 5.5]]))
    spc = sp.csc_matrix(np.array([[1 + 2j, 0, 0], [0, 0, -1j], [3.0, 0, 0]]))
    splog = sp.csc_matrix(np.array([[True, False], [False, True], [True, True]]))
    obj = sio.matlab.MatlabObject(np.array([[(np.array([[42.0]]),)]], dtype=[('val', object)]), 'myclass') if hasattr(sio.matlab, 'MatlabObject') else None
    d = dict(
        dbl=np.arange(12.0).reshape(3, 4) * 1.5 - 2,
        sgl=np.array([[1.5, -2.25, 3.0], [4.0, 5.5, -6.75]], dtype=np.float32),
        i8=np.array([[-128, 127], [0, -1]], dtype=np.int8),
        u8=np.array([[0, 255], [1, 128]], dtype=np.uint8),
        i16=np.array([[-32768, 32767], [5, -5]], dtype=np.int16),
        u16=np.array([[0, 65535], [7, 300]], dtype=np.uint16),
        i32=np.array([[-2147483648, 2147483647], [11, -11]], dtype=np.int32),
        u32=np.array([[0, 4294967295], [13, 99]], dtype=np.uint32),
        i64=np.array([[-2**62, 2**53], [17, -17]], dtype=np.int64),
        u64=np.array([[0, 2**63], [19, 2**40]], dtype=np.uint64),
        lgc=np.array([[True, False, True], [False, False, True]]),
        s='hello',
        sarr=np.array(['abc', 'xyz']),
        uni='h\u00e9llo \u20ac \u65e5\u672c',
        cplx=np.array([[1 + 2j, -3.5j], [4.0, 0.5 - 0.25j]]),
        nd=np.arange(24.0).reshape(2, 3, 4),
        empty=np.zeros((0, 0)),
        emptystr='',
        sp=spm, spc=spc, splog=splog,
        st=st, starr=starr, cell=cell, nest=nest,
        scalar=3.5,
        vec=np.array([1.0, 2.0, 3.0]),
    )
    if obj is not None:
        d['obj'] = obj
    return d

d = mat_dict()
sio.savemat(P('scinet_v5.mat'), d, do_compression=False)
sio.savemat(P('scinet_v5z.mat'), d, do_compression=True)
sio.savemat(P('scinet_v5_col.mat'), {'vec': np.array([1.0, 2.0, 3.0])}, oned_as='column')
sio.savemat(P('scinet_v4.mat'), dict(
    dbl=np.arange(12.0).reshape(3, 4) * 1.5 - 2,
    cplx=np.array([[1 + 2j, -3.5j], [4.0, 0.5 - 0.25j]]),
    s='hello v4',
    sp=sp.csc_matrix(np.array([[1.0, 0, 0, 2.5, 0], [0, 0, -3.0, 0, 0], [0, 4.0, 0, 0, 0], [0, 0, 0, 0, 5.5]])),
), format='4')
print('whosmat v5', sio.whosmat(P('scinet_v5z.mat')))

# ------------------------------------------------------------------ Matrix Market
A = np.array([[1.5, 0, 0, -2.0], [0, 3.25, 0, 0], [7.0, 0, 0, 1e-300], [0, 0, 4.0, 0], [0, 6.0, 0, 0]])
sio.mmwrite(P('mm_general.mtx'), sp.coo_matrix(A), comment='general real', symmetry='general')
S = np.array([[4.0, 1.0, 0], [1.0, 5.0, 2.0], [0, 2.0, 6.0]])
sio.mmwrite(P('mm_symmetric.mtx'), sp.coo_matrix(S), symmetry='symmetric')
K = np.array([[0, 2.0, -1.0], [-2.0, 0, 3.0], [1.0, -3.0, 0]])
sio.mmwrite(P('mm_skew.mtx'), sp.coo_matrix(K), symmetry='skew-symmetric')
sio.mmwrite(P('mm_pattern.mtx'), sp.coo_matrix(A), field='pattern')
sio.mmwrite(P('mm_integer.mtx'), sp.coo_matrix(np.array([[1, 0, -3], [0, 12345678901, 0]], dtype=np.int64)), field='integer')
H = np.array([[2.0, 1 - 1j, 0], [1 + 1j, 3.0, 2j], [0, -2j, 1.0]])
sio.mmwrite(P('mm_hermitian.mtx'), sp.coo_matrix(H), symmetry='hermitian')
sio.mmwrite(P('mm_array.mtx'), np.arange(6.0).reshape(2, 3) / 4)
sio.mmwrite(P('mm_array_sym.mtx'), S, symmetry='symmetric')
sio.mmwrite(P('mm_array_complex.mtx'), np.array([[1 + 1j, 2], [3j, -4 - 0.5j]]))
for n in ['mm_general', 'mm_symmetric', 'mm_skew', 'mm_pattern', 'mm_integer', 'mm_hermitian', 'mm_array', 'mm_array_sym', 'mm_array_complex']:
    print(n, sio.mminfo(P(n + '.mtx')))

# ------------------------------------------------------------------ Harwell-Boeing
HB = sp.csc_matrix(np.array([[1.0, 0, 0, 2.5], [0, -3.75, 0, 0], [6.0, 0, 1e-10, 0], [0, 4.0e20, 0, 7.0]]))
sio.hb_write(P('hb_rua.rua'), HB)
print(open(P('hb_rua.rua')).read())
# hand-written RSA with D exponents packed without separating spaces, lower triangle only
rsa = (
    "Symmetric 3x3 test matrix, packed D exponents                           RSA3    \n"
    "             4             1             1             2\n"
    "RSA                        3             3             5             0\n"
    "(4I4)           (5I4)           (3D12.4)            \n"
    "   1   3   5   6\n"
    "   1   2   2   3   3\n"
    "  0.4000D+01  0.1000D+01  0.5000D+01\n"
    " -0.2000D+01  0.6000D+01\n")
open(P('hb_rsa.rsa'), 'w', newline='\n').write(rsa)
pua = (
    "Pattern matrix                                                          PUA1    \n"
    "             2             1             1             0\n"
    "PUA                        3             4             5             0\n"
    "(5I3)           (5I3)                               \n"
    "  1  3  4  5  6\n"
    "  1  3  2  1  3\n")
open(P('hb_pua.pua'), 'w', newline='\n').write(pua)

# ------------------------------------------------------------------ NetCDF
for ver in (1, 2):
    f = netcdf_file(P(f'nc_v{ver}.nc'), 'w', version=ver)
    f.history = 'Created for SciNet tests'
    f.version = np.int32(3)
    f.coeffs = np.array([0.5, 1.5, -2.0])
    f.createDimension('time', None)
    f.createDimension('x', 3)
    f.createDimension('strlen', 5)
    # No 0-d variable here: scipy 1.17.1 lays a scalar variable out inside the record section
    # of a file that has record variables, overwriting record 1 of the first one.
    t = f.createVariable('time', 'd', ('time',)); t.units = 'days since 2000-01-01'
    t[:] = np.arange(4.0) * 0.5
    temp = f.createVariable('temp', 'f', ('time', 'x')); temp.units = 'K'
    temp[:] = (np.arange(12.0).reshape(4, 3) + 270.25).astype(np.float32)
    pk = f.createVariable('packed', 'h', ('time',))
    pk.scale_factor = 0.01; pk.add_offset = 10.0; pk._FillValue = np.int16(-999)
    pk[:] = np.array([100, -999, 250, -32], dtype=np.int16)
    lat = f.createVariable('lat', 'i', ('x',)); lat[:] = np.array([-90, 0, 90], dtype=np.int32)
    b = f.createVariable('flags', 'b', ('x',)); b[:] = np.array([-1, 0, 127], dtype=np.int8)
    nm = f.createVariable('name', 'c', ('x', 'strlen'))
    nm[:] = np.array([[c.encode('latin1') for c in w] for w in ['alpha', 'beta\0', 'gamma']], dtype='S1')
    f.close()
    print('nc', ver, os.path.getsize(P(f'nc_v{ver}.nc')))

# ------------------------------------------------------------------ WAV
n = 50
tt = np.arange(n)
wavfile.write(P('wav_u8.wav'), 8000, (128 + 100 * np.sin(tt / 3)).astype(np.uint8))
st16 = np.stack([(10000 * np.sin(tt / 5)).astype(np.int16), (-20000 * np.cos(tt / 7)).astype(np.int16)], axis=1)
wavfile.write(P('wav_i16_stereo.wav'), 44100, st16)
wavfile.write(P('wav_i32.wav'), 22050, (2_000_000_000 * np.sin(tt / 4)).astype(np.int32))
wavfile.write(P('wav_f32.wav'), 16000, (0.75 * np.sin(tt / 6)).astype(np.float32))
wavfile.write(P('wav_f64_3ch.wav'), 48000, np.stack([np.sin(tt / 3), np.cos(tt / 3), tt / n], axis=1))

# hand-made: odd-sized unknown chunk before 'data' (padding byte), 16-bit mono, 5 samples
samples = np.array([0, 1000, -1000, 32767, -32768], dtype='<i2').tobytes()
fmt = struct.pack('<HHIIHH', 1, 1, 11025, 11025 * 2, 2, 16)
junk = b'abc'  # odd -> one pad byte
body = b'WAVE' + b'fmt ' + struct.pack('<I', 16) + fmt + b'junk' + struct.pack('<I', 3) + junk + b'\0' \
    + b'data' + struct.pack('<I', len(samples)) + samples
open(P('wav_oddchunk.wav'), 'wb').write(b'RIFF' + struct.pack('<I', len(body)) + body)
print('oddchunk', wavfile.read(P('wav_oddchunk.wav')))

# ------------------------------------------------------------------ Fortran
with FortranFile(P('fortran_mixed4.dat'), 'w') as f:
    f.write_record(np.array([1, -2, 3, 2147483647], dtype=np.int32))
    f.write_record(np.array([0.5, -1.25, 1e300], dtype=np.float64))
    f.write_record(np.array([7, 8], dtype=np.int32), np.array([2.5, 3.5, 4.5], dtype=np.float32))
    f.write_record(np.array([2**40, -5], dtype=np.int64))
with FortranFile(P('fortran_h8.dat'), 'w', header_dtype=np.uint64) as f:
    f.write_record(np.arange(5, dtype=np.float64))
    f.write_record(np.array([42], dtype=np.int32))
print('done')
