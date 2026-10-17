"""Generates the HDF5 / MATLAB v7.3 / NetCDF-4 fixtures (h5_*, mat73_*, nc4_*) used by Hdf5Tests,
MatFileV73Tests and NetCdf4Tests.

Made with h5py 3.16.0 (HDF5 2.0.0), netCDF4 1.7.5 (netCDF-C 4.10.1), numpy 2.4.4, scipy 1.17.1.

MATLAB itself is not available here and SciPy cannot write v7.3, so mat73_scinet.mat is written
with h5py following MATLAB's v7.3 layout exactly as MATLAB documents/produces it: a 512-byte user
block holding the 128-byte MAT header (version 0x0200), a MATLAB_class attribute on every
variable, data stored with the dimensions reversed (column-major), char as UTF-16 uint16 with
MATLAB_int_decode=2, empty arrays as their uint64 dimensions with MATLAB_empty=1, cells as object
references into /#refs#, structs as groups with a MATLAB_fields attribute (struct arrays: one
reference dataset per field), sparse as a group (data, ir, jc, MATLAB_sparse=rows), complex as
a {real, imag} compound. It holds the same variables as scinet_v5.mat (see gen_fixtures.py), so
the tests can check that both loaders agree. scipy/testhdf5_7.4_GLNX86.mat is a real
MATLAB-written v7.3 file copied from SciPy's test data (BSD-3-Clause).
"""
import os
import numpy as np
import h5py
import scipy.sparse as sp

OUT = os.path.dirname(os.path.abspath(__file__))
P = lambda n: os.path.join(OUT, n)


# ------------------------------------------------------------------ generic HDF5, superblock v0 (libver earliest)
def common(f, latest):
    rng = np.random.default_rng(7)
    g = f.create_group('num')
    g['i1'] = np.array([-128, -1, 0, 1, 127], dtype='i1')
    g['u1'] = np.array([0, 1, 200, 255], dtype='u1')
    g['i2'] = np.array([-32768, -2, 3, 32767], dtype='<i2')
    g['u2be'] = np.array([0, 1, 65535, 4660], dtype='>u2')
    g['i4be'] = np.array([[-2147483648, 5], [7, 2147483647]], dtype='>i4')
    g['u4'] = np.array([0, 4294967295, 12345], dtype='<u4')
    g['i8'] = np.array([-2**62, -1, 2**53, 9], dtype='<i8')
    g['u8be'] = np.array([0, 2**63, 2**40 + 1], dtype='>u8')
    g['f2'] = np.array([0.5, -1.25, 65504.0, 1e-3], dtype='<f2')
    g['f4be'] = np.array([[1.5, -2.25, 3.0], [4.0e10, -5.5e-10, 0.0]], dtype='>f4')
    g['f8'] = np.arange(24.0).reshape(2, 3, 4) * 0.5 - 3.0
    g['f8be'] = np.array([np.pi, -np.e, 1e-300, 1e300], dtype='>f8')
    g['bool'] = np.array([True, False, True, True])
    g['scalar'] = np.float64(2.5)
    g['empty'] = np.zeros((0, 3))

    s = f.create_group('str')
    s['fixed'] = np.array([b'alpha', b'be', b'gamma12345'], dtype='S10')
    s['vlen_utf8'] = np.array(['héllo', '', '日本語'], dtype=h5py.string_dtype('utf-8'))
    s['vlen_ascii'] = np.array([b'abc', b'de'], dtype=h5py.string_dtype('ascii'))
    s.create_dataset('scalar_vlen', data='scalar text', dtype=h5py.string_dtype('utf-8'))

    c = f.create_group('cmp')
    ct = np.dtype([('x', '<f8'), ('n', '>i4'), ('tag', 'S5'), ('v', '<f4', (2,))])
    c['rec'] = np.array([(1.5, -7, b'ab', (1.0, 2.0)), (-2.0, 9, b'xyz', (3.0, -4.0))], dtype=ct)
    et = h5py.enum_dtype({'RED': 0, 'GREEN': 1, 'BLUE': 42}, basetype='i2')
    c.create_dataset('enum', data=np.array([0, 42, 1, 42], dtype='i2'), dtype=et)
    vt = h5py.vlen_dtype(np.dtype('i4'))
    vl = c.create_dataset('vlen', (3,), dtype=vt)
    vl[0] = [1, 2, 3]; vl[1] = []; vl[2] = [-5]
    c['arr'] = np.array([[1, 2, 3], [4, 5, 6]], dtype='i2')  # plain 2-D for reference targets
    refs = c.create_dataset('refs', (3,), dtype=h5py.ref_dtype)
    refs[0] = f['num/f8'].ref; refs[1] = c['arr'].ref; refs[2] = f['str'].ref
    c.attrs.create('ref_attr', f['num/i1'].ref, dtype=h5py.ref_dtype)

    l = f.create_group('layout')
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_layout(h5py.h5d.COMPACT)
    space = h5py.h5s.create_simple((6,))
    dsid = h5py.h5d.create(l.id, b'compact', h5py.h5t.NATIVE_INT32, space, dcpl=dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(6, dtype='i4') * 11)
    l['contig'] = np.arange(30.0).reshape(5, 6)
    big = (np.arange(37 * 23).reshape(37, 23) % 17 - 8).astype('<i4')
    l.create_dataset('chunked', data=big, chunks=(8, 5))
    l.create_dataset('deflate_shuffle', data=big.astype('<f8') / 4, chunks=(10, 7), compression='gzip', compression_opts=6, shuffle=True)
    l.create_dataset('fletcher', data=big[:, :4].astype('>i2'), chunks=(9, 4), fletcher32=True, compression='gzip')
    sparse = l.create_dataset('missing_chunks', (20, 20), dtype='<f4', chunks=(5, 5), fillvalue=-1.5)
    sparse[0:5, 0:5] = 1.0
    sparse[12, 17] = 7.0
    l.create_dataset('lzf', data=np.arange(100.0), chunks=(50,), compression='lzf')
    l.create_dataset('resizable', data=np.arange(10, dtype='<i8'), chunks=(4,), maxshape=(None,))
    l.create_dataset('two_unlimited', data=np.arange(35, dtype='<i2').reshape(5, 7), chunks=(2, 3), maxshape=(None, None))

    links = f.create_group('links')
    links['soft'] = h5py.SoftLink('/num/i1')
    links['hard'] = f['num/u1']
    sub = links.create_group('deep/er')
    sub['leaf'] = np.array([42.0])

    a = f.create_group('attrs')
    for i in range(20):
        a.attrs[f'a{i:02d}'] = np.float64(i) * 1.5
    a.attrs['text'] = 'attribute text'
    a.attrs['fixed'] = np.bytes_(b'fixed')
    a.attrs['ints'] = np.array([1, 2, 3], dtype='>i8')
    a.attrs['strs'] = np.array(['x', 'yy'], dtype=h5py.string_dtype())
    a.attrs['empty'] = h5py.Empty('f8')
    a.attrs['flag'] = np.bool_(True)

    for i in range(12):  # enough links for dense link storage under libver latest
        f.create_group(f'many/g{i:02d}')


with h5py.File(P('h5_v0.h5'), 'w', libver='earliest') as f:
    common(f, False)

with h5py.File(P('h5_v2.h5'), 'w', libver=('v108', 'v108')) as f:
    f['data'] = np.arange(12.0).reshape(3, 4)
    for i in range(12):
        f['data'].attrs[f'k{i:02d}'] = i
    f.create_dataset('chunked', data=np.arange(100, dtype='<i4'), chunks=(16,), compression='gzip', shuffle=True)

with h5py.File(P('h5_latest.h5'), 'w', libver='latest') as f:
    common(f, True)
    x = f.create_group('index')
    x.create_dataset('single', data=np.arange(12.0).reshape(3, 4), chunks=(3, 4), compression='gzip')
    x.create_dataset('single_raw', data=np.arange(6, dtype='i4'), chunks=(6,))
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_chunk((4,))
    dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
    dsid = h5py.h5d.create(x.id, b'implicit', h5py.h5t.STD_I32LE, h5py.h5s.create_simple((10,)), dcpl=dcpl)
    dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.arange(10, dtype='<i4') * 3)
    x.create_dataset('fixed', data=np.arange(50, dtype='<f8'), chunks=(7,), compression='gzip')
    x.create_dataset('fixed_paged', data=(np.arange(2000) % 251).astype('u1'), chunks=(1,))
    ea = x.create_dataset('ext', (0,), maxshape=(None,), dtype='<i4', chunks=(3,))
    ea.resize((700,)); ea[:] = np.arange(700, dtype='<i4') * 2 - 7
    ea2 = x.create_dataset('ext2d', (4, 0), maxshape=(4, None), dtype='<f4', chunks=(2, 3), compression='gzip')
    ea2.resize((4, 50)); ea2[:] = np.arange(200, dtype='<f4').reshape(4, 50) / 8
    bt = x.create_dataset('bt2', (0, 0), maxshape=(None, None), dtype='<i8', chunks=(2, 2), compression='gzip', shuffle=True)
    bt.resize((9, 5)); bt[:] = np.arange(45, dtype='<i8').reshape(9, 5) * 1000
    edge = x.create_dataset('edge_unfiltered', (7, 5), dtype='<i4', chunks=(3, 2), compression='gzip')
    edge[:] = np.arange(35, dtype='<i4').reshape(7, 5)
    t = f.create_group('ordered', track_order=True)
    for n in ['zeta', 'alpha', 'mid']:
        t[n] = np.array([len(n)])
    t.attrs['z'] = 1; t.attrs['a'] = 2
    f['ctype'] = np.dtype([('a', '<f8'), ('b', '<i2')])  # committed datatype, shared by the dataset and its attribute
    cd = f.create_dataset('committed', (3,), dtype=f['ctype'])
    cd[:] = np.array([(1.5, 2), (3.5, 4), (5.5, 6)], dtype=f['ctype'].dtype)
    cd.attrs.create('shared_attr', np.array([(9.0, 8)], dtype=f['ctype'].dtype), dtype=f['ctype'])
    f['attrs'].attrs['huge'] = np.arange(8200, dtype='<f8')  # > 64 KiB: a huge object outside the fractal heap

# 4-byte offsets and lengths
fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
fcpl.set_sizes(4, 4)
fid = h5py.h5f.create(P('h5_off4.h5').encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl)
with h5py.File(fid) as f:
    f['v'] = np.array([1.0, 2.0, 3.0])
    f.create_dataset('c', data=np.arange(40, dtype='<i2'), chunks=(8,), compression='gzip')
    f['g/s'] = np.array([b'four'], dtype='S4')
    f['g'].attrs['n'] = np.int32(-3)

with h5py.File(P('h5_latest.h5'), 'r') as f:
    print('latest superblock', f.id.get_create_plist().get_version())
    for n in ['single', 'single_raw', 'implicit', 'fixed', 'fixed_paged', 'ext', 'ext2d', 'bt2', 'edge_unfiltered']:
        d = f['index/' + n]
        print(n, d.shape, d.chunks, d.compression, d[()].ravel()[:6])
    print('many is dense?', len(f['many']), list(f['ordered']), list(f['ordered'].attrs))


# ------------------------------------------------------------------ MATLAB v7.3 (h5py, MATLAB layout)
def mat_dict():  # same as gen_fixtures.py, without the old-style object
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
    return dict(
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
        uni='héllo € 日本',
        cplx=np.array([[1 + 2j, -3.5j], [4.0, 0.5 - 0.25j]]),
        nd=np.arange(24.0).reshape(2, 3, 4),
        empty=np.zeros((0, 0)),
        emptystr='',
        sp=spm, spc=spc, splog=splog,
        st=st, starr=starr, cell=cell, nest=nest,
        scalar=3.5,
        vec=np.array([1.0, 2.0, 3.0]),
    )


CLASS = {np.dtype('f8'): 'double', np.dtype('f4'): 'single', np.dtype('i1'): 'int8', np.dtype('u1'): 'uint8',
         np.dtype('i2'): 'int16', np.dtype('u2'): 'uint16', np.dtype('i4'): 'int32', np.dtype('u4'): 'uint32',
         np.dtype('i8'): 'int64', np.dtype('u8'): 'uint64', np.dtype('bool'): 'logical', np.dtype('c16'): 'double'}


def write_mat73(path, variables):
    names = iter([chr(97 + i) + chr(97 + j) for i in range(26) for j in range(26)])
    with h5py.File(path, 'w', userblock_size=512, libver='earliest') as f:
        refs = f.create_group('#refs#')

        def cls(o, name):
            o.attrs.create('MATLAB_class', np.bytes_(name))

        def empty(parent, name, dims, mclass):
            d = parent.create_dataset(name, data=np.array(dims, dtype='u8'))
            cls(d, mclass)
            d.attrs.create('MATLAB_empty', np.uint8(1))
            return d

        def put(parent, name, v):
            if isinstance(v, str):
                if v == '':
                    return empty(parent, name, [0, 0], 'char')
                units = np.frombuffer(v.encode('utf-16-le'), dtype='<u2')
                d = parent.create_dataset(name, data=units.reshape(len(units), 1))
                cls(d, 'char'); d.attrs.create('MATLAB_int_decode', np.int32(2))
                return d
            if sp.issparse(v):
                v = sp.csc_matrix(v)
                g = parent.create_group(name)
                kind = 'logical' if v.dtype == bool else 'double'
                cls(g, kind)
                g.attrs.create('MATLAB_sparse', np.uint64(v.shape[0]))
                if np.iscomplexobj(v.data):
                    data = np.zeros(v.nnz, dtype=[('real', '<f8'), ('imag', '<f8')])
                    data['real'] = v.data.real; data['imag'] = v.data.imag
                else:
                    data = v.data.astype('u1' if kind == 'logical' else '<f8')
                g['data'] = data
                g['ir'] = v.indices.astype('u8')
                g['jc'] = v.indptr.astype('u8')
                return g
            if isinstance(v, dict):
                g = parent.create_group(name)
                cls(g, 'struct')
                fields(g, list(v))
                for k, x in v.items():
                    put(g, k, x)
                return g
            if isinstance(v, (float, np.floating, np.integer, int)):
                v = np.array([[v]], dtype=np.asarray(v).dtype)
            v = np.asarray(v)
            if v.dtype.kind == 'U':
                rows = [list(s) for s in v.ravel()]
                chars = np.array([[ord(c) for c in r] for r in rows], dtype='<u2')  # MATLAB dims (n, len)
                d = parent.create_dataset(name, data=chars.T)
                cls(d, 'char'); d.attrs.create('MATLAB_int_decode', np.int32(2))
                return d
            if v.ndim == 1:
                v = v.reshape(1, -1)
            if v.dtype.names is not None:  # struct array: one reference dataset per field
                g = parent.create_group(name)
                cls(g, 'struct')
                fields(g, list(v.dtype.names))
                for k in v.dtype.names:
                    col = np.empty(v.shape, dtype=object)
                    for idx in np.ndindex(v.shape):
                        col[idx] = v[idx][k]
                    rd = g.create_dataset(k, v.shape[::-1], dtype=h5py.ref_dtype)
                    rd[...] = cellrefs(col).T
                return g
            if v.dtype == object:
                d = parent.create_dataset(name, v.shape[::-1], dtype=h5py.ref_dtype)
                d[...] = cellrefs(v).T
                cls(d, 'cell')
                return d
            if v.size == 0:
                return empty(parent, name, list(v.shape), CLASS[v.dtype])
            data = v.T
            if np.iscomplexobj(v):
                c = np.zeros(data.shape, dtype=[('real', '<f8'), ('imag', '<f8')])
                c['real'] = data.real; c['imag'] = data.imag
                data = c
            elif v.dtype == bool:
                data = data.astype('u1')
            d = parent.create_dataset(name, data=data)
            cls(d, CLASS[v.dtype])
            return d

        def cellrefs(objs):
            out = np.empty(objs.shape, dtype=h5py.ref_dtype)
            for idx in np.ndindex(objs.shape):
                out[idx] = put(refs, next(names), objs[idx]).ref
            return out

        def fields(g, keys):
            arr = np.empty(len(keys), dtype=object)
            for i, k in enumerate(keys):
                arr[i] = np.array([c.encode() for c in k], dtype='S1')
            g.attrs.create('MATLAB_fields', arr, dtype=h5py.vlen_dtype(np.dtype('S1')))

        for k, v in variables.items():
            put(f, k, v)
        fh = f.create_group('fh')
        cls(fh, 'function_handle')
        sobj = f.create_dataset('strobj', data=np.array([[3707764736, 2, 1, 1, 1, 1]], dtype='u4').T)
        cls(sobj, 'string'); sobj.attrs.create('MATLAB_object_decode', np.int32(3))
        g = f.create_dataset('gl', data=np.array([[7.0]]))
        cls(g, 'double'); g.attrs.create('MATLAB_global', np.uint8(1))
    head = b'MATLAB 7.3 MAT-file, Platform: GLNXA64, Created on: Sat Oct 17 12:00:00 2026 HDF5 schema 1.00 .'
    head = head.ljust(116, b' ') + b'\0' * 8 + b'\x00\x02' + b'IM'
    with open(path, 'r+b') as fh:
        fh.write(head)


write_mat73(P('mat73_scinet.mat'), mat_dict())
with h5py.File(P('mat73_scinet.mat'), 'r') as f:
    print('mat73 userblock', f.userblock_size, sorted(f.keys()))


# ------------------------------------------------------------------ NetCDF-4 (netCDF4-python)
import netCDF4

with netCDF4.Dataset(P('nc4_classic.nc'), 'w', format='NETCDF4_CLASSIC') as ds:
    ds.title = 'SciNet NetCDF-4 classic model'
    ds.version = np.int32(3)
    ds.createDimension('time', None)
    ds.createDimension('lat', 3)
    ds.createDimension('lon', 4)
    ds.createDimension('strlen', 6)
    t = ds.createVariable('time', 'f8', ('time',))
    t.units = 'hours since 2000-01-01'
    t[:] = [0.0, 6.0]
    la = ds.createVariable('lat', 'f4', ('lat',)); la[:] = [-10.0, 0.0, 10.0]; la.units = 'degrees_north'
    lo = ds.createVariable('lon', 'f4', ('lon',)); lo[:] = [100.0, 110.0, 120.0, 130.0]
    tp = ds.createVariable('temp', 'i2', ('time', 'lat', 'lon'), zlib=True, complevel=4, shuffle=True, fill_value=np.int16(-999))
    tp.scale_factor = np.float32(0.01)
    tp.add_offset = np.float32(273.15)
    tp.set_auto_maskandscale(False)
    raw = (np.arange(24, dtype='i2').reshape(2, 3, 4) * 37 - 300)
    raw[1, 2, 3] = -999
    tp[:] = raw
    sc = ds.createVariable('scalar', 'f8'); sc[...] = 42.5
    nm = ds.createVariable('name', 'S1', ('lat', 'strlen'))
    nm[:] = netCDF4.stringtochar(np.array(['south', 'eq', 'north'], dtype='S6'))
    b = ds.createVariable('flags', 'i1', ('lon',)); b[:] = [-1, 0, 1, 127]

with netCDF4.Dataset(P('nc4_enhanced.nc'), 'w', format='NETCDF4') as ds:
    ds.createDimension('x', 5)
    ds.createDimension('rec', None)
    ds.history = 'created for SciNet tests'
    ds.setncattr_string('tags', ['a', 'bb'])
    v = ds.createVariable('u16', 'u2', ('x',)); v[:] = [0, 1, 65535, 300, 7]
    v = ds.createVariable('i64', 'i8', ('x',)); v[:] = [-2**40, 0, 1, 2**52, -5]
    v = ds.createVariable('u64', 'u8', ('x',)); v[:] = [0, 2**63, 3, 4, 5]
    v = ds.createVariable('u8', 'u1', ('x',)); v[:] = [0, 255, 128, 1, 2]
    v = ds.createVariable('u32', 'u4', ('x',)); v[:] = [0, 4294967295, 7, 8, 9]
    s = ds.createVariable('words', str, ('x',))
    for i, w in enumerate(['one', 'two', 'über', '', 'five']):
        s[i] = w
    r = ds.createVariable('series', 'f4', ('rec', 'x'), chunksizes=(2, 5))
    r[0:3, :] = np.arange(15, dtype='f4').reshape(3, 5) + 0.25
    g = ds.createGroup('forecast')
    g.createDimension('step', 2)
    g.model = 'toy'
    fv = g.createVariable('wind', 'f8', ('step', 'x'), fill_value=-1e30)
    fv[0, :] = np.arange(5) * 1.5
    sub = g.createGroup('inner')
    sub.createVariable('k', 'i4', ())[...] = 11

for n in ['nc4_classic.nc', 'nc4_enhanced.nc']:
    with netCDF4.Dataset(P(n)) as ds:
        ds.set_auto_mask(False)
        print(n, ds.data_model, list(ds.dimensions.items()), {k: (v.dtype, v.dimensions, v.ncattrs()) for k, v in ds.variables.items()})
        for k, v in ds.variables.items():
            print('  ', k, v[...].ravel()[:8].tolist() if v.dtype != str else v[...].tolist())
        if n == 'nc4_enhanced.nc':
            print('  forecast/wind', ds['forecast/wind'][...].tolist(), ds['forecast/inner/k'][...])

for n in sorted(os.listdir(OUT)):
    if n.startswith(('h5_', 'mat73_', 'nc4_')):
        print(n, os.path.getsize(P(n)))
