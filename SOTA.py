try: import cPickle
except ImportError: import _pickle as cPickle
import ctypes
import heapq
import json
import math
import multiprocessing.sharedctypes
import sys
import timeit

from array import array

import numpy, numpy.fft
numpy_fftpack_lite = getattr(numpy.fft, 'fftpack_lite', None)

try: from scipy.fftpack._fftpack import drfft
except ImportError: drfft = {}.get(None)

if sys.version_info >= (3, 0): exec("print_ = lambda *args, **kwargs: print(*args, **kwargs) and None or (kwargs.get('file') or sys.stdout).flush(); xrange = range")
else: print_ = __import__('__builtin__').__dict__['print']

NUMBA_CACHE = True
NUMBA_EAGER = False
try:
	import numba  # OPTIONAL (~3x speedup)
	try: numba.__version__  # I commented out the version determination in my local system, so make sure it's set before using cache=True
	except AttributeError: numba.__version__ = '0+unknown'
except ImportError: pass

try:
	from scipy.special import ndtr
except ImportError as e:
	def polyval(x, c, out=None):
		k = len(c) - 1
		if out is None:
			out = numpy.zeros_like(x)
		else:
			if out is x:
				x = numpy.copy(x)
			out[...] = 0
		out += c[k]
		while k:
			out *= x
			out += c[k - 1]
			k -= 1
		return out
	def erf_approx(x, out=None):
		negate = x > 0
		x = numpy.abs(x, x if x is out else None)
		out = polyval(x, (1, 0.0705230784, 0.0422820123, 0.0092705272, 0.0001520143, 0.0002765672, 0.0000430638), out)
		out = numpy.power(out, -16, out)
		out -= 1
		out[negate] = -out[negate]
		return out
	def ndtr(x, out=None):
		out = numpy.divide(x, 1.4142135623730950488, out)
		erf_approx(out, out)
		out += 1
		out /= 2
		return out

def slotted_getstate(cls, self):
	nonpickled = getattr(cls, '__nonpickled_slots__', ())
	result = {}
	for field in cls.__slots__:
		if field not in nonpickled:
			try: result[field] = cls.__getattribute__(self, field)
			except AttributeError: pass
	return result

def slotted_setstate(cls, self, state):
	for field in state:
		cls.__setattr__(self, field, state[field])

class Array(object):
	__nonpickled_slots__ = ('ndarray',)
	__slots__ = ('_impl', '_len', '_cap', '_off', 'dtype', '_default_value', '_min_size_on_reallocate') + __nonpickled_slots__
	def __init__(self, dtype, default_value=None):
		self.dtype = dtype
		self.ndarray = {}.get(None)
		self._off = 0
		self._len = 0
		self._cap = 0
		self._impl = {}.get(None)
		self._default_value = default_value
		self._min_size_on_reallocate = 0
		self.resize(0)
		self._postinit()
	@classmethod
	def fromitems(cls, dtype, items):
		n = len(items)
		result = cls(dtype)
		result.ensure_size(n)[:n] = items
		return result
	def _postinit(self):
		self.ndarray = Array._get_ndarray(self)
	def __getstate__(self): cls = Array; return slotted_getstate(cls, self)
	def __setstate__(self, state): cls = Array; slotted_setstate(cls, self, state); cls._postinit(self)
	@staticmethod
	def create_buffer(typecode, capacity, initializer={}.get(None)):
		result = multiprocessing.sharedctypes._new_value(multiprocessing.sharedctypes.typecode_to_type.get(typecode, typecode) * capacity)
		if initializer is not None:
			numpy.frombuffer(result, typecode, capacity)[:capacity] = initializer
		return result
	def resize(self, length):
		old_len = self._len
		old_ndarray = self.ndarray
		need_to_copy_over = False
		if self._impl is None or self._cap < length:
			capacity = old_len + (old_len >> 1)
			if capacity < length: capacity = length
			self._impl = Array.create_buffer(self._typecode, capacity) if capacity > 0 else {}.get(None)
			self._cap = capacity
			self._off = 0
			need_to_copy_over = True
		self._len = length
		self._postinit()
		result = self.ndarray
		min_len = old_len
		if min_len > length:
			min_len = length
		if min_len and need_to_copy_over:
			result[:min_len] = old_ndarray[:min_len]
		if min_len < length and self._default_value is not None:
			result[min_len:length] = self._default_value
		return result
	def switch_buffer(self, buffer, buffer_offset, buffer_capacity):
		assert buffer_offset >= 0 and buffer_capacity >= 0 and buffer_offset + buffer_capacity <= len(buffer) and buffer_capacity >= self._len
		old_ndarray = self.ndarray
		self._impl = buffer
		self._off = buffer_offset
		self._cap = buffer_capacity
		self._postinit()
		self.ndarray[:] = old_ndarray
	@staticmethod
	def compute_type_code(dtype):
		if dtype == float: type_code = 'd'
		elif dtype == int: type_code = 'l'  # 'q' not supported in Python 2
		elif dtype == bool: type_code = 'B'  # one-byte bools
		else: raise ValueError(dtype)
		return type_code
	@property
	def _typecode(self):
		return Array.compute_type_code(self.dtype)
	def assert_size(self, n):
		assert n <= self._len
		return self.ensure_size(n)
	def ensure_size(self, n, actually_allocate=True):
		assert n >= 0, "negative size is probably a bug somewhere"
		if actually_allocate:
			m = self._len
			if n > m:
				result = self.resize(n if n >= self._min_size_on_reallocate else self._min_size_on_reallocate)
			else:
				result = self.ndarray
			return result
		else:
			self._min_size_on_reallocate = n
	def _get_ndarray(self):
		impl = self._impl
		return numpy.frombuffer(impl, self.dtype, self._len, self._off * numpy.dtype(self.dtype).itemsize) if impl is not None else None
	def tolist(self): return list(self)
	def __getslice__(self, i, j, stride=1):
		if i is None: i = 0
		if i > self._len: i = self._len
		if j is None or j > self._len: j = self._len
		assert 0 <= i <= j
		return self._impl[self._off + i : self._off + j : stride]
	def __getitem__(self, i):
		if isinstance(i, slice): return self.__getslice__(i.start, i.stop, i.step)
		if i >= self._len: raise IndexError('invalid index')  # list() relies on this to find where it ends
		assert 0 <= i < self._len
		return self._impl[self._off + i]
	def __setitem__(self, i, value):
		assert 0 <= i < self._len
		self._impl[self._off + i] = value
	def __len__(self): return self._len
	def __repr__(self): return repr(self.tolist())

def with_numba(generator={}.get(None)):
	try: result = numba
	except NameError: result = {}.get(None)
	return generator(result) if generator is not None else result

def signature(*args, **kwargs):
	kwargs.setdefault('nopython', True)
	kwargs.setdefault('cache', NUMBA_CACHE)
	eager = NUMBA_EAGER
	if not eager: args = ()
	jitter = numba.jit(*args, **kwargs) if with_numba() else None
	def wrapper(f):
		if jitter:
			if eager: tprev = timeit.default_timer(); print_("Compiling with Numba: %s..." % (f.__name__,), end=' ', file=sys.stderr)
			try:
				f = jitter(f)
			finally:
				if eager and 'tprev' in locals(): print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev
		return f
	return wrapper

def numba_single_overload_entrypoint(f):
	try: overloads = f.overloads
	except AttributeError: overloads = {}.get(None)
	if overloads is not None and len(overloads) == 1:
		[overload] = f.overloads.values()
		f = overload.entry_point
	return f

def fftpack_lite_rfftb(buf, s, temp_buf=[()]):
	n = len(buf)
	m = (n - 1) * 2
	temp = temp_buf[0]
	if m >= len(temp): temp_buf[0] = temp = numpy.empty(m * 2, buf.dtype)
	numpy.divide(buf, m, temp[0:n])
	temp[n:m] = 0
	result = (numpy_fftpack_lite.rfftb if numpy_fftpack_lite is not None else numpy.fft.irfft)(temp[0:m], s)
	if numpy_fftpack_lite is None:
		result *= s
	return result

def fftpack_drfftf(buf, s, drfft=drfft):
	return drfft(buf, None, 1, 0, 1)

@signature('void(double[::1], double[::1], double[::1])')
def fftpack_multiply(a, b, result):
	n = len(result)
	if n >= 1:
		result[0] = a[0] * b[0]
		if n >= 2:
			complex = numpy.complex128
			numpy.multiply(a[1:-1].view(complex), b[1:-1].view(complex), result[1:-1].view(complex))
			result[-1] = a[-1] * b[-1]

def fftpack_drfftb(buf, s, drfft=drfft):
	return drfft(buf, None, -1, 1, 1)

def with_scope(obj, func):
	with obj: return func(obj)

def discretize_up(t, dt, divide=numpy.divide, ceil=numpy.ceil):
	r = divide(t, dt)
	return ceil(r, r)

def discretize_down(t, dt, divide=numpy.divide, floor=numpy.floor):
	r = divide(t, dt)
	return floor(r, r)

def arange_len(start, stop, step, ceil=math.ceil, int=int):
	return int(ceil((stop - start) / step))

EARTH_RADIUS_MM = 6.371009

def geographic_to_cartesian(coord):
	c1 = numpy.cos((coord[0] / 180) * numpy.pi)
	return numpy.asarray((c1 * numpy.cos((coord[1] / 180) * numpy.pi), c1 * numpy.sin((coord[1] / 180) * numpy.pi), numpy.sin((coord[0] / 180) * numpy.pi)), float)

def cartesian_to_geographic(x, y, z):
	r = numpy.asarray((numpy.arcsin(z / numpy.linalg.norm((x, y, z))), numpy.arctan2(y, x)))
	r *= 180 / numpy.pi
	return r

class RSet(set):
	def _not_implemented(self, *args): raise NotImplementedError()
	list(map(lambda key, locals_=locals(), not_implemented=_not_implemented: locals_.setdefault(key, not_implemented), filter(lambda key: key not in object.__dict__ and key not in ('copy',), set.__dict__.keys())))
	def __init__(self, current={}.get(None), parent={}.get(None)):
		# if parent is not None: assert isinstance(parent, RSet)
		set.__init__(self, current) if current is not None else set.__init__(self)
		self._parent = parent
	def __contains__(self, item, set_contains=set.__contains__, set_iter=set.__iter__, set_update=set.update):
		found = set_contains(self, item)
		if not found:
			node = self._parent
			if node is not None:
				found = found or set_contains(node, item)
				if not found:
					parent = node._parent
					while not found and parent is not None:
						found = found or set_contains(parent, item)
						set_update(node, set_iter(parent))
						node._parent = parent = parent._parent
		return found

class Records(object):
	def __len__(self): return NotImplemented
	def __getitem__(self, index):
		result = []
		for (k, v) in self.__dict__.items():
			result.append((k, v[index]))
		return result
	def __setitem__(self, index, value):
		for (k, v) in value:
			assert k in self.__dict__
			self.__dict__[k] = v
	def __detitem__(self, index):
		for (k, v) in self.__dict__.items():
			del v[index]

class Edges(Records):
	def __init__(self, n):
		self.id              = [None] * n
		self.startNodeId     = [None] * n
		self.endNodeId       = [None] * n
		self.tmin            = [0.0 ] * n
		self.lanes           = [1   ] * n
		self.hmm             = [None] * n
		self.length          = [0.0 ] * n
		self.tidist_override = [None] * n
		self.begin           = [0   ] * n
		self.end             = [0   ] * n
		self.geom_points     = [None] * n
	def __len__(self): return len(self.id)

class Nodes(Records):
	def __init__(self, n):
		self.outgoing = list(map(lambda _: [], range(n)))
		self.incoming = list(map(lambda _: [], range(n)))
	def __len__(self): return len(self.outgoing)

class Network(object):
	def __init__(self, edges):
		self.edges = edges
		self.nodes = Nodes(len(numpy.union1d(self.edges.begin, self.edges.end)))
		for i in xrange(len(self.edges)):
			j = i if True else 0
			self.nodes.outgoing[self.edges.begin[i]].append(j)
			self.nodes.incoming[self.edges.end  [i]].append(j)

	def print_graph(self, stdout, visited_iedges=None):
		print_("digraph network", file=stdout)
		print_("{", file=stdout)
		print_("\tgraph[rankdir=LR]", file=stdout);
		for (eij, edge) in enumerate(self.edges):
			print_("\t%s -> %s [penwidth=%s, label=\"%s\"];" % (edge.startNodeId, edge.endNodeId, 1 + int(visited_iedges[eij] if visited_iedges is not None else 0), edge.id))
		print_("}", file=stdout)
	def __getstate__(self):
		stderr = sys.stderr if False else None
		if stderr:
			from timeit import default_timer as timer
			tprev = timer()
		result = cPickle.dumps(self.__dict__)
		if stderr:
			print_("Pickling Network... %.0f ms" % ((timer() - tprev) * 1000,), file=stderr)
		return result
	def __setstate__(self, state):
		stderr = sys.stderr if False else None
		if stderr:
			from timeit import default_timer as timer
			tprev = timer()
		loaded = cPickle.loads(state)
		result = self.__dict__.update(loaded)
		if stderr:
			print_("Unpickling Network... %.0f ms" % ((timer() - tprev) * 1000,), file=stderr)
		return result
	@staticmethod
	def discretize_edges(edges_hmm, edges_tmin, dt,
		inf=float('inf'),
		tzmax=8.292362,
		alias_output_with_input_at_extra_memory_cost=True,
		len=len, float=float, zip=zip, arange_len=arange_len, ndtr=ndtr,
		numpy_empty=numpy.empty, numpy_ones=numpy.ones, numpy_zeros=numpy.zeros,
		discretize_down=discretize_down,
		suppress_calculation=False):
		tmins = discretize_down(edges_tmin, dt) * dt
		arrs_specs = []; arrs_specs_append = arrs_specs.append
		for tmin, edge_hmm in zip(tmins, edges_hmm):
			tmin = float(tmin)
			arr_specs = []; arr_specs_append = arr_specs.append
			for hmm_node in edge_hmm:
				tstart = (tmin - hmm_node[1]) / hmm_node[2]
				tstep = dt / hmm_node[2]
				tend = tstart + tstep / 2
				if tend < tzmax: tend = tzmax
				arr_specs_append((arange_len(-tstart, -tend, -tstep), -tstart, -tstep, hmm_node[3]))
			arrs_specs_append(arr_specs)
		all_components_length = 0
		for arr_specs in arrs_specs:
			for arr_spec in arr_specs:
				all_components_length += arr_spec[0]
		all_components = (numpy_ones if not suppress_calculation else numpy_empty)(all_components_length)
		ntotal = 0
		for arr_specs in arrs_specs:
			assert len(arr_specs) > 0, "Mixture model cannot be empty"
			# Sort in descending order so we can use the first component's portion as the return buffer
			arr_specs.sort(key=lambda arr_spec: ~arr_spec[0])

			npartial = 0
			for arr_spec in arr_specs:
				npartial += arr_spec[0]
			components = all_components[ntotal : ntotal + npartial]
			ntotal += npartial
			offset = 0
			nmax = 0
			for (n, tstart, tstep, prob) in arr_specs:
				if nmax < n: nmax = n
				component = components[offset : offset + n]
				if not suppress_calculation:
					# NOTE: Floating-point round-off error is introduced here for performance (removes 1 extra pass)
					# To remove the round-off error, set component[0] = 0 and then do component += tstep
					# Equivalent to: component[:] = numpy.arange(n) * tstep + tstart
					component[0] = tstart / tstep
					component = component.cumsum(out=component)
					component *= tstep
				#assert numpy.allclose(component, numpy.arange(tstart, tstart + (n - 0.5) * tstep, tstep, float))
				component[0] = inf
				offset += n
			if not suppress_calculation:
				components = ndtr(components, components)
			if alias_output_with_input_at_extra_memory_cost:
				mixture = components[:nmax]  # we can let this alias, since the longest component was first...
			else:
				mixture = (numpy_zeros if not suppress_calculation else numpy_empty)(nmax)
			offset = 0
			for (n, tstart, tstep, prob) in arr_specs:
				r = components[offset : offset + n]
				r_slice_delayed = r[+1:]
				if not suppress_calculation:
					r[:-1] -= r_slice_delayed  # NOTE: arrays ALIAS! Do NOT reverse the order of subtraction!
					r *= prob #/ numpy.sum(r)
					if (not alias_output_with_input_at_extra_memory_cost) or (offset > 0):
						mixture[:len(r)] += r
				offset += n
			yield mixture

	@staticmethod
	def make_id(primary, secondary): return (primary, secondary)

	@classmethod
	def remove_bad_edges(self, edges, min_sdev, simulate):
		edges_ids = edges.id
		edges_startNodeId = edges.startNodeId
		edges_endNodeId = edges.endNodeId
		edges_tmin = edges.tmin
		edges_hmm = edges.hmm
		edges_length = edges.length
		if simulate:
			speed_limits = [
				# mi/h
				5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
				15, 15, 15, 15, 15, 15, 15, 15,
				25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
				25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25,
				35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35, 35,
				50, 50, 50, 50,
				55, 55, 55, 55, 55, 55, 55, 55,
				65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65
			]
			pos_inf = float('inf')
			neg_inf = -pos_inf
			for i in xrange(len(speed_limits)):
				speed_limits[i] = speed_limits[i] * 1609.344 / 3600
			for i in xrange(len(edges)):
				if not (neg_inf < edges_tmin[i] < pos_inf):
					edges_tmin[i] = edges_length[i] / speed_limits[((edges_ids[i] // 2) + (edges_startNodeId[i] // 2) + (edges_endNodeId[i] // 2)) % len(speed_limits)]
			typical_modes = []
			for i in xrange(len(edges)):
				edge_tmin = edges_tmin[i]
				if neg_inf < edge_tmin < pos_inf:
					for hmm_node in edges_hmm[i]:
						if hmm_node[2] >= min_sdev:
							(mode, mean, sdev, prob) = hmm_node
							typical_modes.append((mode, mean / edge_tmin, sdev / edge_tmin, prob))
			mode_go = u'go'
			typical_modes.append((mode_go, 1.0, 0.1, 1))
			typical_modes.append((mode_go, 1.0, 0.2, 1))
			typical_modes.append((mode_go, 1.0, 0.3, 1))
			typical_modes.append((mode_go, 1.0, 0.5, 1))
			typical_modes.append((mode_go, 1.2, 0.8, 1))
			typical_modes.append((mode_go, 1.2, 1.0, 1))
			typical_modes.append((mode_go, 1.2, 1.2, 1))
			typical_modes.append((mode_go, 1.5, 1.4, 1))
			typical_modes.append((mode_go, 1.5, 1.6, 1))
			typical_modes.append((mode_go, 2.0, 1.8, 1))
			mode_names = sorted(frozenset(map(lambda item: item[0], typical_modes)))
			mode_names_indices = dict(map(lambda p: p[::-1], enumerate(mode_names)))
			typical_modes_sortable = numpy.asarray(list(map(lambda p: (mode_names_indices[p[0]],) + p[-1:-4:-1], typical_modes)), float)
			typical_modes_sorted_reverse = typical_modes_sortable[numpy.lexsort(typical_modes_sortable.T, 0)][::-1, ::-1]
			ntypical_modes = len(typical_modes)
			if ntypical_modes > 0:
				for (edge_hmm, edge_id, edge_startNodeId, edge_endNodeId, edge_tmin) in zip(
					edges_hmm, edges_ids, edges_startNodeId, edges_endNodeId, edges_tmin
				):
					if len(edge_hmm) == 0:  # Make sure at least one distribution component exists...
						edge_hmm.append([None, None, None, None])
					kn = len(edge_hmm)
					ki = 0
					while ki < kn:
						k = edge_hmm[ki]
						if (k[2] is None or k[2] < min_sdev and (k[3] > 0 or ki == 0)):
							j = (edge_id[0] + edge_startNodeId[0] + edge_endNodeId[0] + ki) % ntypical_modes
							typical_modes_sorted_reverse_j = tuple(typical_modes_sorted_reverse[j].tolist())
							k[0] = mode_names[int(typical_modes_sorted_reverse_j[3])]
							k[1] = typical_modes_sorted_reverse_j[0] * edge_tmin
							k[2] = typical_modes_sorted_reverse_j[1] * edge_tmin
							k[3] = typical_modes_sorted_reverse_j[2]
						ki += 1
					sum = 0
					for k in edge_hmm:
						sum += k[3]
					if sum > 0:
						ki = 0
						while ki < kn:
							edge_hmm[ki][3] /= sum
							ki += 1
		else:
			j = 0
			for i in xrange(len(edges)):
				if edges_hmm[i][0][2] >= min_sdev:
					(edges[i], edges[j]) = (edges[j], edges[i])
					j += 1
			del edges[j:]


	@classmethod
	def load_edges(self, network_json_file):
		with numpy.errstate(over='raise') as es:
			def object_pairs_hook(items, make_id=self.make_id):
				result = None
				for (k, v) in items:
					ki = k
					if ki == u'lat':
						return items
					elif ki == u'primary':
						for (k2, v2) in items:
							if k2 == u'primary': a = v2
							elif k2 == u'secondary': b = v2
						return make_id(a, b)
					elif ki == u'points':
						# this is a geometry object
						result = []
						for obj in v:
							for (k2, v2) in obj:
								if k2 == u'lat': a = v2
								elif k2 == u'lon': b = v2
								else: pass
							result.append((a, b))
						return result
					if result is None:
						result = {ki: v}
					else:
						result[ki] = v
				if result is None: result = {}
				return result
			if True:
				odict_pop = dict.pop
			else:
				def odict_pop(odict, key, default=None):
					for i, p in enumerate(odict):
						if key == p[0]:
							odict[i] = odict[-1]
							del odict[-1]
							return p[1]
					return default
			def json_scan_all(f):
				scanner = json.JSONDecoder(object_pairs_hook=object_pairs_hook).scan_once
				s = f.read()
				result = []
				i = 0
				n = len(s)
				while 0 <= i and i < n:
					(parsed, i) = scanner(s, i)
					result.append(parsed)
					i = s.find('{', i)
				return result

			network_json = json_scan_all(network_json_file)
			edges = Edges(len(network_json))
			node_indices = {}
			def define_node(id_):
				index = node_indices.get(id_)
				if index is None: node_indices[id_] = index = len(node_indices)
				return index
			def canonicalize_id(id_):
				if isinstance(id_, int):
					return (id_, 0)
				if isinstance(id_, tuple):
					return id_
				return tuple(id_)
			for iedge, edge in enumerate(network_json):
				edgeId      = canonicalize_id(odict_pop(edge, u'id'))
				startNodeId = canonicalize_id(odict_pop(edge, u'startNodeId', None) or odict_pop(edge, u'startNodeID'))
				endNodeId   = canonicalize_id(odict_pop(edge, u'endNodeId'  , None) or odict_pop(edge, u'endNodeID'  ))
				length      = odict_pop(edge, u'length')
				hmm         = odict_pop(edge, u'hmm', None)
				edges.id              [iedge] = edgeId
				edges.startNodeId     [iedge] = startNodeId
				edges.endNodeId       [iedge] = endNodeId
				edges.tmin            [iedge] = length / odict_pop(edge, u'speedLimit', 1)
				edges.lanes           [iedge] = odict_pop(edge, u'lanes', None) or odict_pop(edge, u'numLanes', None) or 1
				edges.hmm             [iedge] = list(map(lambda p: [p[u'mode'], p[u'mean'], p[u'sdev'] if u'sdev' in p else p[u'cov'] ** 0.5, p[u'prob']], hmm if hmm is not None else []))
				edges.length          [iedge] = length
				edges.tidist_override [iedge] = None
				edges.begin           [iedge] = define_node(startNodeId)
				edges.end             [iedge] = define_node(endNodeId)
				edges.geom_points     [iedge] = odict_pop(edge, u'geom')
			counter = {}
			for geom_points in edges.geom_points:
				key = tuple(geom_points)
				counter[key] = counter.get(key, 0) + 1
			for geom_points in edges.geom_points:
				if counter[tuple(geom_points)] > 1 and len(geom_points) > 1:
					pdiff = numpy.subtract(geographic_to_cartesian(geom_points[-1]), geographic_to_cartesian(geom_points[+0]))
					for j in xrange(len(geom_points)):
						geom_point = geom_points[j]
						phere = geographic_to_cartesian(geom_point)
						pperp = numpy.cross(pdiff, phere)
						pperp /= numpy.linalg.norm(pperp)
						pperp *= 0.000005 / EARTH_RADIUS_MM
						geom_points[j] = tuple(cartesian_to_geographic(*(phere + pperp))) + geom_point[2:]
			return (edges, node_indices)

@signature('(intp, intp, intp, intp)')
def zdconvolution(an, bn, i=None, j=None):
	result = []
	token = 0
	while True:
		im1_0 = i - (i != 0)
		jm1_0 = j - (j != 0)
		v = token
		if v:
			v >>= 1
		else:
			bnm1 = bn - 1
			v = im1_0 ^ jm1_0
			v = v if v < bnm1 else bnm1

			if True:
			#	# fast flp2() method for Python
			#	v = (((v - v) + 1) << v.bit_length()) >> 1
			#else:  # slow flp2() method for other languages
				k = 1
				while k < 64:
					v |= v >> k
					k <<= 1
				v = v - (v >> 1)
		vm1_c = ~(v - 1)
		a1 = im1_0 & vm1_c if v else i
		a2 = jm1_0 & vm1_c if v else j
		b1 = v
		b2 = v << 1 if v else (i < j) + 0
		a1_returned = a1 if a1 < an else an
		a2_returned = a2 if a2 < an else an
		b1_returned = b1
		b2_returned = b2 if b2 < bn else bn
		output = (
			a1_returned,
			a2_returned,
			b1_returned,
			b2_returned,
			a1 + b1,
			a2_returned - a1_returned,
			b2_returned - b1_returned
		)
		if output[0] == output[1]: break
		result.append(output)
		token = v
		if not token: break
	result.reverse()  # optional
	return result

@with_numba
def convolve_into(numba):
	if numba:
		@signature('(double[::1], intp, intp, double[::1], intp, intp, double[::1], intp, intp, intp, bool_)')
		def convolve_into(a, ai, an, b, bi, bn, c, ci, cn, coffset, accumulate_instead_of_assigning=False):
			if an == -1: an = len(a) - ai
			if bn == -1: bn = len(b) - bi
			if cn == -1:
				cn = an + bn - 1
				cn_lim = len(c) + coffset - ci
				if cn_lim < 0: cn_lim = 0
				if cn > cn_lim: cn = cn_lim
			assert ai + an <= len(a)
			dot = numpy.dot
			bnn = len(b)
			for i in xrange(ci, ci + cn):
				j1 = i + 1 - bn
				if j1 < 0: j1 = 0
				j2 = i + 1
				if j2 > an: j2 = an
				#j1 = max(i + 1 - bn, 0)
				#j2 = min(i + 1, an)
				if 0:
					v = dot(a[ai + j1 : ai + j2], b[bi + i - j1 : bi + i - j2 - bnn : -1])  # convolve
				else:
					v = 0.
					for j in xrange(j1, j2):
						# v += a[ai + j] * b[bi + bn - (i + 1) + j]  # correlate
						v += a[ai + j] * b[bi + i - j]   # convolve
				k = i - coffset
				assert 0 <= k < len(c)
				if accumulate_instead_of_assigning:
					c[k] += v
				else:
					c[k] = v
	else:
		def direct_convolution_cost(an, bn, ci, cn):
			nmin = min(an, bn)
			tot = max(an + bn - 1, 0)
			cl = min(ci, nmin)
			cr = min(tot - ci - cn, nmin)
			return (nmin * (nmin + abs(an - bn)) - (cl * (cl + 1) + cr * (cr + 1)) // 2, tot)
		def convolve_into(a, ai, an, b, bi, bn, c, ci, cn, coffset, accumulate_instead_of_assigning=False):
			if an == -1: an = len(a) - ai
			if bn == -1: bn = len(b) - bi
			if cn == -1:
				cn = an + bn - 1
				cn_lim = len(c) + coffset - ci
				if cn_lim < 0: cn_lim = 0
				if cn > cn_lim: cn = cn_lim
			if an <= 2 or bn <= 2 or (an + bn - 1) * 4 >= cn or direct_convolution_cost(an, bn, 0, an + bn - 1)[0] * 2 >= sum(direct_convolution_cost(an, bn, ci, cn)):
				convolution = numpy.core.multiarray.correlate2(a[ai : ai + an], b[bi : bi + bn][::-1], 2)
				if accumulate_instead_of_assigning:
					c[ci - coffset : ci + cn - coffset] += convolution[ci : ci + cn]
				else:
					c[ci - coffset : ci + cn - coffset] = convolution[ci : ci + cn]
			else:
				assert ai + an <= len(a)
				dot = numpy.dot
				bnn = len(b)
				for i in xrange(ci, ci + cn):
					j1 = i + 1 - bn
					if j1 < 0: j1 = 0
					j2 = i + 1
					if j2 > an: j2 = an
					v = dot(a[ai + j1 : ai + j2], b[bi + i - j1 : bi + i - j2 - bnn : -1])  # convolve
					k = i - coffset
					assert 0 <= k < len(c)
					if accumulate_instead_of_assigning:
						c[k] += v
					else:
						c[k] = v
	return convolve_into

@signature('void(int_, int_, int_, bool_, int_, bool_, int_[::1], int_[::1], int_[::1], int_, bool_)')
def heap_sift(r, begin, end, down, arity, min_heap, heap, heap_index_to_node, heap_node_to_index, j_if_auto_detect_direction, swap_elements):
	i = begin
	j = j_if_auto_detect_direction
	if swap_elements:
		a = heap_index_to_node[i]
		b = heap_index_to_node[j]
		tmp = heap_node_to_index[a]
		heap_node_to_index[a] = heap_node_to_index[b]
		heap_node_to_index[b] = tmp
		heap_index_to_node[i] = b
		heap_index_to_node[j] = a
		tmp = heap[i]
		heap[i] = heap[j]
		heap[j] = tmp
	if j >= 0:
		k1 = j if min_heap else i
		k2 = i if min_heap else j
		down = heap[k1] < heap[k2] or heap[k1] <= heap[k2] and heap_index_to_node[k1] < heap_index_to_node[k2]
	down_xor_min_heap = down != min_heap
	i = r
	while True:
		if down:
			b = (i - begin) * arity + 1 + begin
			if b > end: b = end
			e = arity + b
			if e > end: e = end
			j = b
			if min_heap: k2 = j
			else: k1 = j
			b += 1
			while b < e:
				if min_heap: k1 = b
				else: k2 = b
				if heap[k1] < heap[k2] or heap[k1] <= heap[k2] and heap_index_to_node[k1] < heap_index_to_node[k2]:
					j = b
				b += 1
		else:
			j = begin + ((i - begin - 1) // arity if i != begin else 0)
		if j == end: break
		if down_xor_min_heap: k1 = i; k2 = j
		else: k1 = j; k2 = i
		if not (heap[k1] < heap[k2] or heap[k1] <= heap[k2] and heap_index_to_node[k1] < heap_index_to_node[k2]):
			break

		a = heap_index_to_node[i]
		b = heap_index_to_node[j]
		tmp = heap_node_to_index[a]
		heap_node_to_index[a] = heap_node_to_index[b]
		heap_node_to_index[b] = tmp
		heap_index_to_node[i] = b
		heap_index_to_node[j] = a
		tmp = heap[i]
		heap[i] = heap[j]
		heap[j] = tmp

		if r == i: r = j
		i = j
	r = i

def dijkstra(network, incoming, init, edge_lengths, revisit, ibudget, min_itimes_to_dest):
	children_nodes = network.nodes.incoming if incoming else network.nodes.outgoing
	return dijkstra_impl(
		numpy.concatenate(tuple(children_nodes)).astype(numpy.int32),
		numpy.cumsum(list(map(len, children_nodes))),
		numpy.asarray(network.edges.begin if incoming else network.edges.end, int),
		init,
		numpy.asarray(edge_lengths, int),
		revisit,
		ibudget,
		numpy.asarray(min_itimes_to_dest, int))

@signature('(int_[::1], int_[::1], int_[::1], int_, int_[::1], bool_, int_, int_[::1])')
def dijkstra_impl(children_nodes_concat, children_nodes_ends, edges_terminals, init, edge_lengths, revisit, ibudget, min_itimes_to_dest):
	# The reason this procedure is convoluted is that we want it to be optimizable with Numba.
	# Originally, it used to be a modular combination of 3 different things:
	# A heap implementation, Dijkstra's algorithm, and a visitor object.
	# We've just inlined all the code so that the compiler can compile all of it at once.

	stack = []
	invalid_budget = ibudget + 1
	node_count = len(children_nodes_ends)
	visited_inodes = numpy.full(node_count, invalid_budget, numpy.int32)
	visited_iedges = numpy.full(len(edges_terminals), invalid_budget, numpy.int32)

	arrs = numpy.full(node_count * 3, -1, numpy.int32)
	min_heap = True
	arity = 2
	heap = arrs[0 * node_count : 1 * node_count]
	heap_node_to_index = arrs[1 * node_count : 2 * node_count]
	heap_index_to_node = arrs[2 * node_count : 3 * node_count]

	n = 0
	heap[0] = 0
	heap_index_to_node[0] = init
	heap_node_to_index[init] = 0
	n += 1
	while True:
		if n <= 0: break
		k = n - 1
		heap_sift(0, 0, k, False, arity, min_heap, heap, heap_index_to_node, heap_node_to_index, k, True)
		ti = heap[k]
		inode = heap_index_to_node[k]
		n -= 1
		heap_node_to_index[inode] = -1
		if revisit or visited_inodes[inode] == invalid_budget:
			visited_inodes[inode] = ti
			stack.append((inode, ibudget + 1 - ti))
			children_nodes_index = children_nodes_ends[inode - 1] if inode > 0 else 0
			children_nodes_end = children_nodes_ends[inode]
			while children_nodes_index < children_nodes_end:
				eij = children_nodes_concat[children_nodes_index]
				jnode = edges_terminals[eij]
				tj = ti + edge_lengths[eij]
				if tj + min_itimes_to_dest[jnode] < ibudget + 1:
					visited_iedges[eij] = ti
					if revisit or visited_inodes[jnode] == invalid_budget:
						i = heap_node_to_index[jnode]
						j_if_auto_detect_direction = -1
						if i == -1:
							i = n
							n += 1
							heap_index_to_node[i] = jnode
							heap_node_to_index[jnode] = i
							j_if_auto_detect_direction = ((i - 1) // arity if i != 0 else 0)
						else:
							old = heap[i]
							assert old is not None
							if tj < old: pass  # we will sift in this case
							else: i = -1  # suppress sift
						if i >= 0:
							heap[i] = tj
							heap_sift(i, 0, n, False, arity, min_heap, heap, heap_index_to_node, heap_node_to_index, j_if_auto_detect_direction, False)
				children_nodes_index += 1
	num_itimes = numpy.zeros(node_count, numpy.int32)
	for (i, tij) in stack:
		if num_itimes[i] < tij: num_itimes[i] = tij
	return (stack, visited_inodes, visited_iedges, num_itimes)

class Policy(object):
	__nonpickled_slots__ = (
		# Assigned in __init__()
		'ue', 'uv', 'we',
		'cached_edges_self', 'cached_edges_neighbor', 'cached_edges_tidist', 'cached_neighbors', 'cached_edge_ffts', 'cached_tijoffsets'
	)
	__slots__ = __nonpickled_slots__ + (
		# Assigned in __init__()
		'cache_ffts', 'transpose_graph', 'convolver', 'discretization', 'min_itimes_to_dest', 'network', 'prev_eij_ends', 'progress', 'suppress_calculation', 'temp_buffer_pairs', 'timins', 'tiprevs', 'zero_delay_convolution',
		# Assigned in prepare()
		'final_itimes',
	)
	def __init__(self, network, idst, discretization, zero_delay_convolution, cache_ffts, transpose_graph=False, suppress_calculation=False, stderr=None):
		self.network = network
		self.discretization = float(discretization if discretization is not None else numpy.min(network.edges.tmin))
		self.zero_delay_convolution = zero_delay_convolution
		self.suppress_calculation = suppress_calculation
		self.progress = 0
		self.cache_ffts = cache_ffts
		self.transpose_graph = transpose_graph
		self.temp_buffer_pairs = []  # we keep multiple buffers to avoid creating new array slice objects at every step
		convolvers = []
		if drfft:
			convolvers.append((
				{}.get(None),
				fftpack_drfftf,
				numba_single_overload_entrypoint(fftpack_multiply),
				fftpack_drfftb,
				float
			))
		convolvers.append((
			numpy_fftpack_lite.rffti if numpy_fftpack_lite is not None else numpy.positive,
			numpy_fftpack_lite.rfftf if numpy_fftpack_lite is not None else numpy.fft.rfft,
			numpy.multiply,
			fftpack_lite_rfftb,
			float
		))
		self.convolver = convolvers[0]

		if stderr is not None: tprev = timeit.default_timer(); print_("Computing minimum travel times...", end=' ', file=stderr)
		timins = discretize_down(network.edges.tmin, self.discretization).astype(int) if True else [0]
		assert numpy.all(timins), "Illegal edge travel time %s < discretization interval %s" % (numpy.min(timins), self.discretization)
		max_possible_budget = (1 << ((1 << 5) - 2)) - 2
		(_, min_itimes_to_dest, _, _) = dijkstra(network, not self.transpose_graph, idst, timins, False, max_possible_budget, [0] * len(network.nodes))
		self.min_itimes_to_dest = Array.fromitems(int, min_itimes_to_dest)
		self.timins             = Array.fromitems(int, timins)
		self.prev_eij_ends      = Array.fromitems(int, [0] * len(network.edges))
		self.tiprevs            = Array.fromitems(int, [0] * len(network.nodes))
		if stderr is not None: print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=stderr); del tprev

		if stderr is not None: tprev = timeit.default_timer(); print_("Initializing vertices and edges...", end=' ', file=stderr)
		self._postinit()
		if stderr is not None: print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=stderr); del tprev
	def _postinit(self):
		network = self.network
		nnodes = len(network.nodes)
		nedges = len(network.edges)
		self.ue = list(map(lambda _: Array(float,  0.0), network.edges)) if True else [Array(float)]
		self.we = list(map(lambda _: Array(bool, False), network.edges)) if True else [Array(bool )]  # whether each edge is optimal at each time
		self.uv = list(map(lambda tioffset: Array(float, float('NaN') if tioffset > 0 else 1.0), self.min_itimes_to_dest)) if True else [Array(float)]
		self.cached_edges_tidist   = list(network.edges.tidist_override)
		self.cached_neighbors      = network.nodes.outgoing if not self.transpose_graph else network.nodes.incoming
		self.cached_edges_self     = network.edges.begin    if not self.transpose_graph else network.edges.end
		self.cached_edges_neighbor = network.edges.end      if not self.transpose_graph else network.edges.begin
		self.cached_edge_ffts      = {} if self.cache_ffts else None
		self.cached_tijoffsets     = Array.fromitems(int, self.timins.assert_size(nedges) + self.min_itimes_to_dest.assert_size(nnodes)[self.cached_edges_neighbor])
	def __getstate__(self):
		cls = Policy
		array_fields = sorted(frozenset(Array.__slots__) - frozenset(Array.__nonpickled_slots__))
		arrays = list(map(lambda field: [], array_fields))
		for arr in (self.ue, self.we, self.uv):
			for item in arr:
				for i, field in enumerate(array_fields):
					arrays[i].append(getattr(item, field))
		return (slotted_getstate(cls, self), arrays)
	def __setstate__(self, compound_state):
		cls = Policy
		(state, arrays) = compound_state
		offsets = list(map(lambda array: 0, arrays))
		slotted_setstate(cls, self, state)
		cls._postinit(self)
		array_fields = sorted(frozenset(Array.__slots__) - frozenset(Array.__nonpickled_slots__))
		for arr in (self.ue, self.we, self.uv):
			for item in arr:
				for i, field in enumerate(array_fields):
					setattr(item, field, arrays[i][offsets[i]])
					offsets[i] += 1
				item._postinit()
	def prepare(self, isrc, tbudget, preallocate_aggressively, prediscretize, stderr=None):
		# preallocate_aggressively = {-1: minimize dynamic memory usage, 0: minimize initialization latency, 1: maximize speed}
		network = self.network
		ibudget = int(discretize_up(numpy.asarray([tbudget], float), self.discretization))
		if stderr is not None: tprev = timeit.default_timer(); print_("Computing memory requirements...", end=' ', file=stderr)
		(_, _, visited_iedges, final_itimes) = self._dijkstra(isrc, ibudget, False)
		eused = numpy.flatnonzero((0 <= visited_iedges) & (visited_iedges <= ibudget)).tolist()
		self.final_itimes = final_itimes.tolist()
		if stderr is not None: print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=stderr); del tprev

		if prediscretize:
			if stderr is not None: tprev = timeit.default_timer(); print_("Discretizing edges...", end=' ', file=stderr)
			for eiused, tidist in zip(eused, network.discretize_edges(
				list(map(network.edges.hmm.__getitem__, eused)),
				list(map(network.edges.tmin.__getitem__, eused)),
				self.discretization,
				suppress_calculation=self.suppress_calculation
			)):
				self.cached_edges_tidist[eiused] = tidist
			if stderr is not None: print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=stderr); del tprev

		if stderr is not None: tprev = timeit.default_timer(); print_("Pre-allocating buffers (if requested)...", end=' ', file=stderr)
		total_progress = 0
		# uv[i][t] should be the probability of reaching the destination from node i in <= t steps (so for T = 0 we get uv[idst] == [1.0])
		# Rationale: uv[i] should be the convolution of edge[i,j] with uv[j], with no elements missing.
		single_buffer = True
		combined_uv = {}.get(None)
		combined_ue = {}.get(None)
		combined_we = {}.get(None)
		for pass_ in (False, True):
			combined_v_size = 0
			for i in xrange(len(self.min_itimes_to_dest)):
				m = max(self.final_itimes[i] - self.min_itimes_to_dest[i], 0)
				if preallocate_aggressively >= 0:
					if preallocate_aggressively > 0:
						if pass_ and single_buffer: self.uv[i].switch_buffer(combined_uv, combined_v_size, m)
						combined_v_size += m
					if pass_: self.uv[i].ensure_size(m, preallocate_aggressively > 0)
			if not pass_ and preallocate_aggressively > 0 and single_buffer:
				combined_uv = Array.create_buffer(Array.compute_type_code(float), combined_v_size)
			combined_e_size = 0
			for eij in eused:
				m = max(self.final_itimes[self.cached_edges_self[eij]] - (self.timins[eij] + self.min_itimes_to_dest[self.cached_edges_neighbor[eij]]), 0)
				if pass_: total_progress += m
				if preallocate_aggressively >= 0:
					if preallocate_aggressively > 0:
						if pass_ and single_buffer: self.ue[eij].switch_buffer(combined_ue, combined_e_size, m)
						if pass_ and single_buffer: self.we[eij].switch_buffer(combined_we, combined_e_size, m)
						combined_e_size += m
					if pass_: self.ue[eij].ensure_size(m, preallocate_aggressively > 0)
					if pass_: self.we[eij].ensure_size(m, preallocate_aggressively > 0)
			if not pass_ and preallocate_aggressively > 0 and single_buffer:
				combined_ue = Array.create_buffer(Array.compute_type_code(float), combined_e_size)
				combined_we = Array.create_buffer(Array.compute_type_code(bool ), combined_e_size)
		if stderr is not None: print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=stderr); del tprev
		return (eused, ibudget, total_progress)
	def _dijkstra(self, isrc, ibudget, revisit):
		return dijkstra(self.network, self.transpose_graph, isrc, self.timins, revisit, ibudget, self.min_itimes_to_dest)
	def compute_optimal_update_order(self, isrc, ibudget, stderr=None):
		if stderr is not None: tprev = timeit.default_timer(); print_("Computing optimal update order...", end=' ', file=stderr)
		stack = self._dijkstra(isrc, ibudget, True)[0]
		if stderr is not None: print_(int((timeit.default_timer() - tprev) * 1000), "ms", file=stderr); del tprev
		return stack
	def step(self, i, ti, edges_seen=None,
		len=len, complex=complex, numpy_maximum=numpy.maximum, numpy_empty=numpy.empty, numpy_greater_equal=numpy.greater_equal,
		zdconvolution=numba_single_overload_entrypoint(zdconvolution), convolve_into=numba_single_overload_entrypoint(convolve_into)):
		min_itimes_to_dest = self.min_itimes_to_dest
		tioffset = min_itimes_to_dest[i]
		uvi_size_prev = self.tiprevs[i] - tioffset
		if uvi_size_prev < 0: uvi_size_prev = 0
		uvi_size = ti - tioffset
		if uvi_size < 0: uvi_size = 0
		uv = self.uv
		uvi = uv[i].assert_size(uvi_size)
		if tioffset > 0:
			suppress_calculation = self.suppress_calculation
			deferred_ge = []
			deferred_ge_append = deferred_ge.append
			progress = self.progress
			final_itimes = self.final_itimes
			edges_end = self.cached_edges_neighbor
			ue = self.ue
			we = self.we
			prev_eij_ends = self.prev_eij_ends
			edges_tidist = self.cached_edges_tidist
			tijoffsets = self.cached_tijoffsets
			zero_delay_convolution = self.zero_delay_convolution
			temp_buffer_pairs = None   # if uninitialized, we also need to initialize other FFT-only variables
			uvi[uvi_size_prev : uvi_size] = 0  # initialize to zero first...
			for eij in self.cached_neighbors[i]:
				eij_begin = prev_eij_ends[eij]
				tijoffset = tijoffsets[eij]
				eij_end = ti - tijoffset
				if eij_begin < eij_end:
					etij = edges_tidist[eij]
					if etij is None:
						network = self.network
						[etij] = network.discretize_edges(network.edges.hmm[eij : eij + 1], network.edges.tmin[eij : eij + 1], self.discretization, suppress_calculation=suppress_calculation)
						edges_tidist[eij] = etij
					etij_len = len(etij)  # this information actually is necessary (unlike most other lengths), since it affects the convolution sub-blocks we do
					n = eij_end - eij_begin
					assert n >= 0
					j = edges_end[eij]
					uvj = uv[j].ndarray
					assert uvj is not None
					uvj_begin = eij_begin - etij_len
					if uvj_begin < 0: uvj_begin = 0
					ueij_len = final_itimes[i] - tijoffset
					weij = we[eij].assert_size(eij_end)
					if zero_delay_convolution:
						ueij = ue[eij].assert_size(ueij_len)
					if 1:  # to suppress computations, set this to 0
						if zero_delay_convolution:
							conv_list = zdconvolution(final_itimes[j] - min_itimes_to_dest[j], etij_len, eij_begin, eij_end)
							ueij_slice = ueij[eij_begin : eij_end]
						else:
							an = eij_end - uvj_begin
							bn = etij_len
							if bn > eij_end: bn = eij_end
							if bn < 0: bn = 0
							conv_list = [(uvj_begin, eij_end, 0, bn, eij_begin - uvj_begin, an, bn)]
							ueij_slice = None
						for (a1, a2, b1, b2, c1, an, bn) in conv_list:  # PERF: This loop seems to take too much time outside the convolutions themselves
							cn = int(an + bn - 1)
							min_len = bn if bn < an else an
							assert min_len > 0, "Cannot convolve empty arrays"
							if zero_delay_convolution:
								conv_begin = 0
								conv_length = ueij_len - c1
								if conv_length > cn:
									conv_length = cn
							else:
								conv_begin = eij_begin - uvj_begin
								conv_length = n
							if min_len >= 0x80:
								if temp_buffer_pairs is None:
									edge_ffts = self.cached_edge_ffts
									temp_buffer_pairs = self.temp_buffer_pairs
									(conv_initialize, conv_forward, conv_multiply, conv_backward, conv_dual_dtype) = self.convolver
								m_log2 = cn.bit_length()
								m = 1 << m_log2
								s = conv_initialize(m) if conv_initialize is not None else None
								try:
									temp_buffer_pair = temp_buffer_pairs[m_log2]
								except IndexError:
									while m_log2 >= len(temp_buffer_pairs):
										mi = 1 << len(temp_buffer_pairs)
										temp_buffer = numpy.empty(mi + mi, complex).view(conv_dual_dtype)
										temp_buffer_pairs.append((temp_buffer[0 : mi], temp_buffer[mi : mi + mi]))
									temp_buffer_pair = temp_buffer_pairs[m_log2]
								uvj_slice_padded = temp_buffer_pair[0]
								uvj_slice = uvj[a1:a2]
								uvj_slice_padded_dft = None
								if not suppress_calculation:
									uvj_slice_padded[0 : an] = uvj_slice
									uvj_slice_padded[an : m] = 0
									uvj_slice_padded_dft = conv_forward(uvj_slice_padded, s)
								if uvj_slice_padded_dft is None: uvj_slice_padded_dft = uvj_slice_padded
								b_key = (eij, b1, b1 + bn, m) if edge_ffts is not None else None
								etij_slice_padded_dft = edge_ffts.get(b_key) if edge_ffts else None
								if etij_slice_padded_dft is None:
									etij_slice_padded = temp_buffer_pair[1]
									etij_slice = etij[b1:b2]
									etij_slice_padded_dft = None
									if not suppress_calculation:
										etij_slice_padded[0 : bn] = etij_slice
										etij_slice_padded[bn : m] = 0
										etij_slice_padded_dft = conv_forward(etij_slice_padded, s)
									if etij_slice_padded_dft is None: etij_slice_padded_dft = etij_slice_padded
									if edge_ffts is not None:
										edge_ffts[b_key] = +etij_slice_padded_dft
								conv_dft = None
								if not suppress_calculation:
									conv_dft = conv_multiply(etij_slice_padded_dft, uvj_slice_padded_dft, uvj_slice_padded_dft)
								if conv_dft is None: conv_dft = uvj_slice_padded_dft
								conv = None
								if not suppress_calculation:
									conv = conv_backward(conv_dft, s)
								if conv is None: conv = conv_dft
								conv = conv[conv_begin : conv_begin + conv_length]
							else:
								if zero_delay_convolution:
									conv = ueij
									coffset = conv_begin - c1
								else:
									conv = numpy_empty(conv_length)
									coffset = conv_begin
								if not suppress_calculation:
									convolve_into(uvj, a1, an, etij, b1, bn, conv, conv_begin, conv_length, coffset, zero_delay_convolution)
								if zero_delay_convolution:
									conv = None
							if conv is not None:
								if zero_delay_convolution:
									ueij[c1 : c1 + conv_length] += conv
								else:
									ueij_slice = conv
						uvi_prevoffset = eij_begin + tijoffset - tioffset
						uvi_offset = uvi_prevoffset + n
						uvi_slice = uvi[uvi_prevoffset : uvi_offset]
						if not suppress_calculation:
							numpy_maximum(ueij_slice, uvi_slice, uvi_slice)
						deferred_ge_append((
							ueij_slice,
							uvi_slice,
							weij[eij_begin : eij_end]
						))
					if edges_seen is not None:
						edges_seen(eij)
					self.progress += n
					prev_eij_ends[eij] = eij_end
			for (arg1, arg2, arg3) in deferred_ge:
				if not suppress_calculation:
					numpy_greater_equal(arg1, arg2, arg3)
		self.tiprevs[i] = ti

class Path(object):
	kronecker_delta = numpy.asarray([1.0])
	kronecker_delta.flags['WRITEABLE'] = False
	@staticmethod
	def convolve(a, b, correlate2=numpy.core.multiarray.correlate2):
		return correlate2(a, b[::-1], 2)
	def __init__(self, policy, tibudget_max):
		self.policy = policy
		self.tibudget_max = tibudget_max
		self.seen_paths = {}
		self.path_tree_root = []
		self.pq = []
	def __nonzero__(self): return not not self.pq
	__bool__ = __nonzero__
	def start(self, isrc, tibudget):
		if tibudget > self.tibudget_max: raise ValueError("unprepared for this time budget; convolution data may have been discarded")
		self.tibudget = tibudget
		del self.pq[:]
		uvsrc = self.policy.uv[isrc][tibudget - self.policy.min_itimes_to_dest[isrc]:]
		self.pq.append((
			-(max(uvsrc) if len(uvsrc) > 0 else 0.0),
			((), ~len(self.policy.network.edges)), isrc,
			0, RSet({isrc}), self.kronecker_delta, self.path_tree_root
		))
		self.found_reliability = 0
	def step(self, edge_filter={}.get(None)):
		# Note:
		#   This procedure does NOT automatically exclude duplicate nodes in the path.
		#   Use the filter mechanism to suppress these.
		#   Note that if you do not suppress paths that re-visit the same nodes,
		#   you can easily get exponential-time behavior. (!)
		result = None
		policy = self.policy
		edges_tidist = policy.cached_edges_tidist
		min_itimes_to_dest = policy.min_itimes_to_dest
		edges_neighbors = policy.cached_edges_neighbor
		neighbors = policy.cached_neighbors
		timins = policy.timins
		uv = policy.uv
		pq = self.pq
		convolve = self.convolve
		while pq:
			(negative_reliability, path_so_far, i, timin_elapsed, path_node_set, tidist_curr, path_tree_node_parent) = heapq.heappop(pq) if 1 else pq[0]
			reliability = -negative_reliability
			if reliability >= self.found_reliability:
				reached_destination = min_itimes_to_dest[i] <= 0
				if reached_destination:
					self.found_reliability = reliability
				else:
					for k, eij in enumerate(neighbors[i]):
						j = edges_neighbors[eij]
						uvj_ndarray = uv[j].ndarray
						timin_elapsed_next = timin_elapsed + timins[eij]
						timin_elapsed_and_remaining = timin_elapsed_next + min_itimes_to_dest[j]
						timax_left_next_minus_offset = self.tibudget - timin_elapsed_and_remaining
						if timax_left_next_minus_offset < 0: continue
						if edge_filter and edge_filter(eij, j, path_so_far, path_node_set, reliability, timin_elapsed, tidist_curr) is False: continue
						while k >= len(path_tree_node_parent):
							path_tree_node_parent.append({}.get(None))
						path_tree_node = path_tree_node_parent[k]
						if path_tree_node is None:
							path_tree_node_next = []
							path_node_set_next = RSet({j}, path_node_set)
							etij = edges_tidist[eij]
							if etij is None:  # TODO: PERF: We really shouldn't have to re-discretize edges... we should be able to leverage what was already discretized by the policy.
								network = self.policy.network
								suppress_calculation = False
								[etij] = network.discretize_edges(network.edges.hmm[eij : eij + 1], network.edges.tmin[eij : eij + 1], self.policy.discretization, suppress_calculation=suppress_calculation)
								edges_tidist[eij] = etij
							tidist_next = convolve(tidist_curr, etij)[:self.tibudget_max - timin_elapsed_and_remaining + 1]
							reliabilities = convolve(tidist_next, uvj_ndarray[:self.tibudget_max - timin_elapsed_and_remaining + 1])
							path_tree_node_parent[k] = (path_tree_node_next, path_node_set_next, tidist_next, reliabilities)
						else:
							(path_tree_node_next, path_node_set_next, tidist_next, reliabilities) = path_tree_node
						reliability_next = float(reliabilities[timax_left_next_minus_offset])
						assert 1 or numpy.any(numpy.isclose(reliability_next, numpy.dot(
							tidist_next[max(timax_left_next_minus_offset + 1 - len(uvj_ndarray), 0) : timax_left_next_minus_offset + 1],
							uvj_ndarray[max(timax_left_next_minus_offset + 1 - len(tidist_next), 0) : timax_left_next_minus_offset + 1][::-1]
							).item(), 1E-7, 1E-10))
						heapq.heappush(pq, (-reliability_next, (path_so_far, eij), j, timin_elapsed_next, path_node_set_next, tidist_next, path_tree_node_next))
				result = (reached_destination, path_so_far, reliability)
				break
		return result
