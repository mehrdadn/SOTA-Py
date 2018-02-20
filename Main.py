import argparse
import collections
import copy
import ctypes
import heapq
import os
import sys
import timeit
import threading

if sys.version_info >= (3, 0): exec("print_ = lambda *args, **kwargs: print(*args, **kwargs) and None or (kwargs.get('file') or sys.stdout).flush(); xrange = range")
else: print_ = __import__('__builtin__').__dict__['print']

timer = timeit.default_timer

try:
	SetConsoleCtrlHandler_body_new = b'\xC2\x08\x00' if ctypes.sizeof(ctypes.c_void_p) == 4 else b'\xC3'
	try: SetConsoleCtrlHandler_body = (lambda kernel32: (lambda pSetConsoleCtrlHandler:
		kernel32.VirtualProtect(pSetConsoleCtrlHandler, ctypes.c_size_t(1), 0x40, ctypes.byref(ctypes.c_uint32(0)))
		and (ctypes.c_char * 3).from_address(pSetConsoleCtrlHandler.value)
	)(ctypes.cast(kernel32.SetConsoleCtrlHandler, ctypes.c_void_p)))(ctypes.windll.kernel32)
	except: SetConsoleCtrlHandler_body = None
	if SetConsoleCtrlHandler_body:
		SetConsoleCtrlHandler_body_old = SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)]
		SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)] = SetConsoleCtrlHandler_body_new
	try:
		if 'numpy' not in sys.modules and __name__ == '__main__': tprev = timer(); print_("Importing numpy...", end=' ', file=sys.stderr)
		import numpy
		if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev

		if 'scipy.special' not in sys.modules and __name__ == '__main__': tprev = timer(); print_("Importing scipy.special...", end=' ', file=sys.stderr)
		import scipy.special
		if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev
	finally:
		if SetConsoleCtrlHandler_body:
			SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)] = SetConsoleCtrlHandler_body_old
except ImportError as e:
	pass

if 'numba' not in sys.modules and __name__ == '__main__': tprev = timer(); print_("Importing numba...", end=' ', file=sys.stderr)
try:
	existed = 'pkg_resources' in sys.modules
	if not existed: sys.modules['pkg_resources'] = None
	try: import numba
	finally: existed and sys.modules.pop('pkg_resources'); del existed
except ImportError: pass
if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev

if __name__ == '__main__': tprev = timer(); print_("Other imports...", end=' ', file=sys.stderr)
import SOTA

pyglet = {}.get(None)
pyglet_win32 = {}.get(None)
try:
	import pyglet
	if sys.platform == 'win32': import pyglet.libs.win32 as pyglet_win32
except ImportError: pass

def is_debugger_present():
	import platform
	if platform.system() == 'Windows':
		import ctypes
		return ctypes.windll.kernel32.IsDebuggerPresent()
	return __debug__

def is_slow_display():
	result = False
	try:
		SM_REMOTESESSION = 1 << 12
		result = ctypes.windll.user32.GetSystemMetrics(SM_REMOTESESSION)
	except AttributeError: pass
	return result

def make_glMultiDrawArrays(glDrawArrays):
	'''
	//MSVC: /O1 /Wall /wd4668 /wd4711 /wd4127 /EHsc /link /DefaultLib:OpenGL32.lib /Machine:I386 /Entry:main
	//Compiler: 0

	// "%VS90COMNTOOLS%..\..\VC\bin\amd64\dumpbin" /disasm:bytes "Temp.exe"

	// I386:  "U\x8b\xecS\x8b]\x18\x85\xdbt\x1dV\x8bu\x10W\x8b}\x14\xff7\xff6\xffu\x0c\xffU\x08\x83\xc6\x04\x83\xc7\x04Ku\xed_^[]\xc3"
	// AMD64: "HSUVWATH\x83\xec \x8bt$pI\x8b\xd9I\x8b\xf8\x85\xf6\x8b\xeaL\x8b\xe1t\x16D\x8b\x03\x8b\x17\x8b\xcdA\xff\xd4H\x83\xc7\x04H\x83\xc3\x04\xff\xceu\xeaH\x83\xc4 A\\_^][\xc3"

	#include <Windows.h>
	#include <GL/gl.h>

	__declspec(dllexport)
	void __cdecl glMultiDrawArrays(void __stdcall glDrawArrays(GLenum mode, GLint first, GLsizei count), GLenum mode, GLint *first, GLsizei *count, GLsizei primcount)
	{
		while (primcount)
		{
			glDrawArrays(mode, *first, *count);
			++first;
			++count;
			--primcount;
		}
	}

	int main() { glMultiDrawArrays(glDrawArrays, 0, 0, 0, 0); return 0; }
	'''
	code32 = b"U\x8b\xecS\x8b]\x18\x85\xdbt\x1dV\x8bu\x10W\x8b}\x14\xff7\xff6\xffu\x0c\xffU\x08\x83\xc6\x04\x83\xc7\x04Ku\xed_^[]\xc3"
	code64 = b"HSUVWATH\x83\xec \x8bt$pI\x8b\xd9I\x8b\xf8\x85\xf6\x8b\xeaL\x8b\xe1t\x16D\x8b\x03\x8b\x17\x8b\xcdA\xff\xd4H\x83\xc7\x04H\x83\xc3\x04\xff\xceu\xeaH\x83\xc4 A\\_^][\xc3"
	code = code32 if ctypes.sizeof(ctypes.c_void_p) <= ctypes.sizeof(ctypes.c_int) else code64
	glMultiDrawArrays_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint)
	bincode = ctypes.create_string_buffer(code)
	def callback(mode, first, count, primcount):
		first_p = ctypes.cast(first, ctypes.POINTER(ctypes.c_int))
		count_p = ctypes.cast(count, ctypes.POINTER(ctypes.c_uint))
		for i in range(primcount):
			glDrawArrays(mode, first_p[i], count_p[i])
	success = False
	if not success:
		try:
			old = ctypes.c_uint()
			success = ctypes.windll.kernel32.VirtualProtect(bincode, ctypes.c_uint(len(bincode)), 0x40, ctypes.byref(old)) != 0
		except AttributeError: pass
	if not success:
		try:
			success = ctypes.pythonapi.mprotect(bincode, len(buffer), 0x1 | 0x2 | 0x4) == 0
		except AttributeError:
			pass
	if success:
		def callback(mode, first, count, primcount, glMultiDrawArrays=glMultiDrawArrays_t(ctypes.addressof(bincode)), _keep_alive=bincode):
			return glMultiDrawArrays(glDrawArrays, mode, first, count, primcount)
	return callback

class NetworkGUI(object):
	def __init__(self, edges, gif=None):
		self.prev_update_duration = -1  # used to indicate that we haven't started
		self.prev_update_time = 0
		self.gif = gif
		self.alpha_processed = 1
		self.invalidated = 1
		if not edges:
			self.window = None
		else:
			self.gl = gl = pyglet.gl
			if gl.gl_info.have_version(1, 4):
				self.glMultiDrawArrays = gl.glMultiDrawArrays
			else:
				self.glMultiDrawArrays = make_glMultiDrawArrays(gl.glDrawArrays)
			width = 480
			self.window = pyglet.window.Window(resizable=True, config=gl.Config(double_buffer=is_slow_display()), width=width, height=int(width / 1.025) if width else None)
			self.window.event(self.on_draw)
			self.captured_frames = 0
			self.all_points = []
			edge_lines = []
			for iedge, (lanes, points) in enumerate(zip(edges.lanes, edges.geom_points)):
				lines = []
				for point in points:
					curr = (point[0], point[1])
					self.all_points.append(curr)
					lines.append(curr)
				edge_lines.append((lanes, lines, iedge))
			edge_lines.sort(key=lambda p: (-p[0], iedge))
			self.lines = []
			edge_original_order = []
			edge_lanes = [None] * len(edge_lines)
			edge_indices_to_line_starts = [None] * len(edge_lines)
			edge_indices_to_line_counts = [None] * len(edge_lines)
			for (lanes, lines, iedge) in edge_lines:
				nprev = len(self.lines)
				self.lines.extend(lines)
				edge_original_order.append(iedge)
				edge_indices_to_line_starts[iedge] = nprev
				edge_indices_to_line_counts[iedge] = len(lines)
				edge_lanes[iedge] = lanes
			self.edge_original_order = numpy.asarray(edge_original_order, int)
			self.edge_indices_to_line_starts = numpy.asarray(edge_indices_to_line_starts, gl.GLint)
			self.edge_indices_to_line_counts = numpy.asarray(edge_indices_to_line_counts, gl.GLsizei)
			self.edge_lanes = numpy.asarray(edge_lanes, int)
			min_latlon = numpy.min(self.all_points, 0)
			max_latlon = numpy.max(self.all_points, 0)
			center_geographic = numpy.mean([min_latlon, max_latlon], 0)
			center_normalized = SOTA.geographic_to_cartesian(center_geographic)
			center_normalized[0:3] /= numpy.linalg.norm(center_normalized[0:3])

			all_points_squashed = numpy.copy(self.all_points)
			all_points_squashed[:, 1] = center_geographic[1]  # set the longitude to the center line

			self.lines_array = numpy.vstack(SOTA.geographic_to_cartesian(numpy.asarray(self.lines, float).T) * SOTA.EARTH_RADIUS_MM).T.copy()
			self.optimality = [0.0] * len(edge_lines)
			self.edge_orders = numpy.asarray([0] * len(edge_lines), float)
			self.line_colors = numpy.asarray((lambda gray, alpha: [(gray, gray, gray, alpha)] * len(self.lines))(0.5, 0.125), float)
			self.edge_colors = list(map(lambda i, n: self.line_colors[i : i + n], self.edge_indices_to_line_starts, self.edge_indices_to_line_counts))
			self.layers = []
			gl.glEnable(gl.GL_BLEND); gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
			gl.glEnable(gl.GL_LINE_SMOOTH); gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
			gl.glEnable(gl.GL_POINT_SMOOTH); gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
			gl.glEnable(gl.GL_POLYGON_SMOOTH); gl.glHint(gl.GL_POLYGON_SMOOTH_HINT, gl.GL_NICEST)
			gl.glEnable(gl.GL_COLOR_MATERIAL); gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)

			use_vbo = False

			color_pointer = self.line_colors.ctypes.data
			if use_vbo:
				cbuf = gl.GLuint()
				gl.glGenBuffers(1, ctypes.byref(cbuf));
				gl.glBindBuffer(gl.GL_ARRAY_BUFFER, cbuf.value)
				gl.glBufferData(gl.GL_ARRAY_BUFFER, self.line_colors.size * ctypes.sizeof(ctypes.c_double), color_pointer, gl.GL_DYNAMIC_DRAW);
				color_pointer = 0
			gl.glColorPointer(self.line_colors.size // len(self.lines), gl.GL_DOUBLE, 0, color_pointer)

			vertex_pointer = self.lines_array.ctypes.data
			if use_vbo:
				vbuf = gl.GLuint()
				gl.glGenBuffers(1, ctypes.byref(vbuf));
				gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbuf.value)
				gl.glBufferData(gl.GL_ARRAY_BUFFER, self.lines_array.size * ctypes.sizeof(ctypes.c_double), vertex_pointer, gl.GL_STATIC_DRAW);
				vertex_pointer = 0
			gl.glVertexPointer(self.lines_array.size // len(self.lines), gl.GL_DOUBLE, 0, vertex_pointer)

			self.perspective_degrees = 60
			# HACK: This is heuristic for measuring the distance to the farthest visible point from the line going to the center of the Earth
			# Instead of measuring Euclidean distance from the line, which requires minimizing an L-infinity norm, we measure distance from its projection to the surface
			# It'll give a bad distance if we have a very huge viewing angle to cover (like, say, half of the earth)
			self.altitude_as_fraction_of_radius = numpy.max(numpy.linalg.norm(center_normalized - SOTA.geographic_to_cartesian(numpy.asarray(all_points_squashed, float).T).T, axis=1)) / numpy.tan(self.perspective_degrees / 2 * numpy.pi / 180)

			self.center = center_normalized * SOTA.EARTH_RADIUS_MM
			self.unsort_edges()
	def __nonzero__(self): return not not self.window
	__bool__ = __nonzero__
	def unsort_edges(self):
		self.isorted_edges = self.edge_original_order
	def sort_edges(self):  # can slow down rendering
		self.isorted_edges = numpy.lexsort((self.edge_original_order.argsort(), self.edge_orders))
		self.invalidated += 1
	def on_draw(self):
		gl = self.gl
		gl.glMatrixMode(gl.GL_PROJECTION)
		gl.glLoadIdentity()
		display = self.window.get_size()
		gl.gluPerspective(self.perspective_degrees, display[0] * 1.0 / display[1], self.altitude_as_fraction_of_radius * SOTA.EARTH_RADIUS_MM / 2, ((1 + self.altitude_as_fraction_of_radius) * SOTA.EARTH_RADIUS_MM) * 2);

		gl.glMatrixMode(gl.GL_MODELVIEW)
		gl.glLoadIdentity()
		den = self.center[3] if len(self.center) > 3 else 1
		gl.gluLookAt((self.center[0] / den) * (1 + self.altitude_as_fraction_of_radius), (self.center[1] / den) * (1 + self.altitude_as_fraction_of_radius), (self.center[2] / den) * (1 + self.altitude_as_fraction_of_radius), 0, 0, 0, 0, 0, 1)

		# gl.glLineWidth(1.0 / (1 << 2))

		gl.glClearColor(1, 1, 1, 0)
		gl.glColor4d(0, 0, 0, 1)
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

		GLint_p = ctypes.POINTER(gl.GLint)
		GLsizei_p = ctypes.POINTER(gl.GLsizei)
		GLdouble_p = ctypes.POINTER(gl.GLdouble)
		GLuint_p = ctypes.POINTER(gl.GLuint)
		glMultiDrawArrays = self.glMultiDrawArrays
		glLineWidth = gl.glLineWidth

		prev_line_width = ctypes.c_float(); gl.glGetFloatv(gl.GL_LINE_WIDTH, ctypes.byref(prev_line_width))
		prev_color_array = ctypes.c_ubyte(); gl.glGetBooleanv(gl.GL_COLOR_ARRAY, ctypes.byref(prev_color_array))
		prev_vertex_array = ctypes.c_ubyte(); gl.glGetBooleanv(gl.GL_VERTEX_ARRAY, ctypes.byref(prev_vertex_array))

		edge_lanes_sorted = self.edge_lanes[self.isorted_edges]
		edge_lanes_changes = numpy.flatnonzero(numpy.hstack((edge_lanes_sorted[0], numpy.diff(edge_lanes_sorted))))
		isorted_lines_starts = self.edge_indices_to_line_starts[self.isorted_edges]
		isorted_lines_counts = self.edge_indices_to_line_counts[self.isorted_edges]
		gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
		gl.glEnableClientState(gl.GL_COLOR_ARRAY)
		mode = gl.GL_LINE_STRIP
		for i in range(len(edge_lanes_changes)):
			ilinewidth_change = edge_lanes_changes[i]
			inext_linewidth_change = edge_lanes_changes[i + 1] if i + 1 < len(edge_lanes_changes) else len(edge_lanes_sorted)
			glLineWidth(self.lanes_to_line_width(edge_lanes_sorted[ilinewidth_change]))
			glMultiDrawArrays(
				mode,
				ctypes.cast(isorted_lines_starts[ilinewidth_change : inext_linewidth_change].ctypes.data, GLint_p),
				ctypes.cast(isorted_lines_counts[ilinewidth_change : inext_linewidth_change].ctypes.data, GLsizei_p),
				inext_linewidth_change - ilinewidth_change)
		gl.glDisableClientState(gl.GL_COLOR_ARRAY)
		for (mode, path, reversed, color, lanes) in self.layers:
			glLineWidth(self.lanes_to_line_width(lanes))
			gl.glColor4d(*color)
			segment_starts = self.edge_indices_to_line_starts[path]
			segment_counts = self.edge_indices_to_line_counts[path]
			if reversed:
				segment_starts = segment_starts[::-1]
				segment_counts = segment_counts[::-1]
			segment_indices = numpy.ones(segment_counts.sum(), numpy.int32)
			if len(segment_indices) > 0:
				segment_indices[0] = 0
				segment_indices[segment_counts.cumsum()[:-1]] -= segment_counts[:-1]
			segment_indices.cumsum(out=segment_indices)
			segment_indices += numpy.repeat(segment_starts, segment_counts)
			gl.glDrawElements(mode, len(segment_indices), gl.GL_UNSIGNED_INT, ctypes.cast(segment_indices.ctypes.data, GLuint_p))

		gl.glLineWidth(prev_line_width.value)
		(gl.glEnableClientState if prev_vertex_array.value else gl.glDisableClientState)(gl.GL_VERTEX_ARRAY)
		(gl.glEnableClientState if prev_color_array.value else gl.glDisableClientState)(gl.GL_COLOR_ARRAY)
	@staticmethod
	def lanes_to_line_width(v):
		return v ** 0.75
	@property
	def possibly_needs_drawing(self):
		return self.invalidated and (pyglet_win32 is None or self.window and not pyglet_win32._user32.IsIconic(self.window._hwnd))
	def is_out_of_date(self, update_interval_sec):
		prev_update_duration = self.prev_update_duration * 4
		if update_interval_sec < prev_update_duration: update_interval_sec = prev_update_duration
		return timer() - self.prev_update_time >= update_interval_sec
	def add_path(self, path, reversed, color, lanes, index={}.get(None)):
		if self.window:
			self.layers.insert(index if index is not None else len(self.layers), (self.gl.GL_LINE_STRIP, path, reversed, color, lanes))
			self.invalidated += 1
	def update_edges(self, eijs, color=None, order=None, lanes=None):
		if self.window:
			for eij in eijs:
				if color is not None: self.edge_colors[eij][:] = color
				if order is not None: self.edge_orders[eij] = order
				if lanes is not None: self.edge_lanes[eij] = lanes
			self.invalidated += len(eijs)
	def refresh(self, resort=False, wait_for_next_event=False):
		result = 0
		if self.window:
			now0 = timer()
			if self.prev_update_duration < 0: # we haven't started yet...
				self.prev_update_duration = 0
				if pyglet:
					pyglet.app.event_loop.has_exit = False
					pyglet.app.event_loop.is_running = True
					pyglet.app.event_loop._legacy_setup()
					pyglet.app.platform_event_loop.start()
				result += timer() - now0
			if resort: self.sort_edges()
			if self.possibly_needs_drawing:
				self.window.dispatch_event('on_draw')
				if self.window.config.double_buffer: self.window.flip()
				else: pyglet.gl.glFlush()
				self.invalidated = 0
				self.captured_frames += 1
				if self.gif:
					buffer = numpy.empty((self.window.height, self.window.width, 4), numpy.uint8)
					self.gl.glReadPixels(0, 0, self.window.width, self.window.height, self.gl.GL_RGBA, self.gl.GL_UNSIGNED_BYTE, ctypes.cast(buffer.ctypes.data, ctypes.c_void_p))
					self.gif.append_data(buffer[::-1, ...])
			now1 = timer()
			has_exit = pyglet.app.event_loop.has_exit
			if not has_exit:
				idle = pyglet.app.event_loop.idle()
				if not wait_for_next_event: idle = 0
				pyglet.app.platform_event_loop.step(idle)
				has_exit = pyglet.app.event_loop.has_exit
			if has_exit:
				pyglet.app.platform_event_loop.stop()
				self.window = None
			now2 = timer()
			self.prev_update_duration = now1 - now0
			result += now2 - now1
			self.prev_update_time = now2
		return result

def parse_id(s):
	if s is not None:
		i = s.rfind('.')
		if i < 0: result = (int(s), int("0"))
		else: result = (int(s[:i]), int(s[i + 1:] or "0"))
	else:
		result = None
	return result

def hue2rgb(p, q, t):
	if (t < 0): t += 1
	if (t > 1): t -= 1
	if (t < 1./6): return p + (q - p) * 6 * t
	if (t < 1./2): return q
	if (t < 2./3): return p + (q - p) * (2./3 - t) * 6
	return p

def hsl2rgb(h, s, l):
	if s == 0:
		r = g = b = l
	else:
		q = l * (1 + s) if l < 0.5 else l + s - l * s
		p = 2 * l - q
		r = hue2rgb(p, q, h + 1./3)
		g = hue2rgb(p, q, h)
		b = hue2rgb(p, q, h - 1./3)
	return (r, g, b)

def gl_errcheck(result, func, arguments):
	gl = pyglet.gl
	context = gl.current_context
	if not context:
		raise gl.GLException('No GL context; create a Window first')
	if not context._gl_begin:
		error = gl.glGetError()
		if error:
			raise gl.GLException(ctypes.cast(gl.gluErrorString(error), ctypes.c_char_p).value)
		return result

def strict_slice(s, i, j):
	if not (0 <= i <= j <= len(s)):
		raise IndexError("invalid slice")
	return s[i:j]

class PolicyGUIUpdater(object):
	def __init__(self, sota_policy, transpose_graph, gui):
		self.sota_policy = sota_policy
		self.gui = gui
		self.tijoffsets = list(map(int, sota_policy.timins + numpy.asarray(sota_policy.min_itimes_to_dest, int)[sota_policy.network.edges.end if not transpose_graph else sota_policy.network.edges.begin]))
		self.last_edge_update_times_minus_tijoffset = list(map(lambda v: 0 - v, self.tijoffsets))
		self.next_edge_update_times_minus_tijoffset = list(map(lambda v: 0 - v, self.tijoffsets))
		self.nprev_dirty_edges = 0
		self.dirty_iedges = []
		self.barray = {}.get(None)
		self.revision = 0
	def update(self, tinow={}.get(None), umr_sum=numpy.core._methods.umr_sum if hasattr(numpy.core._methods, 'umr_sum') else numpy.core._methods.um.add.reduce if hasattr(numpy.core._methods, 'um') else numpy.core.umath.add.reduce):
		old_revision = self.revision
		revision = old_revision + 1
		self.revision = revision
		gui = self.gui
		if gui:
			barray = self.barray
			sota_policy = self.sota_policy
			last_edge_update_times_minus_tijoffset = self.last_edge_update_times_minus_tijoffset
			next_edge_update_times_minus_tijoffset = self.next_edge_update_times_minus_tijoffset
			if barray is None: self.barray = barray = [old_revision] * len(sota_policy.network.edges)
			alpha_processed = gui.alpha_processed
			active_color = numpy.asarray([1, 0.5, 0, alpha_processed], float)
			slice_ = Ellipsis  # (slice(None), slice(0, 4))
			edge_colors = gui.edge_colors
			optimality = gui.optimality
			edge_orders = gui.edge_orders
			optimality_fractions = numpy.divide(optimality, next_edge_update_times_minus_tijoffset).tolist()
			for eij in self.dirty_iedges:
				if barray[eij] >= revision:
					continue
				barray[eij] = revision
				ti_minus_tijoffset = next_edge_update_times_minus_tijoffset[eij]
				tijprev_minus_tijoffset = last_edge_update_times_minus_tijoffset[eij]
				not_equal = ti_minus_tijoffset != tijprev_minus_tijoffset
				if not_equal:
					assert ti_minus_tijoffset > 0
					we_slice = sota_policy.we[eij].ndarray[tijprev_minus_tijoffset : ti_minus_tijoffset]
					if not sota_policy.suppress_calculation:
						optimality[eij] += umr_sum(we_slice)
					last_edge_update_times_minus_tijoffset[eij] = ti_minus_tijoffset
				optimality_fraction = optimality_fractions[eij]
				#assert 0 <= optimality_fraction <= 1
				if tinow is not None and not_equal:
					edge_colors[eij][slice_] = active_color
				elif 0:
					c1 = 1. / 3
					c2 = 2. / 3
					c3 = 3. / 3
					weight = 0
					assert 0 <= weight <= 1
					edge_colors[eij][slice_] = hsl2rgb((1 - weight) * (optimality_fraction * c1 + (1 - optimality_fraction) * c2) + weight * c3, 1., optimality_fraction * 0.5 + (1 - optimality_fraction) * 0.0) + (alpha_processed,)
				else:
					edge_colors[eij][slice_] = (0, optimality_fraction, 0, alpha_processed)
				edge_orders[eij] = 2 #+ optimality_fraction  # sub-ordering slows rendering down because of thickness variations
			del self.dirty_iedges[:self.nprev_dirty_edges]
			self.nprev_dirty_edges = len(self.dirty_iedges)
			gui.invalidated += len(self.dirty_iedges)
	def mark_dirty_edges(self, ti, eijs):
		next_edge_update_times_minus_tijoffset = self.next_edge_update_times_minus_tijoffset
		last_edge_update_times_minus_tijoffset = self.last_edge_update_times_minus_tijoffset
		tijoffsets = self.tijoffsets
		dirty_iedges_append = self.dirty_iedges.append
		for eij in eijs:
			if next_edge_update_times_minus_tijoffset[eij] == last_edge_update_times_minus_tijoffset[eij]:
				dirty_iedges_append(eij)
			next_edge_update_times_minus_tijoffset[eij] = ti - tijoffsets[eij]

class ThreadingPipe(object):
	def __init__(self, read_semaphore, read_deque, write_semaphore, write_deque):
		self.sr = read_semaphore
		self.qr = read_deque
		self.sw = write_semaphore
		self.qw = write_deque
	def poll(self, timeout=0):
		if timeout not in (0, None): raise ValueError("only %s and %s are supported timeouts" % (0, None))
		popped = False
		if True:   # Try avoiding the semaphore
			try:
				value = self.qr.popleft()
				popped = True
			except IndexError: pass
		if popped:
			self.qr.appendleft(value)
			result = popped
		else:
			result = self.sr.acquire(timeout != 0)
			if result: self.sr.release()
		return result
	def recv(self):
		self.sr.acquire()
		return self.qr.popleft()
	def send(self, obj, do_copy=True):
		self.qw.append(copy.deepcopy(obj) if do_copy else obj)
		self.sw.release()

def make_threading_pipes(duplex=True):
	(s1, q1) = (threading.Semaphore(0), collections.deque())
	(s2, q2) = (threading.Semaphore(0) if duplex else None, collections.deque() if duplex else None)
	return (ThreadingPipe(s1, q1, s2 if duplex else None, q2 if duplex else None), ThreadingPipe(s2 if duplex else None, q2 if duplex else None, s1, q1))

if __name__ in ('__parents_main__', '__mp_main__'): print_("Communicating computational state to worker process...", end=" ", file=sys.stderr)
def worker_process(sota_policy, isrc, tibudget, tibudget_end_exclusive, pipe, my_listening={}.get(None), other_listening={}.get(None), tprev=timer()):
	stderr = sys.stderr
	# The try-finally blocks here are (imperfect) means for ensuring the parent process always receives a signal, even if we error
	prev_other_listening = other_listening.value - 1 if other_listening is not None else {}.get(None)
	try:
		try:  # compute policy
			if __name__ in ('__parents_main__', '__mp_main__'): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr)
			active_gui_edges = []
			stack = sota_policy.compute_optimal_update_order(isrc, tibudget, stderr=stderr)
			ti = {}.get(None)
			empty = False
			while not empty:
				empty = len(stack) == 0
				if not empty:
					(i, ti) = stack.pop()
					sota_policy.step(i, ti, active_gui_edges.append)
				is_other_ready = empty
				if not is_other_ready:
					if other_listening is None:
						is_other_listening = True
					else:
						curr_other_listening = other_listening.value
						if curr_other_listening != prev_other_listening:
							is_other_ready = True
							prev_other_listening = curr_other_listening
				if is_other_ready:
					pipe.send((ti, active_gui_edges, sota_policy.progress))
					del active_gui_edges[:]
		finally: pipe.send(b'')

		# compute path
		sota_path = SOTA.Path(sota_policy, tibudget)
		pending_items = []
		path = []
		seen_paths = {}
		new_edges_seen = {0 for _ in ()}
		def consider_edge(eij, j, path_so_far, path_node_set, reliability, timin_elapsed, tidist_curr):
			new_edges_seen.add(eij)
			return j not in path_node_set
		tibudget_current = tibudget
		empty = False
		while not empty:
			empty = tibudget_current == tibudget_end_exclusive
			if not empty: sota_path.start(isrc, tibudget_current)
			while 1:
				if not empty:
					found = sota_path.step(consider_edge)
					if found is None: break
					(reached_destination, path_so_far, reliability) = found
					if reached_destination:
						rlist = path_so_far
						assert len(path) == 0
						while rlist[1] >= 0:
							path.append(rlist[1])
							rlist = rlist[0]
						path.reverse()
						path_key = tuple(path)
						del path[:]
						path_key_or_index = seen_paths.get(path_key, path_key)
						if path_key_or_index is path_key:
							seen_paths[path_key] = len(seen_paths)
						pending_items.append((tibudget_current, path_key_or_index, reliability))
				is_other_ready = empty
				if not is_other_ready:
					if other_listening is None:
						is_other_listening = True
					else:
						curr_other_listening = other_listening.value
						if curr_other_listening != prev_other_listening:
							is_other_ready = True
							prev_other_listening = curr_other_listening
				if is_other_ready:
					pipe.send((new_edges_seen, pending_items))
					new_edges_seen.clear()
					del pending_items[:]
				if empty: break
			tibudget_current = tibudget_current - 1 if tibudget_current > tibudget_end_exclusive else tibudget_current + 1
	finally: pipe.send(b'')

def get_multiprocessing():
	global multiprocessing
	try: multiprocessing
	except NameError: import multiprocessing
	return multiprocessing

def apply_async(multiprocess, func, args={}.get(None)):
	multiprocessing = get_multiprocessing()
	(listening1, listening2) = (multiprocessing.Value('l', 0), multiprocessing.Value('l', 0))
	(piper, pipew) = multiprocessing.Pipe(True) if multiprocess else make_threading_pipes()
	kwargs = {'pipe': pipew, 'my_listening': listening2, 'other_listening': listening1}
	if multiprocess:
		process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
		process.start()
	else:
		thread = threading.Thread(target=func, args=args, kwargs=kwargs)
		thread.setDaemon(True)
		thread.start()
	return (piper, listening1, listening2)

def main(program, *args):
	os.environ['SDL_VIDEO_CENTERED'] = '1'
	numpy.seterr(all='raise')
	multiprocess = get_multiprocessing().cpu_count() > 1
	stdin  = sys.stdin
	stdout = sys.stdout
	stderr = sys.stderr
	program_name = os.path.basename(program)
	argparser = argparse.ArgumentParser(
		program_name,
		usage=None,
		description=None,
		epilog=None,
		parents=[],
		formatter_class=argparse.RawTextHelpFormatter)
	argparser.add_argument('-v', '--verbose', required=False, action='store_const', const=True, help="verbose output")
	argparser.add_argument('--network', required=False, help="the routing network", default=None)
	argparser.add_argument('--source', required=False, help="the routing source", default=None)
	argparser.add_argument('--dest', required=False, help="the routing destination", default=None)
	argparser.add_argument('--budget', required=False, help="the time budget (in real time)", default=None)
	argparsed = argparser.parse_args(args)
	if len(args) == 0: return argparser.print_usage(sys.stderr)

	Network = SOTA.Network

	# For SF, optimized C++ takes 39 ms to load everything, whereas optimized CPython takes 892 ms and PyPy takes 758 ms
	if stderr is not None: tprev = timer(); print_("Reading network...", end=" ", file=stderr)

	network_input_file = open(argparsed.network, 'r') if argparsed.network is not None else stdin
	try: loaded_edges = Network.load_edges(network_input_file)
	finally: network_input_file is not stdin and network_input_file.close()

	(edges, node_indices) = loaded_edges
	Network.remove_bad_edges(edges, 0.1 ** 0.5 * 2, True)
	if stderr is not None: print_(int((timer() - tprev) * 1000), "ms", file=stderr); del tprev

	if stderr is not None: tprev = timer(); print_("Loading network...", end=" ", file=stderr)
	network = Network(edges)
	discretization = numpy.min(network.edges.tmin)
	del edges

	src = parse_id(argparsed.source)
	dst = parse_id(argparsed.dest)
	tbudget = float(argparsed.budget) if argparsed.budget is not None else None

	isrc = node_indices[network.make_id(*src)] if src is not None else None
	idst = node_indices[network.make_id(*dst)] if dst is not None else None
	transpose_graph = False
	if transpose_graph: (isrc, idst) = (idst, isrc)

	gui_update_interval_sec = 1. / (60 if multiprocess else 25)
	gif_name = None #'sota_captured.gif'
	gif = {}.get(None)
	try:
		if gif_name is not None:
			import imageio
			gif = imageio.get_writer(gif_name, mode='I', subrectangles=True, duration=gui_update_interval_sec)
	except ImportError: pass
	gui = NetworkGUI(network.edges if pyglet else {}.get(None), gif)
	try:
		if pyglet and pyglet.gl:
			for value in pyglet.gl.__dict__.values():
				if isinstance(value, ctypes._CFuncPtr) and value.errcheck == pyglet.gl.lib.errcheck:
					value.errcheck = gl_errcheck
		gui.refresh()

		if stderr is not None: print_(int((timer() - tprev) * 1000), "ms", file=stderr); del tprev

		if isrc is not None and idst is not None:
			gui_policy_update_interval_sec = gui_update_interval_sec
			gui_path_update_interval_sec = gui_update_interval_sec
			time_start = timer()
			sota_policy = SOTA.Policy(network, idst, discretization, False, False, transpose_graph, suppress_calculation=False, stderr=stderr)
			print_("Routing from #%s to #%s with budget T = %s/%s = %s steps..." % (isrc, idst, tbudget, discretization, int(SOTA.discretize_up([tbudget], discretization)[0])), file=stderr)
			(visited_iedges, tibudget, total_progress) = sota_policy.prepare(isrc, tbudget, 1, False, stderr=stderr)
			(pipe, my_listening, other_listening) = apply_async(multiprocess, worker_process, (sota_policy, isrc, tibudget, sota_policy.min_itimes_to_dest[isrc] - 1))
			alpha = 0.5
			init_progress_time = timer()
			prev_progress_time = init_progress_time
			gui.update_edges(visited_iedges, (0, 0, 0, alpha), 1)
			prev_progress_time_discount = gui.refresh(True)
			init_progress = 0
			prev_progress = init_progress
			gui_policy_updater = PolicyGUIUpdater(sota_policy, sota_policy.transpose_graph, gui)
			while 1:
				my_listening.value += 1
				obj = pipe.recv()
				if not obj: break
				(ti, active_gui_edges, progress) = obj
				gui_policy_updater.mark_dirty_edges(ti, active_gui_edges)
				time_now1 = timer()
				if gui.is_out_of_date(gui_policy_update_interval_sec):
					gui_policy_updater.update(ti)
					prev_progress_time_discount += gui.refresh()
				if progress == 0 or progress - prev_progress >= 500000:
					time_now2 = timer()
					time_elapsed = time_now1 - init_progress_time - prev_progress_time_discount
					time_total = time_elapsed * total_progress / (progress - init_progress) if progress > prev_progress else float('inf')
					print_("Updated node to t = %.2f seconds = %d steps (%.1f%% complete, %.1f/%.1f compute sec remaining)" % (
						ti * discretization, ti, progress * 100.0 / (total_progress if total_progress else 1), time_total - time_elapsed, time_total
					), file=stderr)
					prev_progress_time = time_now1 + (timer() - time_now2)
					prev_progress_time_discount = 0
					prev_progress = progress
					
			gui_policy_updater.update()
			gui.refresh()
			time_policy = timer() - time_start
			print_("Policy: %s" % (repr(max(sota_policy.uv[isrc].tolist()[tibudget - sota_policy.min_itimes_to_dest[isrc]:] + [0.0])),), file=stderr)
			print_("Policy time: %s seconds" % (time_policy,), file=stderr)

			time_start = timer()
			seen_paths_and_reliabilities = {}
			seen_paths = []
			old_edges_seen = []
			new_edges_seen = {0 for _ in ()}
			OLD_EDGE_GUI_UPDATE = ((1, 0.75, 0, gui.alpha_processed), 3, 6)
			NEW_EDGE_GUI_UPDATE = ((1, 0.25, 0, gui.alpha_processed), 4, 4)
			while 1:
				my_listening.value += 1
				obj = pipe.recv()
				if not obj: break
				(new_edges_seen_delta, pending_items) = obj
				new_edges_seen.update(new_edges_seen_delta); del new_edges_seen_delta
				if gui.is_out_of_date(gui_path_update_interval_sec):
					new_edges_seen.difference_update(old_edges_seen)
					gui.update_edges(old_edges_seen, *OLD_EDGE_GUI_UPDATE)
					gui.update_edges(new_edges_seen, *NEW_EDGE_GUI_UPDATE)
					gui.refresh(len(new_edges_seen) > 0)
					old_edges_seen[:] = new_edges_seen
				for item in pending_items:
					(tibudget_current, path_key_or_index, reliability) = item
					path_key = path_key_or_index
					if isinstance(path_key_or_index, int):
						path_key = seen_paths[path_key_or_index]
					else:
						seen_paths.append(path_key)
						print_("Found new path for tibudget = %s; examining other time budgets..." % (tibudget_current,))
						seen_paths_and_reliabilities[path_key] = (len(seen_paths_and_reliabilities), [])
						gui.add_path(list(path_key), sota_policy.transpose_graph, (1 - reliability, 0, reliability, gui.alpha_processed * 1.0 / 2), 8, 0)
						if gui.is_out_of_date(gui_path_update_interval_sec):
							gui.refresh(True)
					seen_paths_and_reliabilities[path_key][1].append(reliability)
			old_edges_seen[len(old_edges_seen):] = new_edges_seen
			gui.update_edges(old_edges_seen, *OLD_EDGE_GUI_UPDATE)
			del old_edges_seen[:]
			for (path, (index, reliabilities)) in sorted(seen_paths_and_reliabilities.items(), key=lambda p: p[1][0]):
				print_("Path: [%s .. %s] %s" % (reliabilities[0], reliabilities[-1], list(map(sota_policy.network.edges.id.__getitem__, path))), file=stdout)
			gui.refresh(True)
			time_path = timer() - time_start
			print_("Path   time: %s seconds" % (time_path,), file=stderr)
			print_("Total  time: %s seconds" % (time_path + time_policy,), file=stderr)
		while gui:
			gui.refresh(True, True)
	finally:
		if gui is not None and gui.gif:
			gui.gif.close()
			gui.gif = None
if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev

if __name__ == '__main__':
	raise SystemExit(main(*sys.argv))
