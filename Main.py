import argparse
import ctypes
import heapq
import imp
import os
import sys
import timeit

if sys.version_info >= (3, 0): exec("print_ = print; xrange = range")
else: print_ = __import__('__builtin__').__dict__['print']

timer = timeit.default_timer

if 'numpy' not in sys.modules: tprev = timer(); print_("Importing numpy...", end=' ', file=sys.stderr)
import numpy
if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev

if 'scipy.special' not in sys.modules: tprev = timer(); print_("Importing scipy.special...", end=' ', file=sys.stderr)
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
		import scipy.special
	finally:
		if SetConsoleCtrlHandler_body:
			SetConsoleCtrlHandler_body[0:len(SetConsoleCtrlHandler_body_new)] = SetConsoleCtrlHandler_body_old
except ImportError as e:
	pass
if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev

if 'numba' not in sys.modules: tprev = timer(); print_("Importing numba...", end=' ', file=sys.stderr)
try: import numba
except ImportError: pass
if 'tprev' in locals(): print_(int((timer() - tprev) * 1000), "ms", file=sys.stderr); del tprev

import SOTA

pyglet = {}.get(None)
pyglet_win32 = {}.get(None)
try: import pyglet, pyglet.libs.win32 as pyglet_win32
except ImportError: pass

def is_debugger_present():
	import platform
	if platform.system() == 'Windows':
		import ctypes
		return ctypes.windll.kernel32.IsDebuggerPresent()
	return __debug__

class NetworkGUI(object):
	def __init__(self, edges, gif=None):
		self.prev_update_duration = -1  # used to indicate that we haven't started
		self.prev_update_time = 0
		self.gif = gif
		self.alpha_processed = 1
		if not edges:
			self.window = None
		else:
			self.gl = gl = pyglet.gl
			width = 480
			self.window = pyglet.window.Window(resizable=True, config=gl.Config(double_buffer=False), width=width, height=int(width / 1.025) if width else None)
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
		glMultiDrawArrays = gl.glMultiDrawArrays
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
		gl.glColorPointer(self.line_colors.size // len(self.lines), gl.GL_DOUBLE, 0, self.line_colors.ctypes.data)
		gl.glVertexPointer(self.lines_array.size // len(self.lines), gl.GL_DOUBLE, 0, self.lines_array.ctypes.data)
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
		for (mode, path, color, lanes) in self.layers:
			glLineWidth(self.lanes_to_line_width(lanes))
			gl.glColor4d(*color)
			segment_starts = self.edge_indices_to_line_starts[path]
			segment_counts = self.edge_indices_to_line_counts[path]
			segment_indices = numpy.ones(segment_counts.sum(), numpy.int32)
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
	def is_possibly_visible(self):
		return pyglet_win32 is None or self.window and not pyglet_win32._user32.IsIconic(self.window._hwnd)
	def is_out_of_date(self, update_interval_sec):
		if update_interval_sec < self.prev_update_duration: update_interval_sec = self.prev_update_duration
		return timer() - self.prev_update_time >= update_interval_sec
	def add_path(self, path, color, lanes, index={}.get(None)):
		if self:
			self.layers.insert(index if index is not None else len(self.layers), (self.gl.GL_LINE_STRIP, path, color, lanes))
	def update_edges(self, eijs, color=None, order=None, lanes=None):
		if self:
			for eij in eijs:
				if color is not None: self.edge_colors[eij][:] = color
				if order is not None: self.edge_orders[eij] = order
				if lanes is not None: self.edge_lanes[eij] = lanes
	def refresh(self, resort=False, wait_for_next_event=False):
		if self.window and self.prev_update_duration < 0:  # we haven't started yet...
			self.prev_update_duration = 0
			if pyglet:
				pyglet.app.event_loop.has_exit = False
				pyglet.app.event_loop.is_running = True
				pyglet.app.event_loop._legacy_setup()
				pyglet.app.platform_event_loop.start()
		if self:
			now = timer()
			if resort: self.sort_edges()
			self.prev_update_duration = timer() - now
			if self.is_possibly_visible:
				self.window.dispatch_event('on_draw')
				self.window.flip()
				self.captured_frames += 1
				buffer = numpy.empty((self.window.height, self.window.width, 4), numpy.uint8)
				self.gl.glReadPixels(0, 0, self.window.width, self.window.height, self.gl.GL_RGBA, self.gl.GL_UNSIGNED_BYTE, ctypes.cast(buffer.ctypes.data, ctypes.c_void_p))
				if self.gif: self.gif.append_data(buffer[::-1, ...])
			has_exit = pyglet.app.event_loop.has_exit
			if not has_exit:
				idle = pyglet.app.event_loop.idle()
				if not wait_for_next_event: idle = 0
				pyglet.app.platform_event_loop.step(idle)
				has_exit = pyglet.app.event_loop.has_exit
			if has_exit:
				pyglet.app.platform_event_loop.stop()
				self.window = None
			self.prev_update_time = timer()

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
	def __init__(self, sota_policy, gui):
		self.sota_policy = sota_policy
		self.gui = gui
		self.tijoffsets = list(map(int, sota_policy.timins + numpy.asarray(sota_policy.min_itimes_to_dest, int)[sota_policy.network.edges.end]))
		self.last_edge_update_times_minus_tijoffset = list(map(lambda v: 0 - v, self.tijoffsets))
		self.next_edge_update_times_minus_tijoffset = list(map(lambda v: 0 - v, self.tijoffsets))
		self.nprev_dirty_edges = 0
		self.dirty_iedges = []
		self.barray = {}.get(None)
	def update(self, tinow={}.get(None), umr_sum=numpy.core._methods.umr_sum):
		gui = self.gui
		if gui:
			barray = self.barray
			sota_policy = self.sota_policy
			last_edge_update_times_minus_tijoffset = self.last_edge_update_times_minus_tijoffset
			next_edge_update_times_minus_tijoffset = self.next_edge_update_times_minus_tijoffset
			if barray is None: self.barray = barray = [0] * len(sota_policy.network.edges)
			alpha_processed = gui.alpha_processed
			active_color = numpy.asarray([1, 0.5, 0, alpha_processed], float)
			slice_ = Ellipsis  # (slice(None), slice(0, 4))
			edge_colors = gui.edge_colors
			optimality = gui.optimality
			edge_orders = gui.edge_orders
			optimality_fractions = numpy.divide(optimality, next_edge_update_times_minus_tijoffset).tolist()
			for eij in self.dirty_iedges:
				if barray[eij]:
					continue
				barray[eij] = 1
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
			for eij in self.dirty_iedges:
				barray[eij] = 0
			del self.dirty_iedges[:self.nprev_dirty_edges]
			self.nprev_dirty_edges = len(self.dirty_iedges)
			gui.refresh()
	def mark_dirty_edges(self, ti, eijs):
		next_edge_update_times_minus_tijoffset = self.next_edge_update_times_minus_tijoffset
		last_edge_update_times_minus_tijoffset = self.last_edge_update_times_minus_tijoffset
		tijoffsets = self.tijoffsets
		dirty_iedges_append = self.dirty_iedges.append
		for eij in eijs:
			if next_edge_update_times_minus_tijoffset[eij] == last_edge_update_times_minus_tijoffset[eij]:
				dirty_iedges_append(eij)
			next_edge_update_times_minus_tijoffset[eij] = ti - tijoffsets[eij]

def main(program, *args):
	os.environ['SDL_VIDEO_CENTERED'] = '1'
	numpy.seterr(all='raise')

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

	gui_update_interval_sec = 1. / 25
	gif_name = None #'sota_captured.gif'
	gif = {}.get(None)
	try:
		if gif_name is not None:
			import imageio
			gif = imageio.get_writer(gif_name, mode='I', subrectangles=True, duration=gui_update_interval_sec)
	except ImportError: pass
	try:
		gui = NetworkGUI(network.edges if pyglet else {}.get(None), gif)
		if pyglet and pyglet.gl:
			for value in pyglet.gl.__dict__.values():
				if isinstance(value, ctypes._CFuncPtr) and value.errcheck == pyglet.gl.lib.errcheck:
					value.errcheck = gl_errcheck
		gui.refresh()

		if stderr is not None: print_(int((timer() - tprev) * 1000), "ms", file=stderr); del tprev

		if isrc is not None and idst is not None:
			gui_policy_update_interval_sec = gui_update_interval_sec
			gui_path_update_interval_sec = gui_update_interval_sec
			time_total = 0
			time_start = timer()
			sota_policy = SOTA.Policy(network, idst, discretization, True, False, suppress_calculation=False, stderr=stderr)
			print_("Routing from #%s to #%s with budget T = %s/%s = %s steps..." % (isrc, idst, tbudget, discretization, int(SOTA.discretize_up([tbudget], discretization)[0])), file=stderr)
			prev_progress_time = timer()
			(stack, visited_iedges, tibudget) = sota_policy.prepare(isrc, tbudget, 1, False, stderr=stderr)
			alpha = 0.5
			gui.update_edges(visited_iedges, (0, 0, 0, alpha), 1)
			gui.refresh(True)
			prev_progress = 0
			active_gui_edges = list()
			gui_policy_updater = PolicyGUIUpdater(sota_policy, gui)
			stack.reverse()
			for (i, ti) in stack:
				sota_policy.step(i, ti, active_gui_edges.append if gui else None)
				gui_policy_updater.mark_dirty_edges(ti, active_gui_edges)
				del active_gui_edges[:]
				time_now = timer()
				if gui.is_out_of_date(gui_policy_update_interval_sec):
					gui_policy_updater.update(ti)
				if len(stack) == 0 or sota_policy.progress == 0 or sota_policy.progress - prev_progress >= 500000:
					print_("Updated node %s to t = %.2f seconds = %d steps (%.0f node-steps/min)" % (i, ti * discretization, ti, discretization * (sota_policy.progress - prev_progress) / (time_now - prev_progress_time) / 60), file=stderr)
					prev_progress_time = time_now
					prev_progress = sota_policy.progress
			gui_policy_updater.update()
			update_pause_sec2 = 1.0 / 60
			if 1:
				sota_path = SOTA.Path(sota_policy, tibudget)
				seen_edges = {0 for _ in ()}
				seen_paths = {}
				for tibudget_current in range(tibudget, sota_policy.min_itimes_to_dest[isrc] - 1, -1):
					sota_path.start(isrc, tibudget_current)
					new_edges_seen = []
					while 1:
						found = sota_path.step(lambda eij, *args: new_edges_seen.append(eij))
						i = 0
						while i < len(new_edges_seen):
							if new_edges_seen[i] in seen_edges:
								new_edges_seen[i] = new_edges_seen[-1]
								new_edges_seen.pop()
								continue
							i += 1
						seen_edges.update(new_edges_seen)
						gui.update_edges(new_edges_seen, (1, 0.75, 0, gui.alpha_processed), 3, 6)
						if gui.is_out_of_date(gui_path_update_interval_sec):
							gui.refresh(len(new_edges_seen) > 0)
						del new_edges_seen[:]
						if found is None: break
						(reached_destination, path_so_far, reliability) = found
						if not reached_destination: continue
						path = []
						while path_so_far[1] is not None:
							path.append(path_so_far[1])
							path_so_far = path_so_far[0]
						path.reverse()
						path_key = tuple(path)
						if path_key not in seen_paths:
							seen_paths[path_key] = (len(seen_paths), [])
							gui.add_path(path, (1 - reliability, 0, reliability, gui.alpha_processed * 1.0 / 2), 8, 0)
							if gui.is_out_of_date(gui_path_update_interval_sec):
								gui.refresh(True)
						seen_paths[path_key][1].append(reliability)
				for (path, (index, reliabilities)) in sorted(seen_paths.items(), key=lambda p: p[1][0]):
					print_("Path: [%s .. %s] %s" % (reliabilities[0], reliabilities[-1], list(map(sota_policy.network.edges.id.__getitem__, path))), file=stdout)
			gui.refresh(True)
			time_total += timer() - time_start
			print_("Policy: %s" % (repr(max(sota_policy.uv[isrc].tolist()[tibudget - sota_policy.min_itimes_to_dest[isrc]:] + [0.0])),), file=stderr)
			print_("Time: %s seconds" % (time_total,), file=stderr)
		while gui:
			gui.refresh(True, True)
	finally:
		if gui.gif:
			gui.gif.close()
			gui.gif = None

if __name__ == '__main__':
	raise SystemExit(main(*sys.argv))
