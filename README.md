# What is SOTA-Py?

SOTA-Py is a Python-based solver for the policy- and path-based "SOTA" problems,
using the algorithm(s) described in
[*Tractable Pathfinding for the Stochastic On-Time Arrival Problem*](https://link.springer.com/chapter/10.1007/978-3-319-38851-9_16) (also in the corresponding [arXiv preprint](https://arxiv.org/abs/1408.4490))
and previous works referenced therein.

What is the SOTA problem? Read on...

# Theory (in plain English)

## What is the ***Stochastic On-Time Arrival*** problem (SOTA)?

It's the ***reliable routing*** problem:

> ### *How do you travel from point A to point B **in T time** under traffic?*  

For example, you might have a meeting in San Jose at 3pm, and one to reach in San Francisco at 4pm.  
Or you might need to get from your house to the airport in less than 1 hour.

### Doesn't Google Maps already solve this?

No. It doesn't let you specify a time budget.
It only lets you specify a departure or arrival time, but not both.

What it (probably) gives you is the path with the *least expected (average) time* to your destination.

### But so what? 30 minutes or 60 minutes—isn't there a *single* best path?

No. That would only be the case if traffic was perfectly predictable.

If you don't have a lot of time, you might need to take a riskier path
(e.g. a highway), otherwise you might have no chance of reaching your destination on time.
But if you have a lot of time, you might take a safer path (like side roads) that no one uses,
to avoid suddenly getting stuck in the middle of, say, a highway, due to traffic.

That means your time budget can affect your route.

## Policy- vs. Path-based Routing

### What is the *policy-based* SOTA problem?

It is the case of the SOTA problem where you decide which road to take based on how much time you have left.
You'd probably need a navigation device for this, since there are too many possibilities in the "policy" to print on paper.

This is what you'd prefer to do, because it can potentially give better results depending on whether you get lucky/unlucky with traffic.

This is a dynamic-programming problem, because the probability of reaching your destination on time
is just the maximum probability of reaching it from each road at the next intersection.

### What is the *path-based* SOTA problem?

It is the SOTA problem in the case where you statically decide on the entire path before you depart.  
You can just print out a map for this on paper, the old-fashioned way.

This is—counterintuitively!—a much tougher problem than finding the policy.
Even though the solution looks simpler (it's just a path rather than a policy),
it's much harder to compute.
Why? Intuitively, it's because after you travel a bit,
you won't necessarily be on the most optimal path anymore,
so you can't make that assumption to simplify the problem initially.  
By contrast, in the policy-based scenario, you always assume that your future actions are optimal,
so you have an optimal subproblem structure to exploit.

## The Algorithm

The (unhelpful) ultra-short version is that Dijkstra's algorithm is used for policy-based SOTA and A* is used for path-based SOTA.

The (more helpful) short version is:
- For the policy computation, a re-visiting variant of Dijkstra's algorithm
is used to obtain an optimal ordering for computing the reliabilty of each node,
and a technique known as *zero-delay convolution* is used to perform cross-dependent convolutions incrementally
to keep the time complexity quasilinear in the time budget. (A naive FFT would not do this.)  
- For the path computation, the computed policy is used as an (admissible) heuristic in A*.
Note that this choice of a heuristic is critical. A poor heuristic can easily result in exponential running time.

For the long version, please see the paper linked above, and others referenced inside.
The paper should (hopefully) be quite easy to follow and understand, especially as far as research papers go.

Note that the pre-processing algorithms from the paper (such as *arc-potentials*) are **not** implemented,
but they should be far easier to implement than the pathfinding algorithms themselves.

## The Traffic Model

This code models the travel time across every road as a mixture of Gaussian distributions (GMM)
("censored" to strictly positive travel times).
It discretizes the distributions and solves the discrete problem in discrete-time.

Obviously, realistic travel times are not normally distributed. But that's the model of the data I had.
Getting good traffic data is hard, and encoding data efficiently is also hard.
If you don't like the current model, you'd have to change the code to accommodate other models.

# The Code

## Inputs

### Dependencies

- [**NumPy**](http://www.numpy.org) is the only hard external dependency.  
- [**Numba**](//numba.pydata.org), if available, is used for compiling Python to native code (≈ 3× speedup).  
- [**PyGLET**](http://www.pyglet.org), if available, is used for rendering the results on the screen via OpenGL.
- [**SciPy**](//www.scipy.org), if available, is used for slightly faster FFTs.  

### Map File Format

The road network and traffic data is assumed to be a concatentation of JSON objects, each as follows:

	{
		"id": [10000, 1],
		"startNodeId": [1000, 0],
		"endNodeId": [1001, 0],
		"geom": { "points": [
			{"lat": 37.7, "lon": -122.4},
			{"lat": 37.8, "lon": -122.5}
		]},
		"length": 12,
		"speedLimit": 11.2,
		"lanes": 1,
		"hmm": [
			{"mode": "go", "mean": 1.2, "cov": 1.5, "prob": 0.85},
			{"mode": "stop", "mean": 7, "cov": 0.1, "prob": 1.5E-1}
		]
	}

Note the following:

- The HMM directly represents *travel times* for various "modes" of travel (stop, go, etc.) for the Gaussian mixture model.
- The HMM is "optional". If missing, pseudo-random data is generated.
- The length and speed limit are divided to obtain the *minimum* travel time across each edge
  (we assume an ideal world where everyone abides by the speed limit).
  Therefore, their individual values are not relevant; only their ratio is relevant.
- The number of lanes is only for rendering purposes.
- Every ID is assumed to be of the form [primary, secondary], where the secondary number is small.  
  The secondary component is intended to distinguish different segments of the same road for each edge.
- A minimum covariance is enforced in the code. (If your variance is too low, you may need to change this.)
- No comma or brackets should delimit these objects, so the full file isn't strictly JSON.
- For hand-checking simple cases, I recommend you set the length to be a multiple of the speed limit
  in order to avoid floating-point round-off error.

## Maintenance (or: why is the code ugly?)

This code isn't intended to finish any job for you. It's certainly not production-quality.
It's just meant to help any researchers working on this topic get started and/or cross-check their algorithm correctness.

Given that it's not meant to be used in any production,
I don't plan on actively maintaining it unless I encounter bugs (or if I see enough interest from others).

## Example

There's no short "getting started" code example, sorry.
The main startup file is basically a (very) long example.

### Usage

It's pretty self-explanatory:

	python Main.py --source=65280072.0 --dest=65345534.0 --budget=1800 --network="data/sf.osm.json"

The time discretization interval is automatically chosen to be the globally minimum travel time across
any edge in the network,
since it should be as large as possible (for speed) and smaller than the travel time of every edge.
You would need to change this in the code for greater accuracy.

Note that a time budget that is too high can cause the pathfinding algorithm to thrash exponentially, because
the reliability of *every* path reaches 100% as your time budget increases, and the algorithm ends up
trying all of them.  
However, realistically, you would not need to run this algorithm for very high time budgets.
A classical path would already be reliable enough.

### Demo

Note that you (obviously) need both a map and traffic data to run this code.
Unfortunately I can't release the dataset I used in the paper,
but I have a sample map from **OpenStreetMap**, and the code attempts to naively fill in missing traffic data,
so that should be good enough to get started.

Here's an example of what one can get in 15 seconds on my machine. The code runs in two phases:

- As time increases, the optimal policy is computed for reachable roads farther and farther from the destination
  (highlighted), until the source is reached.  
  Roads that can never be used to reach the destination on time are not examined.
- Once the policy is determined, the optimal path for each time budget *up to* the one requested is determined,
  in order from high to low time budget.  
  This is to demonstrate the fact that the optimal path can change depending on the time budget.

![Animation](doc/SOTA-Demo.gif)

# Contact

## Licensing

Please refer to the license file.

For attribution, a reference to the aforementioned article (which this code is based on) would be kindly appreciated.

## Questions/Comments

If you find a bug, have questions, would like to contribute,
or the like, feel free to open a GitHub issue/pull request/etc.

For private inquiries (e.g. commercial licensing requests), you can find my contact
information if you search around (e.g. see the paper linked above).
