---
title: Approximating arbitrary functions with shallow neural nets
date: 2023-09-14 03:00:00 -500
categories: [general]
math: true
---

I've been recently studying the material in Prof. Florian Marquadt's excellent [Advanced Machine Learning lecture series](https://pad.gwdg.de/s/2021_AdvancedMachineLearningForScience). The lectures are specially worth your time if your background is in Physics. Now, since I am the kind of person that can only fixate something after working on it myself, I've been solving most of his take-home problems. I will be posting my solutions to problems that I find most insightful. 

This particular problem [(1.1)](https://pad.gwdg.de/s/FUUwJ6c21#Problem-1-Expressivity-of-shallow-Neural-Networks) asks for proving that any continuous function in either $\mathbb{R}$ or $\mathbb{R}^2$ can be approximated by a neural net. This is actually a subcase of the well known fact that neural nets can approximate arbitrarily well any continuous function $f: \mathbb{R}^m \rightarrow \mathbb{R}^n$ inside a bounded domain. [This property](https://link.springer.com/article/10.1007/BF02551274) was proven by G. Cybenko in the late 1980s, showing that even a very shallow architecture consisting of a network with a single hidden layer can do the job.

## One-dimensional case
To get the main idea behind the proof, let's first focus on 1D mappings, i.e., $y = f(x)$ with $x, y \in \mathbb{R}$. To simplify things further, let's restrict $x$ to lie inside the unit interval, $x \in [0,1]$. It is very easy to generalize these restrictions to more complex cases once you understand the overall strategy.

We will focus on using a ReLU nonlinearity, although other nonlinear activation functions such as $\tanh$ or sigmoid works just as well with minimal modifications. Our shallow network would look like the following

![net](/assets/shallownet3.png)

The basic idea in constructing an approximation to $f$ is to partition the unit interval into $n$ arbitrarily small intervals $I_j = [\frac{j}{n},\frac{j+1}{n}]$ with $j=0,...,n-1$ while also "partitioning" the neurons of the hidden layer into $n$ sets $H_j$ again for $j=0,...,n-1$. Now, each set of hidden neurons $H_j$ is used to approximate $f$ only near the $j$-th interval, i.e., inside $I_{j-1}$, $I_j$ and $I_{j+1}$; otherwise they should output $0$. Finally, since the outputs of all hidden neurons are summed together to produce $y$, the network approximation of $f$ should work for any $x$ on the unit interval.

But how do we approximate $f$ on each interval $I_j$? Luckily, that requires only 3 hidden neurons when using the ReLU activation function. The idea is that each of the 3 neurons belonging to $H_j$ output a ReLU activation that, when summed together, produce a sharp spike at the center of $I_j$ with height equal to 1. We then scale this spike to match the function we want to approximate, i.e., multiply by $f(\frac{j+1/2}{n})$. Finally, we superimpose all spikes together, effectively producing a linear interpolation of $f$ that matches the function exactly at the points 
$\frac{j+1/2}{n}$ for $j=0,...,n-1$.

For each set $H_j$, the activations of the three neurons on it are given by

Neuron 1: $$a_j^-(x) = ReLU\left(x-\frac{j-1/2}{n}\right)$$

Neuron 2: $$a_j(x) = ReLU\left(x-\frac{j+1/2}{n}\right)$$

Neuron 3: $$a_j^+(x) = ReLU\left(x-\frac{j+3/2}{n}\right)$$

The terms inside parentheses can be obtained by applying biases in the connections between the input layer and the hidden layer. In the connection to the output layer, these three activations are scaled and summed as

$$F_j(x) = f\left(\frac{j+1/2}{n}\right)[a^-_j(x) -2 a_j(x) + a^+_j(x)]$$

It can be easily seen that the function $F_j(x)$ is a symmetric triangular peak centered at $x=\frac{j+1/2}{n}$, with a base equal to the line segment $\Delta x = [\frac{j-1/2}{n},\frac{j+3/2}{n}]$ and height $y = f(\frac{j+1/2}{n})$. 

The  activation of the neuron in the output layer can then be written as 

$$F(x) = \sum_{j=1}^n F_j(x)$$

Now, to see why $F$ is a linear interpolation of $f$, notice that for $x=\frac{j+1/2}{n}$, $F(x)$ matches exactly with $f(x)$ since the only contributing peak with a nonzero contribution is the term $F_j(x)$. For any $x$ between two consecutive peak centers, e.g., $x \in [\frac{j+1/2}{n},\frac{j+3/2}{n}]$, the only non-zero contributions are $F_j(x)$ and $F_{j+1}(x)$. But since both $F_j$ and $F_{j+1}$ are linear and monotone in this interval, their sum must be a line connecting the peak apexes.

## Two-dimensional case

A very similar construction works for proving that a shallow net can approximate any continuous function on the two-dimensional plane, i.e., $f: 
 \mathcal{C} \rightarrow \mathbb{R}$ with $\mathcal{C} \subset \mathbb{R}^2$. We will now use sigmoid activation functions $\sigma(a) = 1/[1+\exp(-a)]$ inside our network (although ReLU or $\tanh$ would work as well).

The idea is the same as before. We discretize region $\mathcal{C}$ into a fine grid of points and partition the neurons in the hidden layer so that each set of neurons is used to approximate $f$ inside each point in the grid, outputting a null contribution for $\mathbf{x}$ far from it. To approximate $f$ inside each grid point, we use a superposition of sigmoid activations to produce a sharp peak centered on the grid point.

For simplicity, let's assume  $\mathcal{C} = [0,1]\times[0,1]$. Now, let's show how to produce a sharp peak centered at $[0,0]$. Let choose $\mathbf{w} = [w, 0]$ as the weights of the 2 connections between the input vector $\mathbf{x}=[x_1,x_2]$ and a neuron in the hidden layer.

We first produce a function which is $0$ everywhere except on strip along the $x_2$-axis with a adjustable width. Such function can be written as the following superposition of sigmoids

$$ s(\mathbf{x}) = \sigma(\mathbf{w}\cdot\mathbf{x}+b) + \sigma(-\mathbf{w}\cdot\mathbf{x}+b) - 1 = \sigma(wx_1+ b)+\sigma(-wx_1+b) - 1$$ 

Supposing $w$ is large enough and $b>0$, the two sigmoids will produce step-like functions. It can be clearly seen that their sum $s$ is given by

* $s(x_1) \approx 0$ for $x_1<-b/w$
* $s(x_1) \approx 1$ for $-b/w< x_1< b/w$
* $s(x_1) \approx 0$ for $b/w< x_1$

So, the graph of $s$ has the shape of an infinite strip along the $x_2$ axis with width $2b/w$ and height $1$ provided the weights $w$ are large enough. 

To go from a vertical strip to a spike centered at $[0,0]$, we superimpose many "rotated" strips together. By rotation we mean we apply an usual rotation $\theta$ to the vector $\mathbf{w}=[w,0]$ as $\mathbf{w_{\theta}}=[w\cos\theta, w\sin\theta]$. Now, the superposition given by

$$ s(\mathbf{x}) = \sigma(\mathbf{w}\cdot\mathbf{x}+b) + \sigma(-\mathbf{w}\cdot\mathbf{x}+b) - 1 $$

By definition of inner product, this gives approximately

* $s_\theta(\mathbf{x}) \approx 0$ for $x_1\cos\theta + x_2\sin\theta<-b/w$
* $s_\theta(\mathbf{x})\approx 1$ for $-b/w< x_1\cos\theta  + x_2\sin\theta< b$
* $s_\theta(\mathbf{x})\approx 0$ for $b/w< x_1\cos\theta + x_2\sin\theta$

Since $x_1\cos\theta + x_2\sin\theta$ is the projection of $\mathbf{x}$ in the unit vector $[\cos\theta,\sin\theta]$, $s(\mathbf{x})$ is only non-zero for  $\mathbf{x}$ lying inside an infinite strip of width $b/w$ rotated from the $x_2$ axis by $\theta$.

Finally, we superimpose ${N_\theta}$ rotated strips $s_{\theta_n}$ to produce a peak $P$ around $[0,0]$ with radius $r\approx b/w$:

$$ P(\mathbf{x}) = \frac{1}{N_\theta}\sum_{n=1}^{N_\theta} s_{\theta_n}(\mathbf{x})$$

Now that we have a peak with unit height, we can scale and shift it accordinly to match the value of the function $f$ anywhere inside $\mathcal{C}$. Suppose our grid consists of $N^2$ points $\mathbf{x}_{ij}$ with $i,j = 0,1,..,N$. Then, the output of the neural net can be written as

$$ F(\mathbf{x}) = \sum_{i,j} g_{ij}P(\mathbf{x}-\mathbf{x}_{ij})$$

The coefficients $g_{ij}$ need to abjusted to match the values of the function $f$ on the gridpoints such that $F(\mathbf{x_{ij}}) \approx f(\mathbf{x_{ij}} )$. If the overlap between peaks is sufficiently small i.e., $b/w$ is much less than the spacing between grid points, then $g_{ij} \approx f(\mathbf{x_{ij}})$.

## Formulation in Fourier space

The above discussion leaves open the question of how to effectively compute the weights $g_{ij}$ on the general case. To obtain that, we will employ a Fourier decomposition of the equation 

$$ F(\mathbf{x}) = \sum_{i,j} g_{ij}P(\mathbf{x}-\mathbf{x}_{ij}).$$

The reason for that is because this equation is actually a discrete approximation of the integral equation

$$ f(\mathbf{x}) = \int g(\mathbf{x'})P(\mathbf{x}-\mathbf{x'})d\mathbf{x'}$$

In this form, our problem is to determine a function $g$ that when convoluted with $P$ (the kernel) produces exactly $f$. However, in practive, producing such exact approximation of $f$ would require a network with an infinite number of neurons! Instead, the idea is to use Fourier theory to compute the weights $g(\mathbf{x'})$ and then resort back to a sum with finitely many terms as our approximation. 

Luckily, the Fourier transform of a convolution is the product of the transforms. Thus, in Fourier space, our equation is

$$ \tilde{f}(\mathbf{q}) = \tilde{g}(\mathbf{q})\tilde{P}(\mathbf{q})$$

Thus, the coefficients are given by $\tilde{g}(\mathbf{q}) =  \tilde{f}(\mathbf{q})/\tilde{P}(\mathbf{q})$ in Fourier space. In real space, the coefficients are given by the inverse Fourier transform:

$$ g(\mathbf{x}) =\int\frac{\tilde{f}(\mathbf{q})}{\tilde{P}(\mathbf{q})}e^{2\pi i\mathbf{q}\cdot\mathbf{x}}d\mathbf{q}$$

Here, we assumed that  $\tilde{P}(\mathbf{q})$ has no zeros in Fourier space. If there are zeros, we have to assume that the singularities of the function $\tilde{f}(\mathbf{q})/\tilde{P}(\mathbf{q})$ are integrable so that the above integral is finite. We could then use techniques such as [Cauchy's principal value](https://en.wikipedia.org/wiki/Cauchy_principal_value) to evaluate the integral.

Having an explicit form for the function $g(\mathbf{x})$, our final expression for the approximation of $f$ by the neural net is

$$ F(\mathbf{x}) = \sum_{i,j} g(\mathbf{x_{ij}})P(\mathbf{x}-\mathbf{x}_{ij}).$$





