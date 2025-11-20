
# Convergence and Stability

## Rates of Convergence

In the following, we adopt the convention $e_{k} := |x^{*} - x_{k}|$, where $x^{*}$ is the desired value and $x_{k}$ is the $k^{\text{th}}$ iterate. 

*Linear Convergence.* Iterates $\{ x_{k} \}_{k}$ converge linearly with rate $\rho$ if

$$
\lim_{ k \to \infty } \frac{e_{k+1}}{e_{k}} = \rho < 1
$$

*Superlinear Convergence.* Iterates $\{ x_{k} \}_{k}$ converge superlinearly with order $p>1$ if 

$$
\lim_{ k \to \infty } \frac{e_{k+1}}{e_{k}^{p}} = M < 1
$$

*Quadratic Convergence.* Iterates $\{ x_{k} \}_{k}$ converge quadratically if

$$
\lim_{ k \to \infty } \frac{e_{k+1}}{e_{k}^{2}} = M < \infty
$$

## Numerical Stability

*Forward Stability.* Let $f$ be an algorithm and $\tilde{f}$ be its floating point approximation. Then $f$ is forward stable if 

$$
\frac{\|f(x) - \tilde{f}(x)\|}{\|f(x)\|} = \mathcal{O}(\varepsilon_{\text{mach}})
$$

*Backwards Stable.* Let $f$ be an algorithm and $\tilde{f}$ be its floating point approximation. Then $f$ is backward stable if 

$$
\frac{\|x-\tilde{x}\|}{\|x\|} = \mathcal{O}(\varepsilon_{\text{mach}}), 
\quad \tilde{f}(x) = f(\tilde{x})
$$

> [!note] Intuition: Forward Stability
> Forward stability is "almost the right solution for the right problem", while backward stability is "the right solution for almost the right problem."

# Rootfinding

## Bisection Search 

**Theorem (Existence of A Root).** Let $f:[a,b]\to \mathbb{R}$ be a continuous function. If $f(a) f(b) \leq 0$, then there exists a root $x^{*}\in[a,b]$. 

*Proof.* Apply the intermediate value theorem. $\blacksquare$

**Bisection Search.** Given continuous $f:[a,b]\to \mathbb{R}$ with $f(a)f(b) \leq 0$, bisection iteratively updates the feasible region with the midpoint of the existing region. The convergence is linear. 

```python3
def bisection_search(f, a, b, tol=1e-12): 
    """
    Arguments: 
        f: continuous function on [a,b]
        a, b: reals satisfying f(a)f(b) <= 0

    Returns:
        A root of f in [a,b]
    """
    if f(a) * f(b) > 0:
        raise ValueError("f(a)f(b) must be non-negative")

    while abs(f(a)) > tol:
        c = (a + b) / 2
        if f(a) * f(c) <= 0:
            b = c
        else:
            a = c

    return a
```

## Quasi-Newton Methods

**Newton's Method.** Let $f \in C^{2}(x^{*}\pm d)$ where $x^{*}$ is a simple root. Suppose 

$$
\left| \frac{f''(x)}{f'(y)} \right| 
\leq K
\quad
(x,y \in [x^{*}\pm d])
$$

for some $K>0$. If $x_{0} \in (x^{*}\pm h)$ where $h=\min\left( d, \frac{1}{K} \right)$, then Newton's iterates 

$$
x_{k+1} = x_{k} - \frac{f(x_{k})}{f'(x_{k})}
$$

converges quadratically. 

> [!note] Intuition: Newton's Method
> Newton's approximates the function with a tangent line at each iteration. It thus converges when the function is "approximately linear", i.e. the curvature $f''$ is small (bounded) relative to its slope $f'$. The iterations come from $0\approx f(x_{0})+f'(x_{0})(x - x_{0})$. 

**Secant Method.** In the context of Newton's method, suppose the derivatives $f'$ are not available. Using finite-difference approximations yields iterates

$$
x_{k+1} = x_{k} - \frac{x_{k} - x_{k-1}}{f(x_{k}) - f(x_{k-1})} f(x_{k})
$$

which converges superlinearly with order $\frac{1+\sqrt{ 5 }}{2} \approx 1.6$. 

**Broyden's Method.** Generalizes quasi-Newton methods for higher dimensional domains. 

## Fixed-Point Iterations

*Lemma*. If $g$ is continuous and $x_{k+1} = g(x_{k})$ converges to $x^{*}$ then $x^{*}$ is a fixed point. 

**Theorem (Fixed Point).** If $g\in C([a,b] \to [a,b])$ then there exists $x^{*} \in[a,b]$. Furthermore, if $g'$ exists and $\|g'\|_{\infty} \leq \rho < 1$ then the fixed-point iterations converge linearly with rate $\rho$, and the fixed point is unique. 

*Proof.* For existence, apply the Intermediate Value Theorem. For convergence and uniqueness, apply first-order Taylor expansion. $\blacksquare$

> [!note] Contraction Mapping Theorem
> This theorem generalizes to general complete metric spaces via the Banach fixed-point theorem (also known as the Contraction Mapping Theorem). 

# Floating-Point Analysis

## Representation

*Floating Point Representation.* A *normal* FPN has form 

$$
(-1)^{s} \times (1.b_{1}b_{2}\dots b_{p})_{2} \times 2^{E}
$$

Here, $s \in \{ 0,1 \}$ is the signed bit, $E$ is the exponent, $1.b_{1}b_{2}\dots b_{p}$ is the significand. 
- 64-bit double precision: $p=52,E=11$. 
- For normal floats, $-1023<E<1024$, so only $2046/2048$ values are used. The $2$ encodings encode *subnormal numbers* and $\pm \infty$/NaN. 
- Subnormal numbers: $(-1)^{s}\times(0.b_{1}b_{2}\dots b_{p})_{2} \times 2^{-2022}$. 
- Convention: $\operatorname{fl}(x)$ is the correctly-rounded floating-point representation of $x$, i.e. the closest FPN to $x$ with ties broken to the number with $p ^{\text{th}}$-bit $0$. 

*Fundamental Axiom of Floating Point Arithmetic.* If $x,y$ are floating point numbers, 

$$
\operatorname{fl}(x \odot y) = (x \odot y)(1 + \delta), 
\quad |\delta| < \varepsilon_{\text{mach}}
$$

where $\varepsilon_{\text{mach}} = 2^{-53} \approx 1.1\times 10^{-16}$ is the machine error. 

## Cancellation Error

*Cancellation Error.* If $\hat{x}=1+\delta_{1}$ and $\hat{y} =1+\delta_{2}$ then possibly $\hat{x} - \hat{y} \not\approx x - y$ due to cancellation. 

> [!example] Cancellation in Quadratic Formula
> Consider the problem of finding the smaller root in the quadratic formula, i.e. $x=1-\sqrt{ 1-z }$ for $z$ small. There is large cancellation since $\sqrt{ 1-z } \approx 1$. However we can reformulate
> 
> $$
> 1 - \sqrt{ 1-z } = (1 - \sqrt{ 1-z }) \left( \frac{1+\sqrt{ 1-z }}{1 + \sqrt{ 1-z }} \right) = \frac{z}{1+\sqrt{ 1-z }}
> $$
> 
> which is approximately $z /2$, even with cancellation in the denominator. 

> [!example] Cancellation Error in Finite-Difference Approximations
> From second-order Taylor expansion, we can find that if $|f''|\leq C$, then 
> 
> $$
> \frac{f(x+h) - f(x)}{h} = \frac{f(x) - f(x-h)}{h} = f'(x) + \mathcal{O}(Ch)
> $$
> 
> But taking a third-order Taylor expansion around $x\pm h$ yields 
> 
> $$
> \frac{f(x+h)-f(x-h)}{2h} = f'(x) + \mathcal{O}(Ch^{2})
> $$
> 
> which is more accurate. 

# Solving Linear Systems

## Low-Rank Approximations

**Theorem (Eckart-Young-Minsky).** If $A_{t}$ is the $t$-truncated SVD of $A$, then $A_{t}$ is the best rank-$t$ approximation of $A$ in the operator 2-norm and Frobenius norm: 

$$
\begin{align}
\|A - A_{t}\|_{2} &= \min_{\operatorname{rank}(B)=t} \|A - B\|_{2} = \sigma_{t+1} \\
\|A - A_{t}\|_{F} &= \min_{\operatorname{rank}(B)=t} \|A-B\|_{F} = \sqrt{ \sum_{i=t+1}^{r} \sigma_{i}^{2} }
\end{align}
$$

where $r=\operatorname{rank}(A)$. 

*Lemma (Truncated SVD).* The $t$-truncated SVD for $A \in \mathbb{R}^{m\times n}$ requires $\mathcal{O}(t(n+m))$ space to store. Furthermore, it takes $\mathcal{O}(mn)$ for matrix-vector multiplication, since

$$
A_{t}x = \left( \sum_{i=1}^{t} \sigma_{i} u_{i} v_{i}^{\top} \right) x
= \sum_{i=1}^{t} \sigma_{i} u_{i} (v_{i}^{\top} x)
$$

## Condition Number

*Condition Number.* If $A\in \mathbb{R}^{n\times n}$ is invertible, then $\kappa_{2}(A)=\|A\|_{2}\|A^{-1}\|_{2}= \sigma_{1} /\sigma_{r}$. If $A$ is not invertible, then one can define $\kappa_{2}^{+}(A)=\|A\|\|A^{+}\|=\sigma_{1} /\sigma_{r}$. 

> [!note] Intuition: Condition Number
> Both $\kappa_{2}$ and $\kappa_{2}^{+}$ give bounds on the amplification of relative error. The motivation is that we want to measure the relative error in $y=Ax$ derived from matrix error, $\hat{A}=A+\delta A$. Then 
> 
> $$
> \frac{\|y - \hat{y}\|}{\|y\|} 
> = \frac{\|(\delta A)x\|}{\|y\|}
> = \frac{\|(\delta A)A^{-1}y\|}{\|y\|}
> \leq \frac{\|\delta A\|}{\|A\|} \|A\| \|A^{-1}\|
> $$
> There are similar bounds if we also consider error in $y$ derived from errors in $A$ and $x$ together, or even error in $x$ from $A$ and $y$. 

## Back-substitution

*Back-substitution.* Solve a triangular system. 

```python3
def solve_triangular(R, b, lower=False):
	"""
	Solves triangular system Tx = b. 
	Applies backsubstitution. 
	
	Arguments: 
		A: ndarray of shape (n,n)
		b: ndarray of shape (n,)
	"""
    n = b.size
    x = np.zeros_like(b)
	
	if lower:
		R = np.array(R.T, dtype=float, copy=True)
	else: 
		R = np.array(R, dtype=float, copy=True)

    if A[n-1, n-1] == 0:
        raise ValueError

    for i in range(n-1, 0, -1):
        x[i] = A[i, i]/b[i]
        for j in range (i-1, 0, -1):
            A[i, i] += A[j, i]*x[i]

    return x
```

## QR Decomposition

*QR Decomposition.* An algorithm which solves $Ax=b$ by decomposing $A=QR$ for $Q \in \mathbb{R}^{m\times m}$ orthogonal and $R\in \mathbb{R}^{m\times n}$ upper triangular, and solving $Rx=Q^{\top}b$ using back-substitution. Time complexity is $\mathcal{O}(mn ^{2})$ for factorization and $\mathcal{O}(mn)$ for back-substitution. 

*Reduced/Thin/Economy QR.* If $m\gg n$, it is more cost-efficient to form

$$
A = \begin{bmatrix}
\hat{Q} & Q_{0}
\end{bmatrix}
\begin{bmatrix}
\hat{R} \\
0
\end{bmatrix}
= \hat{Q}\hat{R}
$$

where $\hat{Q}\in \mathbb{R}^{m\times n}$ has orthonormal columns and $R\in \mathbb{R}^{n\times n}$ is upper triangular.

> [!question] To which matrices should you apply QR by Householder? 
> Matrices $A \in \mathbb{R}^{m\times n}$ with full column rank, i.e. $m \geq n$ and linearly independent columns. However, heuristics exist for $m<n$ and rank-deficient cases (see `python` code). 

> If $A \in \mathbb{R}^{m\times n}$ has full column rank, then $R$ has full column rank and $\hat{R}$ is invertible. If $m=n$ then $R$ is invertible itself. 

> [!abstract] Reduced QR by Householder
> The basic idea is to find $Q_{k}$ with orthonormal columns such that $Q_{n}\dots Q_{1}A=R$, and thus $A=Q_{1}^{\top}\dots Q_{n}^{\top}R$. It proceeds as follows: 
> 1. Let $A_{0}=A$. We will inductively preserve the following structure in $A_{k}$: 
> $$
> A_{k}
> = 
> \begin{bmatrix}
> R_{k} & A_{:,1: k} \\
> 0 & A_{:, k+1:m}
> \end{bmatrix}
> \equiv 
> \begin{bmatrix}
> R_{k} & B_{1} \\
> 0 & B_{2}
> \end{bmatrix}
> $$
> 2. Take $A_{k-1}$. We need $Q_{k-1}$ with orthonormal columns so that $A_{k}:=Q_{k}A_{k-1}$ satisfies
> $$
> A_{k} = Q_{k-1} A_{k-1}
> = \begin{bmatrix}
> I_{k-1} & 0 \\
> 0 & H
> \end{bmatrix}
> \begin{bmatrix}
> R_{k-1} & B_{1} \\
> 0 & B_{2}
> \end{bmatrix}
> = 
> \begin{bmatrix}
> R_{k-1} & 0 \\
> 0 & HB_{2}
> \end{bmatrix}
> $$
> 3. Let $y=A_{k,k:m}$. We need $Hy \propto e_{1}$ to "zero-out" the last $m-k$ entries of the $k^{\text{th}}$ column. $H$ satisfies $H^{\top}H=I$, so $\|Hy\|=\|y\|\implies Hy =\pm\|y\|e_{1}$. If we pick $v=\|y\|e_{1}-y$ and $u=v /\|v\|$, then we must have $H=I-2u u^{\top}$ to send $Hy=\|y\|e_{1}$. 
> 4. To avoid cancellation error when computing $v$, choose $v=-\operatorname{sgn}(y)\|y\|e_{1}-y$ which is equivalent to $v = \operatorname{sgn}(y)\|y\|e_{1}+y$ since the signs cancel in $vv^{\top}$. It can easily be verified each constructed $H$ actually satisfies $H^{\top}H=I$. 
> 5. To solve $Ax=b$, store the Householder vectors $\{ u_{k} \in \mathbb{R}^{m-k+1} \}_{k}$. Then, solve $Rx=Q_{n}\dots Q_{1}b$ via $b_{k:m}\leftarrow b_{k:m}-2u_{k}u_{k}^{\top}b_{k:m}$. Conclude with back-substitution. 

*Partial Pivoting.* To improve numerical stability, before forming each $A_{k}$ we permute the remaining $n-k+1$ columns to select the largest $\ell ^{2}$ norm. We do this by initializing $c_{l}=\|A_{:,l}\|^{2}$ and then updating $c_{l}\leftarrow c_{l}-A_{k,l}^{2}$ after applying the Householder reflector. At the end, we obtain $A\Pi=Q^{\top}R$ from which we can invert the permutation matrix. 

```python3
import numpy as np
from scipy.linalg import solve_triangular

def solve_qr(A, b, tol=None):
    """
    Solve Ax = b using Householder QR with column pivoting.
    If A is not full rank, uses rank heuristic. 

    Arguments:
        A: ndarray of shape (m, n)
        b: ndarray of shape (m,)

    Returns:
        If m >= n: least-squares solution
        If m <  n: minimum-norm solution
    """
    A = np.array(A, dtype=float, copy=True)
    b = np.array(b, dtype=float, copy=True)
    m, n = A.shape
    kmax = min(m, n)

    piv = np.arange(n)
    col2 = np.sum(A*A, axis=0).astype(float)

    reflectors = []

    for k in range(kmax):
        # pivot max-norm column in A[k:, k:]
        j = k + np.argmax(col2[k:])
        if j != k:
            A[:, [k, j]] = A[:, [j, k]]
            col2[[k, j]] = col2[[j, k]]
            piv[[k, j]] = piv[[j, k]]

        # construct Householder reflector for column k
        y = A[k:, k]
        y_norm = np.linalg.norm(y)
        if y_norm == 0.0:
            v = np.zeros_like(y)
            beta = 0.0
        else:
            alpha = -np.copysign(y_norm, y[0])
            v = y.copy()
            v[0] -= alpha
            v_norm = np.linalg.norm(v)
            if v_norm == 0.0:
                v[:] = 0.0
                beta = 0.0
            else:
                v /= v_norm
                beta = 2.0

        reflectors.append((k, v, beta))

        # apply reflector to trailing block
        if beta != 0.0:
            A[k:, k:] -= beta * np.outer(v, v @ A[k:, k:])

        # cheap column-norm updates for selection
        if k + 1 < n:
            col2[k+1:] -= A[k, k+1:]**2
            col2[k+1:] = np.maximum(col2[k+1:], 0.0)

    y = b.copy()
    for (k, v, beta) in reflectors:
        if beta != 0.0:
            y[k:] -= beta * v * (v @ y[k:])

    R = np.triu(A[:kmax, :n])

    # rank heuristic
    if kmax > 0:
        diag = np.abs(np.diag(R[:kmax, :kmax]))
        if tol is None:
            eps = np.finfo(float).eps
            scale = diag.max() if diag.size else 1.0
            tol = max(m, n) * scale * eps
        r = int(np.sum(diag > tol))
    else:
        r = 0

    x = np.zeros(n)

    if m >= n:
        z = np.zeros(n)
        if r > 0:
            z[:r] = solve_triangular(R[:r, :r], y[:r], lower=False)
        x[piv] = z

    else:
        z = np.zeros(n)
        if r > 0:
            z[:r] = solve_triangular(R[:r, :r], y[:r], lower=False)
        x[piv] = z

    return x
```

> [!question] What is the time complexity of QR by Householder? 
> Asymptotically $\mathcal{O}(mn ^{2})$, since updating $A_{k}$ takes $\mathcal{O}(mn)$ due to vector-matrix multiplication, and there are $n$ updates. 

*Lemma (Stability of QR by Householder).* There exists $\delta A$ such that $\hat{Q}\hat{R}=A+\delta A$ and $\|\delta A\| /\|A\|=\mathcal{O}(\varepsilon_{\text{mach}})$. In other words, the algorithm is backwards stable. 

*Lemma (Orthogonality of $Q$).* In QR by Householder, $\|Q^{\top}Q-I\|=\mathcal{O}(\varepsilon_{\text{mach}})$. 

## LU Decomposition

*LU Decomposition.* An algorithm which solves $Ax=b$ for square $A \in \mathbb{R}^{n\times n}$ by factoring $A=LU$ where $L,U\in \mathbb{R}^{n\times n}$ are lower and upper triangular, respectively. 

> [!abstract] LU by Gauss Transformations
> 
> The idea is to form $L_{n-1}\dots L_{1}A=U$ and solve $x=L_{1}^{-1}\dots L_{n-1}^{-1}Ub$ by back-substitution. 
> 1. Let $A_{0}=A$. To form $A_{k}$ for $k=1,\dots,n$ we want to apply $L_{k-1}A_{k-1}$ to "zero-out" the bottom $\texttt{k+1:n}$ entries. In other words, the $k ^{\text{th}}$ column $x_{k}$ of $A_{k-1}$ satisfies 
> $$
> x_{k} = 
> \begin{bmatrix} 
> a_{1,k} \\
> \vdots \\
> a_{k,k} \\
> a_{k+1,k} \\
> \vdots \\
> a_{n,k}
> \end{bmatrix}
> \stackrel{L_{k}}{\longrightarrow}
> \begin{bmatrix} 
> a_{1,k} \\
> \vdots \\
> a_{k,k} \\
> 0 \\
> \vdots \\
> 0
> \end{bmatrix}
> \implies
> L_{k} = \begin{bmatrix}
> 1 &  &  &  &  &  \\
>  & \ddots &  &  &  &  \\
>  &  &  & 1 &  &  &  \\
>  &  &  & -l_{k+1,k} & \ddots &  &  \\
>   &  &  & \vdots &  & \ddots &  \\
>  &  &  & -l_{n,k} &  &  & 1
> \end{bmatrix}
> = I - l_{k} e_{k}^{\top}
> $$
> where $l_{j,k}=a_{j,k} /a_{k,k}$ for all $k<j\leq n$ and $0$ elsewhere. 
> 2. Each $L_{k}$ is a *Gauss transformation*, and an important observation is that $L_{k}^{-1}=I-l_{k}e_{k}^{\top}$, while $L_{1}^{-1}\dots L_{n-1}^{-1}=I+l_{1}e_{1}^{\top}+\dots+l_{n-1}e_{n-1}^{\top}$. (The idea is $l_{j}e_{k}^{\top}=0$ whenever $j<k$.) 
> 3. Finally we evaluate $y=Ub$ and then $y+l_{1}y_{1}+l_{2}y_{2}+\dots+l_{n-1}y_{n-1}$. 

> [!question] Why are there $n-1$ matrices in LU but $n$ in QR? 
> LU applies only to square matrices, so there are no entries below the $n ^{\text{th}}$ row of the $n ^{\text{th}}$ column. On the other hand, since $m \geq n$ generally in QR, we will need to zero entries explicitly even in the $n ^{\text{th}}$ column.

*Growth Factor.* If the pivots for LU are very small, then numerical instability may occur since we divide by a tiny number to compute $l_{k}$. The growth factor measures this blowup, 

$$
G(A) = \frac{\|L\|_{\infty} \|U\|_{\infty}}{\|A\|_{\infty}}
$$

where $\|X\|_{\infty}$ is the max-norm on the vectorization of $X$. If $G(A)$ is not too large then LU is backwards stable. 

**Wilkinson's Theorem.** For some $P,Q$ permutation matrices, $G(PAQ)=\mathcal{O}(n ^{1/2+1/4\log n})$. 

> [!note] Partial Pivoting
> In practice we only use partial pivoting, i.e. $PA=LU$. Partial pivots require $\mathcal{O}(n ^{2})$ which is negligible relative to the LU cost itself of $\mathcal{O}(n ^{3})$, but complete pivoting requires $\mathcal{O}(n ^{3})$. 

> [!abstract] Partial Pivoting by Max Entry
> The most popular technique. For each $k$, 
> 1. Find $i=\arg\max_{i \geq k}|a_{i,k}|$. 
> 2. Swap rows $i$ and $k$ for the current matrix. 
> 3. Apply the Gauss transformation. The invariant is $L_{n-1}P_{n-1}\dots L_{1}P_{1}A=U$.
> 
> The key idea is that $PLP ^{-1}=PLP$ is still a Gauss transformation if $L$ is one. 

```python3
import numpy as np
from scipy.linalg import solve_triangular

def solve_lu(A, b, tol=1e-12):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = A.shape
    if m != n: 
        raise ValueError("A must be square")

    for k in range(n-1): 
        j = k + np.argmax(np.abs(A[k:, k]))
        if abs(A[j, k]) < tol:
            raise ValueError("A is singular")
        if j != k: 
            A[[k, j]] = A[[j, k]]
            b[[k, j]] = b[[j, k]]

        for i in range(k+1, n): 
            l = A[i, k] / A[k, k]
            A[i, k:] -= l * A[k, k:]
            b[i] -= l * b[k]

    if abs(A[n-1, n-1]) < tol: 
        raise ValueError("A is singular or ill-conditioned")

    x = solve_triangular(A, b, lower=False) 
    return x
```

> [!question] What is the complexity of LU? 
> The factorization requires $\mathcal{O}(n ^{3})$ time, which is around twice as fast (due to better constant factors) than QR. In fact `np.linalg.solve` implements LU with partial pivoting. 

> [!abstract] Block LU
> Suppose we're given an invertible block matrix $A$ and seek its LU factorization, 
> 
> $$
> A = \begin{bmatrix}
> A_{11} & A_{12} \\
> A_{21} & A_{22}
> \end{bmatrix}
> = \begin{bmatrix}
> L_{11} & 0 \\
> L_{21} & L_{22}
> \end{bmatrix}
> \begin{bmatrix}
> U_{11} & U_{12} \\
> 0 & U_{22}
> \end{bmatrix}
> $$
> 
> Here $A_{11} \in \mathbb{R}^{n\times n}$ and $A_{22} \in \mathbb{R}^{p\times p}$ but note $A_{21}$ and $A_{12}$ need not be square. If $A_{11}$ is invertible,
> 1. Factorize $A_{11}=L_{11}U_{11}$. Complexity $\mathcal{O}(n ^{3})$. 
> 2. Solve $L_{11}U_{12}=A_{12}$ by Gaussian elimination for $U_{12}$. Complexity $\mathcal{O}(n ^{2})$. 
> 3. Solve $L_{21}U_{11}=A_{21}$ by Gaussian elimination for $L_{21}$. Complexity $\mathcal{O}(n ^{2})$. 
> 4. Factor $S=A_{22}-L_{21}U_{12}$ to obtain $L_{22}$ and $U_{22}$. Complexity $\mathcal{O}(p ^{3})$. 
> 
> The benefits of Block LU are efficient factorization updates (for instance, if only $A_{22}$ changes) and parallel computation of independent blocks. The matrix $S$ is the *Schur complement.* 

## Cholesky Decomposition

**Theorem.** Every SPD matrix has a unique Cholesky decomposition $A=R^{\top}R$ for $R$ upper triangular with strictly positive diagonal entries, and $\|R\|^{2}=\|A\|$. 

# Optimization

## Unconstrained Optimization

*Lemma (Optimality Conditions).* Let $f:\mathbb{R}^{n}\to \mathbb{R}$ be a function. 
- First-Order Test: $\nabla f(x^{*})=0$ for a local minimum (necessary). 
- Second-Order Test: $\nabla ^{2} f(x^{*})$ is symmetric PSD for a local minimum (necessary, sufficient). If additionally $\nabla ^{2}f(x)$ is PSD for all $x$, then the minimum is unique and global, and in fact $f$ is strictly convex. 

> [!abstract] Quasi-Newton Methods in High-Dimensions
> From the Taylor expansion, we can find roots to $f:\mathbb{R}^{n}\to \mathbb{R}$ under appropriate smoothness conditions. For $f\in C^{3}$, 
> 
> $$
> f(x+p) \approx f(x) + \nabla f(x)^{\top}p + \frac{1}{2}p^{\top}\nabla ^{2}f(x)p
> $$
> 
> This second-order approximation vanishes when 
> 
> $$
> \nabla_{p}f(x+p)
> = \nabla f(x) + \nabla ^{2}f(x) p
> = 0
> \implies
> p = -(\nabla ^{2}f(x))^{-1} \nabla f(x)
> $$
> 
> Quasi-Newton iterates thus take the form 
> 
> $$
> x_{k+1} = x_{k} - \alpha_{k} B_{k}^{-1} \nabla f(x_{k})
> \equiv x_{k} + \alpha_{k} p_{k}
> $$
> 
> where $B_{k}\to \nabla ^{2}f(x_{k})$ in norm, $p_{k}=-B_{k}^{-1}\nabla f(x_{k})$ is the step, and $B_{k}$ is SPD so that $p_{k}$ is a descent direction for properly chosen $\alpha_{k}$. An appropriate step size $\alpha_{k}$ must exist if small enough, so a common approach is *backtracking line search*: halve an initial guess until a stopping condition is met. 

*Armijo Condition.* The stopping condition satisfies $f(x_{k}+\alpha_{k}p_{k})\leq f(x_{k})+c \alpha_{k}\nabla f(x_{k})^{\top}p_{k}$ where $c\in(0,1)$ is constant. The idea is to always ensure at least a constant fraction of the linear prediction term for a decrease, preventing arbitrary small perturbations. 

*Gradient Descent.* When $B_{k}=I$, we have gradient descent. Since $I$ does not tend to $\nabla ^{2}f$, descent is not guaranteed. For convex functions, convergence is sublinear, $\|e_{k}\|=\mathcal{O}(1 /k)$. The advantage is that the per-iteration update is cheap, requiring only $\mathcal{O}(n)$ to compute gradients. 

$$
x_{k+1} = x_{k} - \alpha_{k} \nabla f(x_{k})
$$

*Newton's Method.* When $B_{k}=\nabla ^{2}f(x_{k})$, we have Newton's method. If $\nabla ^{2}f$ satisfies the second-order test and is locally Lipschitz, then quadratic convergence is guaranteed. The per-iteration cost is impractical, though, at $\mathcal{O}(n ^{3})$ per iteration. 

$$
x_{k+1} = x_{k} - \alpha_{k} \nabla ^{2}f(x_{k})^{-1} \nabla f(x_{k})
$$

**Theorem (Superlinear Convergence).** If $f \in C^{3}(\mathbb{R}^{n} \to \mathbb{R})$, $\|(\nabla ^{2}f)^{-1}\|\leq B$ locally around $x^{*}$, and $B_{k} \to \nabla ^{2}f(x_{k})$, then $x_{k} \to x^{*}$ superlinearly. 

> [!question] What are conditions for choosing $B_{k}$? 
> To ensure each iterate has a descent direction, we want $B_{k}$ (thus $B_{k}^{-1}$) to be SPD: 
> 
> $$
> f(x + \alpha p) \approx f(x) + \alpha \nabla f(x)^{\top} p
> = f(x) - \alpha \nabla f(x)^{\top} B_{k}^{-1} \nabla f(x) < f(x)
> $$
> 
> for $\alpha$ small enough. Furthermore if $B_{k}$ approximates the Hessian then 
> 
> $$
> \nabla f(x_{k+1}) \approx \nabla f(x_{k}) + \nabla ^{2}f(x_{k})(x_{k+1} - x_{k})
> \approx \nabla f(x_{k}) + B_{k} (x_{k+1} - x_{k})
> $$
> 
> and hence $B_{k}s_{k} = y_{k}$, where $s_{k}=x_{k+1}-x_{k}$ while $y_{k}=\nabla f(x_{k+1}) - \nabla f(x_{k})$. However, even with these $n$ constraints, any SPD $B_{k}$ still has $n(n+1) /2$ degrees of freedom to be chosen. 

*SR1.* A cheap update is $B_{k+1}=B_{k}+\beta_{k}v_{k}v_{k}^{\top}$ for $v_{k}=y_{k}-B_{k}s_{k}$ and $\beta_{k}=1 /v_{k}^{\top}s_{k}$. With some regulatory conditions it converges superlinearly, but $\beta_{k}$ might become negative, causing $B_{k}$ to no longer remain SPD. By Sherman-Morrison, the inverses can be quickly updated, however. 

*BFGS.* If $B_{k}$ is SPD then the following update preserves this property: 

$$
B_{k+1} = B_{k} - \frac{B_{k}s_{k}s_{k}^{\top}B_{k}}{s_{k}^{\top}B_{k}s_{k}} + \frac{y_{k}y_{k}^{\top}}{y_{k}^{\top}s_{k}}
\equiv B_{k} - \frac{B_{k}s_{k}s_{k}^{\top}B_{k}}{s_{k}^{\top}B_{k}s_{k}} + \rho_{k} y_{k}y_{k}^{\top}
$$

And the inverses are updated as (assume $H_{0}=I$)

$$
\begin{aligned}
H_{k+1} &= (I - \rho_{k} s_{k}y_{k}^{\top}) H_{k} (I - \rho_{k} y_{k}s_{k}^{\top}) + \rho_{k} s_{k}s_{k}^{\top} \\
&= (I-\rho_{k} s_{k}y_{k}^{\top})\cdots(I-\rho_{1} s_{1}y_{1}^{\top})(I-\rho_{1} y_{1}s_{1})\cdots(I-\rho_{k} y_{k}s_{k}^{\top}) + \sum_{i=1}^{k} \rho_{i} s_{i}s_{i}^{\top} \\
\end{aligned}
$$

A further improvement known as *L-BFGS* only takes the last $m$ pairs, 

$$
H_{k+1} = \prod_{i=1}^{m} (I - \rho_{k-i+1} s_{k-i+1} y_{k-i+1}^{\top}) \cdot \prod_{i=1}^{m} (I - \rho_{i}y_{i}s_{i}^{\top}) + \sum_{i=1}^{k} \rho_{i} s_{i} s_{i}^{\top}
$$

It may be the most widely-used algorithm for unconstrained optimization. 

$$
\begin{align}
& \mathrm{Initialize}\; B_{0} \leftarrow I,\; x \leftarrow x_{0} \\
& \mathbf{for}\; k=0,1,2,\dots \; \mathbf{do} \\
& \qquad \mathrm{Solve}\; p_{k} \leftarrow H_{k} \nabla f(x_{k}) \text{ iteratively} \\
& \qquad \mathrm{Find}\; \alpha_{k} \; \text{using Armijo backtracking line search} \\
& \qquad \mathrm{Set}\; x_{k+1} \leftarrow x_{k} + \alpha_{k} p_{k} \\
& \qquad \mathrm{Compute}\; s_{k+1} \leftarrow x_{k+1} - x_{k},\; y_{k+1} \leftarrow \nabla f(x_{k+1}) - \nabla f(x_{k}) \\
& \qquad \mathrm{Update}\; B_{k+1} \;\text{by the BFGS rank-2 update} \\
& \mathbf{end\; for}
\end{align}
$$

The parameter $m$ is usually chosen to be small (i.e. $m \leq 20$) so per-iteration is $\mathcal{O}(n)$. If the derivatives are not provided then finite-difference approximations can estimate them.

# Eigenstuff Computation

## Eigenvalues

### Schur Decomposition

> [!question] How do you find eigenvalues of $A \in \mathbb{C}^{n \times n}$? 
> The key idea is to compute the Schur factorization via 
> 
> $$
> Q_{j}^{\dagger} \dots Q_{1}^{\dagger} A Q_{1}\dots Q_{j} \stackrel{j \to \infty}{\longrightarrow} T
> $$
> 
> where each $Q_{j}$ is unitary. Practical algorithms split this process into two steps: 
> 1. Reduction to Hessenberg. Apply unitary transformations to get $H$ which is diagonal plus one subdiagonal. Costs $\mathcal{O}(n ^{3})$, about 2x as much as Householder.
> 2. QR Algorithm. Reduce $H$ to upper triangular. If $A$ is symmetric, $H$ is tridiagonal, and the convergence is very quick.

> [!note] Existence of Hessenberg Decomposition
> Every $A \in \mathbb{R}^{n\times n}$ admits a Hessenberg $H$ and an orthogonal $Q$ such that $H=Q^{\top}AQ$. 

**Phase 1.** Find $Q_{1},\dots,Q_{n-2}$ orthogonal such that $Q_{j}$ zeros out the necessary elements of column $j$. In particular, 

$$
Q_{j} = \begin{bmatrix}
I_{j} & 0 \\
0 & F_{n-j}
\end{bmatrix}
\implies Q_{1}\dots Q_{n-2} A Q_{1}\dots Q_{n-2} = H
$$

where $F_{j}$ is the same as in Householder. Note $Q_{j}$ is symmetric. 

*Lemma.* If $A$ is symmetric then $H=Q^{\top}AQ$ is symmetric as well and therefore must be tridiagonal, such as 

$$
H = \begin{bmatrix}
\times & \times & 0 & \cdots & 0 \\
\times & \times & \times & \ddots & \vdots \\
0 & \times & \times & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & \times \\
0 & \cdots & 0 & \times & \times
\end{bmatrix}
$$

**Phase 2.** Find a QR factorization $H=\hat{Q}T$. Then $T=\hat{Q}^{\top}H$ has diagonal elements equal to the eigenvalues of $A$.

> [!question] What are the time complexities? 
> 1. Phase 1: Costs $\mathcal{O}(n ^{2})$ per multiplication, so about $\mathcal{O}(n ^{3})$ with 2x leading constant factors as QR by Householder. 
> 2. Phase 2: Costs $\mathcal{O}(n ^{2})$ to factorize a Hessenberg matrix, then $\mathcal{O}(n ^{2})$ for matrix multiplication thanks to Hessenberg property and since only the diagonal matters. 

### Shifted QR Algorithm

**Theorem (Convergence).** Under appropriate conditions $A^{(k)} \to T$ with the eigenvalues of $A$ on the diagonal. 

$$
\begin{align}
& A^{(0)} \leftarrow A \\
& \mathbf{for}\; k = 1, 2, 3, \dots \; \mathbf{do} \\
& \qquad \text{Compute QR factorization:}\; Q^{(k)}R^{(k)} = A^{(k-1)} \\
& \qquad \text{Form new matrix:}\; A^{(k)} \leftarrow R^{(k)}Q^{(k)} \\
& \mathbf{end\; for}
\end{align}
$$

The idea is once we show $V^{(k)}=A^{k}V^{(0)}$ has its first column converge to an eigenvector (true if we assume $\lambda_{1}$ is dominant), then because $Q$ is orthogonal, none of the others columns can converge to the same eigenvector, and we proceed inductively. However for numerical stability we need to normalize the columns (e.g. by computing another QR), i.e. 

$$
\begin{align}
& \text{Require: Matrix } A \in \mathbb{R}^{n \times n}, \text{ initial } Q^{(1)} \in \mathbb{R}^{n \times m} \text{ ON columns} \\
& \mathbf{for}\; k = 1, 2, \dots \; \mathbf{do} \\
& \qquad Z \leftarrow AQ^{(k)}  \\
& \qquad \text{Compute QR: } Q^{(k+1)}R^{(k)} = Z \\
& \mathbf{end\; for}
\end{align}
$$

Incorporate three more tricks to make it practical: 
1. Reduce to Hessenberg structure, which is preserved by iterations, 
2. Factor $A^{(k-1)}-\mu ^{(k)} I$ instead of $A^{(k-1)}$ where $\mu ^{(k)}$ is some well-chosen pivot e.g. $A^{(k-1)}_{nn}$, 
3. Deflate any off-diagonal entries $|a_{ij}| \leq \varepsilon_{\text{mach}}$ to $0$. 

*Shifted QR Algorithm.*

$$
\begin{align}
& \text{Reduce } A \text{ to Hessenberg/tridiagonal form: } \tilde{Q}^T A \tilde{Q} = H \\
& \text{Set } A^{(0)} \leftarrow H \\
& \mathbf{for}\; k = 1, 2, \dots \text{ until convergence } \mathbf{do} \\
& \qquad \text{Choose shift } \mu^{(k)} \text{ (e.g., } \mu^{(k)} = A_{nn}^{(k-1)}) \\
& \qquad \text{Compute QR: } Q^{(k)}R^{(k)} = A^{(k-1)} - \mu^{(k)}I \\
& \qquad \text{Form } A^{(k)} = R^{(k)}Q^{(k)} + \mu^{(k)}I \\
& \qquad \mathbf{if}\; \text{any off-diagonal element } |A_{j,j+1}^{(k)}| < \varepsilon_{\text{mach}} \; \mathbf{then} \\
& \qquad \qquad \text{Set } A_{j,j+1}^{(k)} = A_{j+1,j}^{(k)} = 0 \text{ (deflation)} \\
& \qquad \qquad \text{Apply QR algorithm recursively to the decoupled blocks} \\
& \qquad \mathbf{end\; if} \\
& \mathbf{end\; for}
\end{align}
$$

> [!question] What is the complexity of practical QR algorithm? 
> 1. Hessenberg reduction: $\mathcal{O}(n ^{3})$. 
> 2. Per-iteration: $\mathcal{O}(n ^{2})$, or $\mathcal{O}(n)$ if symmetric (tridiagonal). 
> 3. Total: $\mathcal{O}(n ^{3})$ in general, dominated by Phase 2. $\mathcal{O}(n ^{3})$ for symmetric, dominated by Phase 1, since 3-4 iterations per eigenvalue. 

## Eigenvectors

### Power Iteration

> [!question] How can we find a single (dominant) eigenvector? 
>  *Power iteration* finds a dominant (strictly largest eigenvalue magnitude) eigenvector if it exists. The iterates are $v_{k}=A^{k}v_{0}$ with normalization at each step.

**Theorem.** If $|\lambda_{1}|>|\lambda_{2}|\geq\dots$ are eigenvalues, so that $\lambda_{1}$ is a dominant eigenvalue with eigenvector $q_{1}$, and $b_{1}=q_{1}^{\top}v_{0} \neq 0$ (the guess needs to be nonzero in the direction of the $q_{1}$), then power iteration converges linearly to $q_{1}$ with rate $|\lambda_{2} /\lambda_{1}|$. 

*Proof.* Expand $v_{k}$ in the eigenbasis, then factor out $\lambda_{1}^{k}$. $\blacksquare$

### Inverse Iteration

> [!question] How can we find the entire spectrum? 
> *Inverse iteration* finds nearby eigenvalues from a guess by using the characteristic matrix and power iteration. Let $B=(A - \mu I)^{-1}$ and by substituting via Schur's $A=QTQ^{\top}$ we see that $B$ has the same eigenvalues as $(T - \mu I)^{-1}$, which are $(\lambda_{i}-\mu)^{-1}$. In other words if $\mu$ is close to $\lambda_{i}$ then it is a dominant eigenvalue of $B$!

**Theorem.** If $q_{I}$ and $q_{J}$ are the closest and second-closest eigenvalues to $\mu$, respectively, then inverse iterations (power iterations on $B$, i.e. $v_{k}=B^{k}v_{0}$) converge to $q_{I}$ linearly with rate $\frac{|\lambda_{I}-\mu|}{|\lambda_{J}-\mu|}$. 

### Rayleigh Iteration

> [!summary] Motivation for the Rayleigh Quotient
> You are given $x$ which is close to an eigenvector. How do you find an (approximate) eigenvector $\alpha$? Formulate as optimization: 
> $$
> \operatorname{minimize} \|Ax - \alpha x\|^{2}
> \implies x^{\top}(\alpha x-Ax) = 0 
> \implies \alpha = \frac{x^{\top}Ax}{x^{\top}x}
> $$
> We denote $r(x)=\alpha$ to be the *Rayleigh quotient*, and if $x=q_{i}$ is an eigenvector then $r(x)=\lambda_{i}$, its corresponding eigenvalue. 

**Theorem.** If $\|x-q_{i}\|\leq \varepsilon$ then $|r(x)-\lambda_{i}|\leq \mathcal{O}(\varepsilon ^{2})$. 

*Proof.* It can be shown that $q_{i}$ are stationary points of $r(x)$, so by Taylor expansion, 

$$
r(x) = r(q_{i}) + \nabla r(q_{i})^{\top}(x - q_{i}) + \mathcal{O}(\|x-q_{i}\|^{2})
= \lambda_{i} + 0 + \mathcal{O}(\varepsilon ^{2})
$$

so that $|r(x) - \lambda_{i}| \leq \mathcal{O}(\varepsilon ^{2})$. $\blacksquare$

> [!note] Summary of Current Progress
> 1. Inverse Iteration: given $\mu \approx \lambda_{i}$, we can find $q_{i} \approx B^{k}v_{0}$ for large $k$. 
> 2. Rayleigh Quotient: given $x\approx q_{i}$, we can find $r(x)\approx \lambda_{i}$. 

Combining yields the *Rayleigh quotient iterations*:

$$
\begin{align}
& \mu_0 \leftarrow v_0^T A v_0 \\
& \mathbf{for}\; k = 1, 2, \dots \; \text{until convergence} \; \mathbf{do} \\
& \qquad \mathrm{Solve}\; (A - \mu_{k-1} I) w = v_{k-1} \\
& \qquad v_k \leftarrow w / \|w\| \\
& \qquad \mu_k \leftarrow v_k^T A v_k \\
& \mathbf{end\; for} \\
& \mathbf{return}\; \mu_k \; \text{(eigenvalue)}, v_k \; \text{(eigenvector)}
\end{align}
$$

**Theorem.** If $v_{0}$ is sufficiently close to an eigenvector $q_{j}$ with a *simple* eigenvalue $\lambda_{j}$, then Rayleigh iterations exhibit cubic converge (locally) for both the eigenvalue and eigenvector. 

> [!note] Comparison of Complexities
> 1. Power iteration: $\mathcal{O}(n ^{2})$ per iteration due to MatVec. 
> 2. Inverse iteration: $\mathcal{O}(n ^{3} + kn ^{2})$ for $k$ iterations due Schur and $k$ MatVec operations. 
> 3. Rayleigh iteration: $\mathcal{O}(n ^{3})$ per iteration due to solving the linear system. However if we first reduce $A$ to Hessenberg form, it only takes $\mathcal{O}(n^{2})$ to solve. If further $A$ is symmetric (so $T$ is tridiagonal) then it takes $\mathcal{O}(n)$ to solve. 

# Sparsity and Structure

## Fast MatVec (COO)

*COO (Coordinate) Format.* Given a matrix $A$ which is sparse, it stores tuples $(i,j,A_{ij})$ rather than a full matrix. It requires $\mathcal{O}(\operatorname{nnz}(A))$ space, where $\operatorname{nnz}$ is the number of nonzero elements. 

*Sparse MatVec.* Given $A \in \mathbb{R}^{m\times n}$, calculate $b=Ax$ by $b_{i}=\sum_{(i,j,v)}vx_{j}$. Takes $\mathcal{O}(\operatorname{nnz}(A))$ time. 

*Structured MatVec.* Given $A$ Toeplitz, circulant, etc., requires $\mathcal{O}(n\log n)$ by FFT. 

## Arnoldi Iteration

> [!question] Can we compute Hessenberg using MatVec? 
> Suppose $A=QHQ^{\top}$ computed as before. Then $AQ=QH$ so writing $Q$ as a matrix of orthogonal column vectors, we can show 
> $$
> q_{j+1} = \frac{1}{h_{j+1,j}} \left[ Aq_{j} - \sum_{i=1}^{j} h_{ij}q_{j} \right] 
> $$
> Here, $h_{ij}=q_{i}^{\top}Aq_{j}$, and $h_{j+1,j}$ is chosen so that $q_{j+1}$ is normalized. The interpretation is that we must take $v = Aq_{j}$, then orthonormalize it against $q_{1},\dots,q_{j}$ (similar to Gram-Schmidt). This is the *Arnoldi iteration*, which returns $q_{1},\dots,q_{m+1}$ and $H_{m}$ after $m$ iterations. 

> [!note] Early Stopping in Hessenberg
> Importantly, if $m\ll n$ then stopping early yields a *partial Hessenberg* matrix — important for large matrices. Arnoldi is rarely used in full, because it is not numerically stable. 

*Arnoldi Relation.* After $m<n$ iterations, $AQ_{m}=Q_{m+1}\tilde{H}_m$ where $Q_{m}\in \mathbb{R}^{n\times m}$ is the first $m$ orthonormal vectors (similarly for $Q^{m+1}\in \mathbb{R}^{n\times(m+1)}$), and $\tilde{H}_m\in \mathbb{R}^{(m+1)\times m}$ is upper Hessenberg with an extra row. 

*Lemma.* $H_{m}=Q_{m}^{\top}AQ_{m}$, since 

$$
Q_{m}^{\top}AQ_{m} = Q_{m}^{\top}Q_{m+1}\tilde{H}_{m+1} = \begin{bmatrix}
I_{m} & 0
\end{bmatrix}
\tilde{H}_{m} = H_{m}
$$

## Lanczos Iteration

*Lanczos.* The special case of Arnoldi iterations when $A$ is symmetric. Then $H_{m}=Q_{m}^{\top}AQ_{m}$ is also symmetric, so 

$$
H_{m} = T_m = \begin{bmatrix}
\alpha_1 & \beta_1 & 0 & \cdots & 0 \\
\beta_1 & \alpha_2 & \beta_2 & \cdots & \vdots \\
0 & \beta_2 & \alpha_3 & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & \beta_{m-1} \\
0 & \cdots & 0 & \beta_{m-1} & \alpha_m
\end{bmatrix}
$$

The iterates simplify significantly: only need previous two vectors, $q_{m}$ and $q_{m-1}$, so $\mathcal{O}(n)$ space requires, and each iteration requires only one MatVec plus $\mathcal{O}(n)$ additional work. So total cost is $\mathcal{O}(\operatorname{matvec}(A)+n)$ per iteration. 

$$
\begin{align}
& \beta_0 \leftarrow 0, q_0 \leftarrow 0, q_1 \leftarrow b/\|b\| \\
& \mathbf{for}\; m = 1, 2, 3, \dots \; \mathbf{do} \\
& \qquad v \leftarrow A q_m \\
& \qquad \alpha_m \leftarrow q_m^T v \\
& \qquad v \leftarrow v - \beta_{m-1} q_{m-1} - \alpha_m q_m \\
& \qquad \beta_m \leftarrow \|v\| \\
& \qquad q_{m+1} \leftarrow v / \beta_m \\
& \mathbf{end\; for} \\
& \mathbf{return}\; q_1, \dots, q_{m+1}, T_m
\end{align}
$$

## Krylov Methods

*Ritz Values.* Eigenvalues of $H_{m}$ are called Ritz values of $A$. They provide good approximations to extreme eigenvalues, in particular $\lambda_{\max}$ and $\lambda_{\min}$. 

*Lemma.* If $(\lambda,v)$ is an eigenpair of $A$ with $v \in \operatorname{span}(q_{1},\dots,q_{m})=\operatorname{range}(Q_{m})$ then $\lambda$ is an eigenvalue of $H_{m}$. 

*Krylov Subspace.* Given $A\in \mathbb{R}^{n\times n}$ and $b \in \mathbb{R}^{n}$, the Krylov subspace is 

$$
K_{m}(A,b) = \operatorname{span}(b,Ab, \dots, A^{m-1}b)
= \operatorname{span}(\underbrace{ q_{1},\dots,q_{m} }_{ \text{ON basis} })
$$

since $q_{j+1}$ is built from $Aq_{j}$ (minus orthogonalization). Here $q_{1}=b /\|b\|$ just as in Gram-Schmidt orthonormalization. 

> [!note] Krylov Methods for Eigenvalue Computation.
> Combining Arnoldi and Lanczos,  
> $$
> \begin{align}
> & \mathbf{Require:}\; \text{Matrix } A \in \mathbb{R}^{n \times n}, \text{ number of iterations } m \ll n \\
> & \mathbf{Step\; 1:}\; \text{Run Arnoldi (or Lanczos if } A \text{ is symmetric) for } m \text{ iterations} \\
> & \qquad \rightarrow \text{ Obtain } Q_m \text{ and } H_m \text{ (or } T_m \text{ if symmetric)} \\
> & \mathbf{Step\; 2:}\; \text{Compute eigenvalues of } H_m \text{ (or } T_m \text{) using the shifted QR algorithm} \\
> & \qquad \rightarrow \text{ These are the Ritz values (approximations to extreme eigenvalues of } A\text{)} \\
> & \mathbf{return}\; \text{Approximate extreme eigenvalues}
> \end{align}
> $$
> The complexity is $\mathcal{O}(m \cdot \operatorname{matvec}(A)+nm ^{2})$ for step 1 and $\mathcal{O}(m ^{3})$ for general matrices, while $\mathcal{O}(m \cdot \operatorname{matvec}(A) + nm)$ for step 1 and $\mathcal{O}(m ^{2})$ for step 2 if $A$ is symmetric. 

> [!note] Krylov Methods for Linear Systems
> Solve $Ax=b$ as follows: 
> 1. Build an ON basis $Q_{m}$ for the Krylov subspace $K_{m}(A,b)$. 
> 2. Choose some optimal solution from the subspace, $x_{m}\in K_{m}(A,b)$. Methods include GMRES, conjugate gradient, etc. 

### GMRES

*Generalized Minimal Residual Method (GMRES)*. The key insight is $x \in K_{m}(A,b)$ can be reparametrized as $x=Q_{m}y$ for $y \in R^{m}$, 

$$
\min_{x \in K_{m}(A,b)} \|Ax-b\|^{2}
= \min_{y \in R^{m}} \|AQ_{m}y - b\|^{2}
= \min_{y \in \mathbb{R}^{m}} \|\tilde{H}_{m}y - \|b\|e_{1}\|^{2}
\quad
$$

which comes from the Arnoldi relation $AQ_{m}=Q_{m+1}\tilde{H}_{m}$. The algorithm is: 

$$
\begin{align}
& \mathbf{Require}\; \text{Matrix } A \in \mathbb{R}^{n \times n}, \text{ right-hand side } b \in \mathbb{R}^n \\
& q_1 \leftarrow b/\|b\| \\
& \mathbf{for}\; m = 1, 2, 3, \dots \; \text{until convergence} \; \mathbf{do} \\
& \qquad \text{Run one step of Arnoldi to get } q_{m+1} \text{ and row } m \text{ of } \tilde{H}_m \\
& \qquad \text{Solve } y_m = \arg \min_{y \in \mathbb{R}^m} \|\tilde{H}_m y - \|b\|e_1\|_2 \\
& \qquad x_m \leftarrow Q_m y_m \\
& \qquad \mathbf{if}\; \|Ax_m - b\| < \text{tolerance} \; \mathbf{then} \\
& \qquad \qquad \mathbf{break} \\
& \qquad \mathbf{end\; if} \\
& \mathbf{end\; for} \\
& \mathbf{return}\; x_m
\end{align}
$$

> [!question] What is the complexity of GMRES? 
> The total is $\mathcal{O}(m ^{2}n)$ time and $\mathcal{O}(mn)$ storage. Reason: each Arnoldi step requires $\mathcal{O}(mn+\operatorname{matvec}(A))$. Solving least-squares requires $\mathcal{O}(m)$ using Givens rotations, if we exploit the fact that we only need 1 additional rotation per increment of $m$. Forming the solution $x_{m}=Q_{m}y_{m}$ then takes $\mathcal{O}(mn)$ per iteration. 

The space and time complexity are prohibitive since they grow in the number of iterates $m$, so practically use *restarted GMRES* (restart the iterations periodically to control costs). 

### MINRES 

*MINRES (Minimum Residual).* Apply GMRES to symmetric matrices. 

> [!question] What is the time complexity of MINRES? 
> The total is $\mathcal{O}(mn)$. Arnoldi reduces to Lanczos now, so one iterate takes $\mathcal{O}(n)$. The solve now requires $\mathcal{O}(1)$ since we only have one additional (2x2) Givens rotation. The $x_{m}$ update now is a three-term update so requires $\mathcal{O}(n)$ storage. 

### CG Method

> [!abstract] Purpose of Conjugate Gradient
> Our goal is to solve the problem $Ax=b$ for SPD $A \in \mathbb{R}^{n\times n}$. 

*Conjugate Gradient.* For SPD matrices $A \in \mathbb{R}^{n\times n}$, Krylov subspace methods have us minimize the error in the $A$-norm, 

$$
x_{m} = \arg\min_{x \in K_{m}(A,b)} \| x - x^{*}\|_{A}
= \arg\min_{x \in K_{m}(A,b)} \phi(x)
$$

where $\phi(x)=\frac{1}{2}x^{\top}Ax-b^{\top}x$ is "energy-minimization", since 

$$
\begin{align}
\|x - x^{*}\|_{A}^{2}
&= (x - x^{*})^{\top}A(x - x^{*})
= x^{\top}Ax - 2x^{\top}Ax^{*} + C \\
&= x^{\top}Ax - 2x^{\top}b + C
= 2\phi(x) + C
\end{align}
$$

where $C$ is constant with respect to $x$. There is a unique minimizer since $\nabla ^{2}\phi \succ 0$. The point is that we use the idea of a Krylov basis to solve this optimization problem more efficiently. 

> [!note] Conjugate Basis
> Two vectors are conjugate with respect to SPD $A$ if $u^{\top}Av=0$. If we have $\{ p_{i} \}_{i=1}^{n}$ which are mutually $A$-conjugate vectors, then this collection spans $\mathbb{R}^{n}$. While gradient descent uses $-\nabla \phi(x)$ as a search direction, conjugate gradient descent $A$-conjugate search directions. This provides much faster theoretical convergence than gradient descent!

> [!note] Motivation for Iterates
> Consider the residual at step $k$, $r_{k}=b - Ax_{k}=-\nabla \phi(x_{k})$. We want to choose a new search direction $p_{k}$ which is conjugate to all previous directions, so 
> $$
> p_{k} = r_{k} - \sum_{i=1}^{k-1} \frac{r_{k}^{\top}Ap_{i}}{p_{i}^{\top}Ap_{i}} p_{i}
> $$
> By solving $\phi'(x_{k}+\alpha_{k}p_{k})=0$ for $\alpha_{k}$, we find the optimal step-size to be 
> $$
> \alpha_{k} = \frac{p_{k}^{\top}(b - Ax_{k})}{p_{k}^{\top}Ap_{k}}
> = \frac{p_{k}^{\top}r_{k}}{p_{k}^{\top}Ap_{k}}
> $$
> so you should take a step size proportional to the remaining residual. Finally we just update $x_{k+1}=x_{k}+\alpha_{k}p_{k}$ as usual. 

$$
\begin{align}
& \mathbf{Require}\; \text{SPD matrix } A \in \mathbb{R}^{n \times n}, \text{ right-hand side } b \in \mathbb{R}^n \\
& x_0 \leftarrow 0, r_0 \leftarrow b, p_0 \leftarrow r_0 \\
& \mathbf{for}\; k = 1, 2, 3, \dots \; \text{until convergence} \; \mathbf{do} \\
& \qquad \alpha_k \leftarrow \frac{r_{k-1}^T r_{k-1}}{p_{k-1}^T A p_{k-1}} \\
& \qquad x_k \leftarrow x_{k-1} + \alpha_k p_{k-1} \\
& \qquad r_k \leftarrow r_{k-1} - \alpha_k A p_{k-1} \\
& \qquad \mathbf{if}\; \|r_k\| < \text{tolerance} \; \mathbf{then} \\
& \qquad \qquad \mathbf{break} \\
& \qquad \mathbf{end\; if} \\
& \qquad \beta_k \leftarrow \frac{r_k^T r_k}{r_{k-1}^T r_{k-1}} \\
& \qquad p_k \leftarrow r_k + \beta_k p_{k-1} \\
& \mathbf{end\; for} \\
& \mathbf{return}\; x_k
\end{align}
$$

> [!question] Complexity of Conjugate Gradient? 
> Per iteration, $\mathcal{O}(n)+\text{matvec}(A)$ time and $\mathcal{O}(n)$ space. 

