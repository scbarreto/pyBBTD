# Glossary

```{glossary}
Tensor
  A block of numbers where each value is found by giving one position per direction. A
  table of numbers (a matrix) has two directions, rows and columns; an image with several
  numbers per pixel has three.

Order
  The number of directions of a tensor: an image with several numbers per pixel is a
  third-order tensor, an image with a matrix per pixel is a fourth-order tensor.

Mode
  One direction of a tensor (rows, columns, channels, ...). The *n*-th mode is the *n*-th
  index.

Term
  One of the R parts that a decomposition adds up to explain the data, also called a
  component. In pyBBTD each term is a spatial map times a signature.

Spatial map
  An image saying where a term is present and how strongly: the matrix
  $\mathbf{A}_r \mathbf{B}_r^{\top}$.

Signature
  What a term puts in each pixel where it is present: a vector (BTD, Cov-LL1, Stokes) or a
  matrix (BBTD).

Rank
  For a matrix, the smallest number of (column × row) products that add up to it. A map of
  rank 1 is a single column profile times a row profile; higher ranks allow more complex
  shapes.

Factor matrices
  The matrices A, B, C (and D) that store all the terms. Term r uses one block of columns
  in each of them.

Outer product
  The product $\circ$ that combines a map and a signature into a term: entry (i, j, k) of
  $\mathbf{S} \circ \mathbf{c}$ is $S_{ij}\, c_k$.

BTD
  Block term decomposition: a sum of terms that are each a low-rank block (here, a rank-L
  map) combined with a vector.

LL1
  Short for "rank-(L, L, 1) terms": the map has rank L in the first two modes and the
  signature is a single vector in the third.

BBTD
  Block-block term decomposition: each term combines two low-rank blocks, a spatial map of
  rank L1 and a matrix of rank L2.

CP decomposition
  Canonical polyadic decomposition: a sum of rank-one terms. Every BTD and BBTD can be
  rewritten as a CP decomposition with repeated columns, using the constraint matrices.

Constraint matrix
  A matrix of zeros and ones ($\Theta$, $\Phi$, $\Psi$) that repeats the columns of a factor
  so a BTD or BBTD can be computed as a CP decomposition.

Unfolding
  A tensor rearranged as a matrix by putting one mode along the rows and all the others
  along the columns.

Khatri-Rao product
  The column-wise Kronecker product of two matrices. It appears in the unfoldings of a
  decomposed tensor.

Uniqueness
  A decomposition is (essentially) unique when no other set of terms explains the data,
  apart from reordering the terms or rescaling within each term.

ALS
  Alternating least squares: update one factor matrix at a time by solving a least-squares
  problem with the others fixed.

ADMM
  Alternating direction method of multipliers: an algorithm for problems with
  constraints, used to keep the factors valid (non-negative, PSD, ...).

AO-ADMM
  Alternating optimization with ADMM: update one factor at a time, each update solved
  with a few ADMM iterations.

Fit error
  How far the fitted tensor is from the data, recorded at each iteration in
  `model.fit_error`.

Stokes vector
  Four numbers $[S_0, S_1, S_2, S_3]$ describing the intensity and polarization of light.

Coherency matrix
  A 2 × 2 Hermitian matrix carrying the same information as a Stokes vector. A Stokes
  vector is valid exactly when its coherency matrix is positive semidefinite.

Covariance matrix
  A matrix describing how several signals vary together. It is Hermitian and positive
  semidefinite.

Hermitian
  A complex matrix equal to its conjugate transpose, $\mathbf{M} = \mathbf{M}^H$.

Positive semidefinite
  (PSD) A Hermitian matrix whose eigenvalues are all greater than or equal to zero.

K-means
  A clustering method that groups vectors around R centers. pyBBTD uses it to group
  pixels with similar signatures to build a starting point.

NMF
  Non-negative matrix factorization: writing a non-negative matrix as a product of two
  non-negative matrices. Used to turn each K-means group into non-negative spatial
  factors.

SVD
  Singular value decomposition. Its leading singular vectors give the `"svd"` starting
  point.
```
