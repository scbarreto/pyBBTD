BTD tutorial
===================================

This tutorial provides an introduction to Block Tensor Decomposition (BTD) into rank-(L, L, 1) terms using the pyBBTD library.

Load required libraries
------------------------

.. code:: python3

    import pybbtd.btd as btd
    from pybbtd.solvers import btd_als
    import numpy as np
    
Generate noisy BTD data
-----------------------------------------------------

We first define the dimensions of the tensor and the BTD parameters.

.. code:: python3

    # Define tensor size and BTD parameters
    N1, N2, N3 = 80, 100, 4  # dimensions of the tensor
    R = 3  # number of LL1 terms
    L = 5  # rank of each term

    # Generate BTD model
    X = btd.BTD([N1, N2, N3], R, L, block_mode="LL1")

Then we create the true BTD components and generate the observed tensor with added noise.

.. code:: python3

    # Create ground truth tensor
    A0, B0, C0 = btd_als.init_BTD_factors(X, strat="random")
    # This is the constraint matrix to represent BTD as a CPD model
    theta = X.get_constraint_matrix()

    T_observed = btd.factors_to_tensor(A0, B0, C0, theta, block_mode="LL1") + 1 * 1e-6 * np.random.randn(*X.dims)


Fit a BTD model
-------------------
We now fit a BTD model to the observed tensor with random initialization

.. code:: python3

    # Fit the model using random initialization
    X.fit(T_observed, max_iter=3000, init="random", rel_tol=1e-9, abs_tol=1e-15)

    # Save the fit error for comparison with SVD init
    rand_init_fit_error = X.fit_error

    # Retrieve the estimated factors if necessary
    A_est, B_est, C_est = X.factors

    # Check convergence curve
    draw_metrics.plot_error(X.fit_error)
.. image:: btd_files/fit_error_rand.png

Compare with SVD initialization
---------------------------------

We can also fit the model using SVD initialization and compare the results.

.. code:: python3

    X.fit(T_observed, max_iter=3000, init="svd", rel_tol=1e-9, abs_tol=1e-15)
    svd_init_fit_error = X.fit_error
    draw_metrics.plot_error([svd_init_fit_error, rand_init_fit_error], labels=[r"SVD init", r"Random init"])
    
.. image:: btd_files/fit_error_svd.png