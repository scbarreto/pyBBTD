Stokes Tutorials
===================================

Stokes Tutorial I with Cov-LL1
----------------------------------


This tutorial provides an introduction to the Stokes Block Tensor Decomposition (Stokes-BTD) using the pyBBTD library.

Load required libraries.
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

    import numpy as np
    import pybbtd.stokes as stokes
    import pybbtd.btd as btd
    import pybbtd as 
    
Define Stokes-BTD parameters and spatial map dimensions 

.. code:: python3

    R = 3  # Number of components
    L = 5  # Rank of Spatial maps
    btd._validate_R_L(R, L)

    # Define spatial maps of size 25x25.
    X = stokes.Stokes([25, 25], R, L)

Generate Stokes BTD-LL1 data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code:: python3

    # Create ground truth tensor
    [A0, B0, C0], T0 = X.generate_stokes_tensor()

Then we get the constraint matrix for the equivalent CPD representation of the BTD-Stokes model, and add some noise to the generated tensor.

.. code:: python3

    # theta is the constraint matrix to represent a BTD-LL1 model as a CPD model
    theta = X.get_constraint_matrix()
    Tnoisy = btd.factors_to_tensor(A0, B0, C0, theta, block_mode="LL1") + 1e-6 * np.random.randn(*X.dims)

    # or simply use
    # Tnoisy = T0 +  1e-6 * np.random.randn(*X.dims)

    # We check iff all pixels satisfy the Stokes constraints
    stokes.validate_stokes_tensor(Tnoisy)

Fit a Stokes-BTD model 
^^^^^^^^^^^^^^^^^^^^^^^

We now fit a Stokes-BTD model to the observed tensor.

.. code:: python3

    X.fit(data=Tnoisy,algorithm="ADMM",init="random",max_iter=5000,rho=1,max_admm=1,rel_tol=10**-8,abs_tol=10**-14,admm_tol=10**-10)

    draw_metrics.plot_error(X.fit_error)

.. image:: stokes_files/fit_error_rand.png

Stokes Tutorial II with Cov-LL1
----------------------------------

In this tutorial, we demonstrate how to use the **pyBBTD** library to perform a
**Stokes Block Tensor Decomposition (Stokes-BTD)** with *canonical
polarization states*.

We will load a :class:`numpy.ndarray` representing the acronym **"BBTD"**,
where each letter corresponds to a region with a distinct, canonical
polarization state:

* **Background** — right-handed circularly polarized
* **First B** — fully horizontally polarized
* **Second B** — fully linearly polarized at **+45°**
* **T** — fully right-handed circularly polarized
* **D** — fully linearly polarized at **–45°**

After defining this synthetic dataset, we will fit a **Stokes-BTD model**
to recover the polarization components, and then visualize the reconstructed
Stokes parameters and corresponding polarization ellipses.

Load required libraries
^^^^^^^^^^^^^^^^^^^^^^^
.. code:: python3

    import numpy as np
    import pybbtd.stokes as stokes
    import pybbtd.btd as btd
    from pybbtd.visualization import draw_metrics, draw_stokes

Load the data tensor and define parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

    tensor = np.load("data/BBTD_Stokes.npy")

    # Create Stokes model
    R = 5
    L = 25
    btd._validate_R_L(R, L)
    X = stokes.Stokes([tensor.shape[0], tensor.shape[1]], R, L)

Fit the Stokes-BTD model to the data tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python3

    X.fit(data=tensor,algorithm="ADMM",init="kmeans",max_iter=5000,rho=1,max_admm=2,rel_tol=10**-5, abs_tol=10**-7, admm_tol=10**-5)

Visualize the results
^^^^^^^^^^^^^^^^^^^^^^

We use :func:`~pybbtd.visualization.draw_stokes.plot_stokes_terms` to display the
recovered spatial maps :math:`A_r B_r^{\top}` and the corresponding polarization
ellipses for each component.

.. code:: python3

    Afit, Bfit, Cfit = X.factors[0], X.factors[1], X.factors[2]
    draw_stokes.plot_stokes_terms(Afit, Bfit, Cfit, R, L)

.. image:: stokes_files/bbtd_board_results.png