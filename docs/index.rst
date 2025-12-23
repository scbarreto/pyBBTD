.. pyBBTD documentation master file, created by
   sphinx-quickstart on Tue Sep  2 12:16:27 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyBBTD docs
====================
**pyBBTD** implements two tensor decomposition models: `Cov-LL1 <https://ieeexplore.ieee.org/abstract/document/10446019>`_ and `BBTD <https://hal.science/hal-05059862/>`_.


**Cov-LL1** is a third-order tensor decomposition model specifically tailored for *covariance-valued imaging tensors*, i.e., data in which each pixel is associated with a covariance matrix. It can be seen as a constrained `LL1 model <https://epubs.siam.org/doi/abs/10.1137/070690729>`_.


In mathematical form, Cov-LL1 decomposes a third-order tensor :math:`\mathcal{X}` as:

.. math::

   \mathcal{X} \approx \sum_{r=1}^{R} \mathbf{A}_r \mathbf{B}_r^{\top} \circ \mathbf{c}_r

where:

- :math:`R` is the number of components.

- :math:`\mathbf{A}_r \in \mathbb{R}^{I \times L_1}` and :math:`\mathbf{B}_r \in \mathbb{R}^{J \times L_1}` represent spatial (non-negative) factor matrices of rank :math:`L_1`, and  

- :math:`\mathbf{c}_r \in \mathbb{R}^{K^2}` is a vectorized :math:`K \times K` covariance matrix associated with the :math:`r`-th component.

One can choose the :math:`R` (number of components) and :math:`L_1` (spatial rank) up to certain identifiability conditions (checked by pyBBTD). Here is an illustration of the Cov-LL1 model:

.. image:: index_files/model_covll1.png


**BBTD** is a fourth-order tensor decomposition model that represents data as a sum of two blocks: a spatial component with rank :math:`L_1` and a covariance component with rank :math:`L_2`. The user can choose :math:`R` (number of components), :math:`L_1` (spatial rank), and :math:`L_2` (covariance rank), up to certain identifiability conditions (checked by pyBBTD).


In mathematical form, **BBTD** decomposes a fourth-order tensor :math:`\mathcal{T}` as:

.. math::

   \mathcal{T} \approx \sum_{r=1}^{R} \mathbf{A}_r \mathbf{B}_r^{\top} \circ \mathbf{C}_r\mathbf{C}_r^{H}

where:

- :math:`R` is the number of components.

- :math:`\mathbf{A}_r \in \mathbb{R}^{I \times L_1}` and :math:`\mathbf{B}_r \in \mathbb{R}^{J \times L_1}` represent spatial factor matrices of rank :math:`L_1`, and  

- :math:`\mathbf{C}_r\mathbf{C}_r^{H} \in \mathbb{C}^{K \times K}` is the covariance matrix of the :math:`r`-th component, with each :math:`\mathbf{C}_r \in \mathbb{C}^{K \times L_2}`



The figure below illustrates the structure of the **BBTD** model.


.. image:: index_files/model_bbtd.png

For more details about Cov-LL1 and BBTD models, please refer to the chapters 3 and 4, respectively, of the following `thesis <https://theses.fr/s348607>`_.

################
Guide & Reference
################

.. toctree::
   :maxdepth: 1

   tutorials/index
   reference_api
