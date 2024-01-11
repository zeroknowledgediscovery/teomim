===============
zedstat
===============

.. image:: https://paraknowledge.ai/logo/teomimlogo.png
   :height: 150px
   :align: center 

.. image:: https://zenodo.org/badge/529991779.svg
   :target: https://zenodo.org/badge/latestdoi/529991779

.. class:: no-web no-pdf

:Author: Paraknowledge Corp <research.paraknowledge.ai>
:Description: Digital twin for generating medical histories 
:Documentation: https://zeroknowledgediscovery.github.io/teomim/
:Example: https://github.com/zeroknowledgediscovery/teomim/blob/main/examples/example.ipynb
		
**Usage:**

.. code-block::

   from teomim import teomim
   P=teomim(outfile='out100.csv',num_patients=500)
   P.set_modelpath('adrd')
   P.generate()
   P.evaluate()
   P.quality()

