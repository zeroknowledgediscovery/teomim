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
:Example: https://github.com/zeroknowledgediscovery/zedstat/blob/master/examples/example.ipynb
		
**Usage:**

.. code-block::

   from zedstat import zedstat
   zt=zedstat.processRoc(df=pd.read_csv('roc.csv'),
           order=3, 
           total_samples=100000,
           positive_samples=100,
           alpha=0.01,
           prevalence=.002)

   zt.smooth(STEP=0.001)
   zt.allmeasures(interpolate=True)
   zt.usample(precision=3)
   zt.getBounds()

   print(zt.auc())

   # find the high precision and high sensitivity operating points
   zt.operating_zone(LRminus=.65)
   rf0,txt0=zt.interpret(fpr=zt._operating_zone.fpr.values[0],number_of_positives=10)
   rf1,txt1=zt.interpret(fpr=zt._operating_zone.fpr.values[1],number_of_positives=10)
   display(zt._operating_zone)
   print('high precision operation:\n','\n '.join(txt0))
   print('high recall operation:\n','\n '.join(txt1))
