Learning fair representations is a pre-processing technique that finds a
latent representation which encodes the data well but obfuscates information
about protected attributes

Paper: http://proceedings.mlr.press/v28/zemel13.html

There are 2 implementations on which this code is based on.
<ol>1.https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py
  2.https://github.com/zjelveh/learning-fair-representations
</ol>
I tried to write implementation that would be easy to understand cause implemetations above where quite bad to be honest here is a first line of code from original implemenation   
* Warning: Code below is pretty bad, proceed w/ caution. Hope to refactor in the next 5 years.*

My implementation most importantly **contains docstring** and comments, and is **vectoried**(others had awful for loops)

So enjoy

"""
Date finished - 20.08.2020
"""
