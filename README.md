Learning fair representations is a pre-processing technique that finds a
latent representation which encodes the data well but obfuscates information
about protected attributes

Paper: http://proceedings.mlr.press/v28/zemel13.html

This implementation most importantly **contains docstring** and comments, and is **vectoried**(original had awful for loops)

There are 2 implementations on which this code is based on.
<ol>
<li>https://github.com/Trusted-AI/AIF360/blob/master/aif360/algorithms/preprocessing/lfr.py</li>
<li>https://github.com/zjelveh/learning-fair-representations</li>
</ol>


Date finished - 20.08.2020

# Update (10.03.2025)
I was too dumb to understand that the "awful" for loops were actually what enabled Numba (I had no idea what it was at the time) to translate the code to C. I'm not sure if the vectorization actually resulted in a speedup.
See you in 5 years. Cheers.
