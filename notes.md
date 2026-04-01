## Notes of Phase Spiral Correlations Project with M-SSA

Quick note: I need a new script file with the relevant loading data functions that isn't clogged with a bunch of useless stuff. Have a file called ``load_action_data.py'' that includes load_actions for the test particle sim and load_step_actions for the B2 sim. Make sure that these are stored as dataframes so that we can reference the index numbers. Note that at least in the test particle case, these won't be actual indices like in a normal pandas DF but rather just the numpy array index but since all particles get stored in the same order at each timestep and the total number of particles remains the same, this doesn't matter.

### Ideal M-SSA channels and binning to recover signals

I have often thought about how this problem is not ideal for M-SSA, even though it does reasonably well.
This is because the main signal that M-SSA picks up is that of differential rotation but this has different frequencies for a single channel at different radii.
Since PCs are primarily based on frequencies, this means that M-SSA encodes the same physical signal -- differential rotation -- into many different PCs as opposed to just one or two pairs.

One way around this is to re-run our BFE machinery but to bin by star id instead of by disk region. 
Binning by star id allows us to follow the same group of stars for an extended period of time.
This could be done either by initially choosing to group stars by close physical proximity at the time of the interaction or by action and angle at or before the time of interaction. The former might get a bit more difficult to motivate in the case of multiple interactions or non-impoulsive interactions. Doing this would eliminate the ``rotation'' from the resulting array of summary statistics. 
The result should be that M-SSA will be better able to pick out underlying physical mecahnisms that are not differential rotation.

However, importantly this is not helpful for comparisons to observations.
The differential rotation of the disk is what creates the macro-spiral which we use to date the perturbation.
So the current technique is still useful for that analysis, but it may be missing subtle physical phenomena which the other binning would show us.

I think this alternate binning is likely to be more impactful for pitch and phase angle correlations rather than amplitude. 
It is hard to imagine how large scale features such as the halo would amplify or suppress phase spiral amplitude, especially since the number of stars in each bin would stay constant (by design). At least not in an oscillatory way.


### Interpretation of F and G Matrices

In the F and G Matrices, especially in the test particle simulation, there is often a pattern where for principal components ~2-20 you get high contribution to fewer and fewer channels, with higher PCs keeping their contributions to the inner disk while having very little contribution to the outer disk. This is probably because the inner disk eventually phase mixes completely and the signal stops making sense. So it would be interesting to check if the face-on plots that don't include the upper end of these PCs deviate from a dipole in the inner disk specifically.
