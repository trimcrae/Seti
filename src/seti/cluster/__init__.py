"""Population-level anomaly-clustering test.

Every single-object technosignature channel in this project ran to a
contamination-explained null: any *individual* anomaly can be a blend, a flare, a
reddened star or a noise spike.  This module asks a question that is immune to
those single-object degeneracies: do the weak anomalies **cluster** in position
and velocity beyond what a contaminant population (which traces the ordinary
stellar density) can produce?  A technological population expanding from an origin
would over-cluster in phase space; dust, blends and noise do not.

The rigorous core is a *matched-random null*: the anomaly set is compared not to a
uniform sky but to random subsets of the same parent catalogue matched in the
confounders (magnitude, colour, sky density) that would otherwise make any bright/
nearby subset look clustered.
"""
from .clustering import friends_of_friends, matched_null_clustering
from .phase_space import galactic_xyz, tangential_velocity

__all__ = ["galactic_xyz", "tangential_velocity", "matched_null_clustering",
           "friends_of_friends"]
