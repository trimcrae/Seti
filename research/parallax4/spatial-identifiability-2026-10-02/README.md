# Native scan-geometry spatial identifiability control

AI-assisted by OpenAI Codex; independently reviewed before production repair.

Predeclared baseline: exact Gaia-4 source/transit-ID join of existing SHA256-pinned DR4 prerelease astrometry and DR3 epoch photometry. Keep native times, angles, parallax factors and positive centroid errors; retain original native phi/x separately. Synthetic psi=0.1*(t_yr−median(t_yr)), phi=psi/(1−psi), flux error0.001, hidden D=(300,−200)mas. No parameter tuning to force a verdict.

Production median re-reference preserves affine psi: if M=median(phi), psi_final=(1+M)*psi−M. Each sky column is a combination of position/proper-motion nuisance columns. Two explicit parameter null vectors give indistinguishable along-scan predictions with arbitrary D; a cancelled hidden-blend case and a sky-only injection expose the actual production confidence behavior. The initial report records native geometry and results without preordaining ON_TARGET. Only after actual evidence/review may a small target-coefficient estimability refusal be justified; nuisance-only rank deficiency must not automatically invalidate D.

This is conditional synthetic model identifiability, not an observed blend, measured contamination rate, detection or release calibration. No acquisition, threshold/preregistration/A4/release-gate changes, native scientific-output rewrite or sky rescreen.

Actual baseline CI36980276136/job110752984520 returned ON_TARGET with ul95=0.010447774605436949mas on the cancelled hidden D=(300,−200)mas witness, despite exact rank6/8 and independently verified target null directions. This justified the narrow confidence fix: one direct column-scaled whitened SVD/cutoff for coefficients, covariance, rank/nullspace and rank-based residual DOF; refuse finite D confidence when either target coefficient is nonestimable, while permitting nuisance-only rank defects. Near-affine perturbation1e−6 and nondegenerate amplitude0.2 use the same predeclared golden-ratio chronological-index sine sequence, without verdict tuning. Thresholds and A4/release gates remain byte-identical.
