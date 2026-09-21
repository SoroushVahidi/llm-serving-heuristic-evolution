"""Joint state-policy suitability infrastructure.

f(x, pi) -> (predicted_reward, uncertainty), and the conservative
suitability S(x, pi) = mu(x, pi) - lambda * u(x, pi) built on top of it.

This is the first implementation of the project's stated next research
stage (policy library -> state-policy suitability -> strong selector ->
module credit -> structural synthesis -> new policies -> expanded
library). See docs/current/STATE_POLICY_SUITABILITY_REPORT.md for the
scientific report and docs/current/STATE_POLICY_SUITABILITY_SCHEMA.md for
the dataset/API documentation.
"""
