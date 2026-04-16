# Abstention Policies And Tradeoffs

Abstention is not free. It exchanges prediction volume for accepted-set reliability.

Important tradeoffs:

- higher thresholds often reduce risk but lower coverage
- subgroup rejection rates can become uneven under shift
- accepted-set calibration can improve, worsen, or stay flat depending on the policy
- a policy tuned on reference data may miss target-domain operating points

ShiftStat keeps the tuning logic explicit:

- fixed thresholds are allowed when users want a simple rule
- target-coverage tuning selects the threshold closest to the requested coverage, then prefers lower selective risk
- target-risk tuning chooses the highest-coverage threshold satisfying the risk constraint; if none satisfy it, the lowest-risk threshold is returned and flagged
