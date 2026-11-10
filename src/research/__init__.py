"""Forward validation of frozen research hypotheses ("shadow" models).

Shadow predictions are made at the same moment as the production predictions, from the same data, and stored
append-only under ``local_data/research/`` — never in the production tables, so they can never change what the
website shows. See ``shadow.py`` (recording and settlement), ``registry.py`` (which shadows exist and the frozen
preregistration each one implements), ``variants.py`` (what each shadow computes) and ``evaluate.py`` (paired
comparison against production, with the preregistered "no conclusion before N" rule).
"""
