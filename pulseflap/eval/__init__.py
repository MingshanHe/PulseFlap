"""Statistics used in the paper (aliases of the experiment modules).

    paired(opp_rows, rows)     paired comparison on identical cases: manoeuvre-ratio difference Delta with a
                               bootstrap 95 % CI (resampled within task), crashes, wins (paired_report.paired)
    ratios(rows, medians)      manoeuvre ratios per task (paired_report.ratios)
    clopper_pearson(k, n)      exact binomial 95 % interval of a crash rate (evaluate_v8.clopper_pearson)
    mcnemar_exact(x, y)        exact McNemar test on paired crash outcomes (stepsize_q1.mcnemar_exact)
    holm(p_values)             Holm-Bonferroni adjusted p-values (holm.holm)
"""
from ..spike_steer.paired_report import paired, ratios  # noqa: F401
from ..spike_steer.evaluate_v8 import clopper_pearson  # noqa: F401
from ..spike_steer.stepsize_q1 import mcnemar_exact  # noqa: F401
from ..spike_steer.holm import holm  # noqa: F401
