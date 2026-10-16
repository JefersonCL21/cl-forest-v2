"""cl_allometry: fitting, comparing and keeping allometric models (height, volume, biomass, DBH from the crown) with
their statistics, so their coefficients are always the ones fitted and can be used anywhere.

    from cl_allometry import fit, forms_for
    f = fit("height.curtis", df)          # df with dap_cm and ht_m
    f.coefficients, f.metrics, f.checks, f.predict(dap_cm=25)
"""

from cl_allometry.fit import Fit, cross_validate, fit
from cl_allometry.forms import FORMS, TARGETS, Form, form, forms_for

__all__ = ["FORMS", "TARGETS", "Fit", "Form", "cross_validate", "fit", "form", "forms_for"]
