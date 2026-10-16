"""cl_allometry: fitting, comparing and keeping allometric models (height, volume, biomass, DBH from the crown) with
their statistics, so their coefficients are always the ones fitted and can be used anywhere.

    from cl_allometry import fit, forms_for
    f = fit("height.curtis", df)          # df with dap_cm and ht_m
    f.coefficients, f.metrics, f.checks, f.predict(dap_cm=25)

    from cl_allometry import Registry, open_store
    reg = Registry(open_store("gs://bucket"))
    reg.resolve("ht_m", "Khaya grandifoliola").predict(dap_cm=25)   # the champion of the species
"""

from cl_allometry.fit import Fit, cross_validate, fit
from cl_allometry.forms import FORMS, TARGETS, Form, form, forms_for
from cl_allometry.registry import Filter, Registry, Scope
from cl_allometry.store import open_store

__all__ = ["FORMS", "TARGETS", "Filter", "Fit", "Form", "Registry", "Scope", "cross_validate", "fit", "form",
           "forms_for", "open_store"]
