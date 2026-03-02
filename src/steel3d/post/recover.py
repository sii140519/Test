def recover_element_forces(model, loadcase, u_global):
    out = {}
    for eid, element in model.elements.items():
        out[eid] = element.recover(u_global=u_global, loadcase=loadcase, model=model)
    return out
