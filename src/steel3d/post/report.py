import csv


def print_summary(model, result):
    print("=== Displacements by node ===")
    for nid in sorted(model.nodes):
        dof = model.nodes[nid].dof_index
        vals = result.u[dof]
        print(f"Node {nid}: UX={vals[0]:.6e}, UY={vals[1]:.6e}, UZ={vals[2]:.6e}, RX={vals[3]:.6e}, RY={vals[4]:.6e}, RZ={vals[5]:.6e}")


def write_element_forces_csv(path, result):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["element", "end", "N", "Vy", "Vz", "T", "My", "Mz"])
        for eid, info in result.element_forces.items():
            for end in ("i", "j"):
                row = [eid, end] + [info[end][k] for k in ("N", "Vy", "Vz", "T", "My", "Mz")]
                w.writerow(row)
