from dataclasses import dataclass, field
from typing import Dict

from .material import Material
from .node import Node
from .section import Section


@dataclass
class Model:
    nodes: Dict[int, Node] = field(default_factory=dict)
    materials: Dict[int, Material] = field(default_factory=dict)
    sections: Dict[int, Section] = field(default_factory=dict)
    elements: Dict[int, object] = field(default_factory=dict)

    def add_node(self, node: Node) -> None:
        self.nodes[node.id] = node

    def add_material(self, material: Material) -> None:
        self.materials[material.id] = material

    def add_section(self, section: Section) -> None:
        self.sections[section.id] = section

    def add_element(self, element: object) -> None:
        self.elements[element.id] = element

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("Model has no nodes")
        if not self.elements:
            raise ValueError("Model has no elements")
