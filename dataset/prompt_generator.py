from __future__ import annotations

import random
from typing import Dict, Iterable, List


DEFAULT_TEMPLATES = [
    "A photo of {}",
    "Aircraft with {}",
    "Structural {} on aircraft",
    "Aircraft surface showing {}",
]


def generate_prompt(label: str, templates: Iterable[str] | None = None) -> str:
    """Generate a text prompt from a label."""
    templates_list: List[str] = list(templates) if templates is not None else DEFAULT_TEMPLATES
    template = random.choice(templates_list)
    return template.format(label)


def build_prompts(
    annotation: Dict,
    include_description: bool = True,
    include_definition: bool = True,
) -> List[str]:
    """Build multiple prompts from a rich annotation."""
    label = annotation.get("label") or annotation.get("category_name") or "object"
    zone = annotation.get("zone") or annotation.get("zone_estimation")
    severity = annotation.get("severity") or (annotation.get("risk_assessment") or {}).get("severity_level")
    definition = annotation.get("class_definition")
    description = annotation.get("description")

    prompts: List[str] = []
    prompts.append(f"{label} on aircraft surface")

    if severity:
        prompts.append(f"{severity} severity {label}")
    if zone:
        prompts.append(f"{label} located in {zone} aircraft structure")
    if include_definition and definition:
        prompts.append(str(definition))
    if include_description and description:
        prompts.append(str(description))

    return prompts
