from __future__ import annotations

import random
from typing import Dict, Iterable, List


DEFAULT_TEMPLATES = [
    "A photo of {}",
    "Aircraft with {}",
    "Structural {} on aircraft",
    "Aircraft surface showing {}",
]

PROMPT_MODES = {
    "full",
    "label_only",
    "label_zone",
    "description_only",
    "definition_only",
    "generic",
}


def generate_prompt(label: str, templates: Iterable[str] | None = None) -> str:
    """Generate a text prompt from a label."""
    templates_list: List[str] = list(templates) if templates is not None else DEFAULT_TEMPLATES
    template = random.choice(templates_list)
    return template.format(label)


def build_prompts(
    annotation: Dict,
    include_description: bool = True,
    include_definition: bool = True,
    prompt_mode: str = "full",
) -> List[str]:
    """Build multiple prompts from a rich annotation."""
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode: {prompt_mode}")

    label = annotation.get("label") or annotation.get("category_name") or "object"
    zone = annotation.get("zone") or annotation.get("zone_estimation")
    severity = annotation.get("severity") or (annotation.get("risk_assessment") or {}).get("severity_level")
    definition = annotation.get("class_definition")
    description = annotation.get("description")

    label_prompt = f"{label} on aircraft surface"
    zone_prompt = f"{label} located in {zone} aircraft structure" if zone else None
    severity_prompt = f"{severity} severity {label}" if severity else None

    if prompt_mode == "generic":
        return ["aircraft damage"]
    if prompt_mode == "label_only":
        return [label_prompt]
    if prompt_mode == "label_zone":
        prompts = [label_prompt]
        if zone_prompt:
            prompts.append(zone_prompt)
        return prompts
    if prompt_mode == "description_only":
        return [str(description)] if description else [label_prompt]
    if prompt_mode == "definition_only":
        return [str(definition)] if definition else [label_prompt]

    prompts: List[str] = []
    prompts.append(label_prompt)

    if severity_prompt:
        prompts.append(severity_prompt)
    if zone_prompt:
        prompts.append(zone_prompt)
    if include_definition and definition:
        prompts.append(str(definition))
    if include_description and description:
        prompts.append(str(description))

    return prompts
