import re

from pydantic import BaseModel, Field, create_model

PRESET_ENTITIES: dict[str, str] = {
    "Person": "Full name of a person mentioned in the document.",
    "Organization": "Name of a company, institution, or organization.",
    "Location": "A geographic location such as a city, state, or country.",
    "Date": "A calendar date mentioned in the document.",
    "Amount": "A monetary amount or quantity, including currency symbol if present.",
    "Product": "The name of a product, service, or line item.",
}


def _field_name(entity_name: str, taken: set[str]) -> str:
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", entity_name.strip()).strip("_").lower() or "entity"
    candidate = slug
    suffix = 2
    while candidate in taken:
        candidate = f"{slug}_{suffix}"
        suffix += 1
    return candidate


def build_entity_model(entities: dict[str, str]) -> tuple[type[BaseModel], dict[str, str]]:
    """Builds a pydantic model with one list[str] field per entity type.

    Returns the model plus a mapping of {field_name: original_entity_name}, since entity
    names aren't always valid Python identifiers.
    """
    if not entities:
        raise ValueError("At least one entity type is required")

    field_to_entity: dict[str, str] = {}
    fields: dict[str, tuple] = {}
    for entity_name, description in entities.items():
        field_name = _field_name(entity_name, taken=set(field_to_entity))
        field_to_entity[field_name] = entity_name
        fields[field_name] = (list[str], Field(default_factory=list, description=description))

    model = create_model("ExtractedEntities", **fields)
    return model, field_to_entity
