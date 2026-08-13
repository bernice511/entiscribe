import pytest

from src.entities.schema import build_entity_model


def test_build_entity_model_maps_fields_back_to_entity_names():
    model, field_to_entity = build_entity_model({"Person": "d1", "Organization": "d2"})

    assert set(field_to_entity.values()) == {"Person", "Organization"}
    instance = model()
    for field_name in field_to_entity:
        assert getattr(instance, field_name) == []


def test_build_entity_model_rejects_empty_input():
    with pytest.raises(ValueError):
        build_entity_model({})


def test_build_entity_model_dedupes_slug_collisions():
    _, field_to_entity = build_entity_model({"Amount $": "d1", "Amount!": "d2"})

    assert len(field_to_entity) == 2
    assert set(field_to_entity.values()) == {"Amount $", "Amount!"}
