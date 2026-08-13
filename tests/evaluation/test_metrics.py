import pandas as pd

from src.evaluation.metrics import compare_to_gold, overall_metrics


def test_compare_to_gold_perfect_match():
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Person": "John"}])
    extraction_results = {"a.pdf": {"Person": ["John"]}}

    [metrics] = compare_to_gold(extraction_results, gold_df)

    assert metrics.entity_name == "Person"
    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)
    assert (metrics.true_positives, metrics.false_positives, metrics.false_negatives) == (1, 0, 0)


def test_compare_to_gold_partial_match_across_files():
    gold_df = pd.DataFrame(
        [
            {"file_name": "a.pdf", "Person": "John"},
            {"file_name": "b.pdf", "Person": "Mary"},
        ]
    )
    extraction_results = {
        "a.pdf": {"Person": ["John"]},
        "b.pdf": {"Person": ["Bob"]},
    }

    [metrics] = compare_to_gold(extraction_results, gold_df)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5


def test_compare_to_gold_case_insensitive_and_whitespace_tolerant():
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Person": " John "}])
    extraction_results = {"a.pdf": {"Person": ["john"]}}

    [metrics] = compare_to_gold(extraction_results, gold_df)

    assert metrics.true_positives == 1
    assert metrics.false_positives == 0


def test_compare_to_gold_multi_value_cell_is_comma_split():
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Product": "Taco, Salad"}])
    extraction_results = {"a.pdf": {"Product": ["Taco"]}}

    [metrics] = compare_to_gold(extraction_results, gold_df)

    assert metrics.true_positives == 1
    assert metrics.false_negatives == 1


def test_compare_to_gold_treats_entirely_absent_entity_as_trivially_correct():
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Person": None}])
    extraction_results = {"a.pdf": {"Person": []}}

    [metrics] = compare_to_gold(extraction_results, gold_df)

    assert (metrics.precision, metrics.recall, metrics.f1) == (1.0, 1.0, 1.0)


def test_compare_to_gold_ignores_files_missing_from_gold():
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Person": "John"}])
    extraction_results = {
        "a.pdf": {"Person": ["John"]},
        "unrelated.pdf": {"Person": ["Someone Else"]},
    }

    [metrics] = compare_to_gold(extraction_results, gold_df)

    assert (metrics.true_positives, metrics.false_positives) == (1, 0)


def test_overall_metrics_aggregates_across_entities():
    gold_df = pd.DataFrame([{"file_name": "a.pdf", "Person": "John", "Location": "Boston"}])
    extraction_results = {"a.pdf": {"Person": ["John"], "Location": ["Nowhere"]}}

    per_entity = compare_to_gold(extraction_results, gold_df)
    overall = overall_metrics(per_entity)

    assert overall.entity_name == "overall"
    assert overall.true_positives == 1
    assert overall.false_positives == 1
    assert overall.false_negatives == 1
