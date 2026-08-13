from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer


@dataclass(frozen=True)
class EntityMetrics:
    entity_name: str
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


def _normalize(values) -> list[str]:
    return sorted({str(v).strip().lower() for v in values if v and str(v).strip()})


def _gold_values(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return _normalize(str(cell).split(","))


def compare_to_gold(
    extraction_results: dict[str, dict[str, list[str]]],
    gold_df: pd.DataFrame,
) -> list[EntityMetrics]:
    """Compares extracted entity values against a gold CSV.

    `gold_df` must have a `file_name` column plus one column per entity type, where each
    cell holds the expected value(s) (comma-separated if there is more than one).
    """
    gold_df = gold_df.set_index("file_name")
    entity_names = list(gold_df.columns)
    shared_files = [f for f in extraction_results if f in gold_df.index]

    results = []
    for entity_name in entity_names:
        gold_labels = [_gold_values(gold_df.loc[f, entity_name]) for f in shared_files]
        predicted_labels = [_normalize(extraction_results[f].get(entity_name, [])) for f in shared_files]

        if not any(gold_labels) and not any(predicted_labels):
            results.append(EntityMetrics(entity_name, 1.0, 1.0, 1.0, 0, 0, 0))
            continue

        binarizer = MultiLabelBinarizer()
        binarizer.fit(gold_labels + predicted_labels)
        y_true = binarizer.transform(gold_labels)
        y_pred = binarizer.transform(predicted_labels)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        tn, fp, fn, tp = multilabel_confusion_matrix(y_true, y_pred).sum(axis=0).ravel()

        results.append(
            EntityMetrics(
                entity_name=entity_name,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                true_positives=int(tp),
                false_positives=int(fp),
                false_negatives=int(fn),
            )
        )
    return results


def overall_metrics(entity_metrics: list[EntityMetrics]) -> EntityMetrics:
    tp = sum(m.true_positives for m in entity_metrics)
    fp = sum(m.false_positives for m in entity_metrics)
    fn = sum(m.false_negatives for m in entity_metrics)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EntityMetrics("overall", precision, recall, f1, tp, fp, fn)
