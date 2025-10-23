from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric
)

from test_rag import all_test_cases as test_cases 

metrics = [
    FaithfulnessMetric(),
    ContextualPrecisionMetric(),
    ContextualRecallMetric(),
    AnswerRelevancyMetric()
]

results = evaluate(test_cases, metrics, verbose=True)

print("✅ Avaliação concluída! Resultados salvos em deepeval/report_suit.json")
