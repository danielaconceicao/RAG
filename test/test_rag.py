import os
from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase, ConversationalTestCase, Turn
from deepeval.metrics import (
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    TurnRelevancyMetric
)

# cria a pasta se nao tiver
save_path = "deepeval/result_test_rag.json"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

single_turn_test = [
    LLMTestCase(
        name="definizione covid-19",
        input="Cos'è il COVID-19?",
        actual_output="Il COVID-19 è una malattia respiratoria causata dal virus SARS-CoV-2.",
        retrieval_context=[
            "COVID-19 è il nome della malattia da nuovo coronavirus: 'CO' indica corona, 'VI' virus, 'D' significa disease e 19 si riferisce al 2019."
        ],
        expected_output="Il COVID-19 è una malattia causata dal coronavirus SARS-CoV-2."
    ),
    LLMTestCase(
        name="trasmissione covid-19",
        input="Come si trasmette il virus COVID-19?",
        actual_output="Il virus si trasmette attraverso le goccioline respiratorie emesse da una persona infetta quando tossisce, starnutisce o parla.",
        retrieval_context=[
            "Le goccioline del respiro sono la modalità di trasmissione principale del virus; queste possono passare da una persona all’altra attraverso uno starnuto, un colpo di tosse e contatti diretti personali."
        ],
        expected_output="Il virus si trasmette attraverso le goccioline respiratorie."
    ),
    LLMTestCase(
        name="grafico di avanzamento del caso covid-19",
        input="Cosa mostra il grafico per gli anni 2020-2022?",
        actual_output="Il grafico mostra che i casi sono aumentati nel 2020, sono diminuiti nel 2021 e sono aumentati nuovamente nel 2022.",
        retrieval_context=[
            "L'immagine mostra un grafico che mostra l'evoluzione dei casi di COVID-19 in Italia: aumento nel 2020, diminuzione nel 2021 e nuovo aumento nel 2022."
        ],
        expected_output="I casi sono aumentati nel 2020, sono diminuiti nel 2021 e sono nuovamente aumentati nel 2022."
    ),
    LLMTestCase(
        name="contraddizione incubazione covid-19",
        input="Quanto dura il periodo di incubazione del COVID-19?",
        actual_output="Il periodo di incubazione medio è di circa 7 giorni.",
        retrieval_context=[
            "Secondo gli ultimi dati, il periodo di incubazione del COVID-19 è mediamente di 5 giorni (range 2-14 giorni)."
        ],
        expected_output="Il periodo di incubazione è di 5 giorni."
    ),
]

multi_turn_tests = [
    ConversationalTestCase(
        name="Memória RAG: Definição e Sintomas",
        turns=[
            Turn(
                role="user",
                content="Quali sono i sintomi del COVID-19?",
            ),

            Turn(
                role="assistant",
                content="I sintomi comuni del COVID-19 includono febbre, tosse secca e affaticamento.",
                expected_output="Febbre, tosse e difficoltà respiratorie.",
                retrieval_context=[
                    "I sintomi più comuni del COVID-19 sono febbre, tosse, stanchezza e perdita del gusto o dell'olfatto."
                ],
            ),
            Turn(
                role="user",
                content="Quanto dura la quarantena per quel virus?",
            ),

            Turn(
                role="assistant",
                content="La durata della quarantena dipende dalle normative locali, ma in genere è di circa 10 giorni.",
                expected_output="Circa 10 giorni.",
                retrieval_context=[
                    "Per le persone positive al virus, la durata consigliata della quarantena è di 10 giorni."
                ],
            ),
        ]
    )
]


# métricas
single_metrics = [
    FaithfulnessMetric(),
    ContextualPrecisionMetric(),
    ContextualRecallMetric(),
    AnswerRelevancyMetric()
]
multi_metrics = TurnRelevancyMetric(threshold=0.8)

all_results = []

results_single = evaluate(
    test_cases=single_turn_test, 
    metrics=single_metrics
)
all_results.extend(results_single)


results_multi = evaluate(
    test_cases=multi_turn_tests, 
    metrics=[multi_metrics]
)
all_results.extend(results_multi)

print("Valutazione completata, risultati salvati in deepeval/result_test_rag.json")
