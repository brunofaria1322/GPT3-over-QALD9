import pickle
from tqdm import tqdm
import json
import numpy as np


import warnings

warnings.filterwarnings("ignore")

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_theme(style="white", palette="viridis")

# sparql
from SPARQLWrapper import SPARQLWrapper, JSON

# normalize answers
from dateutil.parser import parse
import string
from nltk.corpus import stopwords
from num2words import num2words
from unidecode import unidecode

# Metrics
from nltk.translate.bleu_score import sentence_bleu

### COSTUM
from dataset_parsing import read_qald, parse_qald

GPT3_TYPES = ["gpt3_davinci002", "gpt3_davinci003"]
GPT3_FT = "gpt3_davinci_ft"

FS_TYPES = ["", "-fs5"]
QUESTIONS_ID = [str(i) for i in range(1, 11)]


def get_scores(y_true, y_pred):
    #
    # y_true = gold standard answers
    # y_pred = system answers
    #
    # recall = number of correct system answers / number of gold standard answers
    # precision = number of correct system answers / number of system answers
    # f1 = 2 * (precision * recall) / (precision + recall)
    # accuracy = number of correct system answers / number of all answers
    #
    # return recall, precision, f1, accuracy

    y_pred = [str(y).lower() for y in y_pred]
    y_true = [str(y).lower() for y in y_true]

    num_system = len(y_pred)
    num_gold = len(y_true)
    num_correct = len(set(y_pred) & set(y_true))

    recall = num_correct / num_gold
    precision = num_correct / num_system

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    # accuracy = num_correct / num_system

    return recall, precision, f1  # , accuracy


def evaluate_sparql_generation(queries):
    """
    Evaluate sparql generation

    :param queries: queries
    """

    print("*** Evaluating sparql generation! ***")

    all_metrics = {}

    for fs in FS_TYPES:
        for gpt3_type in GPT3_TYPES:
            name = f"{gpt3_type}{fs}"
            all_metrics[name] = {q: {} for q in QUESTIONS_ID}

            gpt3_sparql = None

            # load GPT3 sparql
            with open(f"output/{name}/{name}-test-query_dict.pkl", "rb") as f:
                gpt3_sparql = pickle.load(f)

            b1s = []
            b2s = []
            sbs = []
            b1s_np = []
            b2s_np = []
            sbs_np = []

            # evaluate GPT3
            for question in gpt3_sparql:
                for i, query in enumerate(queries):
                    gpt3_query = gpt3_sparql[question][i][0]["text"].lower().split()
                    query = query.lower().split()

                    # Setence BLEU
                    # BLEU-1
                    b1 = sentence_bleu([query], gpt3_query, weights=(1, 0, 0, 0))
                    b1s.append(b1)

                    # BLEU-2
                    b2 = sentence_bleu([query], gpt3_query, weights=(0, 1, 0, 0))
                    b2s.append(b2)

                    # BLEU-4
                    sb = sentence_bleu([query], gpt3_query)
                    sbs.append(sb)

                    # remove prefixes
                    while gpt3_query and gpt3_query[0] == "prefix":
                        gpt3_query = gpt3_query[3:]

                    while query and query[0] == "prefix":
                        query = query[3:]

                    # Setence BLEU
                    # BLEU-1
                    b1_np = sentence_bleu([query], gpt3_query, weights=(1, 0, 0, 0))
                    b1s_np.append(b1_np)

                    # BLEU-2
                    b2_np = sentence_bleu([query], gpt3_query, weights=(0, 1, 0, 0))
                    b2s_np.append(b2_np)

                    # BLEU-4
                    sb_np = sentence_bleu([query], gpt3_query)
                    sbs_np.append(sb_np)

                print(
                    f"{name:20} - {question:<3} => BLEU-1: {np.mean(b1s):.4f} BLEU-2: {np.mean(b2s):.4f} SETENCE-BLEU: {np.mean(sbs):.4f} | BLEU-1 (no prefix): {np.mean(b1s_np):.4f} BLEU-2 (no prefix): {np.mean(b2s_np):.4f} SETENCE-BLEU (no prefix): {np.mean(sbs_np):.4f}"
                )

    # Fine-tuned
    if GPT3_FT:
        name = GPT3_FT
        question = "Q1"  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        gpt3_sparql = None

        # load GPT3 sparql
        with open(f"output/{name}/{name}-test-query_dict.pkl", "rb") as f:
            gpt3_sparql = pickle.load(f)

        #### GPT3
        b1s = []
        b2s = []
        sbs = []
        b1s_np = []
        b2s_np = []
        sbs_np = []

        for i, query in enumerate(queries):
            gpt3_query = gpt3_sparql[question][i][0]["text"].lower().split()
            query = query.lower().split()

            # Setence BLEU
            # BLEU-1
            b1 = sentence_bleu([query], gpt3_query, weights=(1, 0, 0, 0))
            b1s.append(b1)

            # BLEU-2
            b2 = sentence_bleu([query], gpt3_query, weights=(0, 1, 0, 0))
            b2s.append(b2)

            # SETENCE BLEU
            sb = sentence_bleu([query], gpt3_query)
            sbs.append(sb)

            # remove prefixes
            while gpt3_query and gpt3_query[0] == "prefix":
                gpt3_query = gpt3_query[3:]

            while query and query[0] == "prefix":
                query = query[3:]

            # Setence BLEU
            # BLEU-1
            b1_np = sentence_bleu([query], gpt3_query, weights=(1, 0, 0, 0))
            b1s_np.append(b1_np)

            # BLEU-2
            b2_np = sentence_bleu([query], gpt3_query, weights=(0, 1, 0, 0))
            b2s_np.append(b2_np)

            # SETENCE BLEU
            sb_np = sentence_bleu([query], gpt3_query)
            sbs_np.append(sb_np)

        print(
            f"{name:20} - {question:<3} => BLEU-1: {np.mean(b1s):.4f} BLEU-2: {np.mean(b2s):.4f} SETENCE-BLEU: {np.mean(sbs):.4f} | BLEU-1 (no prefix): {np.mean(b1s_np):.4f} BLEU-2 (no prefix): {np.mean(b2s_np):.4f} SETENCE-BLEU (no prefix): {np.mean(sbs_np):.4f}"
        )


def evaluate_sparql_answers(ids, answers):
    """
    Evaluate sparql answers

    :param ids: ids
    :param answers: answers
    """

    print("*** Evaluating sparql answers! ***")

    # indexing for ids
    answers_dict = {id: answer for id, answer in zip(ids, answers)}

    all_metrics = {}

    for gpt3_type in GPT3_TYPES:
        for fs in FS_TYPES:
            name = f"{gpt3_type}{fs}"
            all_metrics[name] = {q: {} for q in QUESTIONS_ID}

            # lists for corpus BLEU

            for q in QUESTIONS_ID:
                # print(f'================== {name} Q{q} ==================')
                all_metrics[name][q] = {id: {} for id in ids}

                # load GPT3 answers
                with open(
                    f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb"
                ) as f:
                    gpt3_sparql_answers = pickle.load(f)

                # print(gpt3_sparql_answers)

                # evaluate GPT3
                for type in gpt3_sparql_answers:
                    for id in gpt3_sparql_answers[type]:
                        if type == "error" or type == "empty":
                            # If there is an error or it is empty, we don't want to evaluate it
                            all_metrics[name][q][id]["precision"] = 0
                            all_metrics[name][q][id]["recall"] = 0
                            all_metrics[name][q][id]["f1"] = 0
                            all_metrics[name][q][id]["BLEU-1"] = 0
                            all_metrics[name][q][id]["BLEU-2"] = 0
                            all_metrics[name][q][id]["SETENCE-BLEU"] = 0
                            continue

                        # get GPT3 answer
                        gpt3_answer = gpt3_sparql_answers[type][id]

                        answer = answers_dict[id]

                        # evaluate: precision, recall, f1

                        recall, precision, f1 = get_scores(answer, gpt3_answer)

                        all_metrics[name][q][id]["precision"] = precision
                        all_metrics[name][q][id]["recall"] = recall
                        all_metrics[name][q][id]["f1"] = f1
                        all_metrics[name][q][id]["BLEU-1"] = sentence_bleu(
                            [answer], gpt3_answer, weights=(1, 0, 0, 0)
                        )
                        all_metrics[name][q][id]["BLEU-2"] = sentence_bleu(
                            [answer], gpt3_answer, weights=(0, 1, 0, 0)
                        )
                        all_metrics[name][q][id]["SETENCE-BLEU"] = sentence_bleu(
                            [answer], gpt3_answer
                        )

                macro_precision = 0
                macro_recall = 0

                mean_b1 = 0
                mean_b2 = 0
                mean_sb = 0

                for id in all_metrics[name][q]:
                    macro_precision += all_metrics[name][q][id]["precision"]
                    macro_recall += all_metrics[name][q][id]["recall"]

                    mean_b1 += all_metrics[name][q][id]["BLEU-1"]
                    mean_b2 += all_metrics[name][q][id]["BLEU-2"]
                    mean_sb += all_metrics[name][q][id]["SETENCE-BLEU"]

                macro_precision /= len(all_metrics[name][q])
                macro_recall /= len(all_metrics[name][q])
                macro_f1 = (
                    2
                    * (macro_precision * macro_recall)
                    / (macro_precision + macro_recall)
                )

                mean_b1 /= len(all_metrics[name][q])
                mean_b2 /= len(all_metrics[name][q])
                mean_sb /= len(all_metrics[name][q])

                print(
                    f"{name:20} - Q{q:<2} => MACRO-PRECISION: {macro_precision:.4f} MACRO-RECALL: {macro_recall:.4f} MACRO-F1: {macro_f1:.4f} MEAN-BLEU-1: {mean_b1:.4f} MEAN-BLEU-2: {mean_b2:.4f} SETENCE-BLEU: {mean_sb:.4f}"
                )

    # Fine-tuned GPT3
    if GPT3_FT:
        name = GPT3_FT
        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        # print(f'================== {name} ==================')
        all_metrics[name] = {q: {id: {} for id in ids}}

        # load GPT3 answers
        with open(f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb") as f:
            gpt3_sparql_answers = pickle.load(f)

        # print(gpt3_sparql_answers)

        # evaluate GPT3
        for type in gpt3_sparql_answers:
            for id in gpt3_sparql_answers[type]:
                if type == "error" or type == "empty":
                    # If there is an error or it is empty, we don't want to evaluate it
                    all_metrics[name][q][id]["precision"] = 0
                    all_metrics[name][q][id]["recall"] = 0
                    all_metrics[name][q][id]["f1"] = 0
                    all_metrics[name][q][id]["BLEU-1"] = 0
                    all_metrics[name][q][id]["BLEU-2"] = 0
                    all_metrics[name][q][id]["SETENCE-BLEU"] = 0
                    continue

                gpt3_answer = gpt3_sparql_answers[type][id]

                answer = answers_dict[id]

                # evaluate: precision, recall, f1

                recall, precision, f1 = get_scores(answer, gpt3_answer)

                all_metrics[name][q][id]["precision"] = precision
                all_metrics[name][q][id]["recall"] = recall
                all_metrics[name][q][id]["f1"] = f1
                all_metrics[name][q][id]["BLEU-1"] = sentence_bleu(
                    [answer], gpt3_answer, weights=(1, 0, 0, 0)
                )
                all_metrics[name][q][id]["BLEU-2"] = sentence_bleu(
                    [answer], gpt3_answer, weights=(0, 1, 0, 0)
                )
                all_metrics[name][q][id]["SETENCE-BLEU"] = sentence_bleu(
                    [answer], gpt3_answer
                )

        macro_precision = 0
        macro_recall = 0

        mean_b1 = 0
        mean_b2 = 0
        mean_sb = 0

        for id in all_metrics[name][q]:
            macro_precision += all_metrics[name][q][id]["precision"]
            macro_recall += all_metrics[name][q][id]["recall"]

            mean_b1 += all_metrics[name][q][id]["BLEU-1"]
            mean_b2 += all_metrics[name][q][id]["BLEU-2"]
            mean_sb += all_metrics[name][q][id]["SETENCE-BLEU"]

        macro_precision /= len(all_metrics[name][q])
        macro_recall /= len(all_metrics[name][q])
        macro_f1 = (
            2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        )

        mean_b1 /= len(all_metrics[name][q])
        mean_b2 /= len(all_metrics[name][q])
        mean_sb /= len(all_metrics[name][q])

        print(
            f"{name:20} - Q{q:<2} => MACRO-PRECISION: {macro_precision:.4f} MACRO-RECALL: {macro_recall:.4f} MACRO-F1: {macro_f1:.4f} MEAN-BLEU-1: {mean_b1:.4f} MEAN-BLEU-2: {mean_b2:.4f} SETENCE-BLEU: {mean_sb:.4f}"
        )


def evaluate_sparql_valid_answers(ids, answers):
    """
    Evaluate sparql answers

    :param ids: ids
    :param answers: answers
    """

    print("*** Evaluating sparql valid answers! ***")

    # indexing for ids
    answers_dict = {id: answer for id, answer in zip(ids, answers)}

    all_metrics = {}

    for gpt3_type in GPT3_TYPES:
        for fs in FS_TYPES:
            name = f"{gpt3_type}{fs}"
            all_metrics[name] = {q: {} for q in QUESTIONS_ID}

            # lists for corpus BLEU

            for q in QUESTIONS_ID:
                # print(f'================== {name} Q{q} ==================')

                # load GPT3 answers
                with open(
                    f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb"
                ) as f:
                    gpt3_sparql_answers = pickle.load(f)

                # print(gpt3_sparql_answers)

                # evaluate GPT3
                for type in gpt3_sparql_answers:
                    if type == "error" or type == "empty":
                        continue

                    for id in gpt3_sparql_answers[type]:
                        all_metrics[name][q][id] = {}

                        # get GPT3 answer
                        gpt3_answer = gpt3_sparql_answers[type][id]

                        answer = answers_dict[id]

                        # evaluate: precision, recall, f1

                        recall, precision, f1 = get_scores(answer, gpt3_answer)

                        all_metrics[name][q][id]["precision"] = precision
                        all_metrics[name][q][id]["recall"] = recall
                        all_metrics[name][q][id]["f1"] = f1
                        all_metrics[name][q][id]["BLEU-1"] = sentence_bleu(
                            [answer], gpt3_answer, weights=(1, 0, 0, 0)
                        )
                        all_metrics[name][q][id]["BLEU-2"] = sentence_bleu(
                            [answer], gpt3_answer, weights=(0, 1, 0, 0)
                        )
                        all_metrics[name][q][id]["SETENCE-BLEU"] = sentence_bleu(
                            [answer], gpt3_answer
                        )

                macro_precision = 0
                macro_recall = 0

                mean_b1 = 0
                mean_b2 = 0
                mean_sb = 0

                for id in all_metrics[name][q]:
                    macro_precision += all_metrics[name][q][id]["precision"]
                    macro_recall += all_metrics[name][q][id]["recall"]

                    mean_b1 += all_metrics[name][q][id]["BLEU-1"]
                    mean_b2 += all_metrics[name][q][id]["BLEU-2"]
                    mean_sb += all_metrics[name][q][id]["SETENCE-BLEU"]

                macro_precision /= len(all_metrics[name][q])
                macro_recall /= len(all_metrics[name][q])
                macro_f1 = (
                    2
                    * (macro_precision * macro_recall)
                    / (macro_precision + macro_recall)
                    if (macro_precision + macro_recall)
                    else 0
                )

                mean_b1 /= len(all_metrics[name][q])
                mean_b2 /= len(all_metrics[name][q])
                mean_sb /= len(all_metrics[name][q])

                print(
                    f"{name:20} - Q{q:<2} {len(all_metrics[name][q]):>3}/150 VALID => MACRO-PRECISION: {macro_precision:.4f} MACRO-RECALL: {macro_recall:.4f} MACRO-F1: {macro_f1:.4f} MEAN-BLEU-1: {mean_b1:.4f} MEAN-BLEU-2: {mean_b2:.4f} SETENCE-BLEU: {mean_sb:.4f}"
                )

    return
    # Fine-tuned GPT3
    if GPT3_FT:
        name = GPT3_FT
        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        # print(f'================== {name} ==================')
        all_metrics[name] = {q: {id: {} for id in ids}}

        # load GPT3 answers
        with open(f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb") as f:
            gpt3_sparql_answers = pickle.load(f)

        # print(gpt3_sparql_answers)

        # evaluate GPT3
        for type in gpt3_sparql_answers:
            if type == "error" or type == "empty":
                continue

            for id in gpt3_sparql_answers[type]:
                gpt3_answer = gpt3_sparql_answers[type][id]

                answer = answers_dict[id]

                # evaluate: precision, recall, f1

                recall, precision, f1 = get_scores(answer, gpt3_answer)

                all_metrics[name][q][id]["precision"] = precision
                all_metrics[name][q][id]["recall"] = recall
                all_metrics[name][q][id]["f1"] = f1
                all_metrics[name][q][id]["BLEU-1"] = sentence_bleu(
                    [answer], gpt3_answer, weights=(1, 0, 0, 0)
                )
                all_metrics[name][q][id]["BLEU-2"] = sentence_bleu(
                    [answer], gpt3_answer, weights=(0, 1, 0, 0)
                )
                all_metrics[name][q][id]["SETENCE-BLEU"] = sentence_bleu(
                    [answer], gpt3_answer
                )

        macro_precision = 0
        macro_recall = 0

        mean_b1 = 0
        mean_b2 = 0
        mean_sb = 0

        for id in all_metrics[name][q]:
            macro_precision += all_metrics[name][q][id]["precision"]
            macro_recall += all_metrics[name][q][id]["recall"]

            mean_b1 += all_metrics[name][q][id]["BLEU-1"]
            mean_b2 += all_metrics[name][q][id]["BLEU-2"]
            mean_sb += all_metrics[name][q][id]["SETENCE-BLEU"]

        macro_precision /= len(all_metrics[name][q])
        macro_recall /= len(all_metrics[name][q])
        macro_f1 = (
            2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)
        )

        mean_b1 /= len(all_metrics[name][q])
        mean_b2 /= len(all_metrics[name][q])
        mean_sb /= len(all_metrics[name][q])

        print(
            f"{name:20} - Q{q:<2} => MACRO-PRECISION: {macro_precision:.4f} MACRO-RECALL: {macro_recall:.4f} MACRO-F1: {macro_f1:.4f} MEAN-BLEU-1: {mean_b1:.4f} MEAN-BLEU-2: {mean_b2:.4f} SETENCE-BLEU: {mean_sb:.4f}"
        )


def parse_date(string, fuzzy=False):
    """
    Parse date

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True

    :return: datetime in extensive form or string
    """

    try:
        a = parse(string, fuzzy=fuzzy)
        return a.strftime("%d %B %Y")

    except:
        return string


def parse_quald_answers(answers):
    """
    Parse QALD answers

    :param answers: answers
    :return: answers_str
    """

    for i in range(len(answers)):
        answers[i] = str(answers[i])
        if answers[i].startswith("http://dbpedia.org"):
            sparql = SPARQLWrapper("http://dbpedia.org/sparql")
            sparql.setQuery(
                f"""
                SELECT ?label
                WHERE{{
                    <{answers[i]}>
                    rdfs:label ?label.
                    FILTER (lang(?label) = 'en')
                }}
            """
            )
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()

            if results["results"]["bindings"]:
                answers[i] = results["results"]["bindings"][0]["label"]["value"]
            else:
                # print(f'No label found for {answers[i]}. Will try to find a name.')
                sparql = SPARQLWrapper("http://dbpedia.org/sparql")
                sparql.setQuery(
                    f"""
                    SELECT ?name
                    WHERE{{
                        <{answers[i]}>
                        foaf:name ?name.
                        FILTER (lang(?name) = 'en')
                    }}
                """
                )
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()

                if results["results"]["bindings"]:
                    answers[i] = results["results"]["bindings"][0]["name"]["value"]
                else:
                    # print(f'No name found for {answers[i]}. Will parse Uri instead.')
                    res = answers[i].split("/")[-1]
                    res = res.replace("_", " ")
                    answers[i] = res.strip()

    return ", ".join(answers)


def normalise_answer(answer):
    """
    Normalise answer

    :param answer: str[], answer to normalise
    :return: str, normalised answer
    """

    # https://www.geeksforgeeks.org/normalizing-textual-data-with-python/

    # replace numbers with words
    answer = " ".join(
        [num2words(word) if word.isdigit() else word for word in answer.split()]
    )

    # replace date
    answer = " ".join([parse_date(word) for word in answer.split()])

    # remove punctuation and lower
    answer = answer.translate(str.maketrans("", "", string.punctuation)).lower()

    # remove accents
    answer = unidecode(answer)

    # remove stopwords
    answer = " ".join(
        [word for word in answer.split() if word not in stopwords.words("english")]
    )

    return answer


def evaluate_direct_answers(ids, answers):
    """
    Evaluate direct answers

    :param ids: ids
    :param answers: answers
    """

    print("*** Evaluating direct answers! ***")

    # check if there is a file with the normalised answers
    try:
        with open("data/normalised_answers.json", "r") as f:
            normalised_answers = json.load(f)
        print("Normalised answers found!")
    except:
        # normalise answers
        print("Normalised answers not found... Normalising answers...")
        normalised_answers = {
            id: normalise_answer(parse_quald_answers(answer))
            for id, answer in tqdm(zip(ids, answers), total=len(ids))
        }

        # save normalised answers
        with open("data/normalised_answers.json", "w") as f:
            json.dump(normalised_answers, f, indent=4)

    # check if data is already treated
    try:
        with open("output/gpt3/treated_direct_answers.json", "r") as f:
            gpt3_treated_direct_answers = json.load(f)
        print("Treated GPT3 answers found!")

    except:
        # load GPT3 answers
        with open("output/gpt3/direct_answers.json", "r") as f:
            gpt3_direct_answers = json.load(f)

        gpt3_treated_direct_answers = {
            id: normalise_answer(" ".join(gpt3_direct_answers[id]))
            for id in gpt3_direct_answers
        }

        # save treated GPT3 answers
        with open("output/gpt3/treated_direct_answers.json", "w") as f:
            json.dump(gpt3_treated_direct_answers, f, indent=4)

    all_metrics = {}

    for id in normalised_answers:
        all_metrics[id] = {}

        gpt3_answer = gpt3_treated_direct_answers[id]
        answer = normalised_answers[id]

        gpt3_answer = gpt3_answer.split()
        answer = answer.split()

        # evaluate: precision, recall, f1
        recall, precision, f1 = get_scores(answer, gpt3_answer)

        all_metrics[id]["precision"] = precision
        all_metrics[id]["recall"] = recall
        all_metrics[id]["f1"] = f1
        all_metrics[id]["BLEU-1"] = sentence_bleu(
            [answer], gpt3_answer, weights=(1, 0, 0, 0)
        )
        all_metrics[id]["BLEU-2"] = sentence_bleu(
            [answer], gpt3_answer, weights=(0, 1, 0, 0)
        )
        all_metrics[id]["SETENCE-BLEU"] = sentence_bleu([answer], gpt3_answer)

    macro_precision = 0
    macro_recall = 0

    mean_b1 = 0
    mean_b2 = 0
    mean_sb = 0

    for id in all_metrics:
        macro_precision += all_metrics[id]["precision"]
        macro_recall += all_metrics[id]["recall"]

        mean_b1 += all_metrics[id]["BLEU-1"]
        mean_b2 += all_metrics[id]["BLEU-2"]
        mean_sb += all_metrics[id]["SETENCE-BLEU"]

    macro_precision /= len(all_metrics)
    macro_recall /= len(all_metrics)
    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall)

    mean_b1 /= len(all_metrics)
    mean_b2 /= len(all_metrics)
    mean_sb /= len(all_metrics)

    print(
        f'{"davinci-002-dir":20} - {"-":<3} => MACRO-PRECISION: {macro_precision:.4f} MACRO-RECALL: {macro_recall:.4f} MACRO-F1: {macro_f1:.4f} MEAN-BLEU-1: {mean_b1:.4f} MEAN-BLEU-2: {mean_b2:.4f} SETENCE-BLEU: {mean_sb:.4f}'
    )


def main():
    qald = read_qald("data/qald_9_test.json")
    ids, questions, keywords, queries, answers = parse_qald(qald)

    # queries = {id: query.strip().replace('\n', '').replace('\t', '') for id, query in zip(ids, queries)}

    # evaluate_sparql_generation(queries)
    # evaluate_sparql_answers(ids, answers)
    evaluate_sparql_valid_answers(ids, answers)
    # evaluate_direct_answers(ids, answers)


if __name__ == "__main__":
    main()
