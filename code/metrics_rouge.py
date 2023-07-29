import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import numpy as np
import pandas as pd
import pickle
import seaborn as sns

sns.set_theme(style="white", palette="viridis")

import sys

sys.setrecursionlimit(5000)  # for rouge

# sparql
from SPARQLWrapper import SPARQLWrapper, JSON

# normalize answers
from dateutil.parser import parse
import string
from nltk.corpus import stopwords
from num2words import num2words
from unidecode import unidecode

### METRICS
from rouge import Rouge

### COSTUM
from dataset_parsing import read_qald, parse_qald

GPT3_TYPES = ["gpt3_davinci002", "gpt3_davinci003"]
GPT3_FT = "gpt3_davinci_ft"

FS_TYPES = ["", "-fs5"]
QUESTIONS_ID = [str(i) for i in range(1, 11)]


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


def evaluate_direct_answers(ids, questions, answers):
    """
    Evaluate direct answers

    :param ids: ids
    :param questions: questions
    :param answers: answers
    """

    print("*** Evaluating direct answers! ***")

    mean_em = 0
    mean_r1_r = 0
    mean_r1_p = 0
    mean_r1_f = 0

    # for answers bigger than 1 word
    long_counter = 0

    mean_r2_r = 0
    mean_r2_p = 0
    mean_r2_f = 0

    mean_rl_r = 0
    mean_rl_p = 0
    mean_rl_f = 0

    #### GPT3
    gpt_metrics = {id: {} for id in ids}

    # load GPT3 answers
    with open("output/gpt3/direct_answers.json", "r") as f:
        gpt3_direct_answers = json.load(f)

    # evaluate GPT3
    for i in range(len(questions)):
        # get GPT3 answer
        gpt3_answer = ", ".join(gpt3_direct_answers[str(ids[i])]).strip()
        answer = parse_quald_answers(answers[i])

        # Normalise answers
        gpt3_answer = normalise_answer(gpt3_answer)
        answer = normalise_answer(answer)

        # Prints
        # print(f"\n\nQ [{i}/{len(questions)}]: {questions[i]}")
        # print(f"A: {answer}")
        # print(f"GPT3: {gpt3_answer}\n")

        # EM
        gpt_metrics[ids[i]]["em"] = 1 if gpt3_answer == answer else 0
        mean_em += gpt_metrics[ids[i]]["em"]

        # ROUGE
        try:
            rouge = Rouge()
            rg = rouge.get_scores(gpt3_answer, answer, avg=True)
        except:
            print("oi?")

        mean_r1_r += rg["rouge-1"]["r"]
        mean_r1_p += rg["rouge-1"]["p"]
        mean_r1_f += rg["rouge-1"]["f"]

        gpt_metrics[ids[i]]["rouge-1"] = rg["rouge-1"]

        # it olny makes sense to calculate rouge-2 and rouge-l if the answer and gpt3_answer are longer than 1 word
        if len(answer.split()) > 1 and len(gpt3_answer.split()) > 1:
            long_counter += 1

            mean_r2_r += rg["rouge-2"]["r"]
            mean_r2_p += rg["rouge-2"]["p"]
            mean_r2_f += rg["rouge-2"]["f"]

            mean_rl_r += rg["rouge-l"]["r"]
            mean_rl_p += rg["rouge-l"]["p"]
            mean_rl_f += rg["rouge-l"]["f"]

            gpt_metrics[ids[i]]["rouge-2"] = rg["rouge-2"]
            gpt_metrics[ids[i]]["rouge-l"] = rg["rouge-l"]

        print(f"Q [{i}/{len(questions)}]:", end=" ")
        print(gpt_metrics[ids[i]])

    mean_em /= len(questions)

    mean_r1_r /= len(questions)
    mean_r1_p /= len(questions)
    mean_r1_f /= len(questions)

    mean_r2_r /= long_counter
    mean_r2_p /= long_counter
    mean_r2_f /= long_counter

    mean_rl_r /= long_counter
    mean_rl_p /= long_counter
    mean_rl_f /= long_counter

    # save GPT3 metrics
    with open("output/metrics/direct.json", "w", encoding="utf-8") as f:
        json.dump(gpt_metrics, f)

    print("GPT3 metrics for direct answers saved")
    print(f"EM: {mean_em}")
    print(
        f"There are {long_counter} of {len(questions)} answers with more than 1 word. Only these were used to calculate rouge-2 and rouge-l"
    )
    print(
        f"ROUGE-1 R: {mean_r1_r:<20} Rouge-2 R: {mean_r2_r:<20} Rouge-L R: {mean_rl_r}"
    )
    print(
        f"ROUGE-1 P: {mean_r1_p:<20} Rouge-2 P: {mean_r2_p:<20} Rouge-L P: {mean_rl_p}"
    )
    print(
        f"ROUGE-1 F: {mean_r1_f:<20} Rouge-2 F: {mean_r2_f:<20} Rouge-L F: {mean_rl_f}"
    )


def evaluate_sparql_generation(ids, queries):
    """
    Evaluate sparql generation

    :param ids: ids
    :param queries: queries
    """

    print("*** Evaluating sparql generation! ***")

    for gpt3_type in GPT3_TYPES:
        for fs in FS_TYPES:
            name = f"{gpt3_type}{fs}"

            print(f"Evaluating {name}!")

            gpt3_sparql = None

            # load GPT3 sparql
            with open(f"output/{name}/{name}-test-query_dict.pkl", "rb") as f:
                gpt3_sparql = pickle.load(f)

            #### GPT3
            gpt_metrics = {q: {id: {} for id in ids} for q in gpt3_sparql}

            # evaluate GPT3
            for question in gpt3_sparql:
                mean_em = 0
                mean_r1_r = 0
                mean_r1_p = 0
                mean_r1_f = 0

                # for answers bigger than 1 word
                long_counter = 0

                mean_r2_r = 0
                mean_r2_p = 0
                mean_r2_f = 0

                mean_rl_r = 0
                mean_rl_p = 0
                mean_rl_f = 0

                for i in range(len(ids)):
                    gpt3_query = (
                        gpt3_sparql[question][0][0]["text"]
                        .strip()
                        .replace("\n", "")
                        .replace("\t", "")
                        .replace("  ", " ")
                    )
                    query = queries[i].strip().replace("\n", "").replace("\t", "")
                    id = ids[i]

                    # print('==================')
                    # print(gpt3_query)
                    # print(query)

                    # EM
                    gpt_metrics[question][id]["em"] = 1 if gpt3_query == query else 0
                    mean_em += gpt_metrics[question][id]["em"]

                    # ROUGE
                    rouge = Rouge()

                    rg = rouge.get_scores(gpt3_query, query, avg=True)
                    mean_r1_r += rg["rouge-1"]["r"]
                    mean_r1_p += rg["rouge-1"]["p"]
                    mean_r1_f += rg["rouge-1"]["f"]

                    gpt_metrics[question][id]["rouge-1"] = rg["rouge-1"]

                    # it olny makes sense to calculate rouge-2 and rouge-l if the answer and gpt3_answer are longer than 1 word
                    if len(query.split()) > 1 and len(gpt3_query.split()) > 1:
                        long_counter += 1

                        mean_r2_r += rg["rouge-2"]["r"]
                        mean_r2_p += rg["rouge-2"]["p"]
                        mean_r2_f += rg["rouge-2"]["f"]

                        mean_rl_r += rg["rouge-l"]["r"]
                        mean_rl_p += rg["rouge-l"]["p"]
                        mean_rl_f += rg["rouge-l"]["f"]

                        gpt_metrics[question][id]["rouge-2"] = rg["rouge-2"]
                        gpt_metrics[question][id]["rouge-l"] = rg["rouge-l"]

                    print(f"ID [{i}/{len(ids)}]:", end=" ")
                    print(gpt_metrics[question][id])

                mean_em /= len(queries)

                mean_r1_r /= len(queries)
                mean_r1_p /= len(queries)
                mean_r1_f /= len(queries)

                mean_r2_r /= long_counter
                mean_r2_p /= long_counter
                mean_r2_f /= long_counter

                mean_rl_r /= long_counter
                mean_rl_p /= long_counter
                mean_rl_f /= long_counter

                print(f"Metrics for question form: {question}")
                print(f"\tEM: {mean_em}")
                print(
                    f"\tThere are {long_counter} of {len(ids)} answers with more than 1 word. Only these were used to calculate rouge-2 and rouge-l"
                )
                print(
                    f"\tROUGE-1 R: {mean_r1_r:<20} Rouge-2 R: {mean_r2_r:<20} Rouge-L R: {mean_rl_r}"
                )
                print(
                    f"\tROUGE-1 P: {mean_r1_p:<20} Rouge-2 P: {mean_r2_p:<20} Rouge-L P: {mean_rl_p}"
                )
                print(
                    f"\tROUGE-1 F: {mean_r1_f:<20} Rouge-2 F: {mean_r2_f:<20} Rouge-L F: {mean_rl_f}"
                )

                # save GPT3 metrics
                with open(
                    f"output/metrics/sparql_generation/{name}/{question}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(gpt_metrics[question], f)

                print(f"GPT3 metrics for {name}-{question} sparql generation saved")

    # Fine-tuned
    if GPT3_FT:
        name = GPT3_FT
        question = "Q1"  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it
        print(f"Evaluating {name}!")

        gpt3_sparql = None

        # load GPT3 sparql
        with open(f"output/{name}/{name}-test-query_dict.pkl", "rb") as f:
            gpt3_sparql = pickle.load(f)

        #### GPT3
        gpt_metrics = {id: {} for id in ids}

        # evaluate GPT3
        mean_em = 0
        mean_r1_r = 0
        mean_r1_p = 0
        mean_r1_f = 0

        # for answers bigger than 1 word
        long_counter = 0

        mean_r2_r = 0
        mean_r2_p = 0
        mean_r2_f = 0

        mean_rl_r = 0
        mean_rl_p = 0
        mean_rl_f = 0

        for i in range(len(ids)):
            gpt3_query = (
                gpt3_sparql[question][0][0]["text"]
                .strip()
                .replace("\n", "")
                .replace("\t", "")
                .replace("  ", " ")
            )
            query = queries[i].strip().replace("\n", "").replace("\t", "")
            id = ids[i]

            # print('==================')
            # print(gpt3_query)
            # print(query)

            # EM
            gpt_metrics[id]["em"] = 1 if gpt3_query == query else 0
            mean_em += gpt_metrics[id]["em"]

            # ROUGE
            rouge = Rouge()

            rg = rouge.get_scores(gpt3_query, query, avg=True)
            mean_r1_r += rg["rouge-1"]["r"]
            mean_r1_p += rg["rouge-1"]["p"]
            mean_r1_f += rg["rouge-1"]["f"]

            gpt_metrics[id]["rouge-1"] = rg["rouge-1"]

            # it olny makes sense to calculate rouge-2 and rouge-l if the answer and gpt3_answer are longer than 1 word
            if len(query.split()) > 1 and len(gpt3_query.split()) > 1:
                long_counter += 1

                mean_r2_r += rg["rouge-2"]["r"]
                mean_r2_p += rg["rouge-2"]["p"]
                mean_r2_f += rg["rouge-2"]["f"]

                mean_rl_r += rg["rouge-l"]["r"]
                mean_rl_p += rg["rouge-l"]["p"]
                mean_rl_f += rg["rouge-l"]["f"]

                gpt_metrics[id]["rouge-2"] = rg["rouge-2"]
                gpt_metrics[id]["rouge-l"] = rg["rouge-l"]

            print(f"ID [{i}/{len(ids)}]:", end=" ")
            print(gpt_metrics[id])

        mean_em /= len(queries)

        mean_r1_r /= len(queries)
        mean_r1_p /= len(queries)
        mean_r1_f /= len(queries)

        mean_r2_r /= long_counter
        mean_r2_p /= long_counter
        mean_r2_f /= long_counter

        mean_rl_r /= long_counter
        mean_rl_p /= long_counter
        mean_rl_f /= long_counter

        print(f"Metrics for question form: {question}")
        print(f"\tEM: {mean_em}")
        print(
            f"\tThere are {long_counter} of {len(ids)} answers with more than 1 word. Only these were used to calculate rouge-2 and rouge-l"
        )
        print(
            f"\tROUGE-1 R: {mean_r1_r:<20} Rouge-2 R: {mean_r2_r:<20} Rouge-L R: {mean_rl_r}"
        )
        print(
            f"\tROUGE-1 P: {mean_r1_p:<20} Rouge-2 P: {mean_r2_p:<20} Rouge-L P: {mean_rl_p}"
        )
        print(
            f"\tROUGE-1 F: {mean_r1_f:<20} Rouge-2 F: {mean_r2_f:<20} Rouge-L F: {mean_rl_f}"
        )

        # save GPT3 metrics
        with open(
            f"output/metrics/sparql_generation/{name}/ft.json", "w", encoding="utf-8"
        ) as f:
            json.dump(gpt_metrics, f)

        print(f"GPT3 metrics for {name} sparql generation saved")


def evaluate_sparql_answers(ids, answers):
    """
    Evaluate sparql answers

    :param ids: ids
    :param answers: answers
    """

    print("*** Evaluating sparql answers! ***")

    # indexing for ids
    answers_dict = {
        id: normalise_answer(parse_quald_answers(answer))
        for id, answer in zip(ids, answers)
    }

    for gpt3_type in GPT3_TYPES:
        for fs in FS_TYPES:
            name = f"{gpt3_type}{fs}"
            for q in QUESTIONS_ID:
                print(f"================== {name} Q{q} ==================")
                gpt3_sparql_answers = None

                # load GPT3 answers
                with open(
                    f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb"
                ) as f:
                    gpt3_sparql_answers = pickle.load(f)

                # print(gpt3_sparql_answers)

                mean_em = 0
                mean_r1_r = 0
                mean_r1_p = 0
                mean_r1_f = 0

                # for answers bigger than 1 word
                long_counter = 0
                all_counter = 0

                mean_r2_r = 0
                mean_r2_p = 0
                mean_r2_f = 0

                mean_rl_r = 0
                mean_rl_p = 0
                mean_rl_f = 0

                #### GPT3
                gpt_metrics = {type: {} for type in gpt3_sparql_answers}

                len_max = 80

                # evaluate GPT3
                for type in gpt3_sparql_answers:
                    for id in gpt3_sparql_answers[type]:
                        gpt_metrics[type][id] = {}

                        if type == "error" or type == "empty":
                            # If there is an error or it is empty, we don't want to evaluate it
                            continue

                        print(f"TYPE: {type:15}ID: {id}")

                        # get GPT3 answer
                        if len(gpt3_sparql_answers[type][id]) > len_max:
                            print(
                                f"GPT3 answer is longer than {len_max} answers. It was truncated since it has {len(gpt3_sparql_answers[type][id])} answers"
                            )
                            gpt3_answer = parse_quald_answers(
                                gpt3_sparql_answers[type][id][:len_max]
                            )
                        else:
                            gpt3_answer = parse_quald_answers(
                                gpt3_sparql_answers[type][id]
                            )
                        # answer = parse_quald_answers(answers_dict[id])

                        # Normalise answers
                        gpt3_answer = normalise_answer(gpt3_answer)
                        # answer = normalise_answer(answer)

                        answer = answers_dict[id]

                        # EM
                        gpt_metrics[type][id]["em"] = 1 if gpt3_answer == answer else 0
                        mean_em += gpt_metrics[type][id]["em"]

                        # ROUGE
                        # print(f'GPT3 answer: {gpt3_answer}')
                        if gpt3_answer == "" or gpt3_answer is None:
                            gpt_metrics[type][id]["rouge-1"] = {"r": 0, "p": 0, "f": 0}
                            continue

                        rouge = Rouge()

                        rg = rouge.get_scores(gpt3_answer, answer, avg=True)
                        mean_r1_r += rg["rouge-1"]["r"]
                        mean_r1_p += rg["rouge-1"]["p"]
                        mean_r1_f += rg["rouge-1"]["f"]

                        gpt_metrics[type][id]["rouge-1"] = rg["rouge-1"]

                        all_counter += 1

                        # it olny makes sense to calculate rouge-2 and rouge-l if the answer and gpt3_answer are longer than 1 word
                        if len(answer.split()) > 1 and len(gpt3_answer.split()) > 1:
                            long_counter += 1

                            mean_r2_r += rg["rouge-2"]["r"]
                            mean_r2_p += rg["rouge-2"]["p"]
                            mean_r2_f += rg["rouge-2"]["f"]

                            mean_rl_r += rg["rouge-l"]["r"]
                            mean_rl_p += rg["rouge-l"]["p"]
                            mean_rl_f += rg["rouge-l"]["f"]

                            gpt_metrics[type][id]["rouge-2"] = rg["rouge-2"]
                            gpt_metrics[type][id]["rouge-l"] = rg["rouge-l"]

                        # print(gpt_metrics[type][id])

                mean_em /= all_counter

                mean_r1_r /= all_counter
                mean_r1_p /= all_counter
                mean_r1_f /= all_counter

                mean_r2_r /= long_counter
                mean_r2_p /= long_counter
                mean_r2_f /= long_counter

                mean_rl_r /= long_counter
                mean_rl_p /= long_counter
                mean_rl_f /= long_counter

                # save GPT3 metrics
                with open(
                    f"output/metrics/sparql_answers/{name}/Q{q}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(gpt_metrics, f)

                print("GPT3 metrics for sparql answers saved")
                print(
                    f"Only {all_counter} of {len(ids)} answers were used (not empty or error)"
                )
                print(f"EM: {mean_em}")
                print(
                    f"There are {long_counter} of {all_counter} answers with more than 1 word. Only these were used to calculate rouge-2 and rouge-l"
                )
                print(
                    f"ROUGE-1 R: {mean_r1_r:<20} Rouge-2 R: {mean_r2_r:<20} Rouge-L R: {mean_rl_r}"
                )
                print(
                    f"ROUGE-1 P: {mean_r1_p:<20} Rouge-2 P: {mean_r2_p:<20} Rouge-L P: {mean_rl_p}"
                )
                print(
                    f"ROUGE-1 F: {mean_r1_f:<20} Rouge-2 F: {mean_r2_f:<20} Rouge-L F: {mean_rl_f}"
                )

    # Fine-tuned GPT3
    if GPT3_FT:
        name = GPT3_FT
        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        print(f"================== {name} ==================")
        gpt3_sparql_answers = None

        # load GPT3 answers
        with open(f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb") as f:
            gpt3_sparql_answers = pickle.load(f)

        # print(gpt3_sparql_answers)

        mean_em = 0
        mean_r1_r = 0
        mean_r1_p = 0
        mean_r1_f = 0

        # for answers bigger than 1 word
        long_counter = 0
        all_counter = 0

        mean_r2_r = 0
        mean_r2_p = 0
        mean_r2_f = 0

        mean_rl_r = 0
        mean_rl_p = 0
        mean_rl_f = 0

        #### GPT3
        gpt_metrics = {type: {} for type in gpt3_sparql_answers}

        len_max = 80

        # evaluate GPT3
        for type in gpt3_sparql_answers:
            for id in gpt3_sparql_answers[type]:
                gpt_metrics[type][id] = {}

                if type == "error" or type == "empty":
                    # If there is an error or it is empty, we don't want to evaluate it
                    continue

                print(f"TYPE: {type:15}ID: {id}")

                # get GPT3 answer
                if len(gpt3_sparql_answers[type][id]) > len_max:
                    print(
                        f"GPT3 answer is longer than {len_max} answers. It was truncated since it has {len(gpt3_sparql_answers[type][id])} answers"
                    )
                    gpt3_answer = parse_quald_answers(
                        gpt3_sparql_answers[type][id][:len_max]
                    )
                else:
                    gpt3_answer = parse_quald_answers(gpt3_sparql_answers[type][id])
                # answer = parse_quald_answers(answers_dict[id])

                # Normalise answers
                gpt3_answer = normalise_answer(gpt3_answer)
                # answer = normalise_answer(answer)

                answer = answers_dict[id]

                # EM
                gpt_metrics[type][id]["em"] = 1 if gpt3_answer == answer else 0
                mean_em += gpt_metrics[type][id]["em"]

                # ROUGE
                # print(f'GPT3 answer: {gpt3_answer}')
                if gpt3_answer == "" or gpt3_answer is None:
                    gpt_metrics[type][id]["rouge-1"] = {"r": 0, "p": 0, "f": 0}
                    continue

                rouge = Rouge()

                rg = rouge.get_scores(gpt3_answer, answer, avg=True)
                mean_r1_r += rg["rouge-1"]["r"]
                mean_r1_p += rg["rouge-1"]["p"]
                mean_r1_f += rg["rouge-1"]["f"]

                gpt_metrics[type][id]["rouge-1"] = rg["rouge-1"]

                all_counter += 1

                # it olny makes sense to calculate rouge-2 and rouge-l if the answer and gpt3_answer are longer than 1 word
                if len(answer.split()) > 1 and len(gpt3_answer.split()) > 1:
                    long_counter += 1

                    mean_r2_r += rg["rouge-2"]["r"]
                    mean_r2_p += rg["rouge-2"]["p"]
                    mean_r2_f += rg["rouge-2"]["f"]

                    mean_rl_r += rg["rouge-l"]["r"]
                    mean_rl_p += rg["rouge-l"]["p"]
                    mean_rl_f += rg["rouge-l"]["f"]

                    gpt_metrics[type][id]["rouge-2"] = rg["rouge-2"]
                    gpt_metrics[type][id]["rouge-l"] = rg["rouge-l"]

                # print(gpt_metrics[type][id])

        mean_em /= all_counter

        mean_r1_r /= all_counter
        mean_r1_p /= all_counter
        mean_r1_f /= all_counter

        mean_r2_r /= long_counter
        mean_r2_p /= long_counter
        mean_r2_f /= long_counter

        mean_rl_r /= long_counter
        mean_rl_p /= long_counter
        mean_rl_f /= long_counter

        # save GPT3 metrics
        with open(
            f"output/metrics/sparql_answers/{name}/ft.json", "w", encoding="utf-8"
        ) as f:
            json.dump(gpt_metrics, f)

        print("GPT3 metrics for sparql answers saved")
        print(
            f"Only {all_counter} of {len(ids)} answers were used (not empty or error)"
        )
        print(f"EM: {mean_em}")
        print(
            f"There are {long_counter} of {all_counter} answers with more than 1 word. Only these were used to calculate rouge-2 and rouge-l"
        )
        print(
            f"ROUGE-1 R: {mean_r1_r:<20} Rouge-2 R: {mean_r2_r:<20} Rouge-L R: {mean_rl_r}"
        )
        print(
            f"ROUGE-1 P: {mean_r1_p:<20} Rouge-2 P: {mean_r2_p:<20} Rouge-L P: {mean_rl_p}"
        )
        print(
            f"ROUGE-1 F: {mean_r1_f:<20} Rouge-2 F: {mean_r2_f:<20} Rouge-L F: {mean_rl_f}"
        )


def main():
    qald = read_qald("data/qald_9_test.json")
    ids, questions, keywords, queries, answers = parse_qald(qald)

    # evaluate_direct_answers(ids, questions, answers)
    evaluate_sparql_generation(ids, queries)
    evaluate_sparql_answers(ids, answers)


def combined_plot(df, y_name, g_type, extra_name=""):
    # sns.set_theme(palette=sns.color_palette("viridis", 3))
    # cp = sns.set_palette(sns.color_palette("viridis", 3))
    cp = sns.color_palette("viridis", 2)
    # sns.set_palette("viridis", 3)
    # sns.palplot(sns.color_palette("viridis", 3))

    plt.figure(figsize=(20, 5))
    plt.ylim(0, 1)
    axe = plt.subplot(111)
    axe = sns.boxplot(
        x="Question id", y=y_name, hue="Engine", data=df, ax=axe, palette=cp
    )

    # add hatch
    path_count = 0
    for patch in axe.patches:
        # print(i, type(patch), patch)
        try:
            patch.get_x()
        except:
            if path_count % 4 > 1:
                patch.set_hatch("///")
            path_count += 1

    # remove legend
    axe.get_legend().remove()

    # legend
    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    l1 = axe.legend(h[0:2], l[0:2], loc="upper left", prop={"size": 15})
    axe.add_artist(l1)

    # Add invisible data to add another legend
    n = [axe.bar(0, 0, color="gray"), axe.bar(0, 0, color="gray", hatch="///")]

    l2 = plt.legend(n, ["zero-shot", "few-shot"], loc="upper right", prop={"size": 15})
    axe.add_artist(l2)

    plt.ylabel("F1-Score")
    plt.xlabel("Question ID")
    plt.savefig(
        f"figures/{g_type}/{y_name}{extra_name}.png",
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.05,
    )
    plt.close()


def generate_sparql_plots():
    leg = [
        f'text-davinci-{aux.split("davinci")[-1].replace("_","")}' for aux in GPT3_TYPES
    ]

    data_r1 = []
    data_r2 = []
    data_rl = []

    for fs in FS_TYPES:
        for ind, engine in enumerate(GPT3_TYPES):
            name = f"{engine}{fs}"
            l_name = f"{leg[ind]}{fs}"
            em = []
            for q in QUESTIONS_ID:
                gpt3_sparql_metrics = None
                # print(f'Loading metrics for {name} Q{q}')

                # load GPT3 answers
                with open(
                    f"output/metrics/sparql_generation/{name}/Q{q}.json", "rb"
                ) as f:
                    gpt3_sparql_metrics = json.load(f)

                # get metrics
                for key in gpt3_sparql_metrics:
                    em.append(gpt3_sparql_metrics[key]["em"])
                    r1 = gpt3_sparql_metrics[key]["rouge-1"]["f"]

                    data_r1.append([f"Q{q}", l_name, r1])

                    if "rouge-2" in gpt3_sparql_metrics[key]:
                        r2 = gpt3_sparql_metrics[key]["rouge-2"]["f"]
                        rl = gpt3_sparql_metrics[key]["rouge-l"]["f"]
                        data_r2.append([f"Q{q}", l_name, r2])
                        data_rl.append([f"Q{q}", l_name, rl])

            # pieplot from binary array em
            # donut_plot([em.count(1), em.count(0)], labels=['EM', 'No EM'], path_figures=f'figures/{name}/sparql_generation_em.png')

    df_r1 = pd.DataFrame(data_r1, columns=["Question id", "Engine", "r1"])

    df_r2 = pd.DataFrame(data_r2, columns=["Question id", "Engine", "r2"])

    df_rl = pd.DataFrame(data_rl, columns=["Question id", "Engine", "rl"])

    # r1
    combined_plot(df_r1, "r1", "sparql_generation")

    # r2
    combined_plot(df_r2, "r2", "sparql_generation")

    # rl
    combined_plot(df_rl, "rl", "sparql_generation")

    ## Fine-tuned
    if GPT3_FT:
        name = GPT3_FT

        em = []
        r1 = []
        r2 = []
        rl = []

        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        gpt3_sparql_metrics = None
        # print(f'Loading metrics for {name} Q{q}')

        # load GPT3 answers
        with open(f"output/metrics/sparql_generation/{name}/ft.json", "rb") as f:
            gpt3_sparql_metrics = json.load(f)

        # get metrics
        for key in gpt3_sparql_metrics:
            em.append(gpt3_sparql_metrics[key]["em"])
            r1.append(gpt3_sparql_metrics[key]["rouge-1"]["f"])

            if "rouge-2" in gpt3_sparql_metrics[key]:
                r2.append(gpt3_sparql_metrics[key]["rouge-2"]["f"])
                rl.append(gpt3_sparql_metrics[key]["rouge-l"]["f"])

        # boxplot rouge for f1
        plt.figure()
        sns.boxplot(data=[r1, r2, rl])
        plt.xticks([0, 1, 2], ["ROUGE-1", "ROUGE-2", "ROUGE-L"])
        plt.ylabel("F1-Score")
        plt.savefig(
            "figures/sparql_generation/rouge_ft.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
        plt.close()


def generate_sparql_answer_plots():
    data_r1 = []
    data_r1_all = []
    data_r2 = []
    data_r2_all = []
    data_rl = []
    data_rl_all = []

    for fs in FS_TYPES:
        for engine in GPT3_TYPES:
            name = f"{engine}{fs}"
            em = []
            for q in QUESTIONS_ID:
                gpt3_sparql_metrics = None
                # print(f'Loading metrics for {name} Q{q}')

                # load GPT3 answers
                with open(f"output/metrics/sparql_answers/{name}/Q{q}.json", "rb") as f:
                    gpt3_sparql_metrics = json.load(f)

                for t in gpt3_sparql_metrics:
                    for id in gpt3_sparql_metrics[t]:
                        if gpt3_sparql_metrics[t][id] == {}:
                            data_r1_all.append([f"Q{q}", name, 0])
                            data_r2_all.append([f"Q{q}", name, 0])
                            data_rl_all.append([f"Q{q}", name, 0])
                            em.append(0)
                            continue

                        em.append(gpt3_sparql_metrics[t][id]["em"])
                        r1 = gpt3_sparql_metrics[t][id]["rouge-1"]["f"]

                        data_r1.append([f"Q{q}", name, r1])
                        data_r1_all.append([f"Q{q}", name, r1])

                        if "rouge-2" in gpt3_sparql_metrics[t][id]:
                            r2 = gpt3_sparql_metrics[t][id]["rouge-2"]["f"]
                            rl = gpt3_sparql_metrics[t][id]["rouge-l"]["f"]
                            data_r2.append([f"Q{q}", name, r2])
                            data_r2_all.append([f"Q{q}", name, r2])
                            data_rl.append([f"Q{q}", name, rl])
                            data_rl_all.append([f"Q{q}", name, r1])
                        else:
                            # if only 1 answer, it will count em as rouge-2 and rouge-l
                            data_r2_all.append([f"Q{q}", name, em[-1]])
                            data_rl_all.append([f"Q{q}", name, em[-1]])

            # pieplot from binary array em
            # donut_plot([em.count(1), em.count(0)], labels=['EM', 'No EM'], path_figures=f'figures/{name}/sparql_answers_em.png')

    df_r1 = pd.DataFrame(data_r1, columns=["Question id", "Engine", "r1"])
    df_r1_all = pd.DataFrame(data_r1_all, columns=["Question id", "Engine", "r1"])

    df_r2 = pd.DataFrame(data_r2, columns=["Question id", "Engine", "r2"])
    df_r2_all = pd.DataFrame(data_r2_all, columns=["Question id", "Engine", "r2"])

    df_rl = pd.DataFrame(data_rl, columns=["Question id", "Engine", "rl"])
    df_rl_all = pd.DataFrame(data_rl_all, columns=["Question id", "Engine", "rl"])

    # r1
    combined_plot(df_r1, "r1", "sparql_answers")
    combined_plot(df_r1_all, "r1", "sparql_answers", "-all")

    # r2
    combined_plot(df_r2, "r2", "sparql_answers")
    combined_plot(df_r2_all, "r2", "sparql_answers", "-all")

    # rl
    combined_plot(df_rl, "rl", "sparql_answers")
    combined_plot(df_rl_all, "rl", "sparql_answers", "-all")

    ## Fine-tuned
    if GPT3_FT:
        name = GPT3_FT

        em = []

        r1 = []
        r1_all = []
        r2 = []
        r2_all = []
        rl = []
        rl_all = []

        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        gpt3_sparql_metrics = None
        # print(f'Loading metrics for {name} Q{q}')

        # load GPT3 answers
        with open(f"output/metrics/sparql_answers/{name}/ft.json", "rb") as f:
            gpt3_sparql_metrics = json.load(f)

        # get metrics
        for t in gpt3_sparql_metrics:
            for id in gpt3_sparql_metrics[t]:
                if gpt3_sparql_metrics[t][id] == {}:
                    r1_all.append(0)
                    r2_all.append(0)
                    rl_all.append(0)
                    em.append(0)
                    continue

                em.append(gpt3_sparql_metrics[t][id]["em"])
                r1.append(gpt3_sparql_metrics[t][id]["rouge-1"]["f"])
                r1_all.append(gpt3_sparql_metrics[t][id]["rouge-1"]["f"])

                if "rouge-2" in gpt3_sparql_metrics[t][id]:
                    r2.append(gpt3_sparql_metrics[t][id]["rouge-2"]["f"])
                    r2_all.append(gpt3_sparql_metrics[t][id]["rouge-2"]["f"])
                    rl.append(gpt3_sparql_metrics[t][id]["rouge-l"]["f"])
                    rl_all.append(gpt3_sparql_metrics[t][id]["rouge-l"]["f"])
                else:
                    # if only 1 answer, it will count em as rouge-2 and rouge-l
                    r2_all.append(em[-1])
                    rl_all.append(em[-1])

        # boxplot rouge for f1
        plt.figure()
        sns.boxplot(data=[r1, r2, rl])
        plt.xticks([0, 1, 2], ["rouge-1", "rouge-2", "rouge-l"])
        plt.ylabel("F1 score")
        plt.savefig(
            "figures/sparql_answers/rouge_ft.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
        plt.close()

        # boxplot rouge for f1 all
        plt.figure()
        sns.boxplot(data=[r1_all, r2_all, rl_all])
        plt.xticks([0, 1, 2], ["rouge-1", "rouge-2", "rouge-l"])
        plt.ylabel("F1 score")
        plt.savefig(
            "figures/sparql_answers/rouge_ft-all.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
        plt.close()

    # generate table with mean and std
    # group by engine and question
    df_r1 = df_r1.groupby(["Engine", "Question id"]).agg(["mean", "std"])
    df_r2 = df_r2.groupby(["Engine", "Question id"]).agg(["mean", "std"])
    df_rl = df_rl.groupby(["Engine", "Question id"]).agg(["mean", "std"])

    # generate table
    test = df_r1.join(df_r2, lsuffix="_r1", rsuffix="_r2")
    test = test.join(df_rl, rsuffix="_rl")

    # add fine-tuned
    if GPT3_FT:
        test.loc[("text-davinci-ft", "-"), :] = [
            np.mean(r1),
            np.std(r1),
            np.mean(r2),
            np.std(r2),
            np.mean(rl),
            np.std(rl),
        ]

    # save test as latex
    test.to_latex(f"figures/sparql_answers.tex", float_format="%.3f")

    # generate table for all with mean and std
    # group by engine and question
    df_r1_all = df_r1_all.groupby(["Engine", "Question id"]).agg(["mean", "std"])
    df_r2_all = df_r2_all.groupby(["Engine", "Question id"]).agg(["mean", "std"])
    df_rl_all = df_rl_all.groupby(["Engine", "Question id"]).agg(["mean", "std"])

    # generate table
    test_all = df_r1_all.join(df_r2_all, lsuffix="_r1", rsuffix="_r2")
    test_all = test_all.join(df_rl_all, rsuffix="_rl")

    # add fine-tuned
    if GPT3_FT:
        test_all.loc[("text-davinci-ft", "-"), :] = [
            np.mean(r1_all),
            np.std(r1_all),
            np.mean(r2_all),
            np.std(r2_all),
            np.mean(rl_all),
            np.std(rl_all),
        ]

    # save test as latex
    test_all.to_latex(f"figures/sparql_answers_all.tex", float_format="%.3f")


def donut_plot(em, labels, path_figures, cmap=None):
    # custom autopct function
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct * total / 100.0))
            return "{p:.2f}%\n({v:d})".format(p=pct, v=val)

        return my_autopct

    # plot
    plt.figure(figsize=(10, 10))
    # donut chart
    if cmap is None:
        patches, texts, autotexts = plt.pie(
            [em.count(1), em.count(0)],
            labels=labels,
            autopct=make_autopct([em.count(1), em.count(0)]),
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
    else:
        patches, texts, autotexts = plt.pie(
            [em.count(1), em.count(0)],
            labels=labels,
            autopct=make_autopct([em.count(1), em.count(0)]),
            colors=cmap.colors,
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
    # autopct text position more to the edge
    for patch, txt in zip(patches, autotexts):
        # the angle at which the text is located
        ang = (patch.theta2 + patch.theta1) / 2.0
        # new coordinates of the text, 0.7 is the distance from the center
        x = patch.r * 0.8 * np.cos(ang * np.pi / 180)
        y = patch.r * 0.8 * np.sin(ang * np.pi / 180)
        txt.set_position((x, y))
    # autopct white
    for autotext in autotexts:
        autotext.set_color("white")
    # draw circle in the middle
    circle = plt.Circle((0, 0), 0.6, color="white")
    p = plt.gcf()
    p.gca().add_artist(circle)
    plt.savefig(path_figures, bbox_inches="tight", dpi=300, pad_inches=0.05)
    plt.close()


def generate_plots():
    ### Direct answers
    print("Evaluating Direct answers...")
    # load GPT3 metrics
    with open("output/metrics/direct.json", "r") as f:
        gpt_metrics = json.load(f)

    em = []

    r1 = []
    r2 = []
    rl = []

    # get metrics
    for key in gpt_metrics:
        em.append(gpt_metrics[key]["em"])
        r1.append(gpt_metrics[key]["rouge-1"]["f"])

        if "rouge-2" in gpt_metrics[key]:
            r2.append(gpt_metrics[key]["rouge-2"]["f"])
            rl.append(gpt_metrics[key]["rouge-l"]["f"])

    donut_plot(em, labels=["EM", "No EM"], path_figures="figures/direct/em.png")

    # boxplot rouge for f1
    plt.figure()
    sns.boxplot(data=[r1, r2, rl])
    plt.xticks([0, 1, 2], ["ROUGE-1", "ROUGE-2", "ROUGE-L"])
    plt.ylabel("F1-Score")
    plt.savefig(
        "figures/direct/rouge.png", bbox_inches="tight", dpi=300, pad_inches=0.05
    )
    plt.close()

    ### Sparql generation
    print("Evaluating Sparql generation...")
    generate_sparql_plots()

    ### Sparql answers
    print("Evaluating Sparql answers...")
    generate_sparql_answer_plots()


if __name__ == "__main__":
    font = {
        "family": "serif",
        #'weight' : 'bold',
        "size": 15,
    }
    matplotlib.rc("font", **font)
    main()
    generate_plots()

    # test string normalisation
    # s = '08/01/2020 was a good day to visit Monção, Portugal, with my 2 dogs.'
    # sn = normalise_answer(s)

    """
    rouge = Rouge()

    #q1 = "PREFIX dct: <http://purl.org/dc/terms/> PREFIX dbc: <http://dbpedia.org/resource/Category:> SELECT DISTINCT ?uri WHERE { ?uri dct:subject dbc:Assassins_of_Julius_Caesar }"

    #q2 = "PREFIX dct: <http://purl.org/dc/terms/> PREFIX dbc: <http://dbpedia.org/resource/Category:> SELECT DISTINCT ?uri WHERE { ?uri dct:asd dbc:xjjpd}"

    a = "Isto e um teste"

    b = "Isto também e um exercicio"

    rg1 = rouge.get_scores(b, a, avg=True)

    rg2 = rouge.get_scores(a, b, avg=True)


    print(rg1)
    print(rg2)
    """
