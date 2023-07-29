import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from tqdm import tqdm, trange

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
# from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction


### COSTUM
from dataset_parsing import read_qald, parse_qald

GPT3_TYPES = ["gpt3_davinci002", "gpt3_davinci003"]
# GPT3_TYPES = ['gpt3_davinci003']

GPT3_FT = "gpt3_davinci_ft"

FS_TYPES = ["", "-fs5"]
QUESTIONS_ID = [str(i) for i in range(1, 11)]

LEN_MAX = 80


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


def evaluate_direct_answers(answers):
    """
    Evaluate direct answers

    :param answers: normalised answers
    :type answers: dict
    """

    print("*** Evaluating direct answers! ***")

    #### GPT3
    gpt_metrics = {}

    # check if data is already treated
    try:
        with open("output/gpt3/treated_direct_answers.json", "r") as f:
            gpt3_treated_direct_answers = json.load(f)

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

    # evaluate GPT3
    all_gpt3_answers = []
    all_answers = []

    for id in answers:
        all_gpt3_answers.append(gpt3_treated_direct_answers[id])
        all_answers.append(answers[id])
        gpt_metrics[id] = {}

        gpt3_answer = gpt3_treated_direct_answers[id]
        answer = answers[id]

        # EM
        gpt_metrics[id]["em"] = 1 if gpt3_answer == answer else 0

        gpt3_answer = gpt3_answer.split()
        answer = answer.split()

        # BLEU
        # Setence BLEU
        sb = sentence_bleu(
            [answer],
            gpt3_answer,
            auto_reweigh=1,
            smoothing_function=SmoothingFunction().method7,
        )

        gpt_metrics[id]["sbleu"] = sb

    # save GPT3 metrics
    with open("output/metrics/bleu_direct.json", "w", encoding="utf-8") as f:
        json.dump(gpt_metrics, f)

    # Corpus BLEU
    cb = corpus_bleu(
        all_answers,
        all_gpt3_answers,
        auto_reweigh=1,
        smoothing_function=SmoothingFunction().method7,
    )

    # save corpus metrics
    with open("output/metrics/corpus_bleu_direct.json", "w", encoding="utf-8") as f:
        json.dump(cb, f)

    print("GPT3 metrics for direct answers saved")


def evaluate_sparql_generation(queries):
    """
    Evaluate sparql generation

    :param queries: queries
    :type queries: dict
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
            gpt_metrics = {q: {id: {} for id in queries} for q in gpt3_sparql}
            corpus_metrics = {}

            # evaluate GPT3
            for question in gpt3_sparql:
                all_gpt3_queries = []
                all_queries = []

                for id in queries:
                    gpt3_query = (
                        gpt3_sparql[question][0][0]["text"]
                        .strip()
                        .replace("\n", "")
                        .replace("\t", "")
                        .replace("  ", " ")
                    )
                    query = queries[id]

                    # EM
                    gpt_metrics[question][id]["em"] = 1 if gpt3_query == query else 0

                    gpt3_query = gpt3_query.split()
                    query = query.split()

                    all_gpt3_queries.append(gpt3_query)
                    all_queries.append(query)

                    # BLEU
                    # Setence BLEU
                    sb = sentence_bleu(
                        [query],
                        gpt3_query,
                        auto_reweigh=1,
                        smoothing_function=SmoothingFunction().method7,
                    )

                    gpt_metrics[question][id]["sbleu"] = sb

                # Corpus BLEU
                cb = corpus_bleu(
                    all_queries,
                    all_gpt3_queries,
                    auto_reweigh=1,
                    smoothing_function=SmoothingFunction().method7,
                )

                corpus_metrics[question] = cb

                # save GPT3 metrics
                with open(
                    f"output/metrics/sparql_generation/{name}/bleu_{question}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(gpt_metrics[question], f, indent=4)

                print(f"GPT3 metrics for {name}-{question} sparql generation saved")

            # save corpus metrics
            with open(
                f"output/metrics/sparql_generation/{name}/corpus_bleu.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(corpus_metrics, f, indent=4)

            print(f"ALL GPT3 metrics for {name} sparql generation saved")

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
        gpt_metrics = {id: {} for id in queries}

        all_gpt3_queries = []
        all_queries = []

        for id in queries:
            gpt3_query = (
                gpt3_sparql[question][0][0]["text"]
                .strip()
                .replace("\n", "")
                .replace("\t", "")
                .replace("  ", " ")
            )
            query = queries[id]

            # EM
            gpt_metrics[id]["em"] = 1 if gpt3_query == query else 0

            gpt3_query = gpt3_query.split()
            query = query.split()

            all_gpt3_queries.append(gpt3_query)
            all_queries.append(query)

            # BLEU
            # Setence BLEU
            sb = sentence_bleu(
                [query],
                gpt3_query,
                auto_reweigh=1,
                smoothing_function=SmoothingFunction().method7,
            )

            gpt_metrics[id]["sbleu"] = sb

        # save GPT3 metrics
        with open(
            f"output/metrics/sparql_generation/{name}/bleu_ft.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(gpt_metrics, f, indent=4)

        print(f"GPT3 metrics for {name} sparql generation saved")

        # Corpus BLEU
        cb = corpus_bleu(
            all_queries,
            all_gpt3_queries,
            auto_reweigh=1,
            smoothing_function=SmoothingFunction().method7,
        )

        # save corpus metrics
        with open(
            f"output/metrics/sparql_generation/{name}/corpus_bleu_ft.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cb, f, indent=4)

        print(f"ALL GPT3 metrics for {name} sparql generation saved")


def evaluate_sparql_answers(answers):
    """
    Evaluate sparql answers

    :param answers: normalised answers7
    :type answers: dict
    """

    print("*** Evaluating sparql answers! ***")

    for gpt3_type in GPT3_TYPES:
        for fs in FS_TYPES:
            name = f"{gpt3_type}{fs}"
            corpus_metrics = {}
            for q in QUESTIONS_ID:
                print(f"================== {name} Q{q} ==================")
                gpt3_sparql_answers = None

                # check if there is a file with the sparql normalised answers

                try:
                    with open(
                        f"output\stats\{name}\{name}-test-Q{q}-treated_answers_dict.pkl",
                        "rb",
                    ) as f:
                        gpt3_sparql_treated_answers = pickle.load(f)

                except:
                    print(
                        f"No file with the sparql normalised answers for {name} Q{q}!"
                    )

                    # load GPT3 answers
                    with open(
                        f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb"
                    ) as f:
                        gpt3_sparql_answers = pickle.load(f)

                    gpt3_sparql_treated_answers = {
                        type: {} for type in gpt3_sparql_answers
                    }
                    for type in tqdm(gpt3_sparql_answers):
                        for id in tqdm(gpt3_sparql_answers[type]):
                            if type == "error" or type == "empty":
                                gpt3_sparql_treated_answers[type][id] = None
                                continue

                            # normalise answers
                            if len(gpt3_sparql_answers[type][id]) > LEN_MAX:
                                print(
                                    f"GPT3 answer is longer than {LEN_MAX} answers. It was truncated since it has {len(gpt3_sparql_answers[type][id])} answers"
                                )
                                gpt3_sparql_treated_answers[type][
                                    id
                                ] = normalise_answer(
                                    parse_quald_answers(
                                        gpt3_sparql_answers[type][id][:LEN_MAX]
                                    )
                                )
                            else:
                                gpt3_sparql_treated_answers[type][
                                    id
                                ] = normalise_answer(
                                    parse_quald_answers(gpt3_sparql_answers[type][id])
                                )

                    # save normalised answers
                    with open(
                        f"output\stats\{name}\{name}-test-Q{q}-treated_answers_dict.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(gpt3_sparql_treated_answers, f)

                all_gpt3_answers = []
                all_answers = []

                #### GPT3
                gpt_metrics = {type: {} for type in gpt3_sparql_treated_answers}

                # evaluate GPT3
                for type in gpt3_sparql_treated_answers:
                    for id in gpt3_sparql_treated_answers[type]:
                        gpt_metrics[type][id] = {}

                        # print(f'TYPE: {type:15}ID: {id}')

                        answer = answers[id]
                        gpt3_answer = gpt3_sparql_treated_answers[type][id]

                        # EM
                        gpt_metrics[type][id]["em"] = 1 if gpt3_answer == answer else 0

                        # BLEU
                        if gpt3_answer == "" or gpt3_answer is None:
                            gpt_metrics[type][id]["sbleu"] = 0
                            continue

                        gpt3_answer = gpt3_answer.split()
                        answer = answer.split()

                        all_gpt3_answers.append(gpt3_answer)
                        all_answers.append(answer)

                        # Setence BLEU
                        sb = sentence_bleu(
                            [answer],
                            gpt3_answer,
                            auto_reweigh=1,
                            smoothing_function=SmoothingFunction().method7,
                        )

                        gpt_metrics[type][id]["sbleu"] = sb

                # save GPT3 metrics
                with open(
                    f"output/metrics/sparql_answers/{name}/bleu_Q{q}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(gpt_metrics, f)

                print("GPT3 metrics for sparql answers saved")

                # Corpus BLEU
                cb = corpus_bleu(
                    all_answers,
                    all_gpt3_answers,
                    auto_reweigh=1,
                    smoothing_function=SmoothingFunction().method7,
                )

                corpus_metrics[q] = cb

            # save corpus metrics
            with open(
                f"output/metrics/sparql_answers/{name}/corpus_bleu.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(corpus_metrics, f)

    # Fine-tuned GPT3
    if GPT3_FT:
        all_gpt3_answers = []
        all_answers = []

        name = GPT3_FT
        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        print(f"================== {name} ==================")
        gpt3_sparql_answers = None

        # check if there is a file with the sparql normalised answers

        try:
            with open(
                f"output\stats\{name}\{name}-test-Q{q}-treated_answers_dict.pkl", "rb"
            ) as f:
                gpt3_sparql_treated_answers = pickle.load(f)

        except:
            print(f"No file with the sparql normalised answers for {name}!")

            # load GPT3 answers
            with open(
                f"output\stats\{name}\{name}-test-Q{q}-answers_dict.pkl", "rb"
            ) as f:
                gpt3_sparql_answers = pickle.load(f)

            gpt3_sparql_treated_answers = {type: {} for type in gpt3_sparql_answers}
            for type in gpt3_sparql_answers:
                for id in gpt3_sparql_answers[type]:
                    if type == "error" or type == "empty":
                        gpt3_sparql_treated_answers[type][id] = None
                        continue

                    # normalise answers
                    if len(gpt3_sparql_answers[type][id]) > LEN_MAX:
                        print(
                            f"GPT3 answer is longer than {LEN_MAX} answers. It was truncated since it has {len(gpt3_sparql_answers[type][id])} answers"
                        )
                        gpt3_sparql_treated_answers[type][id] = normalise_answer(
                            parse_quald_answers(gpt3_sparql_answers[type][id][:LEN_MAX])
                        )
                    else:
                        gpt3_sparql_treated_answers[type][id] = normalise_answer(
                            parse_quald_answers(gpt3_sparql_answers[type][id])
                        )

            # save normalised answers
            with open(
                f"output\stats\{name}\{name}-test-Q{q}-treated_answers_dict.pkl", "wb"
            ) as f:
                pickle.dump(gpt3_sparql_treated_answers, f)

        #### GPT3
        gpt_metrics = {type: {} for type in gpt3_sparql_treated_answers}

        # evaluate GPT3
        for type in gpt3_sparql_treated_answers:
            for id in gpt3_sparql_treated_answers[type]:
                gpt_metrics[type][id] = {}

                if type == "error" or type == "empty":
                    # If there is an error or it is empty, we don't want to evaluate it
                    continue

                # print(f'TYPE: {type:15}ID: {id}')

                gpt3_answer = gpt3_sparql_treated_answers[type][id]
                answer = answers[id]

                # EM
                gpt_metrics[type][id]["em"] = 1 if gpt3_answer == answer else 0

                # BLEU
                if gpt3_answer == "" or gpt3_answer is None:
                    gpt_metrics[type][id]["sbleu"] = 0
                    continue

                gpt3_answer = gpt3_answer.split()
                answer = answer.split()

                all_gpt3_answers.append(gpt3_answer)
                all_answers.append(answer)

                # Setence BLEU
                sb = sentence_bleu(
                    [answer],
                    gpt3_answer,
                    auto_reweigh=1,
                    smoothing_function=SmoothingFunction().method7,
                )

                gpt_metrics[type][id]["sbleu"] = sb

        # save GPT3 metrics
        with open(
            f"output/metrics/sparql_answers/{name}/bleu_ft.json", "w", encoding="utf-8"
        ) as f:
            json.dump(gpt_metrics, f)

        print("GPT3 metrics for sparql answers saved")

        # Corpus BLEU
        cb = corpus_bleu(
            all_answers,
            all_gpt3_answers,
            auto_reweigh=1,
            smoothing_function=SmoothingFunction().method7,
        )

        # save corpus metrics
        with open(
            f"output/metrics/sparql_answers/{name}/corpus_bleu_ft.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cb, f)


def main():
    qald = read_qald("data/qald_9_test.json")
    ids, questions, keywords, queries, answers = parse_qald(qald)

    # check if there is a file with the normalised answers
    try:
        with open("data/normalised_answers.json", "r") as f:
            normalised_answers = json.load(f)
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

    queries = {
        id: query.strip().replace("\n", "").replace("\t", "")
        for id, query in zip(ids, queries)
    }

    evaluate_direct_answers(normalised_answers)
    evaluate_sparql_generation(queries)
    evaluate_sparql_answers(normalised_answers)


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

    plt.ylabel("Score")
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

    data_sb = []

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
                    f"output/metrics/sparql_generation/{name}/bleu_Q{q}.json", "rb"
                ) as f:
                    gpt3_sparql_metrics = json.load(f)

                # get metrics
                for key in gpt3_sparql_metrics:
                    em.append(gpt3_sparql_metrics[key]["em"])
                    sb = gpt3_sparql_metrics[key]["sbleu"]

                    data_sb.append([f"Q{q}", l_name, sb])

            # pieplot from binary array em
            donut_plot(
                em,
                labels=["EM", "No EM"],
                path_figures=f"figures/{name}/sparql_generation_em_{q}.png",
            )

    df_b1 = pd.DataFrame(data_sb, columns=["Question id", "Engine", "sb"])

    # sb
    combined_plot(df_b1, "sb", "sparql_generation")

    ## Fine-tuned
    if GPT3_FT:
        name = GPT3_FT

        em = []
        sb = []

        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        gpt3_sparql_metrics = None
        # print(f'Loading metrics for {name} Q{q}')

        # load GPT3 answers
        with open(f"output/metrics/sparql_generation/{name}/bleu_ft.json", "rb") as f:
            gpt3_sparql_metrics = json.load(f)

        # get metrics
        for key in gpt3_sparql_metrics:
            em.append(gpt3_sparql_metrics[key]["em"])
            sb.append(gpt3_sparql_metrics[key]["sbleu"])

        # boxplot bleu
        plt.figure()
        sns.boxplot(data=[sb])
        plt.xticks([0], ["BLEU"])
        plt.ylabel("Score")
        plt.savefig(
            "figures/sparql_generation/bleu_ft.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
        plt.close()

        donut_plot(
            em,
            labels=["EM", "No EM"],
            path_figures=f"figures/{name}/sparql_generation_em.png",
        )


def generate_sparql_answer_plots():
    data_sb = []

    for fs in FS_TYPES:
        for engine in GPT3_TYPES:
            name = f"{engine}{fs}"
            for q in QUESTIONS_ID:
                gpt3_sparql_metrics = None
                # print(f'Loading metrics for {name} Q{q}')

                # load GPT3 answers
                with open(
                    f"output/metrics/sparql_answers/{name}/bleu_Q{q}.json", "rb"
                ) as f:
                    gpt3_sparql_metrics = json.load(f)

                em = []
                for t in gpt3_sparql_metrics:
                    for id in gpt3_sparql_metrics[t]:
                        if gpt3_sparql_metrics[t][id] == {}:
                            data_sb.append([f"Q{q}", name, 0])
                            em.append(0)
                            continue

                        em.append(gpt3_sparql_metrics[t][id]["em"])
                        sb = gpt3_sparql_metrics[t][id]["sbleu"]

                        data_sb.append([f"Q{q}", name, sb])

                # pieplot from binary array em
                donut_plot(
                    em,
                    labels=["EM", "No EM"],
                    path_figures=f"figures/{name}/sparql_answers_em_{q}.png",
                )
                # donut_plot([em.count(1), em.count(0)], labels=['EM', 'No EM'], path_figures=f'figures/{name}/sparql_answers_em.png')

    df_sb = pd.DataFrame(data_sb, columns=["Question id", "Engine", "sb"])

    # sb
    combined_plot(df_sb, "sb", "sparql_answers")

    ## Fine-tuned
    if GPT3_FT:
        name = GPT3_FT

        em = []

        sb = []

        q = 1  # because of the way the metrics are saved, we need to specify a question form, despite the fact that FT doesn't use it

        gpt3_sparql_metrics = None
        # print(f'Loading metrics for {name} Q{q}')

        # load GPT3 answers
        with open(f"output/metrics/sparql_answers/{name}/bleu_ft.json", "rb") as f:
            gpt3_sparql_metrics = json.load(f)

        # get metrics
        for t in gpt3_sparql_metrics:
            for id in gpt3_sparql_metrics[t]:
                if gpt3_sparql_metrics[t][id] == {}:
                    sb.append(0)
                    em.append(0)
                    continue

                em.append(gpt3_sparql_metrics[t][id]["em"])
                sb.append(gpt3_sparql_metrics[t][id]["sbleu"])

        # boxplot bleu
        plt.figure()
        sns.boxplot(data=[sb])
        plt.xticks([0], ["BLEU"])
        plt.ylabel("Score")
        plt.savefig(
            "figures/sparql_answers/sbleu_ft.png",
            bbox_inches="tight",
            dpi=300,
            pad_inches=0.05,
        )
        plt.close()

        # pieplot from binary array em
        donut_plot(
            em,
            labels=["EM", "No EM"],
            path_figures=f"figures/{name}/sparql_answers_em.png",
        )

    # generate table with mean and std
    # group by engine and question
    df_sb = df_sb.groupby(["Engine", "Question id"]).agg(["mean", "std"])

    # generate table
    test = df_sb

    # add fine-tuned
    if GPT3_FT:
        test.loc[("text-davinci-ft", "-"), :] = [np.mean(sb), np.std(sb)]

    # save test as latex
    test.to_latex(f"figures/sbleu_sparql_answers.tex", float_format="%.3f")


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
    with open("output/metrics/bleu_direct.json", "r") as f:
        gpt_metrics = json.load(f)

    em = []
    sb = []

    # get metrics
    for id in gpt_metrics:
        em.append(gpt_metrics[id]["em"])
        sb.append(gpt_metrics[id]["sbleu"])

    donut_plot(em, labels=["EM", "No EM"], path_figures="figures/direct/em.png")

    # boxplot bleu for f1
    plt.figure()
    sns.boxplot(data=[sb])
    plt.xticks([0], ["BLEU"])
    plt.ylabel("Score")
    plt.savefig(
        "figures/direct/bleu.png", bbox_inches="tight", dpi=300, pad_inches=0.05
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
    # main()
    # generate_plots()

    generate_sparql_plots()
