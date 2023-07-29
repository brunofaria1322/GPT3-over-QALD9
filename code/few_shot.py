import json

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

    except ValueError:
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
                print(f"No label found for {answers[i]}")
                res = answers[i].split("/")[-1]
                res = res.replace("_", " ")
                answers[i] = res.strip()

    return ", ".join(answers)


def evaluate_direct_answers(ids, questions, answers):
    """
    Evaluate direct answers

    :param ids: ids
    :param questions: questions
    :param answers: answers
    """

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
    with open("output/gpt/direct_answers.json", "r") as f:
        gpt3_direct_answers = json.load(f)

    # evaluate GPT3
    for i in range(len(questions)):
        # get GPT3 answer
        gpt3_answer = ", ".join(gpt3_direct_answers[str(ids[i])]).strip()
        answer = parse_quald_answers(answers[i])

        # Normalize answers
        # https://www.geeksforgeeks.org/normalizing-textual-data-with-python/

        # replace date
        answer = " ".join([parse_date(word) for word in answer.split()])
        gpt3_answer = " ".join([parse_date(word) for word in gpt3_answer.split()])

        # remove punctuation and lower
        answer = answer.translate(str.maketrans("", "", string.punctuation)).lower()
        gpt3_answer = gpt3_answer.translate(
            str.maketrans("", "", string.punctuation)
        ).lower()

        # remove accents
        answer = unidecode(answer)
        gpt3_answer = unidecode(gpt3_answer)

        # remove stopwords
        answer = " ".join(
            [word for word in answer.split() if word not in stopwords.words("english")]
        )
        gpt3_answer = " ".join(
            [
                word
                for word in gpt3_answer.split()
                if word not in stopwords.words("english")
            ]
        )

        # replace numbers with words
        answer = " ".join(
            [
                num2words(word).replace(",", "") if word.isdigit() else word
                for word in answer.split()
            ]
        )
        gpt3_answer = " ".join(
            [
                num2words(word).replace(",", "") if word.isdigit() else word
                for word in gpt3_answer.split()
            ]
        )

        # Prints
        print(f"\n\nQ [{i}/{len(questions)}]: {questions[i]}")
        print(f"A: {answer}")
        print(f"GPT3: {gpt3_answer}\n")

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

        # print(gpt_metrics[ids[i]])

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
    with open("output/gpt/direct_metrics.json", "w", encoding="utf-8") as f:
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

    return mean_em, mean_r1_r, mean_r1_p, mean_r1_f


def evaluate_sparql_answers(ids, questions, answers):
    """
    Evaluate sparql answers

    :param ids: ids
    :param questions: questions
    :param answers: answers
    """

    mean_em = 0
    mean_r1_r = 0
    mean_r1_p = 0
    mean_r1_f = 0

    #### GPT3
    gpt_metrics = {id: {} for id in ids}

    # load GPT3 answers
    with open("output/gpt/sparql_answers.json", "r") as f:
        gpt3_sparql_answers = json.load(f)

    # evaluate GPT3
    for i in range(len(questions)):
        # get GPT3 answer
        gpt3_answer = gpt3_sparql_answers[str(ids[i])][0].strip()
        answer = str(answers[i][0]).strip()

        # remove \n
        answer = answer.replace("\n", "")
        gpt3_answer = gpt3_answer.replace("\n", "")

        # EM
        gpt_metrics[ids[i]]["em"] = 1 if gpt3_answer == answer else 0
        mean_em += gpt_metrics[ids[i]]["em"]

        # ROUGE
        rouge = Rouge()

        rg = rouge.get_scores(gpt3_answer, answer, avg=True)
        mean_r1_r += rg["rouge-1"]["r"]
        mean_r1_p += rg["rouge-1"]["p"]
        mean_r1_f += rg["rouge-1"]["f"]

        mean_r2_r += rg["rouge-2"]["r"]
        mean_r2_p += rg["rouge-2"]["p"]
        mean_r2_f += rg["rouge-2"]["f"]

        mean_rl_r += rg["rouge-l"]["r"]
        mean_rl_p += rg["rouge-l"]["p"]
        mean_rl_f += rg["rouge-l"]["f"]

        gpt_metrics[ids[i]]["rouge-1"] = rg["rouge-1"]
        gpt_metrics[ids[i]]["rouge-2"] = rg["rouge-2"]
        gpt_metrics[ids[i]]["rouge-l"] = rg["rouge-l"]

        # print(gpt_metrics[ids[i]])

    mean_em /= len(questions)

    mean_r1_r /= len(questions)
    mean_r1_p /= len(questions)
    mean_r1_f /= len(questions)

    mean_r2_r /= len(questions)
    mean_r2_p /= len(questions)
    mean_r2_f /= len(questions)

    mean_rl_r /= len(questions)
    mean_rl_p /= len(questions)
    mean_rl_f /= len(questions)

    # save GPT3 metrics
    with open("output/gpt/sparql_metrics.json", "w", encoding="utf-8") as f:
        json.dump(gpt_metrics, f)

    print("GPT3 metrics for sparql answers saved")
    print(f"EM: {mean_em}")
    print(
        f"ROUGE-1 R: {mean_r1_r:<20} Rouge-2 R: {mean_r2_r:<20} Rouge-L R: {mean_rl_r}"
    )
    print(
        f"ROUGE-1 P: {mean_r1_p:>20} Rouge-2 P: {mean_r2_p:<20} Rouge-L P: {mean_rl_p}"
    )
    print(
        f"ROUGE-1 F: {mean_r1_f:<20} Rouge-2 F: {mean_r2_f:<20} Rouge-L F: {mean_rl_f}"
    )


def main():
    qald = read_qald("data/qald_9_test.json")
    ids, questions, keywords, queries, answers = parse_qald(qald)

    evaluate_direct_answers(ids, questions, answers)
    # evaluate_sparql_answers(ids, questions, answers)

    """
    print(f"===========================\nTEST BLEU\n===========================")
    ### WITH PYTORCH (has not ROUGE)
    candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
    references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
    print(f"Bleu pytorch: {bleu_score(candidate_corpus, references_corpus)}")
    a = ['My full pytorch test','Another Sentence']
    b = ['My full pytorch test','Completely Different']
    ### WITH BLEU.PY 
    # a: Ref, b: Hypothesis
    # same result as pytorch
    print(f"Bleu like with pytorch example: {BLEU.list_bleu(a, b)}")
    # One sentence
    a = ['My full pytorch test Another Sentence']
    b = ['My full pytorch test Completely Different']
    print(f"Bleu - one sentence: {BLEU.list_bleu(a, b)}")
    print(f"===========================\nTEST ROUGE\n===========================")
    print(ROUGE.Rouge().get_scores(a, b))
    """


if __name__ == "__main__":
    main()
