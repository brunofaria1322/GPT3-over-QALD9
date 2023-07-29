import dataset_parsing as dp
import json

from query_checker import query_checker


def fine_tunning_parsing(path, questions, answers):
    """
    Write questions and answers to jsonl file for fine tunning

    path: path to jsonl file
    questions: questions
    answers: answers (can be direct answer or query)
    """
    with open(path, "w") as f:
        for i in range(len(questions)):
            qi = questions[i]
            ai = answers[i]
            try:
                key, val = query_checker(ai)
            except TypeError:
                key, val = "error", None
            if key == "error" or key == "empty":
                print(f"Error: {val}")
                continue
            f.write(json.dumps({"prompt": qi, "completion": ai + "\n<EOQ>\n"}) + "\n")


def main():
    qald = dp.read_qald("data/qald_9_train.json")
    ids, questions, keywords, queries, answers = dp.parse_qald(qald)
    fine_tunning_parsing("model/ft-qald9_train.jsonl", questions, queries)


if __name__ == "__main__":
    main()
