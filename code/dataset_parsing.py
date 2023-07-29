import json


def parse_qald(qald):
    """
    Parse QALD data

    :param qald: QALD data
    :return: parsed QALD data
    """
    questions = []
    ids = []
    queries = []
    keywords = []
    answers = []
    for element in qald["questions"]:
        # get question id
        ids.append(element["id"])

        # get only english questions
        for en_question in element["question"]:
            if en_question["language"] == "en":
                break
        assert (
            len(element["answers"]) == 1
        ), "More than one element for the key 'answers'"
        # The answer has a lot of problems and different formats
        try:
            if "boolean" in element["answers"][0]:
                # get the answer from the boolean
                answers.append([element["answers"][0]["boolean"]])
            elif "bindings" in element["answers"][0]["results"]:
                # get vars
                try:
                    vars = element["answers"][0]["head"]["vars"]
                except Exception as e:
                    assert False, f"No vars in query: {element['answers'][0]}\n{e}"
                # check the kind of answer
                bindings = element["answers"][0]["results"]["bindings"]
                for key in vars:  # ["uri","result","c","resultCnt","string","date"]:
                    if key in bindings[0]:
                        # get the answer(s) from the bindings
                        answers.append([binding[key]["value"] for binding in bindings])
                        break
            else:
                assert False, f"No key found for question {en_question['string']}"
        except Exception as e:
            print("No answer found -- ", e)
            continue
        # get question
        questions.append(en_question["string"])
        keywords.append(en_question["keywords"])
        # get sparql query
        queries.append(element["query"]["sparql"])
    return ids, questions, keywords, queries, answers


def read_qald(path):
    """
    Read QALD

    :param path: path to file
    :return: QALD data
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    qald = read_qald("data/qald_9_train.json")
    ids, questions, keywords, queries, answers = parse_qald(qald)


if __name__ == "__main__":
    main()
