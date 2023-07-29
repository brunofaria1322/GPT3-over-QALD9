import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed
from dataset_parsing import read_qald, parse_qald


def get_opt_answer(
    question, model_name="facebook/opt-66b", expected_answer=None, seed=42
):
    """
    Test OPT

    :param question: question
    :param model_name: model name
    :param expected_answer: expected answer
    :param seed: seed
    :return: answer
    """
    # set hyperparameters
    set_seed(seed)
    if expected_answer is None:
        answer_length = 1000000
    else:
        answer_length = len(expected_answer) * 2
    # generation
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    prompt = question
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    generated_ids = model.generate(input_ids)
    x = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # generator = pipeline('question-answering', model=model_name, do_sample=True, max_length=answer_length)
    # return generator(question, max_length=answer_length)
    return x


def main():
    qald = read_qald("data/qald_9_test.json")
    questions, queries, answers = parse_qald(qald)

    assert len(questions) == len(queries) == len(answers)

    for i in range(len(questions)):
        print(f"Question ({i}/{len(questions)})")
        question = questions[i]
        # query = queries[i]
        answer = answers[i]

        opt_answer = get_opt_answer(
            # f'The SPARQL query for the question "{question}" over Wikidata is ',
            question,
            model_name="facebook/opt-350m",
            seed=SEED,
        )
        with open(f"output/opt/opt_answers.csv", "a") as f:
            f.write(f"{opt_answer[0]['generated_text']}\n")
    print(answer)
    print(opt_answer)


if __name__ == "__main__":
    SEED = 42
    main()
