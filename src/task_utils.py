import re


def yesno(x):
    if x:
        return "yes"
    else:
        return "no"


class PIQA:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['goal']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        if -1 in examples["label"]:  # test set
            return [""] * len(examples["label"])
        else:
            gt_tuples = [("sol{}".format(label + 1), idx)
                         for idx, label in enumerate(examples['label'])]
            return [examples[k][i] for k, i in gt_tuples]


class HellaSwag:
    @classmethod
    def preprocess(cls, text):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    def get_context(self, examples):
        ctx_zip = zip(examples["activity_label"], examples["ctx_a"],
                      examples["ctx_b"])
        return [
            self.preprocess(a + ": " + b + " " + c.capitalize())
            for a, b, c in ctx_zip
        ]

    def get_target(self, examples):
        labels = examples["label"]
        endings = examples["endings"]
        targets = []
        for idx, label in enumerate(labels):
            target = '' if label == '' else endings[idx][int(label)]
            targets.append(self.preprocess(target))
        return targets


class OpenBookQA:
    def get_context(self, examples):
        return examples['question_stem']

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        targets = []
        for choice, answer in zip(choices, answers):
            answer = ord(answer.strip()) - ord('A')
            targets.append(choice['text'][answer])
        return targets


class ARC:
    def __init__(self):
        self._template = "Question: {}\nAnswer:"

    def get_context(self, examples):
        ctx = examples['question']
        return [self._template.format(c) for c in ctx]

    def get_target(self, examples):
        choices = examples['choices']
        answers = examples['answerKey']
        num_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E"}
        for idx, answer in enumerate(answers):
            answer = num_to_letter.get(answer, answer)
            answer = ord(answer) - ord("A")
            answers[idx] = choices[idx]["text"][answer]
        return answers


class RACE:
    @classmethod
    def doc_to_text(cls, article, question):
        text = "Article: " + article + "\n\n"
        text += "Question: " + question + "\n\n"
        text += "Answer:"
        return text

    def get_context(self, examples):
        return [
            self.doc_to_text(article, question) for article, question in zip(
                examples["article"], examples["question"])
        ]

    def get_target(self, examples):
        answers = examples['answer']
        options = examples['options']
        for idx, answer in enumerate(answers):
            answers[idx] = options[idx][ord(answer) - ord("A")]
        return answers


class SciQ:
    def __init__(self):
        self._template = "{}\nQuestion: {}\nAnswer:"

    def get_context(self, examples):
        sources = examples['support']
        queries = examples['question']
        return [self._template.format(s, q) for s, q in zip(sources, queries)]

    def get_target(self, examples):
        return examples['correct_answer']


class WebQs:
    def get_context(self, examples):
        return [
            "Question: " + question + "\nAnswer:"
            for question in examples["question"]
        ]

    def get_target(self, examples):
        return [" " + answers[0] for answers in examples["answers"]]


class Alpaca:
    def get_context(self, examples):
        return examples["text"]

    def get_target(self, examples):
        return examples["output"]


class StoryCloze:
    pass


class Copa:
    # from superglue
    @classmethod
    def doc_to_text(cls, question, premise):
        connector = {
            "cause": "because",
            "effect": "therefore",
        }[question]
        return premise.strip()[:-1] + f" {connector}"

    @classmethod
    def convert_choice(cls, choice):
        return choice[0].lower() + choice[1:]

    def get_context(self, examples):
        return [
            self.doc_to_text(question=question,
                             premise=premise) for question, premise in zip(
                                 examples["question"], examples["premise"])
        ]

    def get_target(self, examples):
        return [
            " " + self.convert_choice(choice1 if label == 0 else choice2)
            for choice1, choice2, label in zip(
                examples["choice1"], examples["choice2"], examples["label"])
        ]


class TriviaQA:
    def get_context(self, examples):
        return [
            f"Question: {question}\nAnswer:"
            for question in examples["question"]
        ]

    def get_target(self, examples):
        return [" " + answer["value"] for answer in examples["answer"]]


class BoolQ:
    def get_context(self, examples):
        return [
            f"{passage}\nQuestion: {question}?\nAnswer:" for passage, question
            in zip(examples["passage"], examples["question"])
        ]

    def get_target(self, examples):
        return [" " + yesno(label) for label in examples["label"]]


class StoryCloze:
    def get_context(self, examples):
        return [
            " ".join([inp1, inp2, inp3, inp4]) for inp1, inp2, inp3, inp4 in
            zip(examples["input_sentence_1"], examples["input_sentence_2"],
                examples["input_sentence_3"], examples["input_sentence_4"])
        ]
        
    def get_target(self, examples):
        clozes = [[s1, s2] for s1, s2 in zip(examples["sentence_quiz1"], examples["sentence_quiz2"])]

        return [" " + cloze[end - 1] for cloze, end in zip(clozes, examples["answer_right_ending"])]


task_dict = {
    "piqa": PIQA(),
    "hellaswag": HellaSwag(),
    "openbookqa": OpenBookQA(),
    "arc_easy": ARC(),
    "arc_challenge": ARC(),
    "sciq": SciQ(),
    "web_questions": WebQs(),
    "race": RACE(),
    "alpaca": Alpaca(),
    "copa": Copa(),
    "trivia_qa": TriviaQA(),
    "boolq": BoolQ(),
    "story_cloze": StoryCloze(),
}


def map_dataset_name_and_config(args):
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    if args.dataset_name == 'arc_easy':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Easy'
    elif args.dataset_name == 'arc_challenge':
        dataset_name = 'ai2_arc'
        dataset_config_name = 'ARC-Challenge'
    elif args.dataset_name == 'race':
        dataset_config_name = 'high'
    elif args.dataset_name == "alpaca":
        dataset_name = "tatsu-lab/alpaca"
    elif args.dataset_name == "wikitext":
        dataset_name = "wikitext"
        dataset_config_name = "wikitext-2-raw-v1"
    elif args.dataset_name == "copa":
        dataset_name = "super_glue"
        dataset_config_name = "copa"
    elif args.dataset_name == "boolq":
        dataset_name = "super_glue"
        dataset_config_name = "boolq"
    elif args.dataset_name == "trivia_qa":
        dataset_config_name = "unfiltered.nocontext"
    elif args.dataset_name == "story_cloze":
        dataset_name = "story_cloze"
        dataset_config_name = "2016"
    return dataset_name, dataset_config_name


LM_EVAL_TASK_NAME_MAPPING = {"web_questions": "webqs"}
