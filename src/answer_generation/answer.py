from src.answer_generation.answer_gen import *
from src.answer_generation.question_preprocess import *
from src.answer_generation.relevent_sent import *
from src.answer_generation.binary_question_processor import binary_question_transform, check_two_sentence_semantically_equal
import re

def format_answer(answer):
    answer = answer.replace('-LRB- ', '(')
    answer = answer.replace(' -RRB-', ')')
    answer = re.sub(r"\s([,;:\.\?!\'\"])", r"\1", answer)
    answer = answer[0].upper() + answer[1 :]
    return answer


def generate_answer(question, most_relevant_sentence):
    """
    :param question:  question in string
    :param most_relevant_sentence: most relevant sentence in string
    :return: answer in string
    """
    preprocess = q_preprocess(question)
    keywords, q_type, curr_q = preprocess.preprocess()
    print("answer type : {}".format(q_type))

    # Generate answer
    if q_type == "WH_ADV":
        pos_answer = answer_whadv(curr_q[0], most_relevant_sentence)
    elif q_type == "BINARY" or q_type == "EITHER_OR":
        if q_type == "BINARY":
            try:
                transformed_q = binary_question_transform(curr_q[0])
                if check_two_sentence_semantically_equal(transformed_q, most_relevant_sentence):
                    return "YES!"
                else:
                    return "NO!"
            except Exception:
                return most_relevant_sentence
        else:
            try:
                transformed_q1 = binary_question_transform(curr_q[0])
                transformed_q2 = binary_question_transform(curr_q[1])
                if check_two_sentence_semantically_equal(transformed_q1, most_relevant_sentence) and \
                   not check_two_sentence_semantically_equal(transformed_q2, most_relevant_sentence):
                    return "I believe it is the former"
                elif check_two_sentence_semantically_equal(transformed_q2, most_relevant_sentence) and \
                   not check_two_sentence_semantically_equal(transformed_q1, most_relevant_sentence):
                    return "I believe it is the latter"
                else:
                    return most_relevant_sentence
            except Exception:
                return most_relevant_sentence
    elif q_type == "WH_N":
        pos_answer = answer_whn(curr_q[0], most_relevant_sentence)
    elif q_type == "HOW":
        pos_answer = answer_howx(curr_q[0], most_relevant_sentence)
    elif q_type == "WHY":
        pos_answer = answer_why(curr_q[0], most_relevant_sentence)
    else:
        pos_answer = most_relevant_sentence

    # Process answer format
    answer = format_answer(pos_answer)
    return answer

if __name__ == "__main__":
    filename = sys.argv[1]
    q_filename = sys.argv[2]

    with open(q_filename) as qf:
        questions = qf.read().splitlines()
    for i in range(len(questions)):
        answer = ''
        # preprocess questions
        preprocess = q_preprocess(questions[i])
        keywords, q_type, curr_q = preprocess.preprocess()

        # TODO(xuanxuan): find most relevant sentence
        rel_sentence = ''
        
        # Generate answer
        if q_type == "WH_ADV":
            pos_answer = answer_whadv(curr_q[0], rel_sentence)
        elif q_type == "BINARY" or q_type == "EITHER_OR":
            pos_answer = "Not implemented yet!"
        elif q_type == "WH_N":
            pos_answer = answer_whn(curr_q[0], rel_sentence)
        elif q_type == "HOW":
            pos_answer = answer_howx(curr_q[0], rel_sentence)
        elif q_type == "WHY":
            pos_answer = answer_why(curr_q[0], rel_sentence)
        else:
            pos_answer = rel_sentence

        # Process answer format
        answer = format_answer(pos_answer)
        print(answer)