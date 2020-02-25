from nltk import sent_tokenize
import re
import sys

def remove_ref(corpus):
    """
    Helper function to remove non-asii chars and the references/external links/notes at the end of file
    Arg: text(str) - the article to be parsed
    Return: a string of the text without references 
    """
    # remove non-ascii char
    corpus = re.sub(r'[^\x00-\x7f]',r'', corpus)
    # remove references at the end of file
    del_list = ['References', 'See also', 'Notes', 'External Links']
    del_pos = sys.maxsize
    for word in del_list:
        tmp = re.search(r'\n+%s\n+' % re.escape(word), corpus, re.IGNORECASE)
        if tmp is None:
            continue
        del_pos = min(del_pos, tmp.start())
    if del_pos != sys.maxsize:
        corpus = corpus[:del_pos]
    return corpus

def file_to_sentence(file_path):
    """
    Parse the file into a list of useful sentences 
    Arg: file_path(str) - the path of the file needs to be parsed
    Return: a list of sentences(string) 
    """
    with open(file_path) as f:
        corpus = f.read()

    # remove references at the end of file
    corpus = remove_ref(corpus)

    # remove subtitles
    corpus = re.sub(r'\n*[\w+\s+\-*]+\n+', r' ', corpus).strip()

    # tokenize to sentences 
    output = sent_tokenize(corpus)
    return output

def file_to_paragraph(file_path):
    """
    Parse the file into a list of useful paragraphs 
    Arg: file_path(str) - the path of the file needs to be parsed
    Return: a list of paragraphs(str) 
    """
    with open(file_path) as f:
        corpus = f.read()

    # remove references at the end of file
    corpus = remove_ref(corpus)

    # remove subtitles
    corpus = re.sub(r'\n*[\w+\s+\-*]+\n+', r'\n', corpus).strip()

    output = corpus.split('\n')
    return output
    

    
def text_to_sentence(text):
    """
    Parse the processed text without subtitles/references into a list of useful sentences 
    Arg: text(str) - the origianl text
    Return: a list of sentences(string) 
    """
    # remove non-ascii char
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    return sent_tokenize(text)

def file_rmextra(file_path):
    """
    Remove the extra contents in the file 
    Arg: file_path(str) - the path of the file needs to be parsed
    Return: the parsed file as a string 
    """
    with open(file_path) as f:
        corpus = f.read()
    # remove references at the end of file
    corpus = remove_ref(corpus)

    # remove subtitles
    corpus = re.sub(r'\n*[\w+\s+\-*]+\n+', r'\n', corpus).strip()

    return corpus
# for test
# if __name__ == "__main__":
#     t = file_to_paragraph('./set1/a1.txt')[3]
#     print(text_to_sentence(t))