from networkx.algorithms.operators.product import tensor_product
import numpy as np

all_letters = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
n_letters = len(all_letters)

# Don't need this
LABELS = ['diagnose', 'drugname', 'usage', 'quantity', 'date','other']
# https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html

# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# Turn a line into a <line_length>,
def lineToTensor(line):
    # tensor = torch.zeros(len(line), dtype=torch.long)
    tensor = [0] * len(line)
    for li, letter in enumerate(line):
        tensor[li] = letterToIndex(letter)
    # tensor = torch.LongTensor(tensor, dtype=torch.long)
    return tensor

def expand(img):
    h, w, c = img.shape
    append_row = np.full((h//10, w, 3), 255, dtype=np.uint8)
    img = np.append(img, append_row, 0)
    h, w, c = img.shape
    append_col = np.full((h, w//10, 3), 255, dtype=np.uint8)
    img = np.append(img, append_col, 1)
    return img
