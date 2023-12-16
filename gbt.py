import torch

# read the file
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

print("Length of the file: ", len(text))

#create a list with all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

for i in chars:
    print(i, end='')
print('\n',vocab_size)


#create a mapping from characters to integers
stoi = {}
itos = {}

# create the hashmaps
for i, ch in enumerate(chars):
    stoi[ch] = i
for i, ch in enumerate(chars):
    itos[i] = ch

#create a encode/decode function
def encode(phrase):
    encoded = []
    for i in range(len(phrase)):
        encoded.append(stoi[phrase[i]])

    return encoded

def decode(encoded):
    decoded = ""
    for i in encoded:
        decoded+=(itos[i])

    return decoded

print(encode("hii there"))
print(decode(encode("hii there")))

# encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)

# print(data[:1000])

# split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, 10% val
train_data = data[:n]
val_data = data[n:]

block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]

# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print("when input is ",context," the target is: ", target)

torch.manual_seed(1337) # ensures the reproducibility of random processes
batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generates a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)