text = ""

with open("./input.txt", "r") as file:
	text = file.read()
	

print("Total chars: ", len(text))

vocabulary_list = sorted(list(set(text)))

print("Vocabulary: ", "".join(vocabulary_list))
print("Lenght of vocabulary: ", len(vocabulary_list))


stoi = { char: i for i, char in enumerate(vocabulary_list) }
itos = { i: char for i, char in enumerate(vocabulary_list) }

encode = lambda s: [stoi[char] for char in s]
decode = lambda l: "".join([itos[i] for i in l])

print(encode("test encoding"))
print(decode(encode("test encoding")))

import torch

tensor = torch.tensor(encode(text), dtype=torch.long)
print(tensor.shape, tensor.dtype)
print(tensor[:1000])

n = int(0.9 * len(tensor))
train_data, val_data = tensor[:n], tensor[n:]

block_size = 8
print(train_data[:block_size+1])


x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
	context = x[:t + 1]
	target = y[t]

	print(f"when input is {context} then target is {target}")

batch_size = 4 # how many independent sequences will we process in parallel
block_size = 8 # what is the maximum context length for predictions

def get_batch(split):
	data = train_data if split == "train" else val_data
	ix = torch.randint(len(data) - block_size, (batch_size,))
	print("ix", ix)

	x = torch.stack([data[i:i + block_size] for i in ix])
	y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

	return x, y

xb, yb = get_batch("train")
print("inputs:")

print(xb.shape)
print(xb)

print("targets:")
print(yb.shape)
print(yb)

print('------')

for b in range(batch_size):
	for t in range(block_size):
		context = xb[b, :t + 1]
		target = yb[b, t]

		print(f"when input is {context} then target is {target}")


